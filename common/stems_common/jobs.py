"""The contract between the API and a worker: three JSON files in storage per job.

    jobs/{id}/source        the upload, as uploaded                      written by the API
    jobs/{id}/spec.json     what to do                                   written by the API
    jobs/{id}/status.json   where the worker is, with its event history  rewritten by the worker
    jobs/{id}/result.json   the outcome; its presence ends the job       written by the worker
    jobs/{id}/stems/...     the audio, one waveform-peaks file per stem  written by the worker
    jobs/{id}/stems.zip     every stem in one download                   written by the worker

The API and the worker never call each other (PLAN.md section 2), so these files are the
whole interface. Each has a JSON Schema in ``stems_common/schemas``, and both sides go
through the ``validate_*`` functions here: the writer before it writes, the reader after
it reads. Besides the schema they check what a schema cannot: that timestamps parse, that
every key a document names lies inside its own job's folder, and that events are numbered
1, 2, 3 in order.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from stems_common import schemas
from stems_common.stems import PRODUCT_STEMS

CONTRACT_VERSION = 1
SPEC_SCHEMA, STATUS_SCHEMA, RESULT_SCHEMA = "job_spec", "job_status", "job_result"

# Outputs of the optional stages, delivered beside the stems.
EXTRA_STEMS: tuple[str, ...] = ("lead_vocals", "backing_vocals", "vocals_dry")
DELIVERABLE_STEMS: tuple[str, ...] = (*PRODUCT_STEMS, *EXTRA_STEMS)
# The stages a worker reports, in order. `retrying` happens only when the gate asks.
WORKER_STAGES: tuple[str, ...] = (
    "queued",
    "transcoding",
    "analysing",
    "planning",
    "separating",
    "assessing",
    "retrying",
    "packaging",
    "ready",
    "failed",
)
# States a result may have. Each is also a status state; a status may be `running` too.
END_STATES: tuple[str, ...] = ("ready", "needs_review", "failed")
FILE_EXTENSIONS = {"wav": "wav", "flac": "flac", "mp3_320": "mp3", "mp3_192": "mp3"}
MAX_EVENTS = 500


class ContractError(ValueError):
    """A spec, status or result document is not what the contract says."""


# -- keys ----------------------------------------------------------------------------------


def check_job_id(job_id: str) -> str:
    """``job_id`` if it is a canonical UUID. Keys are built from it, so nothing else is let in."""
    try:
        if str(uuid.UUID(job_id)) == job_id:
            return job_id
    except (ValueError, AttributeError, TypeError):
        pass
    raise ContractError(f"not a job id: {job_id!r}")


def job_prefix(job_id: str) -> str:
    return f"jobs/{check_job_id(job_id)}"


def source_key(job_id: str) -> str:
    return f"{job_prefix(job_id)}/source"


def spec_key(job_id: str) -> str:
    return f"{job_prefix(job_id)}/spec.json"


def status_key(job_id: str) -> str:
    return f"{job_prefix(job_id)}/status.json"


def result_key(job_id: str) -> str:
    return f"{job_prefix(job_id)}/result.json"


def zip_key(job_id: str) -> str:
    return f"{job_prefix(job_id)}/stems.zip"


def _check_stem(name: str) -> str:
    if name not in DELIVERABLE_STEMS:
        raise ContractError(f"not a stem name: {name!r}")
    return name


def stem_key(job_id: str, name: str, fmt: str) -> str:
    if fmt not in FILE_EXTENSIONS:
        raise ContractError(f"not an output format: {fmt!r}")
    return f"{job_prefix(job_id)}/stems/{_check_stem(name)}.{FILE_EXTENSIONS[fmt]}"


def peaks_key(job_id: str, name: str) -> str:
    return f"{job_prefix(job_id)}/stems/{_check_stem(name)}.peaks.json"


# -- time ----------------------------------------------------------------------------------


def timestamp(moment: datetime) -> str:
    """UTC, to the millisecond, as every document here writes times."""
    if moment.tzinfo is None:
        raise ContractError("timestamps must carry a time zone")
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


def parse_timestamp(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"not a timestamp: {text!r}") from exc
    if moment.tzinfo is None:
        raise ContractError(f"timestamp without a time zone: {text!r}")
    return moment.astimezone(UTC)


# -- validation ----------------------------------------------------------------------------


def _schema_problems(name: str, document: Any) -> list[str]:
    return schemas.errors(name, document)


def _time_problems(document: dict[str, Any], *fields: str) -> list[str]:
    problems = []
    for field in fields:
        try:
            parse_timestamp(document[field])
        except ContractError as exc:
            problems.append(f"{field}: {exc}")
    return problems


def _inside(job_id: str, key: Any) -> bool:
    return isinstance(key, str) and key.startswith(f"jobs/{job_id}/")


def _raise(what: str, problems: Sequence[str]) -> None:
    if problems:
        raise ContractError(f"not a valid {what}: " + "; ".join(problems))


def validate_spec(document: Any, job_id: str | None = None) -> dict[str, Any]:
    """``document`` if it is a valid spec (for ``job_id``, when given); else ``ContractError``."""
    problems = _schema_problems(SPEC_SCHEMA, document)
    if not problems:
        problems += _time_problems(document, "created_at")
        if job_id is not None and document["job_id"] != job_id:
            problems.append(f"job_id: is {document['job_id']!r}, expected {job_id!r}")
        if not _inside(document["job_id"], document["source"]["key"]):
            problems.append("source/key: is outside the job's own folder")
    _raise("job spec", problems)
    return document


def validate_status(document: Any, job_id: str | None = None) -> dict[str, Any]:
    problems = _schema_problems(STATUS_SCHEMA, document)
    if not problems:
        problems += _time_problems(document, "updated_at")
        if job_id is not None and document["job_id"] != job_id:
            problems.append(f"job_id: is {document['job_id']!r}, expected {job_id!r}")
        numbers = [event["seq"] for event in document["events"]]
        if numbers != list(range(1, len(numbers) + 1)):
            problems.append("events: must be numbered 1, 2, 3, ... in order")
        for index, event in enumerate(document["events"]):
            problems += [f"events/{index}/{p}" for p in _time_problems(event, "at")]
        ended = document["state"] != "running"
        if ended != (document["stage"] in ("ready", "failed")):
            problems.append("stage: only a job that is over is at ready or failed")
        if document["state"] == "failed" and not document.get("error"):
            problems.append("error: a failed job says why")
    _raise("job status", problems)
    return document


def validate_result(document: Any, job_id: str | None = None) -> dict[str, Any]:
    problems = _schema_problems(RESULT_SCHEMA, document)
    if not problems:
        problems += _time_problems(document, "finished_at")
        own = document["job_id"]
        if job_id is not None and own != job_id:
            problems.append(f"job_id: is {own!r}, expected {job_id!r}")
        names = [stem["name"] for stem in document["stems"]]
        if len(set(names)) != len(names):
            problems.append("stems: a stem is listed twice")
        for index, stem in enumerate(document["stems"]):
            for field in ("key", "peaks_key"):
                if not _inside(own, stem[field]):
                    problems.append(f"stems/{index}/{field}: is outside the job's own folder")
        if document.get("zip_key") is not None and not _inside(own, document["zip_key"]):
            problems.append("zip_key: is outside the job's own folder")
    _raise("job result", problems)
    return document


# -- building a spec -----------------------------------------------------------------------


def build_spec(
    job_id: str,
    *,
    created_at: datetime,
    stems: Sequence[str] | None = None,
    text: str | None = None,
    split_vocals: bool = False,
    dereverb: bool = False,
    fmt: str | None = None,
    quality_mode: str = "unspecified",
    source_bytes: int | None = None,
) -> dict[str, Any]:
    """The spec the API writes for a job. Stems are put in display order."""
    chosen = None if stems is None else [s for s in PRODUCT_STEMS if s in set(stems)]
    if stems is not None and len(chosen or []) != len(set(stems)):
        unknown = sorted(set(stems) - set(PRODUCT_STEMS))
        raise ContractError(f"not a valid job spec: unknown stems {unknown}")
    source: dict[str, Any] = {"key": source_key(job_id)}
    if source_bytes is not None:
        source["bytes"] = source_bytes
    spec = {
        "version": CONTRACT_VERSION,
        "job_id": job_id,
        "created_at": timestamp(created_at),
        "source": source,
        "request": {
            "stems": chosen,
            "text": text,
            "split_vocals": split_vocals,
            "dereverb": dereverb,
            "format": fmt,
            "quality_mode": quality_mode,
        },
    }
    return validate_spec(spec, job_id)
