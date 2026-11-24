"""The spec, status and result documents the API and the worker exchange."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta, timezone

import pytest

from stems_common import jobs, schemas
from stems_common.jobs import ContractError

JOB = "0b9c2f6e-58d4-4c58-9d3b-6f3d1c8a7e21"
OTHER = "7d1e9a34-2f0b-4b6c-8c59-1a2b3c4d5e6f"
NOW = datetime(2026, 11, 24, 12, 0, 0, 250000, tzinfo=UTC)
AT = "2026-11-24T12:00:00.250+00:00"


def spec() -> dict:
    return jobs.build_spec(JOB, created_at=NOW, stems=["drums", "vocals"], source_bytes=1234)


def status() -> dict:
    return {
        "version": 1,
        "job_id": JOB,
        "state": "running",
        "stage": "separating",
        "updated_at": AT,
        "progress": 0.5,
        "stems": {"vocals": {"state": "separating", "attempt": 0}, "drums": {"state": "waiting"}},
        "events": [
            {"seq": 1, "at": AT, "stage": "transcoding"},
            {"seq": 2, "at": AT, "stage": "separating", "stem": "vocals", "progress": 0.5},
        ],
        "error": None,
    }


def stem(name: str = "vocals") -> dict:
    return {
        "name": name,
        "key": jobs.stem_key(JOB, name, "flac"),
        "format": "flac",
        "bytes": 1000,
        "peaks_key": jobs.peaks_key(JOB, name),
        "model_ids": ["mel_roformer_kim_vocals"],
        "ensemble_mode": "single_best",
        "attempts": 1,
        "assessment": {"score": 3},
        "warning": None,
    }


def result() -> dict:
    return {
        "version": 1,
        "job_id": JOB,
        "state": "ready",
        "finished_at": AT,
        "error": None,
        "format": "flac",
        "device": "cpu",
        "source": {"duration_seconds": 12.0, "sample_rate_hz": 44100, "channels": 2},
        "stems": [stem("vocals"), stem("drums")],
        "zip_key": jobs.zip_key(JOB),
        "parsed_request": {"raw_text": "x"},
        "track_features": {},
        "plan": {},
        "decisions": [{"kind": "plan", "provider": "fallback", "fallback_reason": "no_key"}],
        "timings": [
            {
                "model_id": "mel_roformer_kim_vocals",
                "stage": "separation",
                "outputs": ["vocals"],
                "device": "cpu",
                "audio_seconds": 12.0,
                "wall_seconds": 1.5,
                "cached": False,
            }
        ],
    }


def failed() -> dict:
    return {
        "version": 1,
        "job_id": JOB,
        "state": "failed",
        "finished_at": AT,
        "error": "the upload is not decodable audio",
        "stems": [],
        "decisions": [],
        "timings": [],
    }


def changed(document: dict, path: str, value) -> dict:
    """A copy with the value at ``a/b/0/c`` replaced (``...`` as the value removes the key)."""
    copy_ = copy.deepcopy(document)
    *parents, last = path.split("/")
    target = copy_
    for part in parents:
        target = target[int(part) if isinstance(target, list) else part]
    key = int(last) if isinstance(target, list) else last
    if value is ...:
        del target[key]
    else:
        target[key] = value
    return copy_


# -- keys --------------------------------------------------------------------------------


def test_every_key_is_inside_the_jobs_own_folder() -> None:
    assert jobs.job_prefix(JOB) == f"jobs/{JOB}"
    assert jobs.source_key(JOB) == f"jobs/{JOB}/source"
    assert jobs.spec_key(JOB) == f"jobs/{JOB}/spec.json"
    assert jobs.status_key(JOB) == f"jobs/{JOB}/status.json"
    assert jobs.result_key(JOB) == f"jobs/{JOB}/result.json"
    assert jobs.zip_key(JOB) == f"jobs/{JOB}/stems.zip"
    assert jobs.stem_key(JOB, "vocals", "mp3_320") == f"jobs/{JOB}/stems/vocals.mp3"
    assert jobs.stem_key(JOB, "vocals_dry", "wav") == f"jobs/{JOB}/stems/vocals_dry.wav"
    assert jobs.peaks_key(JOB, "drums") == f"jobs/{JOB}/stems/drums.peaks.json"


def test_keys_pass_the_storage_layers_own_check() -> None:
    storage = pytest.importorskip("stems_api.storage")
    for key in (
        jobs.source_key(JOB),
        jobs.spec_key(JOB),
        jobs.stem_key(JOB, "backing_vocals", "flac"),
        jobs.peaks_key(JOB, "other"),
        jobs.zip_key(JOB),
    ):
        assert storage.check_key(key) == key
    assert storage.check_prefix(jobs.job_prefix(JOB) + "/") == jobs.job_prefix(JOB)


@pytest.mark.parametrize(
    "job_id",
    [
        "",
        "abc",
        "../" + JOB,
        JOB + "/..",
        JOB.upper(),
        JOB.replace("-", ""),
        "{" + JOB + "}",
        None,
        7,
    ],
)
def test_only_a_canonical_uuid_is_a_job_id(job_id) -> None:
    with pytest.raises(ContractError, match="not a job id"):
        jobs.job_prefix(job_id)
    with pytest.raises(ContractError):
        jobs.spec_key(job_id)


def test_stem_and_format_names_are_checked() -> None:
    for name in ("../spec", "vocals/x", "instrumental", "", "VOCALS"):
        with pytest.raises(ContractError, match="not a stem name"):
            jobs.stem_key(JOB, name, "wav")
        with pytest.raises(ContractError, match="not a stem name"):
            jobs.peaks_key(JOB, name)
    with pytest.raises(ContractError, match="not an output format"):
        jobs.stem_key(JOB, "vocals", "ogg")


def test_timestamps_are_utc_and_round_trip() -> None:
    assert jobs.timestamp(NOW) == AT
    tokyo = timezone(timedelta(hours=9))
    assert jobs.timestamp(NOW.astimezone(tokyo)) == AT
    assert jobs.parse_timestamp(AT) == NOW
    assert jobs.parse_timestamp("2026-11-24T21:00:00.250+09:00") == NOW
    with pytest.raises(ContractError, match="time zone"):
        jobs.timestamp(datetime(2026, 11, 24, 12, 0))
    for bad in ("2026-11-24T12:00:00", "yesterday", "", None, 5):
        with pytest.raises(ContractError):
            jobs.parse_timestamp(bad)


# -- spec --------------------------------------------------------------------------------


def test_build_spec_gives_a_valid_spec_with_stems_in_display_order() -> None:
    document = spec()
    assert document["request"]["stems"] == ["vocals", "drums"]
    assert document["source"] == {"key": f"jobs/{JOB}/source", "bytes": 1234}
    assert document["created_at"] == AT
    assert jobs.validate_spec(document, JOB) is document
    text = jobs.build_spec(JOB, created_at=NOW, text="just the vocals", fmt="flac", dereverb=True)
    assert text["request"] == {
        "stems": None,
        "text": "just the vocals",
        "split_vocals": False,
        "dereverb": True,
        "format": "flac",
        "quality_mode": "unspecified",
    }
    assert "bytes" not in text["source"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},  # neither stems nor text
        {"stems": ["vocals"], "text": "vocals"},  # both
        {"stems": []},
        {"stems": ["vocals", "kazoo"]},
        {"text": ""},
        {"text": "x" * 2001},
        {"stems": ["vocals"], "fmt": "ogg"},
        {"stems": ["vocals"], "quality_mode": "turbo"},
        {"stems": ["vocals"], "source_bytes": 0},
        {"stems": ["vocals"], "split_vocals": "yes"},
    ],
)
def test_build_spec_refuses_a_request_the_worker_could_not_run(arguments) -> None:
    with pytest.raises(ContractError, match="not a valid job spec"):
        jobs.build_spec(JOB, created_at=NOW, **arguments)


@pytest.mark.parametrize(
    ("path", "value", "says"),
    [
        ("version", 2, "version"),
        ("job_id", "abc", "job_id"),
        ("created_at", "2026-11-24 noon", "created_at"),
        ("created_at", "2026-11-24T12:00:00", "time zone"),
        ("source/key", f"jobs/{OTHER}/source", "outside the job's own folder"),
        ("source/key", "/etc/passwd", "outside the job's own folder"),
        ("source/key", ..., "key"),
        ("request/stems", ["vocals", "vocals"], "stems"),
        ("request/format", "unspecified", "format"),
        ("request/dereverb", ..., "dereverb"),
        ("extra", 1, "extra"),
        ("request/extra", 1, "extra"),
    ],
)
def test_a_reader_refuses_a_bad_spec(path, value, says) -> None:
    with pytest.raises(ContractError, match=says):
        jobs.validate_spec(changed(spec(), path, value), JOB)


def test_a_spec_for_another_job_is_refused() -> None:
    with pytest.raises(ContractError, match="expected"):
        jobs.validate_spec(spec(), OTHER)
    for document in (None, [], "spec", 5):
        with pytest.raises(ContractError):
            jobs.validate_spec(document)


# -- status ------------------------------------------------------------------------------


def test_valid_statuses() -> None:
    assert jobs.validate_status(status(), JOB)["stage"] == "separating"
    first = {**status(), "stage": "queued", "stems": {}, "events": [], "progress": None}
    jobs.validate_status(first, JOB)
    done = {**status(), "state": "ready", "stage": "ready"}
    jobs.validate_status(done, JOB)
    review = {**status(), "state": "needs_review", "stage": "ready"}
    jobs.validate_status(review, JOB)
    broke = {**status(), "state": "failed", "stage": "failed", "error": "out of memory"}
    jobs.validate_status(broke, JOB)


def test_every_stage_a_worker_reports_is_in_the_schema_and_fits_the_database() -> None:
    schema_stages = schemas.load("job_status")["properties"]["stage"]["enum"]
    assert tuple(schema_stages) == jobs.WORKER_STAGES
    assert schemas.load("job_status")["properties"]["events"]["maxItems"] == jobs.MAX_EVENTS
    assert schemas.load("job_result")["properties"]["state"]["enum"] == list(jobs.END_STATES)
    names = schemas.load("job_result")["properties"]["stems"]["items"]["properties"]["name"]
    assert tuple(names["enum"]) == jobs.DELIVERABLE_STEMS


@pytest.mark.parametrize(
    ("path", "value", "says"),
    [
        ("state", "queued", "state"),
        ("stage", "uploaded", "stage"),
        ("stage", "ready", "only a job that is over"),
        ("state", "ready", "only a job that is over"),
        ("updated_at", "now", "updated_at"),
        ("progress", 1.5, "progress"),
        ("stems/kazoo", {"state": "waiting"}, "stems"),
        ("stems/vocals/state", "done", "state"),
        ("stems/vocals/attempt", 3, "attempt"),
        ("events/1/seq", 3, "numbered 1, 2, 3"),
        ("events/0/seq", 0, "seq"),
        ("events/0/at", "then", "events/0/at"),
        ("events/0/extra", 1, "extra"),
        ("events", ..., "events"),
        ("job_id", OTHER, "expected"),
    ],
)
def test_a_reader_refuses_a_bad_status(path, value, says) -> None:
    with pytest.raises(ContractError, match=says):
        jobs.validate_status(changed(status(), path, value), JOB)


def test_events_must_be_in_order_and_a_failure_must_say_why() -> None:
    swapped = status()
    swapped["events"].reverse()
    with pytest.raises(ContractError, match="numbered 1, 2, 3"):
        jobs.validate_status(swapped)
    silent = {**status(), "state": "failed", "stage": "failed", "error": None}
    with pytest.raises(ContractError, match="says why"):
        jobs.validate_status(silent)


# -- result ------------------------------------------------------------------------------


def test_valid_results() -> None:
    assert jobs.validate_result(result(), JOB)["state"] == "ready"
    jobs.validate_result(failed(), JOB)
    review = {**failed(), "state": "needs_review", "error": None, "parsed_request": {"x": 1}}
    jobs.validate_result(review, JOB)
    extra = result()
    extra["stems"].append({**stem("lead_vocals"), "assessment": None, "ensemble_mode": None})
    jobs.validate_result(extra, JOB)


@pytest.mark.parametrize(
    ("path", "value", "says"),
    [
        ("stems/0/key", f"jobs/{OTHER}/stems/vocals.flac", "outside the job's own folder"),
        ("stems/1/peaks_key", "jobs/x", "outside the job's own folder"),
        ("zip_key", f"jobs/{OTHER}/stems.zip", "outside the job's own folder"),
        ("stems/1/name", "vocals", "listed twice"),
        ("stems/0/name", "instrumental", "name"),
        ("stems/0/bytes", 0, "bytes"),
        ("stems/0/attempts", 4, "attempts"),
        ("stems/0/model_ids", [], "model_ids"),
        ("stems/0/warning", ..., "warning"),
        ("stems", [], "stems"),  # ready with nothing delivered
        ("format", None, "format"),
        ("source", None, "source"),
        ("finished_at", "later", "finished_at"),
        ("state", "running", "state"),
        ("decisions/0/kind", "guess", "kind"),
        ("timings/0/wall_seconds", -1, "wall_seconds"),
        ("job_id", OTHER, "expected"),
    ],
)
def test_a_reader_refuses_a_bad_result(path, value, says) -> None:
    with pytest.raises(ContractError, match=says):
        jobs.validate_result(changed(result(), path, value), JOB)


def test_a_failed_result_has_an_error_and_no_stems() -> None:
    with pytest.raises(ContractError, match="error"):
        jobs.validate_result({**failed(), "error": None})
    with pytest.raises(ContractError, match="error"):
        jobs.validate_result(changed(failed(), "error", ...))
    with pytest.raises(ContractError, match="stems"):
        jobs.validate_result({**failed(), "stems": [stem()]})
    with pytest.raises(ContractError, match="parsed_request"):
        jobs.validate_result({**failed(), "state": "needs_review", "error": None})


def test_the_three_schemas_are_valid_json_schema() -> None:
    from jsonschema import Draft202012Validator

    for name in (jobs.SPEC_SCHEMA, jobs.STATUS_SCHEMA, jobs.RESULT_SCHEMA):
        Draft202012Validator.check_schema(schemas.load(name))
