"""A job from storage to storage: the worker's entry point.

    python -m stems_worker.jobs JOB_ID

Reads ``jobs/{id}/spec.json`` and the uploaded source, runs the whole pipeline
(``stems_worker.decision.job``), and writes the stems, their waveform peaks, a zip of
them and ``result.json``. As it goes it rewrites ``status.json`` with the stage, the
state of each stem and the history of events, which is all the API ever learns about
progress. The local provider runs this in a subprocess and the Modal provider calls
``run_job`` inside a Modal function.

``result.json`` is written last. If it is already there the job is over and nothing is
done, so a job that is submitted twice, or again after a restart, costs nothing.
"""

from __future__ import annotations

import argparse
import signal
import sys
import tempfile
import time
import traceback
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stems_api.storage import ObjectNotFound, Storage, put_json, read_json
from stems_common import jobs as contract
from stems_common.settings import Defaults, load_defaults
from stems_worker.audio import AudioError, load_audio
from stems_worker.decision import job as file_job
from stems_worker.peaks import compute_peaks

CONTENT_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3_320": "audio/mpeg",
    "mp3_192": "audio/mpeg",
}
# Progress alone (no stage or stem change) rewrites status.json at most this often.
PROGRESS_INTERVAL_SECONDS = 2.0
FAKE_MODELS_ENV = "STEMS_WORKER_FAKE_MODELS"
MAX_ERROR_LENGTH = 2000

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobNotFound(LookupError):
    """There is no spec in storage for the job."""


class StatusWriter:
    """Keeps ``jobs/{id}/status.json`` current. Every write is the whole, validated file."""

    def __init__(
        self,
        storage: Storage,
        job_id: str,
        *,
        clock: Clock = _utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._storage, self._job_id, self._clock, self._monotonic = (
            storage,
            job_id,
            clock,
            monotonic,
        )
        self._state, self._stage = "running", "queued"
        self._progress: float | None = None
        self._stems: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._error: str | None = None
        self._last_progress_write = float("-inf")

    def _event(self, stage: str, stem: str | None, message: str | None, final: bool) -> None:
        # One slot is kept back so that the event that ends the job always fits.
        if len(self._events) >= contract.MAX_EVENTS - (0 if final else 1):
            return
        self._events.append(
            {
                "seq": len(self._events) + 1,
                "at": contract.timestamp(self._clock()),
                "stage": stage,
                "stem": stem,
                "message": message,
                "progress": None,
            }
        )

    def _write(self) -> None:
        document = {
            "version": contract.CONTRACT_VERSION,
            "job_id": self._job_id,
            "state": self._state,
            "stage": self._stage,
            "updated_at": contract.timestamp(self._clock()),
            "progress": self._progress,
            "stems": self._stems,
            "events": self._events,
            "error": self._error,
        }
        put_json(
            self._storage,
            contract.status_key(self._job_id),
            contract.validate_status(document, self._job_id),
        )

    def stage(self, stage: str, *, message: str | None = None) -> None:
        """The job has moved to ``stage``. Saying the same stage twice writes nothing."""
        if stage == self._stage:
            return
        self._stage, self._progress = stage, None
        self._event(stage, None, message, final=False)
        self._write()

    def stems(self, names: tuple[str, ...], state: str) -> None:
        for name in names:
            self._stems[name] = {"state": state, "attempt": 0}
        self._write()

    def stem(self, name: str, state: str, attempt: int = 0, *, stage: str | None = None) -> None:
        """One stem has changed state; ``stage`` moves the job as well."""
        if stage is not None:
            self._stage, self._progress = stage, None
        self._stems[name] = {"state": state, "attempt": attempt}
        self._event(self._stage, name, state, final=False)
        self._write()

    def progress(self, fraction: float) -> None:
        """How far through the current stage. Written only now and then."""
        self._progress = min(1.0, max(0.0, float(fraction)))
        now = self._monotonic()
        if now - self._last_progress_write >= PROGRESS_INTERVAL_SECONDS:
            self._last_progress_write = now
            self._write()

    def finish(self, state: str, error: str | None = None) -> None:
        self._state, self._error = state, error
        self._stage = "failed" if state == "failed" else "ready"
        self._progress = None
        self._event(self._stage, None, error, final=True)
        self._write()


def _error_text(exc: BaseException) -> str:
    """What the user is told. A problem with the audio speaks for itself. Anything else is
    named by its type only: an exception's message can hold paths and other details of
    the server, so it goes to the worker's log with the traceback and not to the browser."""
    if isinstance(exc, AudioError):
        text = str(exc)
    else:
        text = (
            f"The separation failed inside the server ({type(exc).__name__}). "
            "The details are in the server's log."
        )
    return text[:MAX_ERROR_LENGTH] or type(exc).__name__


def _failed(job_id: str, error: str, clock: Clock) -> dict[str, Any]:
    return {
        "version": contract.CONTRACT_VERSION,
        "job_id": job_id,
        "state": "failed",
        "finished_at": contract.timestamp(clock()),
        "error": error,
        "stems": [],
        "decisions": [],
        "timings": [],
    }


def _zip(directory: Path, files: list[Path], fmt: str) -> Path:
    # WAV shrinks a little under deflate; FLAC and MP3 are compressed already.
    method = zipfile.ZIP_DEFLATED if fmt == "wav" else zipfile.ZIP_STORED
    path = directory / "stems.zip"
    with zipfile.ZipFile(path, "w", method) as archive:
        for file in files:
            archive.write(file, arcname=file.name)
    return path


def _package(
    storage: Storage, job_id: str, out_dir: Path, report: Mapping[str, Any], clock: Clock
) -> dict[str, Any]:
    """Upload what the pipeline wrote and describe it as a result document."""
    fmt = report["format"]
    stems: list[dict[str, Any]] = []
    files: list[Path] = []
    entries = [(name, entry, True) for name, entry in report["stems"].items()]
    entries += [(name, entry, False) for name, entry in report["extras"].items()]
    for name, entry, gated in entries:
        path = out_dir / entry["file"]
        files.append(path)
        key, peaks = contract.stem_key(job_id, name, fmt), contract.peaks_key(job_id, name)
        loaded = load_audio(path)
        put_json(storage, peaks, compute_peaks(loaded.samples, loaded.sample_rate))
        with path.open("rb") as handle:
            storage.put(key, handle, CONTENT_TYPES[fmt])
        delivered = entry["attempts"][entry["delivered_attempt"]] if gated else None
        stems.append(
            {
                "name": name,
                "key": key,
                "format": fmt,
                "bytes": path.stat().st_size,
                "peaks_key": peaks,
                "model_ids": list(entry["model_ids"]),
                "ensemble_mode": entry["mode"] if gated else None,
                "attempts": entry["retries"] + 1 if gated else 1,
                "assessment": None
                if delivered is None
                else {k: v for k, v in delivered.items() if k != "mode"},
                "warning": entry["warning"] if gated else None,
            }
        )
    archive = _zip(out_dir, files, fmt)
    with archive.open("rb") as handle:
        storage.put(contract.zip_key(job_id), handle, "application/zip")
    source = report["input"]
    return {
        "version": contract.CONTRACT_VERSION,
        "job_id": job_id,
        "state": "ready",
        "finished_at": contract.timestamp(clock()),
        "error": None,
        "format": fmt,
        "device": report["device"],
        "source": {
            "duration_seconds": source["duration_seconds"],
            "sample_rate_hz": source["source_sample_rate_hz"],
            "channels": source["source_channels"],
        },
        "stems": stems,
        "zip_key": contract.zip_key(job_id),
        "parsed_request": report["parsed_request"],
        "track_features": report["track_features"],
        "plan": report["plan"],
        "decisions": report["decisions"],
        "timings": report["models"],
    }


def _needs_review(job_id: str, report: Mapping[str, Any], clock: Clock) -> dict[str, Any]:
    return {
        "version": contract.CONTRACT_VERSION,
        "job_id": job_id,
        "state": "needs_review",
        "finished_at": contract.timestamp(clock()),
        "error": None,
        "stems": [],
        "parsed_request": report["parsed_request"],
        "decisions": report["decisions"],
        "timings": [],
    }


def run_job(
    job_id: str,
    storage: Storage,
    *,
    defaults: Defaults | None = None,
    clock: Clock = _utcnow,
    device: str | None = None,
    timings_path: Path | None = file_job.DEFAULT_TIMINGS_PATH,
    **pipeline: Any,
) -> dict[str, Any]:
    """Run the job and return its result document, which is also in storage by then.

    ``pipeline`` goes to ``stems_worker.decision.job.run_job``: the Jev ``client``, the
    ``tagger``, and for tests ``registry``, ``factory``, ``fingerprint`` and ``rates``.
    A job that fails still ends with a result (``state: failed``); the only exceptions
    this raises are for a job that has no spec, or a spec that is not valid.

    Nothing made from the upload outlives the job outside storage: the source copy, the
    stems before upload and the cache of model outputs (which lets a retry reuse what was
    already separated) all live in one temporary folder that is removed at the end. The
    timing log keeps a model, a device and two durations per run, and no audio.
    """
    contract.check_job_id(job_id)
    try:
        return contract.validate_result(read_json(storage, contract.result_key(job_id)), job_id)
    except ObjectNotFound:
        pass  # not finished yet: that is the normal case
    try:
        spec = read_json(storage, contract.spec_key(job_id))
    except ObjectNotFound:
        raise JobNotFound(f"no spec in storage for job {job_id}") from None
    spec = contract.validate_spec(spec, job_id)
    defaults = defaults or load_defaults()
    asked = spec["request"]
    status = StatusWriter(storage, job_id, clock=clock)

    def on_stage(stage: str, info: Mapping[str, Any]) -> None:
        stem, attempt = info.get("stem"), info.get("attempt", 0)
        if stage == "separating":
            status.stage("separating")
            status.stems(tuple(info["stems"]), "separating")
        elif stage in ("assessing", "retrying"):
            status.stem(stem, stage, attempt, stage=stage)
        elif stage == "delivered":
            status.stem(stem, info["state"], attempt)
        else:
            status.stage(stage)

    def on_progress(_model_id: str, done: int, total: int) -> None:
        if total > 0:
            status.progress(done / total)

    try:
        status.stage("transcoding")
        with tempfile.TemporaryDirectory(prefix="stems-job-") as work:
            source = Path(work) / "source"
            with storage.open(spec["source"]["key"]) as incoming, source.open("wb") as local:
                while chunk := incoming.read(1024 * 1024):
                    local.write(chunk)
            out_dir = Path(work) / "out"
            report = file_job.run_job(
                source,
                out_dir,
                file_job.JobRequest(
                    text=asked["text"],
                    stems=None if asked["stems"] is None else tuple(asked["stems"]),
                    split_vocals=asked["split_vocals"],
                    dereverb=asked["dereverb"],
                    quality_mode=asked["quality_mode"],
                    fmt=asked["format"],
                    device=device,
                    # In the job's own folder, never the shared cache: retention covers
                    # storage, and a shared cache of separated uploads would escape it.
                    cache_dir=Path(work) / "cache",
                    timings_path=timings_path,
                ),
                defaults=defaults,
                job_id=job_id,
                progress=on_progress,
                on_stage=on_stage,
                max_duration_seconds=defaults.uploads.max_duration_seconds,
                **pipeline,
            )
            if report["status"] == file_job.NEEDS_REVIEW:
                result = _needs_review(job_id, report, clock)
            else:
                status.stage("packaging")
                result = _package(storage, job_id, out_dir, report, clock)
            contract.validate_result(result, job_id)
    except Exception as exc:
        traceback.print_exc()
        result = contract.validate_result(_failed(job_id, _error_text(exc), clock), job_id)
    put_json(storage, contract.result_key(job_id), result)
    status.finish(result["state"], result.get("error"))
    return result


def _terminate(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_worker.jobs",
        description="Run one job from storage: read its spec and source, write its stems.",
    )
    parser.add_argument("job_id")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    # Asked to stop (a deleted job, a shutdown): leave through the normal exits, so the
    # temporary folder holding the upload is removed on the way out.
    signal.signal(signal.SIGTERM, _terminate)

    from stems_api.storage import storage_from_env
    from stems_common.env import get_env

    storage = storage_from_env()
    defaults = load_defaults()
    options: dict[str, Any] = {}
    tagger = None
    if get_env(FAKE_MODELS_ENV):
        # Weightless stand-in models: the end-to-end tests and a first local run.
        from stems_worker.fakes import fake_pipeline

        options, defaults = fake_pipeline(defaults)
        # Stand-in outputs stay out of the separation cache and the timing log.
        options.update(timings_path=None)
    else:
        from stems_models.download import DownloadError
        from stems_worker.decision.jev import JevClient
        from stems_worker.listening.beats import BeatsError, BeatsTagger

        options["client"] = JevClient.from_env(defaults.decision)
        if defaults.listening.enabled:
            try:
                tagger = BeatsTagger.from_registry()
                tagger.load("cpu")
            except (BeatsError, DownloadError, OSError) as exc:
                print(f"worker: no tagger ({exc}); stems will not be listened to", file=sys.stderr)
                tagger = None
        options["tagger"] = tagger
    try:
        result = run_job(args.job_id, storage, defaults=defaults, device=args.device, **options)
    except (JobNotFound, contract.ContractError) as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 1
    finally:
        if tagger is not None:
            tagger.unload()
    print(f"job {args.job_id}: {result['state']}")
    for stem in result["stems"]:
        print(f"  {stem['name']}: {stem['key']}")
    if result.get("error"):
        print(f"  error: {result['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
