"""Bring a job's database row up to date with what its worker wrote to storage.

The API learns about a job only from ``status.json`` and ``result.json`` (PLAN.md section
2). ``JobSync.sync`` reads them for one job in progress and applies what is new: the
stage, events not seen before, and at the end the stems, decisions and timings. It also
asks the provider whether the worker is still alive, so that a worker that died without
writing a result ends the job as failed instead of leaving it running for ever.

Both documents are validated before anything in them is used. A status that does not
validate is ignored (the next one may be fine); a result that does not validate fails the
job, since nothing better will come.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from stems_api.db import system
from stems_api.db.models import DecisionCall, Job, JobEvent, Stem, Timing
from stems_api.providers.base import WorkerProvider
from stems_api.storage import ObjectNotFound, Storage, StorageError, read_json
from stems_common import jobs as contract

log = logging.getLogger("stems_api.sync")

WORKER_LOST = "The worker stopped before it finished this job."
BAD_RESULT = "The worker's result could not be read."
NEEDS_REVIEW = (
    "No stem could be recognised in the request, so nothing was separated. "
    "Choose the stems you want and submit again."
)
IN_PROGRESS = ("queued", "running")


class JobSync:
    def __init__(
        self,
        storage: Storage,
        provider: WorkerProvider,
        clock: Callable[[], datetime],
    ) -> None:
        self._storage, self._provider, self._clock = storage, provider, clock
        # The last valid status of each job in progress: per-stem state and progress,
        # which the database has no columns for.
        self._live: dict[str, dict[str, Any]] = {}
        self._busy: set[str] = set()
        self._lock = threading.Lock()

    # -- what a request may read ---------------------------------------------------------

    def live(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._live.get(job_id)

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._live.pop(job_id, None)

    # -- reading storage -----------------------------------------------------------------

    def _read(self, key: str) -> Any | None:
        """The JSON at ``key``; None if it is not there (yet) or is half-written."""
        try:
            return read_json(self._storage, key)
        except ObjectNotFound:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ...  # there, and not JSON

    def _status(self, job_id: str) -> dict[str, Any] | None:
        document = self._read(contract.status_key(job_id))
        if document is None or document is ...:
            return None
        try:
            return contract.validate_status(document, job_id)
        except contract.ContractError as exc:
            log.warning("ignoring an invalid status for job %s: %s", job_id, exc)
            return None

    # -- applying ------------------------------------------------------------------------

    def _apply_events(self, session: Session, job: Job, status: dict[str, Any]) -> None:
        seen = session.scalar(select(func.max(JobEvent.seq)).where(JobEvent.job_id == job.id))
        for event in status["events"]:
            if event["seq"] > (seen or 0):
                system.add_event(
                    session,
                    job,
                    seq=event["seq"],
                    at=contract.parse_timestamp(event["at"]),
                    stage=event["stage"],
                    stem=event.get("stem"),
                    message=event.get("message"),
                    progress=event.get("progress"),
                )

    def _fail(self, job: Job, error: str, now: datetime) -> None:
        job.status, job.stage, job.error, job.updated_at = "failed", "failed", error, now
        self.forget(job.id)

    def _apply_result(
        self, session: Session, job: Job, result: dict[str, Any], now: datetime
    ) -> None:
        if result["state"] == "failed":
            self._fail(job, result["error"], now)
            return
        if result["state"] == "needs_review":
            job.request = {**job.request, "parsed_request": result["parsed_request"]}
            self._fail(job, NEEDS_REVIEW, now)
            return
        source = result["source"]
        job.source_duration_s = source["duration_seconds"]
        job.source_sample_rate_hz = source["sample_rate_hz"]
        job.source_channels = source["channels"]
        job.track_features, job.plan = result.get("track_features"), result.get("plan")
        job.stems.clear()
        session.flush()
        for stem in result["stems"]:
            job.stems.append(
                Stem(
                    name=stem["name"],
                    object_key=stem["key"],
                    format=stem["format"],
                    bytes=stem["bytes"],
                    model_ids=stem["model_ids"],
                    ensemble_mode=stem["ensemble_mode"],
                    attempts=stem["attempts"],
                    assessment=stem["assessment"],
                    warning=stem["warning"],
                )
            )
        session.execute(delete(DecisionCall).where(DecisionCall.job_id == job.id))
        session.execute(delete(Timing).where(Timing.job_id == job.id))
        for call in result["decisions"]:
            session.add(
                DecisionCall(
                    job_id=job.id,
                    kind=call["kind"],
                    provider=call["provider"],
                    model=call.get("model"),
                    input_tokens=call.get("input_tokens"),
                    output_tokens=call.get("output_tokens"),
                    latency_ms=call.get("latency_ms"),
                    answers=call.get("answers"),
                    fallback_reason=call.get("fallback_reason"),
                )
            )
        for run in result["timings"]:
            session.add(
                Timing(
                    job_id=job.id,
                    model_id=run["model_id"],
                    stem=(",".join(run.get("outputs") or []) or None),
                    device=run["device"],
                    audio_seconds=run["audio_seconds"],
                    wall_seconds=run["wall_seconds"],
                )
            )
        job.status, job.stage, job.error, job.updated_at = "ready", "ready", None, now
        self.forget(job.id)

    def _finish_if_there_is_a_result(self, session: Session, job: Job, now: datetime) -> bool:
        document = self._read(contract.result_key(job.id))
        if document is None:
            return False
        status = self._status(job.id)
        if status is not None:
            self._apply_events(session, job, status)  # the last events, before it closes
        try:
            if document is ...:
                raise contract.ContractError("result.json is not JSON")
            result = contract.validate_result(document, job.id)
        except contract.ContractError as exc:
            log.error("job %s: unreadable result: %s", job.id, exc)
            self._fail(job, BAD_RESULT, now)
            return True
        self._apply_result(session, job, result, now)
        return True

    # -- the entry points ----------------------------------------------------------------

    def sync(self, session: Session, job: Job) -> bool:
        """Update ``job`` from storage. Returns True if the job is now over.

        Safe to call from several threads: a job that is being synced elsewhere is left
        to that caller. The session is flushed, not committed.
        """
        if job.status not in IN_PROGRESS or job.deleted_at is not None:
            return job.status not in IN_PROGRESS
        with self._lock:
            if job.id in self._busy:
                return False
            self._busy.add(job.id)
        try:
            return self._sync(session, job)
        except StorageError as exc:
            log.warning("job %s: storage could not be read: %s", job.id, exc)
            return False
        finally:
            with self._lock:
                self._busy.discard(job.id)

    def _sync(self, session: Session, job: Job) -> bool:
        now = self._clock()
        if self._finish_if_there_is_a_result(session, job, now):
            session.flush()
            return True
        status = self._status(job.id)
        if status is not None and status["state"] == "running":
            if (job.status, job.stage) != ("running", status["stage"]):
                job.status, job.stage, job.updated_at = "running", status["stage"], now
            self._apply_events(session, job, status)
            with self._lock:
                self._live[job.id] = status
        # Is anyone still working on it?
        if job.provider_handle:
            worker = self._provider.status(job.provider_handle)
            if worker.over or worker.state == "unknown":
                # It may have written its result just after the look above.
                if not self._finish_if_there_is_a_result(session, job, now):
                    detail = f" ({worker.detail})" if worker.detail else ""
                    log.error("job %s: worker %s, no result%s", job.id, worker.state, detail)
                    self._fail(job, WORKER_LOST, now)
                session.flush()
                return True
        session.flush()
        return False

    def sync_all(self, sessions: sessionmaker[Session]) -> int:
        """Sync every job in progress, each in its own transaction. Returns how many ended."""
        with sessions() as session:
            job_ids = [job.id for job in system.jobs_in_progress(session)]
        ended = 0
        for job_id in job_ids:
            with sessions() as session:
                job = session.get(Job, job_id)
                if job is not None and self.sync(session, job):
                    ended += 1
                session.commit()
        return ended

    def recover(self, sessions: sessionmaker[Session]) -> int:
        """After a restart: hand every unfinished job back to the provider.

        A job whose worker already finished is closed from its result instead. Submitting
        a job again is safe, because a worker that finds a result does nothing.
        """
        resubmitted = 0
        with sessions() as session:
            for job in system.jobs_in_progress(session):
                now = self._clock()
                if self._finish_if_there_is_a_result(session, job, now):
                    continue
                if job.provider_handle and self._provider.status(job.provider_handle).state in (
                    "queued",
                    "running",
                ):
                    continue
                try:
                    spec = contract.validate_spec(
                        read_json(self._storage, contract.spec_key(job.id)), job.id
                    )
                    job.provider_handle = self._provider.submit(spec)
                    job.provider = self._provider.name
                    resubmitted += 1
                except (StorageError, contract.ContractError, RuntimeError) as exc:
                    log.error("job %s could not be resubmitted: %s", job.id, exc)
                    self._fail(job, WORKER_LOST, now)
            session.commit()
        return resubmitted
