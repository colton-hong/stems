"""The retention sweeper: delete what has outlived ``retention.days``.

For every job past its expiry it stops the worker if one is still going, deletes every
object under the job's prefix, and scrubs the row to ``expired``. It also deletes again
under the prefixes of jobs that ended recently, which removes anything a worker managed
to write after its job was deleted. Running it twice in a row changes nothing more, so
it is run at startup and then on a timer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from stems_api.db import system
from stems_api.db.models import Job
from stems_api.providers.base import WorkerProvider
from stems_api.storage import Storage, StorageError
from stems_common import jobs as contract

log = logging.getLogger("stems_api.sweeper")

# Deleted and expired jobs get their prefix deleted again for this long afterwards.
RECLEAN_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class SweepReport:
    expired_jobs: int = 0
    objects_deleted: int = 0
    failures: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.expired_jobs or self.objects_deleted)


def purge(storage: Storage, provider: WorkerProvider, job: Job) -> int:
    """Stop the job's worker and delete its objects. Returns how many objects went."""
    if job.provider_handle and job.status in ("queued", "running"):
        provider.cancel(job.provider_handle)
    return storage.delete_prefix(contract.job_prefix(job.id))


def sweep(
    sessions: sessionmaker[Session],
    storage: Storage,
    provider: WorkerProvider,
    now: datetime,
) -> SweepReport:
    expired = objects = failures = 0
    with sessions() as session:
        due = [job.id for job in system.jobs_past_expiry(session, now)]
        recent = [job.id for job in system.jobs_ended_since(session, now - RECLEAN_WINDOW)]
    for job_id in due:
        # One transaction per job: a failure on one does not undo the others.
        with sessions() as session:
            job = session.get(Job, job_id)
            if job is None or job.deleted_at is not None:
                continue
            try:
                objects += purge(storage, provider, job)
            except (StorageError, RuntimeError) as exc:
                # The row stays unexpired, so the next sweep tries again.
                log.error("job %s: could not delete its objects: %s", job_id, exc)
                failures += 1
                continue
            system.mark_expired(session, job, now)
            session.commit()
            expired += 1
    for job_id in recent:
        try:
            objects += storage.delete_prefix(contract.job_prefix(job_id))
        except StorageError as exc:
            log.error("job %s: could not re-check its prefix: %s", job_id, exc)
            failures += 1
    return SweepReport(expired_jobs=expired, objects_deleted=objects, failures=failures)
