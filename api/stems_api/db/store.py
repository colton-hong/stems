"""Queries for request handlers.

A handler never selects from ``jobs`` itself. It gets a ``JobStore`` bound to the owner
the cookie identified, and every method of that store filters on the owner: asking for
someone else's job gives the same answer as asking for one that does not exist. The
sweeper and the status poller, which act for no owner, use ``stems_api.db.system``.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Select, delete, select
from sqlalchemy.orm import Session

from stems_api.db.models import DecisionCall, Job, JobEvent, Owner, Stem, Timing

# last_seen_at is refreshed at most this often, so that reading does not mean writing.
TOUCH_AFTER = timedelta(minutes=10)


def new_id() -> str:
    return str(uuid.uuid4())


def new_owner_token() -> str:
    """The secret kept in the browser's cookie: 256 random bits."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """What the database keeps of a cookie secret. The secret is random and long, so a
    plain SHA-256 is enough: there is nothing to guess."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_owner(session: Session, now: datetime) -> tuple[Owner, str]:
    """A new owner and the token to put in its cookie. The token is not stored."""
    token = new_owner_token()
    owner = Owner(id=new_id(), token_hash=hash_token(token), created_at=now, last_seen_at=now)
    session.add(owner)
    session.flush()
    return owner, token


def find_owner(session: Session, token: str | None, now: datetime) -> Owner | None:
    if not token:
        return None
    owner = session.scalar(select(Owner).where(Owner.token_hash == hash_token(token)))
    if owner is not None and now - owner.last_seen_at >= TOUCH_AFTER:
        owner.last_seen_at = now
    return owner


def scrub(session: Session, job: Job, status: str, now: datetime) -> None:
    """Mark a job ``deleted`` or ``expired`` and drop what described its audio.

    The row stays so that the job's page can say what happened. The file name, the
    analysis, the plan, the per-stem rows, the event history and the decision log (whose
    answers are about the request and the audio) all go. Timings stay: a model, a device
    and two durations, kept for the cost figures.
    """
    job.status = status
    job.stage = status
    job.display_name = None
    job.track_features = None
    job.plan = None
    job.error = None
    job.provider_handle = None
    job.updated_at = now
    job.deleted_at = job.deleted_at or now
    job.stems.clear()
    session.execute(delete(JobEvent).where(JobEvent.job_id == job.id))
    session.execute(delete(DecisionCall).where(DecisionCall.job_id == job.id))


class JobStore:
    """One owner's jobs. Every query here carries ``owner_id``."""

    def __init__(self, session: Session, owner_id: str) -> None:
        if not owner_id:
            raise ValueError("a JobStore needs an owner")
        self._session = session
        self.owner_id = owner_id

    def _jobs(self) -> Select[tuple[Job]]:
        return select(Job).where(Job.owner_id == self.owner_id)

    def _owned(self, job_id: str) -> Select[tuple[str]]:
        """The job's id if this owner has it: the filter for child-table queries."""
        return select(Job.id).where(Job.owner_id == self.owner_id, Job.id == job_id)

    # -- jobs ----------------------------------------------------------------------------

    def create(
        self,
        *,
        job_id: str,
        request: dict[str, Any],
        source_key: str,
        now: datetime,
        expires_at: datetime,
        display_name: str | None = None,
        source_bytes: int | None = None,
        source_duration_s: float | None = None,
        source_sample_rate_hz: int | None = None,
        source_channels: int | None = None,
    ) -> Job:
        job = Job(
            id=job_id,
            owner_id=self.owner_id,
            status="queued",
            stage="uploaded",
            request=request,
            display_name=display_name,
            source_key=source_key,
            source_bytes=source_bytes,
            source_duration_s=source_duration_s,
            source_sample_rate_hz=source_sample_rate_hz,
            source_channels=source_channels,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )
        self._session.add(job)
        self._session.flush()
        return job

    def get(self, job_id: str, *, include_ended: bool = True) -> Job | None:
        """The job, or None if it does not exist or is another owner's.

        A deleted or expired job is still returned, as a row that says so; pass
        ``include_ended=False`` to get only jobs whose files may still exist.
        """
        query = self._jobs().where(Job.id == job_id)
        if not include_ended:
            query = query.where(Job.deleted_at.is_(None))
        return self._session.scalar(query)

    def list(self, *, include_ended: bool = False) -> list[Job]:
        """Newest first. Deleted and expired jobs are left out unless asked for."""
        query = self._jobs().order_by(Job.created_at.desc(), Job.id)
        if not include_ended:
            query = query.where(Job.deleted_at.is_(None))
        return list(self._session.scalars(query))

    def mark_deleted(self, job_id: str, now: datetime) -> Job | None:
        """Scrub the job's row. Returns it, or None if there is nothing of this owner's to
        delete. The caller deletes the stored objects."""
        job = self.get(job_id, include_ended=False)
        if job is None:
            return None
        scrub(self._session, job, "deleted", now)
        return job

    def forget_everything(self) -> list[str]:
        """Remove every row of this owner, the owner included. Returns the ids of the jobs
        that were removed, so the caller can delete their objects."""
        job_ids = list(self._session.scalars(select(Job.id).where(Job.owner_id == self.owner_id)))
        for job in self._session.scalars(self._jobs()):
            self._session.delete(job)
        self._session.flush()
        self._session.execute(delete(Owner).where(Owner.id == self.owner_id))
        return job_ids

    # -- what hangs off a job ------------------------------------------------------------

    def stems(self, job_id: str) -> list[Stem]:
        query = select(Stem).where(Stem.job_id.in_(self._owned(job_id))).order_by(Stem.id)
        return list(self._session.scalars(query))

    def stem(self, job_id: str, name: str) -> Stem | None:
        query = select(Stem).where(Stem.job_id.in_(self._owned(job_id)), Stem.name == name)
        return self._session.scalar(query)

    def events(self, job_id: str, *, after_id: int = 0, limit: int = 500) -> list[JobEvent]:
        query = (
            select(JobEvent)
            .where(JobEvent.job_id.in_(self._owned(job_id)), JobEvent.id > after_id)
            .order_by(JobEvent.id)
            .limit(limit)
        )
        return list(self._session.scalars(query))

    def decision_calls(self, job_id: str) -> list[DecisionCall]:
        query = (
            select(DecisionCall)
            .where(DecisionCall.job_id.in_(self._owned(job_id)))
            .order_by(DecisionCall.id)
        )
        return list(self._session.scalars(query))

    def timings(self, job_id: str) -> list[Timing]:
        query = select(Timing).where(Timing.job_id.in_(self._owned(job_id))).order_by(Timing.id)
        return list(self._session.scalars(query))
