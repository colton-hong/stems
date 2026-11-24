"""Queries that act for no owner: the status poller and the retention sweeper.

Request handlers do not use this module: everything a request does goes through
``stems_api.db.store.JobStore``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from stems_api.db.models import Job, JobEvent
from stems_api.db.store import scrub


def jobs_in_progress(session: Session) -> list[Job]:
    """Jobs a worker may still be working on, oldest first."""
    query = (
        select(Job)
        .where(Job.status.in_(("queued", "running")), Job.deleted_at.is_(None))
        .order_by(Job.created_at, Job.id)
    )
    return list(session.scalars(query))


def jobs_past_expiry(session: Session, now: datetime) -> list[Job]:
    """Jobs whose retention has run out and whose files have not been removed yet."""
    query = (
        select(Job)
        .where(Job.expires_at <= now, Job.deleted_at.is_(None))
        .order_by(Job.expires_at, Job.id)
    )
    return list(session.scalars(query))


def jobs_ended_since(session: Session, since: datetime) -> list[Job]:
    """Jobs deleted or expired at or after ``since``."""
    query = select(Job).where(Job.deleted_at >= since).order_by(Job.deleted_at, Job.id)
    return list(session.scalars(query))


def mark_expired(session: Session, job: Job, now: datetime) -> None:
    scrub(session, job, "expired", now)


def add_event(
    session: Session,
    job: Job,
    *,
    at: datetime,
    stage: str,
    stem: str | None = None,
    message: str | None = None,
    progress: float | None = None,
    seq: int | None = None,
) -> JobEvent:
    event = JobEvent(
        job_id=job.id, seq=seq, at=at, stage=stage, stem=stem, message=message, progress=progress
    )
    session.add(event)
    return event
