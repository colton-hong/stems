"""The database tables of PLAN.md section 7.

SQLite by default, and nothing here is specific to it: JSON columns, text ids and UTC
timestamps work the same on Postgres. Times are stored in UTC without an offset (SQLite
has no offset-aware type) and always come back offset-aware.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

# What a job is doing, coarsely. `stage` says where in the pipeline a running job is.
STATUSES = ("queued", "running", "ready", "failed", "expired", "deleted")
# States a job never leaves on its own.
END_STATUSES = ("ready", "failed", "expired", "deleted")
# PLAN.md, "Job stages". The first two are set by the API, the rest reported by the worker.
STAGES = (
    "uploaded",
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
    "expired",
    "deleted",
)
DECISION_KINDS = ("parse", "plan", "gate")


class UTCDateTime(TypeDecorator[datetime]):
    """A timestamp that is always UTC: naive in the database, offset-aware in Python."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must carry a time zone")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[str]: JSON}  # noqa: RUF012


class Owner(Base):
    """A browser. Only the hash of its cookie secret is stored (BLOCKED.md D1)."""

    __tablename__ = "owners"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_owner_created", "owner_id", "created_at"),
        Index("ix_jobs_expires_at", "expires_at"),
        Index("ix_jobs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("owners.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16))
    stage: Mapped[str] = mapped_column(String(16))
    # What was asked for: stems, vocal split, de-reverb, format, quality mode.
    request: Mapped[dict[str, Any]]
    # The uploaded file's name. A label for the UI and for download names, nothing else.
    display_name: Mapped[str | None] = mapped_column(String(255))
    source_key: Mapped[str] = mapped_column(String(512))
    source_bytes: Mapped[int | None] = mapped_column(Integer)
    source_duration_s: Mapped[float | None] = mapped_column(Float)
    source_sample_rate_hz: Mapped[int | None] = mapped_column(Integer)
    source_channels: Mapped[int | None] = mapped_column(Integer)
    track_features: Mapped[dict[str, Any] | None]
    plan: Mapped[dict[str, Any] | None]
    provider: Mapped[str | None] = mapped_column(String(32))
    provider_handle: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    stems: Mapped[list[Stem]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Stem.id"
    )


class Stem(Base):
    __tablename__ = "stems"
    __table_args__ = (Index("ux_stems_job_name", "job_id", "name", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(32))
    object_key: Mapped[str] = mapped_column(String(512))
    format: Mapped[str] = mapped_column(String(16))
    bytes: Mapped[int | None] = mapped_column(Integer)
    model_ids: Mapped[list[str]]
    ensemble_mode: Mapped[str | None] = mapped_column(String(32))
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    assessment: Mapped[dict[str, Any] | None]
    warning: Mapped[str | None] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="stems")


class JobEvent(Base):
    __tablename__ = "job_events"
    __table_args__ = (
        Index("ix_job_events_job", "job_id", "id"),
        Index("ux_job_events_job_seq", "job_id", "seq", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    # The worker's number for this event in its status.json (1, 2, 3, ...), so that a
    # status read twice is recorded once. Null for events the API adds itself.
    seq: Mapped[int | None] = mapped_column(Integer)
    at: Mapped[datetime] = mapped_column(UTCDateTime)
    stage: Mapped[str] = mapped_column(String(16))
    stem: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[float | None] = mapped_column(Float)


class DecisionCall(Base):
    __tablename__ = "decision_calls"
    __table_args__ = (Index("ix_decision_calls_job", "job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(16))
    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    answers: Mapped[dict[str, Any] | None]
    fallback_reason: Mapped[str | None] = mapped_column(Text)


class Timing(Base):
    __tablename__ = "timings"
    __table_args__ = (Index("ix_timings_job", "job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    model_id: Mapped[str] = mapped_column(String(64))
    stem: Mapped[str | None] = mapped_column(String(32))
    device: Mapped[str] = mapped_column(String(16))
    audio_seconds: Mapped[float] = mapped_column(Float)
    wall_seconds: Mapped[float] = mapped_column(Float)
