"""The six tables of PLAN.md section 7.

Written out in full, not generated from the models, so that this revision keeps meaning
the same thing when the models change. ``tests/test_db.py`` checks that applying every
revision gives exactly the schema the models describe.
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _cascade(target: str) -> sa.ForeignKey:
    return sa.ForeignKey(target, ondelete="CASCADE")


def upgrade() -> None:
    op.create_table(
        "owners",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(36), _cascade("owners.id"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("stage", sa.String(16), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("display_name", sa.String(255)),
        sa.Column("source_key", sa.String(512), nullable=False),
        sa.Column("source_bytes", sa.Integer()),
        sa.Column("source_duration_s", sa.Float()),
        sa.Column("source_sample_rate_hz", sa.Integer()),
        sa.Column("source_channels", sa.Integer()),
        sa.Column("track_features", sa.JSON()),
        sa.Column("plan", sa.JSON()),
        sa.Column("provider", sa.String(32)),
        sa.Column("provider_handle", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime()),
    )
    op.create_index("ix_jobs_owner_created", "jobs", ["owner_id", "created_at"])
    op.create_index("ix_jobs_expires_at", "jobs", ["expires_at"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_table(
        "stems",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), _cascade("jobs.id"), nullable=False),
        sa.Column("name", sa.String(32), nullable=False),
        sa.Column("object_key", sa.String(512), nullable=False),
        sa.Column("format", sa.String(16), nullable=False),
        sa.Column("bytes", sa.Integer()),
        sa.Column("model_ids", sa.JSON(), nullable=False),
        sa.Column("ensemble_mode", sa.String(32)),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("assessment", sa.JSON()),
        sa.Column("warning", sa.Text()),
    )
    op.create_index("ux_stems_job_name", "stems", ["job_id", "name"], unique=True)
    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), _cascade("jobs.id"), nullable=False),
        sa.Column("seq", sa.Integer()),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("stage", sa.String(16), nullable=False),
        sa.Column("stem", sa.String(32)),
        sa.Column("message", sa.Text()),
        sa.Column("progress", sa.Float()),
    )
    op.create_index("ix_job_events_job", "job_events", ["job_id", "id"])
    op.create_index("ux_job_events_job_seq", "job_events", ["job_id", "seq"], unique=True)
    op.create_table(
        "decision_calls",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), _cascade("jobs.id"), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("model", sa.String(64)),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("answers", sa.JSON()),
        sa.Column("fallback_reason", sa.Text()),
    )
    op.create_index("ix_decision_calls_job", "decision_calls", ["job_id"])
    op.create_table(
        "timings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), _cascade("jobs.id"), nullable=False),
        sa.Column("model_id", sa.String(64), nullable=False),
        sa.Column("stem", sa.String(32)),
        sa.Column("device", sa.String(16), nullable=False),
        sa.Column("audio_seconds", sa.Float(), nullable=False),
        sa.Column("wall_seconds", sa.Float(), nullable=False),
    )
    op.create_index("ix_timings_job", "timings", ["job_id"])


def downgrade() -> None:
    for table in ("timings", "decision_calls", "job_events", "stems", "jobs", "owners"):
        op.drop_table(table)
