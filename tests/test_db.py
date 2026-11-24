"""Database schema, migrations and owner scoping."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from stems_api.db import migrate, system
from stems_api.db.engine import database_url, make_engine, session_factory
from stems_api.db.models import (
    STAGES,
    STATUSES,
    Base,
    DecisionCall,
    Job,
    JobEvent,
    Owner,
    Stem,
    Timing,
)
from stems_api.db.store import (
    TOUCH_AFTER,
    JobStore,
    create_owner,
    find_owner,
    hash_token,
    new_id,
)
from stems_common.paths import REPO_ROOT

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
WEEK = timedelta(days=7)


@pytest.fixture
def engine():
    engine = make_engine("sqlite://")
    migrate.upgrade(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_factory(engine)() as session:
        yield session


def add_job(session, owner_id: str, *, created: datetime = NOW, name: str = "song.wav") -> Job:
    """A finished job with one row in every child table."""
    job_id = new_id()
    job = JobStore(session, owner_id).create(
        job_id=job_id,
        request={"stems": ["vocals", "drums"], "format": "wav"},
        source_key=f"jobs/{job_id}/source",
        now=created,
        expires_at=created + WEEK,
        display_name=name,
        source_bytes=1234,
        source_duration_s=12.0,
        source_sample_rate_hz=44100,
        source_channels=2,
    )
    job.status, job.stage = "ready", "ready"
    job.track_features = {"genre_hint": "rock"}
    job.plan = {"mode": "single_best"}
    job.stems.append(
        Stem(
            name="vocals",
            object_key=f"jobs/{job_id}/stems/vocals.wav",
            format="wav",
            bytes=999,
            model_ids=["fake4"],
            ensemble_mode="single_best",
            attempts=1,
            assessment={"score": 3},
        )
    )
    system.add_event(session, job, at=created, stage="separating", stem="vocals", progress=0.5)
    session.add(DecisionCall(job_id=job_id, kind="plan", provider="fallback", answers={}))
    session.add(
        Timing(job_id=job_id, model_id="fake4", device="cpu", audio_seconds=12.0, wall_seconds=1.5)
    )
    session.flush()
    return job


# -- schema ------------------------------------------------------------------------------


def plan_tables() -> dict[str, set[str]]:
    """Table name -> column names, read from the table in PLAN.md section 7."""
    text_ = (REPO_ROOT / "PLAN.md").read_text(encoding="utf-8")
    section = text_.split("## 7. Data model")[1].split("\n## ")[0]
    tables: dict[str, set[str]] = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and re.fullmatch(r"`\w+`", cells[0]):
            # A column is a backticked name; what follows one in brackets describes it.
            columns = re.findall(r"`(\w+)`", re.sub(r"\([^)]*\)", "", cells[1]))
            tables[cells[0].strip("`")] = set(columns)
    return tables


def test_schema_has_every_table_and_column_of_plan_section_7(engine) -> None:
    planned = plan_tables()
    assert set(planned) == {"owners", "jobs", "stems", "job_events", "decision_calls", "timings"}
    assert len(planned["jobs"]) == 18  # the parser read the long row whole
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) - {"alembic_version"} == set(planned)
    for table, columns in planned.items():
        actual = {column["name"] for column in inspector.get_columns(table)}
        assert columns <= actual, f"{table} lacks {sorted(columns - actual)}"
    # What the schema adds to the plan's "key columns" is exactly this.
    extra = {
        table: {c["name"] for c in inspector.get_columns(table)} - columns
        for table, columns in planned.items()
    }
    assert extra == {
        "owners": set(),
        "jobs": {"display_name", "source_bytes"},
        "stems": set(),
        "job_events": {"seq"},
        "decision_calls": set(),
        "timings": set(),
    }


def test_migrations_give_exactly_the_schema_of_the_models(engine) -> None:
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        assert compare_metadata(context, Base.metadata) == []
    assert migrate.current_revision(engine) == migrate.head_revision() == "0001"


def test_upgrade_is_idempotent_and_works_on_a_file(tmp_path) -> None:
    path = tmp_path / "nested" / "stems.db"
    engine = make_engine(f"sqlite:///{path}")
    assert migrate.current_revision(engine) is None
    assert migrate.upgrade(engine) == "0001"
    with session_factory(engine)() as session:
        owner, _ = create_owner(session, NOW)
        add_job(session, owner.id)
        session.commit()
    assert migrate.upgrade(engine) == "0001"  # a second run changes nothing
    with session_factory(engine)() as session:
        assert len(session.scalars(select(Job)).all()) == 1
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
    engine.dispose()


def test_database_url_comes_from_the_environment() -> None:
    assert database_url(lambda name: None).endswith("/runs/dev/stems.db")
    assert database_url({"STEMS_DATABASE_URL": "sqlite:///x.db"}.get) == "sqlite:///x.db"


def test_stage_and_status_vocabularies_follow_the_plan() -> None:
    plan = (REPO_ROOT / "PLAN.md").read_text(encoding="utf-8")
    stages = plan.split("### Job stages")[1].split("###")[0]
    assert set(STAGES) <= set(re.findall(r"[a-z]+", stages))
    assert set(STATUSES) <= {*STAGES, "running"}
    longest = max(len(name) for name in (*STAGES, *STATUSES))
    assert longest <= Job.__table__.c.stage.type.length


# -- column behaviour --------------------------------------------------------------------


def test_timestamps_are_utc_and_must_carry_a_zone(session) -> None:
    owner, _ = create_owner(session, NOW)
    tokyo = timezone(timedelta(hours=9))
    job = add_job(session, owner.id, created=datetime(2026, 11, 24, 21, 0, tzinfo=tokyo))
    session.commit()
    session.expire_all()
    stored = session.get(Job, job.id)
    assert stored.created_at == NOW
    assert stored.created_at.tzinfo is UTC
    stored.updated_at = datetime(2026, 11, 24, 12, 0)  # no zone
    with pytest.raises(Exception, match="time zone"):
        session.commit()


def test_json_columns_round_trip(session) -> None:
    owner, _ = create_owner(session, NOW)
    job = add_job(session, owner.id)
    session.commit()
    session.expire_all()
    stored = session.get(Job, job.id)
    assert stored.request == {"stems": ["vocals", "drums"], "format": "wav"}
    assert stored.stems[0].model_ids == ["fake4"]
    assert stored.stems[0].assessment == {"score": 3}


def test_a_job_needs_a_real_owner_and_stem_names_are_unique_per_job(session) -> None:
    owner, _ = create_owner(session, NOW)
    job = add_job(session, owner.id)
    session.commit()
    job.stems.append(Stem(name="vocals", object_key="k", format="wav", model_ids=[]))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    with pytest.raises(IntegrityError):
        JobStore(session, "no-such-owner").create(
            job_id=new_id(), request={}, source_key="k", now=NOW, expires_at=NOW + WEEK
        )
    session.rollback()


def test_deleting_a_job_row_removes_its_children(session) -> None:
    owner, _ = create_owner(session, NOW)
    job = add_job(session, owner.id)
    session.commit()
    session.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job.id})
    session.commit()
    for model in (Stem, JobEvent, DecisionCall, Timing):
        assert session.scalars(select(model)).all() == []


# -- owners ------------------------------------------------------------------------------


def test_only_the_hash_of_an_owner_token_is_stored(session, engine) -> None:
    owner, token = create_owner(session, NOW)
    session.commit()
    assert len(token) >= 43  # 32 random bytes, base64url
    assert owner.token_hash == hash_token(token) != token
    with engine.connect() as connection:
        dump = repr(connection.execute(text("SELECT * FROM owners")).all())
    assert token not in dump
    assert find_owner(session, token, NOW).id == owner.id
    assert find_owner(session, token + "x", NOW) is None
    assert find_owner(session, owner.token_hash, NOW) is None  # the hash is not a token
    assert find_owner(session, "", NOW) is None
    assert find_owner(session, None, NOW) is None
    other, other_token = create_owner(session, NOW)
    assert other_token != token
    assert other.id != owner.id


def test_last_seen_is_refreshed_only_now_and_then(session) -> None:
    _, token = create_owner(session, NOW)
    session.commit()
    soon = NOW + TOUCH_AFTER - timedelta(seconds=1)
    assert find_owner(session, token, soon).last_seen_at == NOW
    assert not session.dirty
    later = NOW + TOUCH_AFTER
    assert find_owner(session, token, later).last_seen_at == later


# -- owner scoping -----------------------------------------------------------------------


@pytest.fixture
def two_owners(session):
    alice, _ = create_owner(session, NOW)
    bob, _ = create_owner(session, NOW)
    alices = add_job(session, alice.id, name="alice.wav")
    bobs = add_job(session, bob.id, name="bob.wav")
    session.commit()
    return alice, bob, alices, bobs


def test_every_read_is_scoped_to_the_owner(session, two_owners) -> None:
    alice, bob, alices, bobs = two_owners
    as_alice, as_bob = JobStore(session, alice.id), JobStore(session, bob.id)
    # Each owner sees their own job in full.
    assert as_alice.get(alices.id).display_name == "alice.wav"
    assert [job.id for job in as_alice.list()] == [alices.id]
    assert [job.id for job in as_alice.list(include_ended=True)] == [alices.id]
    assert len(as_alice.stems(alices.id)) == 1
    assert as_alice.stem(alices.id, "vocals").bytes == 999
    assert len(as_alice.events(alices.id)) == 1
    assert len(as_alice.decision_calls(alices.id)) == 1
    assert len(as_alice.timings(alices.id)) == 1
    # Asking for the other owner's job is the same as asking for one that does not exist.
    for store, foreign in ((as_alice, bobs.id), (as_bob, alices.id), (as_bob, new_id())):
        assert store.get(foreign) is None
        assert store.get(foreign, include_ended=False) is None
        assert store.stems(foreign) == []
        assert store.stem(foreign, "vocals") is None
        assert store.events(foreign) == []
        assert store.decision_calls(foreign) == []
        assert store.timings(foreign) == []


def test_every_write_is_scoped_to_the_owner(session, two_owners) -> None:
    alice, bob, alices, bobs = two_owners
    as_bob = JobStore(session, bob.id)
    assert as_bob.mark_deleted(alices.id, NOW) is None
    session.commit()
    session.expire_all()
    untouched = JobStore(session, alice.id).get(alices.id)
    assert (untouched.status, untouched.display_name, untouched.deleted_at) == (
        "ready",
        "alice.wav",
        None,
    )
    assert len(untouched.stems) == 1
    # Bob forgetting everything removes Bob's rows and none of Alice's.
    assert as_bob.forget_everything() == [bobs.id]
    session.commit()
    assert [owner.id for owner in session.scalars(select(Owner))] == [alice.id]
    assert [job.id for job in session.scalars(select(Job))] == [alices.id]
    assert len(session.scalars(select(Stem)).all()) == 1
    assert len(session.scalars(select(JobEvent)).all()) == 1
    assert len(session.scalars(select(DecisionCall)).all()) == 1
    assert len(session.scalars(select(Timing)).all()) == 1


def test_job_store_has_no_method_that_skips_the_owner() -> None:
    """Every public method builds its query from ``_jobs`` or ``_owned``."""
    import inspect as pyinspect

    assert JobStore.__init__.__code__.co_varnames[:3] == ("self", "session", "owner_id")
    with pytest.raises(ValueError, match="owner"):
        JobStore(None, "")
    for name, method in pyinspect.getmembers(JobStore, pyinspect.isfunction):
        if name.startswith("_"):
            continue
        source = pyinspect.getsource(method)
        scoped = ("self._jobs()", "self._owned(", "self.owner_id", "self.get(")
        assert any(marker in source for marker in scoped), name
        # Nothing selects from jobs without going through one of those.
        assert "select(Job)" not in source, name


def test_list_is_newest_first_and_hides_ended_jobs(session) -> None:
    owner, _ = create_owner(session, NOW)
    store = JobStore(session, owner.id)
    old = add_job(session, owner.id, created=NOW - timedelta(hours=2))
    new = add_job(session, owner.id, created=NOW)
    gone = add_job(session, owner.id, created=NOW - timedelta(hours=1))
    store.mark_deleted(gone.id, NOW)
    session.commit()
    assert [job.id for job in store.list()] == [new.id, old.id]
    assert [job.id for job in store.list(include_ended=True)] == [new.id, gone.id, old.id]


def test_events_after_an_id(session) -> None:
    owner, _ = create_owner(session, NOW)
    job = add_job(session, owner.id)
    for index in range(5):
        system.add_event(session, job, at=NOW, stage="separating", progress=index / 5)
    session.commit()
    store = JobStore(session, owner.id)
    events = store.events(job.id)
    assert len(events) == 6
    assert [event.id for event in events] == sorted(event.id for event in events)
    assert [event.id for event in store.events(job.id, after_id=events[2].id)] == [
        event.id for event in events[3:]
    ]
    assert len(store.events(job.id, limit=2)) == 2


# -- deletion and expiry -----------------------------------------------------------------


def test_mark_deleted_scrubs_the_row_and_is_idempotent(session) -> None:
    owner, _ = create_owner(session, NOW)
    store = JobStore(session, owner.id)
    job = add_job(session, owner.id)
    session.commit()
    later = NOW + timedelta(minutes=5)
    assert store.mark_deleted(job.id, later).id == job.id
    session.commit()
    session.expire_all()
    row = store.get(job.id)
    assert (row.status, row.stage, row.deleted_at, row.updated_at) == (
        "deleted",
        "deleted",
        later,
        later,
    )
    assert (row.display_name, row.track_features, row.plan, row.error) == (None, None, None, None)
    assert store.stems(job.id) == []
    assert store.events(job.id) == []
    assert store.get(job.id, include_ended=False) is None
    # Deleting again finds nothing to delete and changes nothing.
    assert store.mark_deleted(job.id, later + timedelta(days=1)) is None
    assert store.get(job.id).deleted_at == later


def test_system_queries_find_work_in_progress_and_expired_jobs(session) -> None:
    owner, _ = create_owner(session, NOW)
    store = JobStore(session, owner.id)
    running = add_job(session, owner.id, created=NOW - timedelta(days=1))
    running.status, running.stage = "running", "separating"
    queued = add_job(session, owner.id, created=NOW)
    queued.status, queued.stage = "queued", "queued"
    done = add_job(session, owner.id, created=NOW - timedelta(days=8))
    deleted = add_job(session, owner.id, created=NOW - timedelta(days=9))
    store.mark_deleted(deleted.id, NOW)
    session.commit()
    assert [job.id for job in system.jobs_in_progress(session)] == [running.id, queued.id]
    assert [job.id for job in system.jobs_past_expiry(session, NOW)] == [done.id]
    assert system.jobs_past_expiry(session, NOW - timedelta(days=2)) == []
    # Exactly at the expiry counts as expired.
    assert [j.id for j in system.jobs_past_expiry(session, done.expires_at)] == [done.id]
    system.mark_expired(session, done, NOW)
    session.commit()
    assert system.jobs_past_expiry(session, NOW) == []
    row = store.get(done.id)
    assert (row.status, row.display_name, row.deleted_at) == ("expired", None, NOW)
    assert store.stems(done.id) == []
