"""The database connection.

    STEMS_DATABASE_URL   default: sqlite:///<repo>/runs/dev/stems.db

Any SQLAlchemy URL works. SQLite is opened with foreign keys on, and a file database
with write-ahead logging so that the API's reads do not wait on its writes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from stems_common.env import get_env
from stems_common.paths import RUNS_DIR

DATABASE_ENV = "STEMS_DATABASE_URL"
DEFAULT_PATH = RUNS_DIR / "dev" / "stems.db"
MEMORY_URL = "sqlite://"


def database_url(getenv: Callable[[str], str | None] = get_env) -> str:
    return getenv(DATABASE_ENV) or f"sqlite:///{DEFAULT_PATH}"


def make_engine(url: str | None = None) -> Engine:
    """An engine for ``url`` (default: ``database_url()``). Creates a SQLite file's folder."""
    parsed = make_url(url or database_url())
    if parsed.get_backend_name() != "sqlite":
        return create_engine(parsed, pool_pre_ping=True)
    in_memory = parsed.database in (None, "", ":memory:")
    if in_memory:
        # One shared connection: every session sees the same in-memory database.
        engine = create_engine(
            parsed, connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    else:
        Path(parsed.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(parsed, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _configure(connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        if not in_memory:
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    # Objects stay readable after commit: handlers build their replies from them.
    return sessionmaker(engine, expire_on_commit=False)
