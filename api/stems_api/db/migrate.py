"""Bring a database to the current schema.

    python -m stems_api.db.migrate          # upgrade STEMS_DATABASE_URL to the newest revision

The API runs the same upgrade at startup. Revisions live in ``migrations/versions`` and
are applied in order; a database already at the newest one is left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

from stems_api.db.engine import make_engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    return config


def head_revision() -> str:
    head = ScriptDirectory.from_config(_config()).get_current_head()
    assert head is not None, "no migration revisions found"
    return head


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def upgrade(engine: Engine, revision: str = "head") -> str:
    """Apply every revision up to ``revision``. Returns the revision the database is at."""
    config = _config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)
    current = current_revision(engine)
    assert current is not None
    return current


def main() -> int:
    engine = make_engine()
    before = current_revision(engine)
    after = upgrade(engine)
    print(f"database at revision {after}" + ("" if before == after else f" (was {before})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
