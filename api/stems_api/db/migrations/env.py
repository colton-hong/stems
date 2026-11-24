"""Alembic's entry point. Always run through ``stems_api.db.migrate``, which hands over an
open connection; there is no alembic.ini."""

from alembic import context

from stems_api.db.models import Base

connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("run migrations with: python -m stems_api.db.migrate")

# Batch mode: SQLite cannot alter a column in place, so later revisions rebuild the table.
context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
with context.begin_transaction():
    context.run_migrations()
