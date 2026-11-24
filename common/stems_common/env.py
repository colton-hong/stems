"""Read settings the owner supplies: environment variables first, then the ``.env`` file.

Values read here are paths and secrets. They are returned to the caller and never logged.

``STEMS_DOTENV_PATH`` names another file to read in place of the repository's ``.env``.
The test suites point it at a file that does not exist, so that no test, and no process a
test starts, ever picks up the owner's real keys.
"""

from __future__ import annotations

import os
from pathlib import Path

from stems_common.paths import REPO_ROOT

DOTENV_PATH = REPO_ROOT / ".env"
DOTENV_ENV = "STEMS_DOTENV_PATH"


def dotenv_file() -> Path:
    """The ``.env`` file in use: ``STEMS_DOTENV_PATH`` if set, else the repository's."""
    override = os.environ.get(DOTENV_ENV)
    return Path(override) if override else DOTENV_PATH


def parse_dotenv(text: str) -> dict[str, str]:
    """``KEY=value`` lines. Blank lines and ``#`` comments are skipped; quotes are removed."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def get_env(name: str, *, dotenv_path: Path | None = None) -> str | None:
    """The value of ``name``, or None if it is unset or empty in both places."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        text = (dotenv_path or dotenv_file()).read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return None
    return parse_dotenv(text).get(name) or None
