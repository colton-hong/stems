"""Read settings the owner supplies: environment variables first, then the ``.env`` file.

Values read here are paths and secrets. They are returned to the caller and never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

from stems_common.paths import REPO_ROOT

DOTENV_PATH = REPO_ROOT / ".env"


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
        text = (dotenv_path or DOTENV_PATH).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return parse_dotenv(text).get(name) or None
