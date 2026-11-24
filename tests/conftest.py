"""Shared test setup.

The owner's ``.env`` holds real keys. No test may read it: a test that did could send a
request to a paid service, or pass only on this machine. So for the whole run, and for
every process a test starts, the ``.env`` file is one that does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["STEMS_DOTENV_PATH"] = str(Path(__file__).parent / "no-such-file.env")
for _name in ("TYPESAFE_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_AUTH_TOKEN"):
    os.environ.pop(_name, None)
