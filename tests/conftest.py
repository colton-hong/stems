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

# The retention sweeper removes this project's temporary folders (stems-job-*,
# stems-upload-*) once they look two days old, and the tests drive it with clocks set
# days ahead. Pointed at the real temporary directory that deletes the working folder of
# any job running on this machine at the time: it did, on 2026-11-24, to a real job. So
# the whole run, and every process a test starts, gets a temporary directory of its own.
import tempfile  # noqa: E402

_SYSTEM_TEMP = Path(tempfile.gettempdir())
_TEST_TEMP = _SYSTEM_TEMP / f"stems-tests-{os.getuid()}"
_TEST_TEMP.mkdir(mode=0o700, exist_ok=True)
os.environ["TMPDIR"] = str(_TEST_TEMP)
tempfile.tempdir = str(_TEST_TEMP)
