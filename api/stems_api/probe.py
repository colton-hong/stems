"""Look inside an uploaded file, in a process of its own.

    python -m stems_api.probe FILE        # prints one JSON object

Media decoders are large C libraries fed whatever a stranger uploads. Running them here,
in a short-lived child of the API, means a file that makes a decoder crash or hang takes
down this process and not the API: the parent sees no answer, or a timeout, and refuses
the upload. The child reads the header and decodes the first second, nothing more.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from stems_worker.audio import AudioError, SourceInfo, decode_head, probe


class ProbeFailed(Exception):
    """The file is not audio the decoders can read, or reading it did not end well."""


def look(path: Path) -> SourceInfo:
    """Header and first second, in this process. ``ProbeFailed`` if it is not audio."""
    try:
        info = probe(path)
        if info.sample_rate <= 0 or info.channels <= 0:
            raise AudioError("no usable audio stream")
        decode_head(path)
    except AudioError as exc:
        raise ProbeFailed(str(exc)) from exc
    return info


def look_isolated(path: Path, timeout_seconds: float) -> SourceInfo:
    """The same answer as ``look``, from a child process with a time limit."""
    try:
        done = subprocess.run(
            [sys.executable, "-m", "stems_api.probe", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeFailed(f"reading the file took longer than {timeout_seconds:.0f} s") from exc
    try:
        answer = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        # No answer at all: the decoder took the process down with it.
        raise ProbeFailed(f"the decoder stopped unexpectedly (exit {done.returncode})") from exc
    if not answer.get("ok"):
        raise ProbeFailed(str(answer.get("error", "not audio")))
    return SourceInfo(**answer["info"])


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m stems_api.probe FILE", file=sys.stderr)
        return 2
    try:
        info = look(Path(args[0]))
    except ProbeFailed as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 0
    print(json.dumps({"ok": True, "info": asdict(info)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
