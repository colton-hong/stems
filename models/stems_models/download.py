"""Download model files by registry id. Resumable, checksum-pinned, safe to re-run.

    python -m stems_models.download mel_roformer_kim_vocals htdemucs
    python -m stems_models.download --tier candidate --background
    python -m stems_models.download --status

Files land in ``models/weights/<model id>/<filename>`` (or under ``STEMS_MODELS_DIR``) and
are never committed.

Checksums are trust-on-first-use: the first time a URL is downloaded its SHA-256 is written
to ``models/checksums.lock.json``, which is committed. Every later download or load of that
URL must match, so a file that changes upstream or is corrupted on disk is caught instead
of being loaded. Checkpoints are pickle files, so this matters.

Each model's progress is written to ``runs/downloads/<model id>/status.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import certifi

from stems_common.paths import REPO_ROOT, RUNS_DIR
from stems_models.registry import ModelEntry, ModelFile, Registry, load_registry

USER_AGENT = "stems-model-download/0.1"
LOCK_PATH = REPO_ROOT / "models" / "checksums.lock.json"
DOWNLOAD_RUNS_DIR = RUNS_DIR / "downloads"
BLOCK = 1 << 20

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
_PERMANENT_HTTP = {400, 401, 403, 404, 410}

ProgressFn = Callable[[int], None]


class DownloadError(Exception):
    """A file could not be fetched."""


class SizeMismatchError(DownloadError):
    """The server sent a different number of bytes than the registry records."""


class ChecksumMismatchError(DownloadError):
    """A file's SHA-256 differs from the one pinned in the lock file."""


class AlreadyRunningError(DownloadError):
    """Another process is downloading this model right now."""


# --- locations ----------------------------------------------------------------------------


def weights_dir() -> Path:
    override = os.environ.get("STEMS_MODELS_DIR")
    return Path(override).expanduser() if override else REPO_ROOT / "models" / "weights"


def model_dir(model_id: str, directory: Path | None = None) -> Path:
    return (directory or weights_dir()) / model_id


# --- checksums ----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(4 * BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


class ChecksumLock:
    """The committed record of what each URL's content hashed to when first downloaded."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or LOCK_PATH

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, url: str) -> dict[str, Any] | None:
        return self._read().get(url)

    def check_or_record(self, url: str, filename: str, sha256: str, size_bytes: int) -> str:
        """Return "verified" or "recorded". Raise if the pinned checksum differs."""
        entries = self._read()
        known = entries.get(url)
        if known is not None:
            if known["sha256"] != sha256:
                raise ChecksumMismatchError(
                    f"{filename}: SHA-256 {sha256[:16]}... does not match the pinned "
                    f"{known['sha256'][:16]}... for {url}. The file changed upstream or is "
                    f"corrupt. It was not accepted."
                )
            return "verified"
        entries[url] = {"filename": filename, "sha256": sha256, "size_bytes": size_bytes}
        _write_json_atomic(self.path, entries)
        return "recorded"


class _VerifiedCache:
    """Remembers which files on disk were already hashed, keyed by size and mtime.

    Without it every load would re-hash gigabytes. It lives next to the weights and is not
    committed; deleting it only costs time.
    """

    def __init__(self, directory: Path) -> None:
        self.path = directory / ".verified.json"

    def sha256(self, path: Path) -> str:
        stat = path.stat()
        key = str(path.resolve())
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            entries = {}
        hit = entries.get(key)
        if hit and hit["size"] == stat.st_size and hit["mtime_ns"] == stat.st_mtime_ns:
            return hit["sha256"]
        digest = sha256_file(path)
        entries[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
        _write_json_atomic(self.path, entries)
        return digest


# --- fetching one file --------------------------------------------------------------------


def _download(
    url: str,
    part: Path,
    expected_size: int | None,
    *,
    progress: ProgressFn | None,
    max_attempts: int,
    retry_wait: float,
    timeout: float,
) -> None:
    """Fetch ``url`` into ``part``, resuming from whatever ``part`` already holds."""
    failures = 0
    last_error = "no attempt made"
    while True:
        have = part.stat().st_size if part.exists() else 0
        if expected_size is not None:
            if have == expected_size:
                return
            if have > expected_size:
                part.unlink()
                have = 0

        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        started_with = have
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_SSL_CONTEXT) as r:
                content_type = r.headers.get("Content-Type", "")
                if content_type.startswith("text/html") and not part.name.endswith(".html.part"):
                    raise DownloadError(
                        f"{url} returned an HTML page instead of the file "
                        f"(a login, share or error page)"
                    )
                resumed = r.status == 206 and have > 0
                if not resumed:
                    have = 0  # the server ignored the range: start over
                length = r.headers.get("Content-Length")
                target = have + int(length) if length and length.isdigit() else None
                with part.open("ab" if resumed else "wb") as f:
                    while block := r.read(BLOCK):
                        f.write(block)
                        have += len(block)
                        if progress is not None:
                            progress(have)
            if target is None or have >= target:
                if expected_size is None or have == expected_size:
                    return
                raise SizeMismatchError(
                    f"{url}: received {have} bytes but the registry records {expected_size}. "
                    f"The file changed upstream; run `python -m stems_models.check_sources`."
                )
            last_error = f"connection closed after {have} of {target} bytes"
        except urllib.error.HTTPError as e:
            if e.code == 416:  # our partial file is not a prefix of what the server has
                part.unlink(missing_ok=True)
                last_error = "HTTP 416 on resume; restarting"
            elif e.code in _PERMANENT_HTTP:
                raise DownloadError(f"{url}: HTTP {e.code} {e.reason}") from e
            else:
                last_error = f"HTTP {e.code} {e.reason}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_error = f"{type(e).__name__}: {e}"

        now = part.stat().st_size if part.exists() else 0
        failures = 0 if now > started_with else failures + 1  # progress resets the count
        if failures >= max_attempts:
            raise DownloadError(f"{url}: gave up after {max_attempts} attempts ({last_error})")
        time.sleep(retry_wait)


def fetch_file(
    file: ModelFile,
    destination: Path,
    *,
    lock: ChecksumLock,
    cache: _VerifiedCache | None = None,
    progress: ProgressFn | None = None,
    max_attempts: int = 5,
    retry_wait: float = 2.0,
    timeout: float = 60.0,
) -> tuple[Path, str]:
    """Make sure ``file`` is in ``destination`` and matches its pinned checksum.

    Returns the path and what happened: "present" (already there and verified) or
    "downloaded".
    """
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / file.filename
    cache = cache or _VerifiedCache(destination.parent)

    if final.exists():
        lock.check_or_record(file.url, file.filename, cache.sha256(final), final.stat().st_size)
        return final, "present"

    part = final.with_name(final.name + ".part")
    try:
        _download(
            file.url,
            part,
            file.size_bytes,
            progress=progress,
            max_attempts=max_attempts,
            retry_wait=retry_wait,
            timeout=timeout,
        )
        lock.check_or_record(file.url, file.filename, sha256_file(part), part.stat().st_size)
    except (SizeMismatchError, ChecksumMismatchError):
        # Wrong content is never kept: a later run must not resume from it.
        part.unlink(missing_ok=True)
        raise
    os.replace(part, final)
    cache.sha256(final)
    return final, "downloaded"


# --- status files -------------------------------------------------------------------------


def status_path(model_id: str, runs_dir: Path | None = None) -> Path:
    return (runs_dir or DOWNLOAD_RUNS_DIR) / model_id / "status.json"


def read_status(model_id: str, runs_dir: Path | None = None) -> dict[str, Any] | None:
    try:
        return json.loads(status_path(model_id, runs_dir).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class _Status:
    model_id: str
    path: Path
    data: dict[str, Any]
    _last_write: float = 0.0

    @classmethod
    def start(cls, entry: ModelEntry, files: list[ModelFile], runs_dir: Path | None, state: str):
        data = {
            "model_id": entry.id,
            "state": state,
            "pid": os.getpid(),
            "started_at": _now(),
            "updated_at": _now(),
            "error": None,
            "files": [
                {
                    "filename": f.filename,
                    "size_bytes": f.size_bytes,
                    "bytes_done": 0,
                    "state": "pending",
                }
                for f in files
            ],
        }
        status = cls(entry.id, status_path(entry.id, runs_dir), data)
        status.write()
        return status

    def write(self) -> None:
        self.data["updated_at"] = _now()
        _write_json_atomic(self.path, self.data)
        self._last_write = time.monotonic()

    def file(self, filename: str) -> dict[str, Any]:
        return next(f for f in self.data["files"] if f["filename"] == filename)

    def progress_for(self, filename: str) -> ProgressFn:
        record = self.file(filename)

        def report(bytes_done: int) -> None:
            record["bytes_done"] = bytes_done
            if time.monotonic() - self._last_write >= 1.0:
                self.write()

        return report


# --- whole models -------------------------------------------------------------------------


@contextlib.contextmanager
def _model_lock(directory: Path) -> Iterator[None]:
    """Hold an exclusive lock for one model's directory. Released if the process dies."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".lock").open("w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AlreadyRunningError(
                f"another process is already downloading into {directory}"
            ) from None
        yield
    finally:
        handle.close()


def is_downloaded(
    entry: ModelEntry, stem: str | None = None, directory: Path | None = None
) -> bool:
    target = model_dir(entry.id, directory)
    return all((target / f.filename).exists() for f in entry.files_for(stem))


def ensure_model(
    model_id: str,
    *,
    stem: str | None = None,
    registry: Registry | None = None,
    directory: Path | None = None,
    lock: ChecksumLock | None = None,
    runs_dir: Path | None = None,
    max_attempts: int = 5,
    retry_wait: float = 2.0,
) -> dict[str, Path]:
    """Download whatever is missing for ``model_id`` and verify all of it.

    Returns a mapping of filename to path. With ``stem``, only the files that stem needs
    are fetched (for bags that ship one file per stem).
    """
    registry = registry or load_registry()
    entry = registry.get(model_id)
    if not entry.is_loadable:
        raise DownloadError(f"{model_id} has status {entry.status!r}: nothing to download")
    directory = directory or weights_dir()
    target = model_dir(model_id, directory)
    files = entry.files_for(stem)
    lock = lock or ChecksumLock()
    cache = _VerifiedCache(directory)

    with _model_lock(target):
        status = _Status.start(entry, files, runs_dir, "running")
        paths: dict[str, Path] = {}
        try:
            for f in files:
                record = status.file(f.filename)
                record["state"] = "running"
                status.write()
                path, outcome = fetch_file(
                    f,
                    target,
                    lock=lock,
                    cache=cache,
                    progress=status.progress_for(f.filename),
                    max_attempts=max_attempts,
                    retry_wait=retry_wait,
                )
                record["state"] = outcome
                record["bytes_done"] = path.stat().st_size
                paths[f.filename] = path
        except Exception as e:
            status.data["state"] = "failed"
            status.data["error"] = f"{type(e).__name__}: {e}"
            status.write()
            raise
        status.data["state"] = "done"
        status.write()
    return paths


# --- command line -------------------------------------------------------------------------


def select_models(registry: Registry, ids: list[str], tiers: list[str], everything: bool):
    if everything:
        return registry.loadable()
    chosen = [registry.get(model_id) for model_id in ids]
    chosen += [m for m in registry.loadable() if m.tier in tiers and m not in chosen]
    return chosen


def _print_status(models: list[ModelEntry], directory: Path, runs_dir: Path | None) -> None:
    for m in models:
        status = read_status(m.id, runs_dir)
        on_disk = is_downloaded(m, directory=directory)
        state = status["state"] if status else "not started"
        if on_disk and state in ("not started", "done"):
            state = "done"
        detail = ""
        if status and status["state"] in ("running", "failed"):
            done = sum(f["bytes_done"] for f in status["files"])
            total = sum(f["size_bytes"] or 0 for f in status["files"])
            detail = f"  {done / 1e6:.0f} of {total / 1e6:.0f} MB"
            if status["error"]:
                detail += f"  {status['error']}"
        print(f"{m.id:42s} {state}{detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model files by registry id.")
    parser.add_argument("ids", nargs="*", help="registry ids")
    parser.add_argument("--tier", action="append", default=[], help="candidate|baseline|extra")
    parser.add_argument("--all", action="store_true", help="every loadable model")
    parser.add_argument("--stem", help="only the files this stem needs")
    parser.add_argument("--dir", type=Path, help="weights directory (default: models/weights)")
    parser.add_argument("--background", action="store_true", help="detach and return at once")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    args = parser.parse_args(argv)

    registry = load_registry()
    directory = args.dir or weights_dir()
    try:
        models = select_models(registry, args.ids, args.tier, args.all)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    if args.status:
        _print_status(models or registry.loadable(), directory, None)
        return 0
    if not models:
        parser.error("name at least one model id, or use --tier or --all")

    if args.background:
        for m in models:
            if not is_downloaded(m, args.stem, directory):
                _Status.start(m, m.files_for(args.stem), None, "queued")
        log_path = DOWNLOAD_RUNS_DIR / "background.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "stems_models.download", *[m.id for m in models]]
        command += ["--dir", str(directory)]
        if args.stem:
            command += ["--stem", args.stem]
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                cwd=REPO_ROOT,
            )
        print(f"started pid {process.pid} for {len(models)} model(s); log: {log_path}")
        print("check progress with: python -m stems_models.download --status")
        return 0

    failed = 0
    for m in models:
        try:
            paths = ensure_model(m.id, stem=args.stem, registry=registry, directory=directory)
            size = sum(p.stat().st_size for p in paths.values())
            print(f"{_now()} ok      {m.id}  ({size / 1e6:.1f} MB)", flush=True)
        except AlreadyRunningError as e:
            print(f"{_now()} skipped {m.id}: {e}", flush=True)
        except DownloadError as e:
            failed += 1
            print(f"{_now()} FAILED  {m.id}: {e}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
