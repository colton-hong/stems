"""The local provider: each job runs in its own subprocess, one job at a time.

``submit`` puts the job in a queue and returns. A single dispatcher thread takes jobs
off the queue in order, starts ``python -m stems_worker.jobs JOB_ID`` and waits for it,
so the machine's GPU and memory are never asked to hold two songs. The worker process
picks its own device (CUDA, then MPS, then CPU) and finds storage from the same
environment as the API. Its output goes to ``runs/jobs/<job id>.log``.

The queue lives in this process. If the API restarts, queued and running jobs are
unknown to the new provider; the API submits them again at startup, which is safe
because a worker that finds a finished job does no work.
"""

from __future__ import annotations

import contextlib
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from stems_api.providers.base import JobSpec, ProviderError, ProviderHandle, ProviderStatus
from stems_common.jobs import ContractError, check_job_id, validate_spec
from stems_common.paths import REPO_ROOT, RUNS_DIR

DEFAULT_LOG_DIR = RUNS_DIR / "jobs"
HANDLE_PREFIX = "local:"
# How long a worker gets to exit after being asked to, before it is killed.
TERMINATE_GRACE_SECONDS = 5.0


def worker_command(job_id: str) -> list[str]:
    return [sys.executable, "-m", "stems_worker.jobs", job_id]


@dataclass
class _Entry:
    job_id: str
    state: str = "queued"
    detail: str | None = None
    process: subprocess.Popen | None = None
    cancelled: bool = False
    done: threading.Event = field(default_factory=threading.Event)


class LocalProvider:
    name = "local"

    def __init__(
        self,
        *,
        command: Callable[[str], Sequence[str]] = worker_command,
        env: Mapping[str, str] | None = None,
        log_dir: Path | str | None = DEFAULT_LOG_DIR,
        cwd: Path | str = REPO_ROOT,
    ) -> None:
        self._command = command
        self._env = None if env is None else dict(env)
        self._log_dir = None if log_dir is None else Path(log_dir)
        self._cwd = str(cwd)
        self._entries: dict[str, _Entry] = {}
        self._queue: queue.Queue[_Entry | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    # -- WorkerProvider ------------------------------------------------------------------

    def submit(self, spec: JobSpec) -> ProviderHandle:
        # The API side of the contract: nothing is queued that a worker would refuse.
        job_id = validate_spec(spec)["job_id"]
        with self._lock:
            if self._closed:
                raise ProviderError("the local provider has been shut down")
            existing = self._entries.get(job_id)
            if existing is not None and existing.state in ("queued", "running"):
                return HANDLE_PREFIX + job_id  # already on its way: do not run it twice
            entry = _Entry(job_id)
            self._entries[job_id] = entry
            self._queue.put(entry)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._dispatch, name="stems-local-provider", daemon=True
                )
                self._thread.start()
        return HANDLE_PREFIX + job_id

    def status(self, handle: ProviderHandle) -> ProviderStatus:
        entry = self._find(handle)
        if entry is None:
            return ProviderStatus("unknown", "this provider has no record of the job")
        with self._lock:
            return ProviderStatus(entry.state, entry.detail)  # type: ignore[arg-type]

    def cancel(self, handle: ProviderHandle) -> None:
        entry = self._find(handle)
        if entry is None:
            return
        with self._lock:
            if entry.state not in ("queued", "running"):
                return
            entry.cancelled = True
            process = entry.process
            if entry.state == "queued":
                entry.state, entry.detail = "cancelled", "cancelled before it started"
                entry.done.set()
        if process is not None:
            self._stop(process)

    # -- beyond the protocol -------------------------------------------------------------

    def wait(self, handle: ProviderHandle, timeout: float | None = None) -> ProviderStatus:
        """Block until the worker for ``handle`` is over (or ``timeout`` passes)."""
        entry = self._find(handle)
        if entry is not None:
            entry.done.wait(timeout)
        return self.status(handle)

    def forget(self, job_id: str) -> bool:
        """Remove what this provider kept about a job that is gone: its worker's log.

        A log can name the job's files and show a traceback from its audio, so it goes
        when the job goes. Returns whether there was a log to remove.
        """
        if self._log_dir is None:
            return False
        try:
            (self._log_dir / f"{check_job_id(job_id)}.log").unlink()
        except (FileNotFoundError, ContractError):
            return False
        return True

    def close(self) -> None:
        """Cancel everything and stop the dispatcher. For shutdown and for tests."""
        with self._lock:
            self._closed = True
            handles = [HANDLE_PREFIX + job_id for job_id in self._entries]
            thread = self._thread
        for handle in handles:
            self.cancel(handle)
        self._queue.put(None)
        if thread is not None:
            thread.join(timeout=TERMINATE_GRACE_SECONDS * 2)

    # -- internals -----------------------------------------------------------------------

    def _find(self, handle: ProviderHandle) -> _Entry | None:
        if not isinstance(handle, str) or not handle.startswith(HANDLE_PREFIX):
            return None
        with self._lock:
            return self._entries.get(handle[len(HANDLE_PREFIX) :])

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            process.wait(TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                process.kill()

    def _dispatch(self) -> None:
        while True:
            entry = self._queue.get()
            if entry is None:
                return
            self._run(entry)

    def _run(self, entry: _Entry) -> None:
        with self._lock:
            if entry.cancelled:
                return
        log = None
        try:
            if self._log_dir is not None:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                log = (self._log_dir / f"{entry.job_id}.log").open("ab")
            process = subprocess.Popen(
                list(self._command(entry.job_id)),
                cwd=self._cwd,
                env=None if self._env is None else {**os.environ, **self._env},
                stdin=subprocess.DEVNULL,
                stdout=log if log is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            with self._lock:
                entry.state, entry.detail = "failed", f"the worker could not be started: {exc}"
            entry.done.set()
            if log is not None:
                log.close()
            return
        with self._lock:
            entry.process, entry.state = process, "running"
            cancelled = entry.cancelled
        if cancelled:  # cancel() arrived between the check above and the start
            self._stop(process)
        code = process.wait()
        if log is not None:
            log.close()
        with self._lock:
            entry.process = None
            if entry.cancelled:
                entry.state, entry.detail = "cancelled", "stopped on request"
            elif code == 0:
                entry.state, entry.detail = "succeeded", None
            else:
                entry.state, entry.detail = "failed", f"the worker exited with code {code}"
        entry.done.set()
