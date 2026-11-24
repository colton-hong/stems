"""The worker provider interface (PLAN.md section 6).

A provider starts a worker for a job and can say whether that worker is still alive. It
says nothing about the job's progress: that is in the job's ``status.json`` and
``result.json`` in storage, which the worker writes and the API polls. The provider's
answer matters when a worker dies without writing a result.

``submit`` must return quickly. A provider never runs a job in the caller's thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

# What the API writes for a job and a worker reads: a validated ``spec.json`` document.
JobSpec = dict[str, Any]
# Whatever the provider needs to find the worker again. Stored in jobs.provider_handle.
ProviderHandle = str

WorkerState = Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown"]
# States in which no worker will do anything more for the job.
OVER: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


class ProviderError(RuntimeError):
    """The provider could not do what was asked."""


@dataclass(frozen=True)
class ProviderStatus:
    """Whether the worker for a job is alive.

    ``unknown`` means the provider has no record of the handle, for example a local
    worker that was running when the API was restarted. ``succeeded`` means the worker
    exited cleanly, which is not the same as the job being ready: ``result.json`` says.
    """

    state: WorkerState
    detail: str | None = None

    @property
    def over(self) -> bool:
        return self.state in OVER


@runtime_checkable
class WorkerProvider(Protocol):
    name: str

    def submit(self, spec: JobSpec) -> ProviderHandle:
        """Start (or queue) a worker for the job. Returns at once."""

    def status(self, handle: ProviderHandle) -> ProviderStatus: ...

    def cancel(self, handle: ProviderHandle) -> None:
        """Stop the worker if it has not finished. Unknown and finished handles are ignored."""
