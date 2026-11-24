"""Azure Container Apps: a seam, not an implementation (project decision 9).

The build does not depend on Azure GPU quota, so this provider is deliberately empty.
It is here so that adding Azure later means filling in one class and nothing else: the
API already selects it with ``STEMS_WORKER_PROVIDER=azure_container_apps``, and until it
is filled in that selection fails at startup with a message that says so.

To implement it, write the two methods marked below. Everything else can stay.

``_start_job(job_id) -> str``
    Start one execution of a Container Apps job whose image runs
    ``python -m stems_worker.jobs <job_id>`` (the same entry point the local provider
    runs in a subprocess and Modal runs in a function), passing ``job_id`` as an
    argument or environment override. The container needs the storage settings
    (``STEMS_STORAGE_BACKEND=azure_blob`` and its connection string, or ``s3``) and a GPU
    workload profile. With ``azure-mgmt-appcontainers`` this is ``jobs.begin_start`` on a
    manually triggered job. Return the execution's name: it becomes the handle.

``_execution_state(execution_name) -> str``
    Look the execution up (``jobs_executions.list`` or ``job_execution``) and return
    Azure's status string for it: one of the keys of ``STATES`` below.

The quota requests this needs are the owner's to file (BLOCKED.md B9; the research
report's section 6 has the texts).
"""

from __future__ import annotations

from stems_api.providers.base import (
    JobSpec,
    ProviderHandle,
    ProviderStatus,
    WorkerState,
)
from stems_common.jobs import validate_spec

HANDLE_PREFIX = "aca:"
NOT_BUILT = (
    "the azure_container_apps worker provider is a seam and is not implemented: fill in "
    "AzureContainerAppsProvider._start_job and ._execution_state "
    "(api/stems_api/providers/azure_aca.py says how), or set STEMS_WORKER_PROVIDER to "
    "local or modal"
)
# Azure's job-execution statuses, as this interface reports them.
STATES: dict[str, WorkerState] = {
    "Running": "running",
    "Processing": "running",
    "Succeeded": "succeeded",
    "Failed": "failed",
    "Degraded": "failed",
    "Stopped": "cancelled",
    "Unknown": "unknown",
}


class AzureContainerAppsProvider:
    name = "azure_container_apps"

    # -- the two methods to implement ----------------------------------------------------

    def _start_job(self, job_id: str) -> str:
        """Start a Container Apps job execution for ``job_id``; return its name."""
        raise NotImplementedError(NOT_BUILT)

    def _execution_state(self, execution_name: str) -> str:
        """Azure's status string for the execution: a key of ``STATES``."""
        raise NotImplementedError(NOT_BUILT)

    # -- WorkerProvider, already written in terms of those two ---------------------------

    def submit(self, spec: JobSpec) -> ProviderHandle:
        job_id = validate_spec(spec)["job_id"]
        return HANDLE_PREFIX + self._start_job(job_id)

    def status(self, handle: ProviderHandle) -> ProviderStatus:
        if not isinstance(handle, str) or not handle.startswith(HANDLE_PREFIX):
            return ProviderStatus("unknown", "not a Container Apps handle")
        reported = self._execution_state(handle[len(HANDLE_PREFIX) :])
        return ProviderStatus(STATES.get(reported, "unknown"), reported)

    def cancel(self, handle: ProviderHandle) -> None:
        """Stopping an execution is optional for a first version: a job that is deleted
        while it runs has its objects removed by the sweeper's second pass."""
