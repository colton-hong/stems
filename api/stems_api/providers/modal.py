"""The Modal provider: each job is one call of a deployed Modal function.

The function is ``run_job`` in the app ``stems-worker`` (``worker/stems_worker/modal_app.py``,
deployed by the owner with ``modal deploy``). ``submit`` spawns a call and returns its id;
``status`` asks Modal whether that call is still running. The job itself is read from and
written to the shared bucket, so this needs ``STEMS_STORAGE_BACKEND`` to be ``s3`` or
``azure_blob``: a Modal container cannot see this machine's disk.

The Modal SDK is imported only when the provider is used. ``ModalProvider`` talks to it
through the small ``ModalBackend`` interface, which is what the tests replace.

Not run against Modal: there is no account yet (BLOCKED.md B4). What is tested is this
module against a stub of the SDK's calls, using the installed SDK's own classes.
"""

from __future__ import annotations

import contextlib
from typing import Any, Protocol

from stems_api.providers.base import (
    JobSpec,
    ProviderError,
    ProviderHandle,
    ProviderStatus,
    WorkerState,
)
from stems_common.jobs import validate_spec

APP_NAME = "stems-worker"
FUNCTION_NAME = "run_job"
HANDLE_PREFIX = "modal:"


class ModalBackend(Protocol):
    def spawn(self, job_id: str) -> str:
        """Start a call for the job and return the call's id."""

    def poll(self, call_id: str) -> tuple[WorkerState, str | None]:
        """Where the call is, and a detail to log."""

    def cancel(self, call_id: str) -> None: ...


class SdkBackend:
    """``ModalBackend`` on the real SDK: ``Function.from_name(...).spawn``, and
    ``FunctionCall.from_id(...)`` with ``get(timeout=0)`` and ``cancel()``."""

    def __init__(self, app_name: str = APP_NAME, function_name: str = FUNCTION_NAME) -> None:
        self.app_name, self.function_name = app_name, function_name

    @staticmethod
    def _modal() -> Any:
        try:
            import modal
        except ImportError as exc:
            raise ProviderError(
                "the modal worker provider needs the Modal SDK: pip install -e '.[modal]'"
            ) from exc
        return modal

    def spawn(self, job_id: str) -> str:
        modal = self._modal()
        try:
            function = modal.Function.from_name(self.app_name, self.function_name)
            return str(function.spawn(job_id).object_id)
        except modal.exception.Error as exc:
            # The SDK's message never holds the token; its type says what went wrong.
            raise ProviderError(f"Modal did not accept the job ({type(exc).__name__})") from exc

    def poll(self, call_id: str) -> tuple[WorkerState, str | None]:
        modal = self._modal()
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return "running", None  # no result within zero seconds: still going
        except modal.exception.NotFoundError:
            return "unknown", "Modal has no record of the call"
        except Exception as exc:
            # The function raised, was cancelled, or its container was lost.
            return "failed", type(exc).__name__
        return "succeeded", None

    def cancel(self, call_id: str) -> None:
        modal = self._modal()
        # Already over, or never known: then there is nothing left to stop.
        with contextlib.suppress(modal.exception.Error):
            modal.FunctionCall.from_id(call_id).cancel()


class ModalProvider:
    name = "modal"

    def __init__(self, backend: ModalBackend | None = None) -> None:
        self._backend = backend or SdkBackend()

    def submit(self, spec: JobSpec) -> ProviderHandle:
        job_id = validate_spec(spec)["job_id"]
        return HANDLE_PREFIX + self._backend.spawn(job_id)

    def _call_id(self, handle: ProviderHandle) -> str | None:
        if isinstance(handle, str) and handle.startswith(HANDLE_PREFIX):
            return handle[len(HANDLE_PREFIX) :] or None
        return None

    def status(self, handle: ProviderHandle) -> ProviderStatus:
        call_id = self._call_id(handle)
        if call_id is None:
            return ProviderStatus("unknown", "not a Modal handle")
        state, detail = self._backend.poll(call_id)
        return ProviderStatus(state, detail)

    def cancel(self, handle: ProviderHandle) -> None:
        call_id = self._call_id(handle)
        if call_id is not None:
            self._backend.cancel(call_id)
