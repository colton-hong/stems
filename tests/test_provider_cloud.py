"""The Modal provider against a stub of the SDK, and the Azure Container Apps seam."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from stems_api.providers import (
    ProviderConfigError,
    ProviderError,
    ProviderStatus,
    WorkerProvider,
    azure_aca,
    provider_from_env,
)
from stems_api.providers import modal as modal_provider
from stems_api.providers.azure_aca import AzureContainerAppsProvider
from stems_api.providers.modal import ModalProvider, SdkBackend
from stems_common import jobs

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
JOB = "0b9c2f6e-58d4-4c58-9d3b-6f3d1c8a7e21"
FAKE_TOKEN = "ak-fake-token-id"
FAKE_SECRET = "as-fake-token-secret"


def spec(job_id: str = JOB) -> dict:
    return jobs.build_spec(job_id, created_at=NOW, stems=["vocals"])


def env(values: dict[str, str]):
    return lambda name: values.get(name) or None


# -- the provider, on a backend the test controls ----------------------------------------


class FakeBackend:
    def __init__(self) -> None:
        self.spawned: list[str] = []
        self.cancelled: list[str] = []
        self.states: dict[str, tuple[str, str | None]] = {}

    def spawn(self, job_id: str) -> str:
        self.spawned.append(job_id)
        return f"fc-{len(self.spawned)}"

    def poll(self, call_id: str):
        return self.states.get(call_id, ("unknown", "no such call"))

    def cancel(self, call_id: str) -> None:
        self.cancelled.append(call_id)


def test_modal_provider_spawns_polls_and_cancels_by_call_id() -> None:
    backend = FakeBackend()
    provider = ModalProvider(backend)
    assert isinstance(provider, WorkerProvider)
    assert provider.name == "modal"
    handle = provider.submit(spec())
    assert handle == "modal:fc-1"
    assert backend.spawned == [JOB]
    for state, detail in (("running", None), ("succeeded", None), ("failed", "RemoteError")):
        backend.states["fc-1"] = (state, detail)
        assert provider.status(handle) == ProviderStatus(state, detail)
    assert provider.status("modal:fc-9") == ProviderStatus("unknown", "no such call")
    provider.cancel(handle)
    assert backend.cancelled == ["fc-1"]
    # A handle that is not Modal's never reaches the backend.
    for foreign in ("local:" + JOB, "modal:", "", None):
        assert provider.status(foreign).state == "unknown"
        provider.cancel(foreign)
    assert backend.cancelled == ["fc-1"]
    with pytest.raises(jobs.ContractError):
        provider.submit({"job_id": "nonsense"})
    assert backend.spawned == [JOB]


# -- the SDK calls, stubbed --------------------------------------------------------------


class StubCall:
    def __init__(self, outcome=None) -> None:
        self.object_id = "fc-01ABC"
        self.outcome = outcome
        self.get_timeouts: list[float | None] = []
        self.cancelled = 0

    def get(self, timeout=None):
        self.get_timeouts.append(timeout)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def cancel(self):
        self.cancelled += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome


@pytest.fixture
def sdk(monkeypatch):
    """The installed Modal SDK with its two network entry points replaced."""
    modal = pytest.importorskip("modal")
    record = {"from_name": [], "spawned": [], "from_id": [], "call": StubCall()}

    class StubFunction:
        def spawn(self, *args):
            record["spawned"].append(args)
            return record["call"]

    def from_name(app_name, name, **kwargs):
        record["from_name"].append((app_name, name))
        if record.get("refuse") is not None:
            raise record["refuse"]
        return StubFunction()

    def from_id(call_id, **kwargs):
        record["from_id"].append(call_id)
        return record["call"]

    monkeypatch.setattr(modal.Function, "from_name", staticmethod(from_name))
    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(from_id))
    record["modal"] = modal
    return record


def test_sdk_backend_spawns_the_deployed_function_with_the_job_id(sdk) -> None:
    provider = ModalProvider()  # the real backend, on the stubbed SDK
    assert provider.submit(spec()) == "modal:fc-01ABC"
    assert sdk["from_name"] == [("stems-worker", "run_job")]
    assert sdk["spawned"] == [(JOB,)]


def test_sdk_backend_reads_a_calls_state_without_waiting(sdk) -> None:
    modal, backend = sdk["modal"], SdkBackend()
    outcomes = [
        (TimeoutError(), ("running", None)),
        (modal.exception.TimeoutError("not yet"), ("running", None)),
        ("ready", ("succeeded", None)),
        (modal.exception.NotFoundError("gone"), ("unknown", "Modal has no record of the call")),
        (modal.exception.RemoteError("boom"), ("failed", "RemoteError")),
        (RuntimeError("the job raised"), ("failed", "RuntimeError")),
    ]
    for outcome, expected in outcomes:
        sdk["call"] = StubCall(outcome)
        assert backend.poll("fc-01ABC") == expected
        assert sdk["call"].get_timeouts == [0]  # never blocks on the job
    assert set(sdk["from_id"]) == {"fc-01ABC"}


def test_sdk_backend_cancels_and_tolerates_a_call_that_is_already_over(sdk) -> None:
    modal, backend = sdk["modal"], SdkBackend()
    backend.cancel("fc-01ABC")
    assert sdk["call"].cancelled == 1
    sdk["call"] = StubCall(modal.exception.NotFoundError("gone"))
    backend.cancel("fc-01ABC")  # no exception
    assert sdk["call"].cancelled == 1


def test_a_refusal_from_modal_is_a_provider_error_that_shows_no_secret(sdk) -> None:
    sdk["refuse"] = sdk["modal"].exception.AuthError(f"token {FAKE_SECRET} rejected")
    with pytest.raises(ProviderError) as caught:
        ModalProvider().submit(spec())
    assert str(caught.value) == "Modal did not accept the job (AuthError)"
    assert FAKE_SECRET not in str(caught.value)


def test_the_modal_app_declares_the_function_the_provider_looks_up() -> None:
    pytest.importorskip("modal")
    from stems_worker import modal_app

    assert modal_app.APP_NAME == modal_provider.APP_NAME == modal_app.app.name == "stems-worker"
    assert modal_app.FUNCTION_NAME == modal_provider.FUNCTION_NAME == "run_job"
    assert "NOT RUN" in modal_app.__doc__  # it has never been deployed, and says so
    assert ".env" in modal_app._IGNORE  # secrets do not go into the image
    assert "models/weights" in modal_app._IGNORE


# -- choosing a provider -----------------------------------------------------------------


def test_modal_needs_its_token_and_shared_storage(monkeypatch) -> None:
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(name, raising=False)
    base = {"STEMS_WORKER_PROVIDER": "modal"}
    with pytest.raises(ProviderConfigError, match="MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"):
        provider_from_env(env(base))
    with pytest.raises(ProviderConfigError, match="MODAL_TOKEN_SECRET"):
        provider_from_env(env({**base, "MODAL_TOKEN_ID": FAKE_TOKEN}))
    tokens = {**base, "MODAL_TOKEN_ID": FAKE_TOKEN, "MODAL_TOKEN_SECRET": FAKE_SECRET}
    for storage in ({}, {"STEMS_STORAGE_BACKEND": "local"}):
        with pytest.raises(ProviderConfigError, match="needs shared storage") as caught:
            provider_from_env(env({**tokens, **storage}))
        assert FAKE_SECRET not in str(caught.value)
    import os

    assert "MODAL_TOKEN_ID" not in os.environ  # nothing was exported for a refused config
    monkeypatch.setenv("MODAL_TOKEN_ID", "")  # so the test can undo what the factory sets
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "")
    monkeypatch.delenv("MODAL_TOKEN_ID")
    monkeypatch.delenv("MODAL_TOKEN_SECRET")
    provider = provider_from_env(env({**tokens, "STEMS_STORAGE_BACKEND": "s3"}))
    assert isinstance(provider, ModalProvider)
    assert os.environ["MODAL_TOKEN_ID"] == FAKE_TOKEN  # where the SDK looks for it


# -- the Azure Container Apps seam -------------------------------------------------------


def test_selecting_azure_container_apps_fails_with_instructions() -> None:
    with pytest.raises(ProviderConfigError) as caught:
        provider_from_env(env({"STEMS_WORKER_PROVIDER": "azure_container_apps"}))
    message = str(caught.value)
    assert "is a seam and is not implemented" in message
    assert "_start_job" in message
    assert "_execution_state" in message
    assert "api/stems_api/providers/azure_aca.py" in message
    assert "local or modal" in message


def test_the_seam_documents_exactly_two_methods_to_implement() -> None:
    provider = AzureContainerAppsProvider()
    assert provider.name == "azure_container_apps"
    unwritten = [
        name
        for name, member in inspect.getmembers(AzureContainerAppsProvider, inspect.isfunction)
        if "NotImplementedError" in inspect.getsource(member)
    ]
    assert unwritten == ["_execution_state", "_start_job"]
    for name in unwritten:
        assert f"``{name}(" in azure_aca.__doc__  # each is described in the module's docs
        assert getattr(AzureContainerAppsProvider, name).__doc__
    with pytest.raises(NotImplementedError, match="is a seam and is not implemented"):
        provider.submit(spec())
    with pytest.raises(NotImplementedError):
        provider.status("aca:execution-1")
    assert provider.status("local:" + JOB).state == "unknown"


def test_filling_in_the_two_methods_gives_a_working_provider() -> None:
    class Filled(AzureContainerAppsProvider):
        def __init__(self) -> None:
            self.started: list[str] = []
            self.reported = "Running"

        def _start_job(self, job_id: str) -> str:
            self.started.append(job_id)
            return f"stems-worker-{len(self.started)}"

        def _execution_state(self, execution_name: str) -> str:
            return self.reported

    provider = Filled()
    assert isinstance(provider, WorkerProvider)
    handle = provider.submit(spec())
    assert (handle, provider.started) == ("aca:stems-worker-1", [JOB])
    expected = {
        "Running": "running",
        "Processing": "running",
        "Succeeded": "succeeded",
        "Failed": "failed",
        "Degraded": "failed",
        "Stopped": "cancelled",
        "SomethingNew": "unknown",
    }
    for reported, state in expected.items():
        provider.reported = reported
        assert provider.status(handle) == ProviderStatus(state, reported)
    provider.cancel(handle)  # optional in a first version: does nothing, raises nothing
    with pytest.raises(jobs.ContractError):
        provider.submit({"job_id": "nonsense"})
