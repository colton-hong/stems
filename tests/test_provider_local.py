"""The local worker provider: subprocesses, one at a time, never in the caller's thread."""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime

import pytest

from stems_api.providers import (
    LocalProvider,
    ProviderConfigError,
    ProviderError,
    ProviderStatus,
    WorkerProvider,
    provider_from_env,
)
from stems_api.providers.local import worker_command
from stems_common import jobs

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
JOB_A = "0b9c2f6e-58d4-4c58-9d3b-6f3d1c8a7e21"
JOB_B = "7d1e9a34-2f0b-4b6c-8c59-1a2b3c4d5e6f"
JOB_C = "c3a1f5d2-9b47-4e0a-a1c6-5e2d7f8b9a10"
# submit() must come back at once. The slow job in the test below takes 1.5 seconds, so a
# submit or a status call that waited on it would miss this bound six times over.
SUBMIT_BOUND_SECONDS = 0.25


def spec(job_id: str) -> dict:
    return jobs.build_spec(job_id, created_at=NOW, stems=["vocals"])


def script(code: str):
    """A provider command that runs ``code`` with the job id in ``sys.argv[1]``."""
    return lambda job_id: [sys.executable, "-c", code, job_id]


def marker_script(tmp_path, seconds: float = 0.0) -> str:
    """Writes <job>.start and <job>.end with the times it started and finished."""
    return (
        "import sys, time, pathlib\n"
        f"d = pathlib.Path({str(tmp_path)!r})\n"
        "(d / (sys.argv[1] + '.start')).write_text(repr(time.time()))\n"
        f"time.sleep({seconds})\n"
        "(d / (sys.argv[1] + '.end')).write_text(repr(time.time()))\n"
    )


@pytest.fixture
def make_provider(tmp_path):
    made = []

    def make(code: str, **options) -> LocalProvider:
        provider = LocalProvider(command=script(code), log_dir=tmp_path / "logs", **options)
        made.append(provider)
        return provider

    yield make
    for provider in made:
        provider.close()


def test_it_is_a_worker_provider(make_provider) -> None:
    provider = make_provider("pass")
    assert isinstance(provider, WorkerProvider)
    assert provider.name == "local"


def test_a_job_runs_in_a_subprocess_and_its_output_is_logged(make_provider, tmp_path) -> None:
    provider = make_provider("import os, sys; print('worker', sys.argv[1], os.getpid())")
    handle = provider.submit(spec(JOB_A))
    assert handle == f"local:{JOB_A}"
    assert provider.wait(handle, timeout=30) == ProviderStatus("succeeded")
    words = (tmp_path / "logs" / f"{JOB_A}.log").read_text().split()
    assert words[:2] == ["worker", JOB_A]
    import os

    assert int(words[2]) != os.getpid()


def test_submit_returns_at_once_while_a_slow_job_runs(make_provider, tmp_path) -> None:
    provider = make_provider(marker_script(tmp_path, seconds=1.5))
    started = time.perf_counter()
    first = provider.submit(spec(JOB_A))
    assert time.perf_counter() - started < SUBMIT_BOUND_SECONDS
    # Wait until the slow job is really running, then do unrelated things.
    deadline = time.time() + 30
    while not (tmp_path / f"{JOB_A}.start").exists():
        assert time.time() < deadline, "the worker never started"
        time.sleep(0.01)
    assert provider.status(first).state == "running"
    started = time.perf_counter()
    second = provider.submit(spec(JOB_B))
    assert provider.status(second).state == "queued"
    assert provider.status(first).state == "running"
    assert provider.status("local:" + JOB_C).state == "unknown"
    elapsed = time.perf_counter() - started
    assert elapsed < SUBMIT_BOUND_SECONDS, f"the caller was held for {elapsed:.3f}s"
    assert not (tmp_path / f"{JOB_A}.end").exists()  # the slow job is still going


def test_jobs_run_one_at_a_time_in_the_order_submitted(make_provider, tmp_path) -> None:
    provider = make_provider(marker_script(tmp_path, seconds=0.3))
    handles = [provider.submit(spec(job)) for job in (JOB_A, JOB_B, JOB_C)]
    for handle in handles:
        assert provider.wait(handle, timeout=60).state == "succeeded"

    def times(job: str) -> tuple[float, float]:
        return tuple(float((tmp_path / f"{job}.{end}").read_text()) for end in ("start", "end"))

    (a0, a1), (b0, b1), (c0, c1) = times(JOB_A), times(JOB_B), times(JOB_C)
    assert a0 < a1 <= b0 < b1 <= c0 < c1


def test_a_worker_that_fails_is_reported_and_the_queue_moves_on(make_provider) -> None:
    code = f"import sys; sys.exit(3 if sys.argv[1] == {JOB_A!r} else 0)"
    provider = make_provider(code)
    bad, good = provider.submit(spec(JOB_A)), provider.submit(spec(JOB_B))
    assert provider.wait(bad, timeout=30) == ProviderStatus(
        "failed", "the worker exited with code 3"
    )
    assert provider.wait(good, timeout=30).state == "succeeded"
    assert provider.status(bad).over
    assert not ProviderStatus("running").over


def test_a_command_that_cannot_start_is_a_failed_worker(tmp_path) -> None:
    provider = LocalProvider(command=lambda job: [str(tmp_path / "no-such-program")], log_dir=None)
    try:
        status = provider.wait(provider.submit(spec(JOB_A)), timeout=30)
        assert status.state == "failed"
        assert "could not be started" in status.detail
    finally:
        provider.close()


def test_cancel_stops_a_running_worker_and_drops_a_queued_one(make_provider, tmp_path) -> None:
    provider = make_provider(marker_script(tmp_path, seconds=60))
    running, queued = provider.submit(spec(JOB_A)), provider.submit(spec(JOB_B))
    deadline = time.time() + 30
    while not (tmp_path / f"{JOB_A}.start").exists():
        assert time.time() < deadline
        time.sleep(0.01)
    provider.cancel(queued)
    assert provider.status(queued) == ProviderStatus("cancelled", "cancelled before it started")
    started = time.perf_counter()
    provider.cancel(running)
    assert provider.wait(running, timeout=30) == ProviderStatus("cancelled", "stopped on request")
    assert time.perf_counter() - started < 10  # terminated, not waited out
    assert not (tmp_path / f"{JOB_A}.end").exists()
    assert not (tmp_path / f"{JOB_B}.start").exists()  # the cancelled job never ran
    # Cancelling again, or something unknown, is harmless.
    provider.cancel(running)
    provider.cancel(queued)
    provider.cancel("local:" + JOB_C)
    provider.cancel("modal:whatever")


def test_submitting_a_job_twice_does_not_run_it_twice(make_provider, tmp_path) -> None:
    counter = tmp_path / "count"
    code = (
        "import sys, time, pathlib\n"
        f"p = pathlib.Path({str(counter)!r})\n"
        "p.write_text(p.read_text() + 'x' if p.exists() else 'x')\n"
        "time.sleep(0.3)\n"
    )
    provider = make_provider(code)
    first = provider.submit(spec(JOB_A))
    assert provider.submit(spec(JOB_A)) == first  # still queued or running: the same worker
    assert provider.wait(first, timeout=30).state == "succeeded"
    assert counter.read_text() == "x"
    # Once it is over, submitting again starts a new worker (the worker itself is what
    # refuses to redo a finished job).
    again = provider.submit(spec(JOB_A))
    assert provider.wait(again, timeout=30).state == "succeeded"
    assert counter.read_text() == "xx"


def test_the_worker_gets_the_extra_environment(make_provider, tmp_path) -> None:
    out = tmp_path / "env.txt"
    code = f"import os, pathlib; pathlib.Path({str(out)!r}).write_text(os.environ['STEMS_X'])"
    provider = make_provider(code, env={"STEMS_X": "from-the-api"})
    assert provider.wait(provider.submit(spec(JOB_A)), timeout=30).state == "succeeded"
    assert out.read_text() == "from-the-api"


def test_close_stops_everything_and_refuses_new_work(tmp_path) -> None:
    provider = LocalProvider(command=script(marker_script(tmp_path, seconds=60)), log_dir=None)
    handle = provider.submit(spec(JOB_A))
    deadline = time.time() + 30
    while provider.status(handle).state != "running":
        assert time.time() < deadline
        time.sleep(0.01)
    provider.close()
    assert provider.status(handle).state == "cancelled"
    with pytest.raises(ProviderError, match="shut down"):
        provider.submit(spec(JOB_B))


def test_a_bad_job_id_never_reaches_a_command_line(make_provider) -> None:
    provider = make_provider("pass")
    with pytest.raises(jobs.ContractError):
        provider.submit({"job_id": "; rm -rf ~"})
    assert provider.status("local:; rm -rf ~").state == "unknown"


def test_the_default_command_runs_the_worker_module() -> None:
    assert worker_command(JOB_A) == [sys.executable, "-m", "stems_worker.jobs", JOB_A]


def test_provider_from_env() -> None:
    provider = provider_from_env(lambda name: None)
    assert isinstance(provider, LocalProvider)
    provider.close()
    with pytest.raises(ProviderConfigError, match="not a worker provider"):
        provider_from_env({"STEMS_WORKER_PROVIDER": "kubernetes"}.get)
