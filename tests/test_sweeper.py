"""The retention sweeper, on a clock the test moves."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("torch")

import api_helpers as h
from fastapi.testclient import TestClient
from sqlalchemy import select

from stems_api.app import create_app
from stems_api.db.models import DecisionCall, JobEvent, Stem, Timing
from stems_api.storage import StorageError, put_bytes
from stems_api.sweeper import RECLEAN_WINDOW, SweepReport, sweep
from stems_worker import jobs as worker
from stems_worker.fakes import fake_pipeline


@pytest.fixture
def deps(tmp_path):
    return h.make_deps(tmp_path)


@pytest.fixture
def client(deps):
    with TestClient(create_app(deps, background=False)) as client:
        yield client


def ready_job(client, deps) -> str:
    job_id = h.upload(client).json()["id"]
    options, defaults = fake_pipeline(deps.defaults)
    worker.run_job(
        job_id,
        deps.storage,
        defaults=defaults,
        device="cpu",
        cache_dir=None,
        timings_path=None,
        **options,
    )
    deps.provider.states[f"manual:{job_id}"] = "succeeded"
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "ready"
    return job_id


def run(deps) -> SweepReport:
    return sweep(deps.sessions, deps.storage, deps.provider, deps.clock())


def keys_of(deps, job_id: str) -> list[str]:
    return [key for key in h.stored_keys(deps) if key.startswith(f"jobs/{job_id}/")]


def test_a_job_past_retention_loses_its_objects_and_is_marked_expired(client, deps) -> None:
    old = ready_job(client, deps)
    deps.clock.advance(days=3)
    young = ready_job(client, deps)
    assert deps.defaults.retention.days == 7
    assert len(keys_of(deps, old)) == len(keys_of(deps, young)) == 9

    # One second before the old job's seven days are up: nothing happens.
    deps.clock.advance(days=4, seconds=-1)
    assert run(deps) == SweepReport()
    assert len(keys_of(deps, old)) == 9

    deps.clock.advance(seconds=1)  # exactly seven days after the upload
    report = run(deps)
    assert report == SweepReport(expired_jobs=1, objects_deleted=9, failures=0)
    assert report.changed
    assert keys_of(deps, old) == []
    assert len(keys_of(deps, young)) == 9  # four days left
    expired = client.get(f"/api/jobs/{old}").json()
    assert (expired["status"], expired["stage"], expired["name"]) == ("expired", "expired", None)
    assert expired["stems"] == []
    assert expired["seconds_until_expiry"] == 0
    assert [j["id"] for j in client.get("/api/jobs").json()["jobs"]] == [young]
    assert client.post(f"/api/jobs/{old}/links").status_code == 409
    with deps.sessions() as session:
        for model in (Stem, JobEvent, DecisionCall):
            left = session.scalars(select(model.job_id)).unique().all()
            assert old not in left, model.__name__
        # Timings are kept: model, device and durations, nothing about the audio itself.
        assert old in session.scalars(select(Timing.job_id)).all()
    untouched = client.get(f"/api/jobs/{young}").json()
    assert untouched["status"] == "ready"
    assert len(untouched["stems"]) == 2

    # Running it again changes nothing more.
    before = (h.stored_keys(deps), client.get("/api/jobs").json(), h.all_jobs(deps)[0].updated_at)
    second = run(deps)
    assert second == SweepReport()
    assert not second.changed
    assert (
        h.stored_keys(deps),
        client.get("/api/jobs").json(),
        h.all_jobs(deps)[0].updated_at,
    ) == before


def test_links_made_before_expiry_stop_working(client, deps) -> None:
    job_id = ready_job(client, deps)
    deps.clock.advance(days=7, minutes=-5)
    url = client.post(f"/api/jobs/{job_id}/links").json()["stems"][0]["url"]
    assert client.get(url).status_code == 200
    deps.clock.advance(minutes=5)
    run(deps)
    # The link has ten of its fifteen minutes left; the file is gone all the same.
    assert client.get(url).status_code == 404


def test_an_expired_job_still_in_progress_has_its_worker_stopped(client, deps) -> None:
    job_id = h.upload(client).json()["id"]
    deps.provider.states[f"manual:{job_id}"] = "running"
    deps.clock.advance(days=7)
    assert run(deps).expired_jobs == 1
    assert deps.provider.cancelled == [f"manual:{job_id}"]
    assert h.stored_keys(deps) == []
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "expired"


def test_the_sweeper_runs_when_the_api_starts(tmp_path) -> None:
    deps = h.make_deps(tmp_path)
    with TestClient(create_app(deps, background=False)) as client:
        job_id = ready_job(client, deps)
    deps.clock.advance(days=8)
    again = h.make_deps(tmp_path, clock=deps.clock)
    with TestClient(create_app(again, background=False)):
        assert h.stored_keys(again) == []
        assert h.all_jobs(again)[0].status == "expired"
        assert h.all_jobs(again)[0].id == job_id


def test_a_storage_failure_leaves_the_job_for_the_next_sweep(client, deps) -> None:
    broken = ready_job(client, deps)
    fine = ready_job(client, deps)
    deps.clock.advance(days=7)
    real_delete = deps.storage.delete_prefix

    def flaky(prefix: str) -> int:
        if broken in prefix:
            raise StorageError("the bucket did not answer")
        return real_delete(prefix)

    deps.storage.delete_prefix = flaky
    report = run(deps)
    assert (report.expired_jobs, report.failures) == (1, 1)
    statuses = {job.id: job.status for job in h.all_jobs(deps)}
    assert statuses == {broken: "ready", fine: "expired"}  # not marked expired: it still has files
    assert len(keys_of(deps, broken)) == 9
    deps.storage.delete_prefix = real_delete
    assert run(deps) == SweepReport(expired_jobs=1, objects_deleted=9, failures=0)
    assert h.stored_keys(deps) == []
    assert run(deps) == SweepReport()


def test_objects_written_after_a_delete_are_removed_by_the_next_sweeps(client, deps) -> None:
    """A worker that was mid-write when its job was deleted cannot leave audio behind."""
    job_id = ready_job(client, deps)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert h.stored_keys(deps) == []
    stray = f"jobs/{job_id}/stems/vocals.wav"
    put_bytes(deps.storage, stray, b"late write", "audio/wav")
    deps.clock.advance(hours=1)
    assert run(deps) == SweepReport(expired_jobs=0, objects_deleted=1, failures=0)
    assert h.stored_keys(deps) == []
    assert run(deps) == SweepReport()
    # After the window the prefix is no longer visited.
    put_bytes(deps.storage, stray, b"very late", "audio/wav")
    deps.clock.advance(seconds=RECLEAN_WINDOW.total_seconds())
    assert run(deps) == SweepReport()
    assert h.stored_keys(deps) == [stray]


def test_the_maintenance_thread_syncs_and_sweeps_on_its_own(tmp_path) -> None:
    import time

    defaults = h.test_defaults()
    defaults = defaults.model_copy(
        update={"api": defaults.api.model_copy(update={"sweep_interval_seconds": 0.1})}
    )
    deps = h.make_deps(tmp_path, defaults=defaults)
    with TestClient(create_app(deps, background=True)) as client:
        job_id = h.upload(client).json()["id"]
        options, fake_defaults = fake_pipeline(deps.defaults)
        worker.run_job(
            job_id,
            deps.storage,
            defaults=fake_defaults,
            device="cpu",
            cache_dir=None,
            timings_path=None,
            **options,
        )
        deps.provider.states[f"manual:{job_id}"] = "succeeded"

        def wait_for(status: str) -> None:
            deadline = time.time() + 30
            while h.all_jobs(deps)[0].status != status:  # read the row, not the route
                assert time.time() < deadline, f"never became {status}"
                time.sleep(0.02)

        wait_for("ready")  # nobody asked for the job: the thread synced it
        deps.clock.advance(days=7)
        wait_for("expired")
        assert h.stored_keys(deps) == []
