"""The hardening pass (P7.6): what protects the server from an upload and a user's files
from everyone else. docs/security-review.md says what was looked at and why."""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

import api_helpers as h
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from stems_api import probe
from stems_api.app import create_app
from stems_api.providers import LocalProvider
from stems_api.storage import LocalStorage
from stems_api.sweeper import STALE_TEMP, purge, remove_stale_temp
from stems_api.uploads import UploadRejected, inspect
from stems_common import jobs as contract
from stems_common.settings import RateLimitSettings, load_defaults

JOB = "0b9c2f6e-58d4-4c58-9d3b-6f3d1c8a7e21"


def wav(rate: int = 44100, channels: int = 2, seconds: float = 1.0) -> bytes:
    data = np.zeros((int(seconds * rate), channels), dtype="float32")
    data[::7] = 0.1
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def limits():
    return h.test_defaults().uploads


# -- the decoders run somewhere a bad file cannot hurt -----------------------------------


def test_the_probe_command_answers_in_json_for_audio_and_for_anything_else(tmp_path) -> None:
    good = tmp_path / "upload"
    good.write_bytes(wav())
    done = subprocess.run(
        [sys.executable, "-m", "stems_api.probe", str(good)],
        capture_output=True,
        text=True,
        check=True,
    )
    answer = json.loads(done.stdout)
    assert answer["ok"] is True
    assert (answer["info"]["sample_rate"], answer["info"]["channels"]) == (44100, 2)
    bad = tmp_path / "not-audio"
    bad.write_bytes(b"%PDF-1.7 not audio " * 500)
    done = subprocess.run(
        [sys.executable, "-m", "stems_api.probe", str(bad)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(done.stdout)["ok"] is False
    assert probe.main([]) == 2


def test_an_upload_is_checked_in_a_child_process_the_way_the_server_does_it(
    tmp_path, limits
) -> None:
    good = tmp_path / "upload"
    good.write_bytes(wav(seconds=2.0))
    isolated = inspect(good, limits, isolate=True)
    here = inspect(good, limits, isolate=False)
    assert isolated.info == here.info  # the same answer either way
    assert isolated.info.duration_seconds == 2.0
    bad = tmp_path / "renamed.mp3"
    bad.write_bytes(b"%PDF-1.7 not audio " * 500)
    for isolate in (True, False):
        with pytest.raises(UploadRejected) as caught:
            inspect(bad, limits, isolate=isolate)
        assert caught.value.status_code == 415


def test_a_decoder_that_hangs_or_crashes_refuses_the_upload_and_nothing_else(
    tmp_path, limits, monkeypatch
) -> None:
    path = tmp_path / "upload"
    path.write_bytes(wav())

    def hangs(command, **options):
        raise subprocess.TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(probe.subprocess, "run", hangs)
    with pytest.raises(probe.ProbeFailed, match="took longer than 20 s"):
        probe.look_isolated(path, 20)
    with pytest.raises(UploadRejected) as caught:
        inspect(path, limits, isolate=True)
    assert caught.value.status_code == 415

    def crashes(command, **options):
        return subprocess.CompletedProcess(command, -11, stdout="", stderr="Segmentation fault")

    monkeypatch.setattr(probe.subprocess, "run", crashes)
    with pytest.raises(probe.ProbeFailed, match="stopped unexpectedly"):
        probe.look_isolated(path, 20)


def test_the_time_limit_is_real(tmp_path, monkeypatch) -> None:
    """A child that does not answer is stopped at the limit, not waited for."""
    path = tmp_path / "upload"
    path.write_bytes(wav())
    real_run = subprocess.run

    def slow_child(command, **options):
        return real_run([sys.executable, "-c", "import time; time.sleep(30)"], **options)

    monkeypatch.setattr(probe.subprocess, "run", slow_child)
    started = time.perf_counter()
    with pytest.raises(probe.ProbeFailed, match="took longer"):
        probe.look_isolated(path, 0.5)
    assert time.perf_counter() - started < 5


def test_the_server_checks_uploads_in_isolation_by_default(tmp_path) -> None:
    from stems_api.app import Deps

    assert Deps.__dataclass_fields__["isolate_decoding"].default is True
    deps = h.make_deps(tmp_path)
    deps.isolate_decoding = True
    with TestClient(create_app(deps, background=False)) as client:
        assert h.upload(client).status_code == 201
        refused = h.upload(client, b"%PDF-1.7 not audio " * 500, name="renamed.mp3")
        assert refused.status_code == 415


# -- audio that would decode to something enormous ---------------------------------------


@pytest.mark.parametrize(
    ("rate", "channels", "says"),
    [
        (4000, 2, "A sample rate of 4000 Hz is not supported. Use one from 8000 to 192000 Hz."),
        (384000, 2, "A sample rate of 384000 Hz is not supported."),
        (44100, 10, "Audio with 10 channels is not supported. The most is 8."),
    ],
)
def test_extreme_sample_rates_and_channel_counts_are_refused(tmp_path, rate, channels, says):
    deps = h.make_deps(tmp_path)
    with TestClient(create_app(deps, background=False)) as client:
        response = h.upload(client, wav(rate, channels, seconds=0.2))
    assert response.status_code == 415
    assert says in response.json()["detail"]
    assert h.all_jobs(deps) == []
    assert h.stored_keys(deps) == []


def test_ordinary_rates_and_layouts_are_accepted(tmp_path) -> None:
    deps = h.make_deps(tmp_path)
    with TestClient(create_app(deps, background=False)) as client:
        for index, (rate, channels) in enumerate([(8000, 1), (48000, 2), (96000, 2), (44100, 6)]):
            response = h.upload(client, wav(rate, channels, seconds=0.2))
            assert response.status_code == 201, (rate, channels, response.text)
            assert response.json()["source"]["sample_rate_hz"] == rate
            with deps.sessions.begin() as session:  # keep under the per-browser job limit
                from sqlalchemy import update

                from stems_api.db.models import Job

                session.execute(update(Job).values(status="failed", stage="failed"))
            assert index < 4
    limits = load_defaults().uploads
    assert (limits.min_sample_rate_hz, limits.max_sample_rate_hz, limits.max_channels) == (
        8000,
        192000,
        8,
    )
    assert limits.inspect_timeout_seconds == 20


# -- a queue nobody can fill ---------------------------------------------------------------


def test_uploads_are_turned_away_when_the_whole_server_is_busy(tmp_path) -> None:
    base = load_defaults().rate_limits.model_dump()
    limits = RateLimitSettings(**{**base, "max_jobs_in_progress": 2})
    deps = h.make_deps(tmp_path, defaults=h.test_defaults(rate_limits=limits))
    app = create_app(deps, background=False)
    statuses = []
    for _ in range(3):  # three different browsers
        with TestClient(app) as browser:
            response = h.upload(browser)
            statuses.append(response.status_code)
    assert statuses == [201, 201, 503]
    assert response.headers["retry-after"] == "120"
    assert "busy" in response.json()["detail"]
    assert len(h.all_jobs(deps)) == 2
    assert load_defaults().rate_limits.max_jobs_in_progress == 12


# -- headers -----------------------------------------------------------------------------


def test_every_reply_says_how_it_may_be_handled(tmp_path) -> None:
    deps = h.make_deps(tmp_path)
    with TestClient(create_app(deps, background=False)) as client:
        job_id = h.upload(client).json()["id"]
        replies = [
            client.get("/api/health"),
            client.get("/api/config"),
            client.get("/api/jobs"),
            client.get(f"/api/jobs/{job_id}"),
            client.get("/api/jobs/not-a-job"),
            client.post("/api/requests/parse", json={"text": ""}),
            client.delete("/api/me"),
        ]
    for reply in replies:
        assert reply.headers["x-content-type-options"] == "nosniff", reply.url
        assert reply.headers["referrer-policy"] == "no-referrer", reply.url
        # Answers about a browser's jobs are never kept by a cache.
        assert reply.headers["cache-control"] == "no-store", reply.url


# -- what is on disk -----------------------------------------------------------------------


def test_a_storage_folder_this_code_creates_is_private(tmp_path) -> None:
    root = tmp_path / "storage"
    LocalStorage(root, h.SECRET)
    assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
    # One that was already there keeps the permissions its owner gave it.
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    os.chmod(shared, 0o755)
    LocalStorage(shared, h.SECRET)
    assert stat.S_IMODE(os.stat(shared).st_mode) == 0o755


def test_a_jobs_log_goes_when_the_job_goes(tmp_path) -> None:
    logs = tmp_path / "logs"
    provider = LocalProvider(
        command=lambda job: [sys.executable, "-c", "print('uploaded: My Private Song.wav')"],
        log_dir=logs,
    )
    try:
        from datetime import UTC, datetime

        spec = contract.build_spec(JOB, created_at=datetime.now(UTC), stems=["vocals"])
        assert provider.wait(provider.submit(spec), timeout=30).state == "succeeded"
        log = logs / f"{JOB}.log"
        assert "My Private Song" in log.read_text()

        class Row:
            id, status, provider_handle = JOB, "ready", f"local:{JOB}"

        storage = LocalStorage(tmp_path / "storage", h.SECRET)
        assert purge(storage, provider, Row()) == 0
        assert not log.exists()
        assert provider.forget(JOB) is False  # nothing left to forget
        assert provider.forget("../../etc/passwd") is False
    finally:
        provider.close()
    assert LocalProvider(log_dir=None).forget(JOB) is False


def test_the_tests_never_sweep_the_real_temporary_directory(tmp_path) -> None:
    import tempfile
    from datetime import timedelta
    from pathlib import Path

    from stems_api.sweeper import sweep

    # The run has a temporary directory of its own, and so does every process it starts.
    here = Path(tempfile.gettempdir())
    assert here.name == f"stems-tests-{os.getuid()}"
    assert os.environ["TMPDIR"] == str(here)
    # A sweep with a clock a year ahead removes what looks stale in there, and only there.
    deps = h.make_deps(tmp_path)
    mine = here / "stems-job-of-this-test"
    mine.mkdir(exist_ok=True)
    sweep(deps.sessions, deps.storage, deps.provider, deps.clock() + timedelta(days=365))
    assert not mine.exists()
    assert here.exists()


def test_temporary_folders_left_by_a_dead_process_are_removed(tmp_path) -> None:
    now = time.time()
    old = now - STALE_TEMP.total_seconds() - 60
    stale_upload, stale_job = tmp_path / "stems-upload-abc", tmp_path / "stems-job-def"
    fresh, unrelated = tmp_path / "stems-job-running", tmp_path / "someone-elses-folder"
    for folder in (stale_upload, stale_job, fresh, unrelated):
        folder.mkdir()
        (folder / "source").write_bytes(b"audio")
    for folder in (stale_upload, stale_job, unrelated):
        os.utime(folder, (old, old))
    assert remove_stale_temp(now, tmp_path) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "someone-elses-folder",
        "stems-job-running",
    ]
    assert remove_stale_temp(now, tmp_path) == 0
