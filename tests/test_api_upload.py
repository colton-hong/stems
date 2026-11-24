"""POST /jobs and the public routes: the owner cookie, upload checks, queueing."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

import api_helpers as h
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from stems_api.app import OWNER_COOKIE, create_app
from stems_api.db.models import JobEvent, Owner
from stems_api.db.store import hash_token
from stems_api.providers import LocalProvider
from stems_api.storage import read_json
from stems_common import jobs as contract
from stems_common.paths import REPO_ROOT


@pytest.fixture
def deps(tmp_path):
    return h.make_deps(tmp_path)


@pytest.fixture
def client(deps):
    with TestClient(create_app(deps, background=False)) as client:
        yield client


# -- public routes -----------------------------------------------------------------------


def test_health_and_config(client, deps) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    config = client.get("/api/config").json()
    assert config["stems"] == ["vocals", "drums", "bass", "guitar", "piano", "other"]
    assert config["formats"] == ["wav", "flac", "mp3_320", "mp3_192"]
    assert config["uploads"]["max_bytes"] == h.MAX_BYTES
    assert config["uploads"]["max_duration_seconds"] == h.MAX_SECONDS
    assert "mp3" in config["uploads"]["allowed_extensions"]
    assert config["retention_days"] == deps.defaults.retention.days == 7
    assert config["link_ttl_seconds"] == 900
    # Neither route makes an owner or sets a cookie.
    assert OWNER_COOKIE not in client.cookies
    with deps.sessions() as session:
        assert session.scalars(select(Owner)).all() == []


def test_the_api_does_not_load_torch() -> None:
    code = "import sys, stems_api.app; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=False).returncode == 0


# -- a good upload -----------------------------------------------------------------------


def test_an_upload_becomes_a_queued_job(client, deps) -> None:
    data = h.wav_bytes(seconds=2.0)
    response = h.upload(client, data, stems="drums, vocals", format="flac", dereverb="true")
    assert response.status_code == 201, response.text
    job = response.json()
    assert (job["status"], job["stage"], job["name"]) == ("queued", "queued", "My Song.wav")
    assert job["request"] == {
        "stems": ["vocals", "drums"],
        "text": None,
        "split_vocals": False,
        "dereverb": True,
        "format": "flac",
        "quality_mode": "unspecified",
    }
    assert job["source"] == {
        "bytes": len(data),
        "duration_seconds": 2.0,
        "sample_rate_hz": 44100,
        "channels": 2,
    }
    assert job["stems"] == [
        {"name": "vocals", "state": "waiting", "attempt": 0},
        {"name": "drums", "state": "waiting", "attempt": 0},
    ]
    assert job["seconds_until_expiry"] == 7 * 86400
    assert job["error"] is None

    # The source and a valid spec are in storage, under the job's own prefix only.
    job_id = job["id"]
    assert h.stored_keys(deps) == [f"jobs/{job_id}/source", f"jobs/{job_id}/spec.json"]
    spec = contract.validate_spec(read_json(deps.storage, contract.spec_key(job_id)), job_id)
    assert spec["request"] == job["request"]
    assert spec["source"] == {"key": f"jobs/{job_id}/source", "bytes": len(data)}
    with deps.storage.open(contract.source_key(job_id)) as stored:
        assert stored.read() == data
    # The provider was handed that spec, and the row remembers the handle.
    assert deps.provider.submitted == [spec]
    (row,) = h.all_jobs(deps)
    assert (row.provider, row.provider_handle) == ("manual", f"manual:{job_id}")
    assert row.source_key == f"jobs/{job_id}/source"
    with deps.sessions() as session:
        stages = [e.stage for e in session.scalars(select(JobEvent).order_by(JobEvent.id))]
    assert stages == ["uploaded", "queued"]


def test_object_keys_never_contain_the_uploaded_name(client, deps) -> None:
    response = h.upload(client, name="../../Private Demo (final).wav")
    assert response.status_code == 201
    assert response.json()["name"] == "Private Demo (final).wav"  # a label only
    assert not any("Private" in key or "Demo" in key for key in h.stored_keys(deps))


def test_a_request_in_words_is_passed_to_the_worker(client, deps) -> None:
    response = h.upload(client, request_text="  just the vocals please  ")
    assert response.status_code == 201
    assert response.json()["request"]["text"] == "just the vocals please"
    assert response.json()["request"]["stems"] is None
    assert deps.provider.submitted[0]["request"]["text"] == "just the vocals please"
    assert h.upload(client, stems="all").json()["request"]["stems"] == list(
        client.get("/api/config").json()["stems"]
    )


# -- the owner cookie --------------------------------------------------------------------


def test_first_upload_sets_an_http_only_same_site_cookie(client, deps) -> None:
    response = h.upload(client)
    cookie = response.headers["set-cookie"]
    token = client.cookies[OWNER_COOKIE]
    assert cookie.startswith(f"{OWNER_COOKIE}={token};")
    lowered = cookie.lower()
    assert "httponly" in lowered
    assert "samesite=lax" in lowered
    assert "path=/api" in lowered
    assert f"max-age={400 * 86400}" in lowered
    assert "secure" not in lowered  # this request was plain HTTP
    assert len(token) >= 43
    # The database holds the hash of the token and nowhere the token.
    with deps.sessions() as session:
        (owner,) = session.scalars(select(Owner)).all()
        dump = repr(session.execute(text("SELECT * FROM owners")).all())
    assert owner.token_hash == hash_token(token)
    assert token not in dump
    # The same browser keeps its owner: no new cookie, no new owner.
    again = h.upload(client)
    assert "set-cookie" not in again.headers
    with deps.sessions() as session:
        assert len(session.scalars(select(Owner)).all()) == 1
    assert len({job.owner_id for job in h.all_jobs(deps)}) == 1


def test_the_cookie_is_secure_over_https(deps) -> None:
    with TestClient(create_app(deps, background=False), base_url="https://testserver") as client:
        cookie = h.upload(client).headers["set-cookie"].lower()
    assert "secure" in cookie


def test_an_unknown_cookie_is_a_stranger(client, deps) -> None:
    h.upload(client)
    client.cookies.set(OWNER_COOKIE, "not-a-real-token")
    assert client.get("/api/jobs").json() == {"jobs": []}
    with deps.sessions() as session:
        assert len(session.scalars(select(Owner)).all()) == 1  # reading made no owner


# -- refused uploads ---------------------------------------------------------------------


def refused(client, deps, response, status: int, says: str) -> None:
    assert response.status_code == status, response.text
    assert says in response.json()["detail"]
    # Nothing was kept: no job, no object, no owner, no cookie.
    assert h.all_jobs(deps) == []
    assert h.stored_keys(deps) == []
    assert deps.provider.submitted == []
    assert "set-cookie" not in response.headers
    with deps.sessions() as session:
        assert session.scalars(select(Owner)).all() == []


def test_a_file_type_that_is_not_audio_is_refused_by_name(client, deps) -> None:
    refused(client, deps, h.upload(client, name="notes.pdf"), 415, "Unsupported file type")
    refused(client, deps, h.upload(client, name="noextension"), 415, "Unsupported file type")


def test_a_renamed_non_audio_file_is_refused_by_decoding(client, deps) -> None:
    pdf = b"%PDF-1.7\n" + b"1 0 obj << /Type /Catalog >> endobj\n" * 2000
    for name in ("document.mp3", "document.wav", "document.flac", "document.m4a"):
        refused(client, deps, h.upload(client, pdf, name=name), 415, "could not be read as audio")
    text_file = ("lyrics " * 5000).encode()
    refused(client, deps, h.upload(client, text_file, name="song.ogg"), 415, "could not be read")
    # A WAV header with nothing after it parses as audio and decodes to nothing.
    refused(client, deps, h.upload(client, h.wav_bytes()[:44], name="empty.wav"), 415, "could not")


def test_audio_is_accepted_whatever_the_extension_says(client) -> None:
    """The decoders read the content: a WAV named .mp3 is still audio."""
    response = h.upload(client, h.wav_bytes(), name="actually-a-wav.mp3")
    assert response.status_code == 201
    assert response.json()["source"]["sample_rate_hz"] == 44100


def test_an_oversize_upload_is_refused(client, deps) -> None:
    big = h.wav_bytes(seconds=9.5)  # under the duration limit, over the size limit
    assert len(big) > h.MAX_BYTES
    refused(client, deps, h.upload(client, big), 413, "larger than the limit")


def test_an_oversize_upload_is_cut_off_even_without_a_content_length(client, deps) -> None:
    sent = 0

    def body():
        nonlocal sent
        yield b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        for _ in range(4000):  # up to 40 MB if nothing stops it
            sent += 10_000
            yield b"\x00" * 10_000
        yield b"\r\n--x--\r\n"

    headers = {"content-type": "multipart/form-data; boundary=x"}
    response = client.post("/api/jobs", content=body(), headers=headers)
    refused(client, deps, response, 413, "larger than the limit")


def test_an_empty_file_is_refused(client, deps) -> None:
    refused(client, deps, h.upload(client, b""), 400, "empty")


def test_overlong_audio_is_refused(client, deps) -> None:
    # 12 seconds at 8 kHz mono: small in bytes, too long in time.
    long_ = h.wav_bytes(seconds=12.0, sample_rate=8000, channels=1)
    assert len(long_) < h.MAX_BYTES
    refused(client, deps, h.upload(client, long_), 422, "longer than the limit")


@pytest.mark.parametrize(
    ("form", "says"),
    [
        ({"stems": "vocals,kazoo"}, "Unknown stems: kazoo"),
        ({"stems": ""}, "either the stems"),
        ({"stems": "vocals", "request_text": "and drums"}, "either the stems"),
        ({"stems": "vocals", "format": "ogg"}, "Unknown format"),
        ({"stems": "vocals", "quality_mode": "turbo"}, "Unknown quality mode"),
        ({"request_text": "x" * 2001}, "longer than 2000"),
    ],
)
def test_bad_options_are_refused(client, deps, form, says) -> None:
    refused(client, deps, h.upload(client, **form), 422, says)


def test_a_request_without_a_file_is_refused(client, deps) -> None:
    response = client.post("/api/jobs", data={"stems": "vocals"})
    assert response.status_code == 422
    assert h.all_jobs(deps) == []


# -- cross-site requests -----------------------------------------------------------------


def test_unsafe_requests_from_another_site_are_refused(client, deps) -> None:
    files = {"file": ("a.wav", h.wav_bytes(), "audio/wav")}
    evil = client.post(
        "/api/jobs", files=files, data={"stems": "vocals"}, headers={"Origin": "https://evil.test"}
    )
    refused(client, deps, evil, 403, "origin")
    assert client.delete("/api/me", headers={"Origin": "https://evil.test"}).status_code == 403
    for origin in ("http://localhost:3000", "http://testserver"):
        ok = client.post(
            "/api/jobs", files=files, data={"stems": "vocals"}, headers={"Origin": origin}
        )
        assert ok.status_code == 201
    # Reading is not blocked by the guard, and CORS only names the web app.
    assert client.get("/api/health", headers={"Origin": "https://evil.test"}).status_code == 200
    preflight = client.options(
        "/api/jobs",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert preflight.headers["access-control-allow-credentials"] == "true"
    stranger = client.get("/api/health", headers={"Origin": "https://evil.test"})
    assert "access-control-allow-origin" not in stranger.headers


# -- when the job cannot start -----------------------------------------------------------


def test_a_provider_that_refuses_leaves_a_failed_job_and_no_objects(client, deps) -> None:
    deps.provider.refuse = h.refusal()
    response = h.upload(client)
    assert response.status_code == 503
    assert "could not be started" in response.json()["detail"]
    (row,) = h.all_jobs(deps)
    assert (row.status, row.stage) == ("failed", "failed")
    assert "could not be started" in row.error
    assert h.stored_keys(deps) == []
    # The owner can see what happened.
    (listed,) = client.get("/api/jobs").json()["jobs"]
    assert listed["status"] == "failed"


# -- a slow job does not hold up the API -------------------------------------------------


def test_a_slow_job_does_not_delay_other_requests(tmp_path) -> None:
    """With the real local provider and a worker that takes 2 seconds: the upload returns
    and unrelated requests are answered while that worker is still running."""
    marker = tmp_path / "worker"
    code = (
        "import sys, time, pathlib\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "m.with_suffix('.start').write_text('x')\n"
        "time.sleep(2.0)\n"
        "m.with_suffix('.end').write_text('x')\n"
    )
    provider = LocalProvider(command=lambda job: [sys.executable, "-c", code], log_dir=None)
    deps = h.make_deps(tmp_path, provider=provider)
    try:
        with TestClient(create_app(deps, background=False)) as client:
            client.get("/api/health")  # warm up, so the bound below times requests only
            started = time.perf_counter()
            assert h.upload(client).status_code == 201
            upload_seconds = time.perf_counter() - started
            deadline = time.time() + 30
            while not marker.with_suffix(".start").exists():
                assert time.time() < deadline, "the worker never started"
                time.sleep(0.01)
            timings = []
            for path in ("/api/health", "/api/config", "/api/jobs", "/api/health"):
                started = time.perf_counter()
                assert client.get(path).status_code == 200
                timings.append(time.perf_counter() - started)
            still_running = not marker.with_suffix(".end").exists()
        assert still_running, "the slow worker finished before the requests were timed"
        assert upload_seconds < 1.0, f"the upload waited {upload_seconds:.2f}s"
        assert max(timings) < 0.5, f"a request took {max(timings):.2f}s behind a 2 s job"
    finally:
        provider.close()
