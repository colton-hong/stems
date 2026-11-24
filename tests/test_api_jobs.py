"""The job routes: list, status, events, links, downloads, peaks, delete, delete-all."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("torch")

import api_helpers as h
from fastapi.testclient import TestClient
from sqlalchemy import select

from stems_api.app import OWNER_COOKIE, create_app
from stems_api.db.models import DecisionCall, Job, JobEvent, Owner, Stem, Timing
from stems_api.storage import put_bytes, put_json
from stems_api.sync import BAD_RESULT, NEEDS_REVIEW, WORKER_LOST
from stems_common import jobs as contract
from stems_worker import jobs as worker
from stems_worker.fakes import fake_pipeline

UNCHECKED = "This stem could not be quality-checked automatically."


@pytest.fixture
def deps(tmp_path):
    return h.make_deps(tmp_path)


@pytest.fixture
def client(deps):
    with TestClient(create_app(deps, background=False)) as client:
        yield client


@pytest.fixture
def stranger(deps):
    """A second browser on the same API."""
    with TestClient(create_app(deps, background=False)) as client:
        yield client


def run_worker(deps, job_id: str, **overrides) -> dict:
    """Do in this process what the worker subprocess would, then report it as exited."""
    options, defaults = fake_pipeline(deps.defaults)
    settings = {"device": "cpu", "cache_dir": None, "timings_path": None, **options, **overrides}
    result = worker.run_job(job_id, deps.storage, defaults=defaults, **settings)
    deps.provider.states[f"manual:{job_id}"] = "succeeded"
    return result


def ready_job(client, deps, **form) -> dict:
    job_id = h.upload(client, **form).json()["id"]
    run_worker(deps, job_id)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "ready", job
    return job


def count(deps, model, job_id: str) -> int:
    with deps.sessions() as session:
        return len(session.scalars(select(model).where(model.job_id == job_id)).all())


def sse(text: str) -> list[tuple[str, dict]]:
    """A server-sent-event stream as (event name, data) pairs."""
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


# -- from queued to ready ----------------------------------------------------------------


def test_a_job_is_synced_from_what_the_worker_wrote(client, deps) -> None:
    job_id = h.upload(client, name="My Song.wav", format="flac").json()["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    result = run_worker(deps, job_id)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert (job["status"], job["stage"], job["error"], job["progress"]) == (
        "ready",
        "ready",
        None,
        None,
    )
    assert [stem["name"] for stem in job["stems"]] == ["vocals", "drums"]
    for stem, written in zip(job["stems"], result["stems"], strict=True):
        assert stem == {
            "name": written["name"],
            "state": "delivered_with_warning",
            "attempt": 0,
            "format": "flac",
            "bytes": written["bytes"],
            "warning": UNCHECKED,
            "quality": {
                "score": written["assessment"]["assessment"]["score"],
                "score_label": written["assessment"]["assessment"]["score_label"],
                "confidence": written["assessment"]["assessment"]["confidence"],
                "reason_code": written["assessment"]["assessment"]["reason_code"],
            },
            "model_ids": ["fake_bands"],
            "ensemble_mode": "single_best",
        }
    assert job["source"]["duration_seconds"] == 1.0
    # The rows behind it.
    (row,) = h.all_jobs(deps)
    assert [(s.name, s.object_key) for s in row.stems] == [
        ("vocals", f"jobs/{job_id}/stems/vocals.flac"),
        ("drums", f"jobs/{job_id}/stems/drums.flac"),
    ]
    assert row.plan == result["plan"]
    assert row.track_features == result["track_features"]
    assert count(deps, DecisionCall, job_id) == len(result["decisions"]) == 4
    assert count(deps, Timing, job_id) == len(result["timings"]) == 1
    with deps.sessions() as session:
        events = list(session.scalars(select(JobEvent).order_by(JobEvent.id)))
        timing = session.scalars(select(Timing)).one()
    assert [e.stage for e in events if e.seq is None] == ["uploaded", "queued"]
    assert [e.seq for e in events if e.seq is not None] == list(range(1, len(events) - 1))
    assert events[-1].stage == "ready"
    assert (timing.model_id, timing.device, timing.audio_seconds) == ("fake_bands", "cpu", 1.0)
    # Asking again changes nothing: no event is recorded twice.
    before = count(deps, JobEvent, job_id)
    assert client.get(f"/api/jobs/{job_id}").json() == job
    assert count(deps, JobEvent, job_id) == before
    assert count(deps, Stem, job_id) == 2


def test_a_running_job_shows_its_stage_and_each_stem(client, deps) -> None:
    job_id = h.upload(client).json()["id"]
    at = contract.timestamp(deps.clock())
    status = {
        "version": 1,
        "job_id": job_id,
        "state": "running",
        "stage": "assessing",
        "updated_at": at,
        "progress": 0.25,
        "stems": {
            "vocals": {"state": "retrying", "attempt": 1},
            "drums": {"state": "separating", "attempt": 0},
        },
        "events": [
            {"seq": 1, "at": at, "stage": "transcoding"},
            {"seq": 2, "at": at, "stage": "separating"},
            {"seq": 3, "at": at, "stage": "retrying", "stem": "vocals", "message": "retrying"},
        ],
        "error": None,
    }
    put_json(deps.storage, contract.status_key(job_id), status)
    deps.provider.states[f"manual:{job_id}"] = "running"
    job = client.get(f"/api/jobs/{job_id}").json()
    assert (job["status"], job["stage"], job["progress"]) == ("running", "assessing", 0.25)
    assert job["stems"] == [
        {"name": "vocals", "state": "retrying", "attempt": 1},
        {"name": "drums", "state": "separating", "attempt": 0},
    ]
    assert count(deps, JobEvent, job_id) == 5
    # One more event arrives; only that one is added.
    status["events"].append({"seq": 4, "at": at, "stage": "packaging"})
    status["stage"] = "packaging"
    put_json(deps.storage, contract.status_key(job_id), status)
    assert client.get(f"/api/jobs/{job_id}").json()["stage"] == "packaging"
    assert count(deps, JobEvent, job_id) == 6
    # The list shows the same state without reading storage again.
    (listed,) = client.get("/api/jobs").json()["jobs"]
    assert (listed["stage"], listed["stems"][0]["state"]) == ("packaging", "retrying")


def test_a_status_that_is_not_valid_is_ignored(client, deps) -> None:
    job_id = h.upload(client).json()["id"]
    deps.provider.states[f"manual:{job_id}"] = "running"
    for junk in (b"{not json", json.dumps({"job_id": job_id, "state": "ready"}).encode()):
        put_bytes(deps.storage, contract.status_key(job_id), junk, "application/json")
        job = client.get(f"/api/jobs/{job_id}").json()
        assert (job["status"], job["stage"]) == ("queued", "queued")
    # A status written for another job is not believed either.
    other = h.upload(client).json()["id"]
    at = contract.timestamp(deps.clock())
    foreign = {
        "version": 1,
        "job_id": other,
        "state": "running",
        "stage": "packaging",
        "updated_at": at,
        "stems": {},
        "events": [],
    }
    put_json(deps.storage, contract.status_key(job_id), foreign)
    assert client.get(f"/api/jobs/{job_id}").json()["stage"] == "queued"


def test_a_result_that_is_not_valid_fails_the_job(client, deps) -> None:
    job_id = h.upload(client).json()["id"]
    put_json(deps.storage, contract.result_key(job_id), {"job_id": job_id, "state": "ready"})
    job = client.get(f"/api/jobs/{job_id}").json()
    assert (job["status"], job["error"]) == ("failed", BAD_RESULT)
    assert job["stems"] == []


def test_a_worker_that_dies_without_a_result_fails_the_job(client, deps) -> None:
    job_id = h.upload(client).json()["id"]
    deps.provider.states[f"manual:{job_id}"] = "running"
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    for state in ("failed", "succeeded", "cancelled", "unknown"):
        other = h.upload(client).json()["id"] if state != "failed" else job_id
        deps.provider.states[f"manual:{other}"] = state
        if state == "unknown":
            del deps.provider.states[f"manual:{other}"]
        job = client.get(f"/api/jobs/{other}").json()
        assert (job["status"], job["stage"], job["error"]) == ("failed", "failed", WORKER_LOST)
        deps.sync.forget(other)


def test_a_failed_job_carries_the_workers_message(client, deps) -> None:
    job_id = h.upload(client).json()["id"]

    def broken(model_id, stems):
        raise RuntimeError("the GPU fell over")

    run_worker(deps, job_id, factory=broken)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert (job["status"], job["error"]) == ("failed", "RuntimeError: the GPU fell over")
    assert job["stems"] == []
    assert client.post(f"/api/jobs/{job_id}/links").status_code == 409


def test_text_that_names_no_stem_ends_as_failed_with_the_parse_shown(client, deps) -> None:
    job_id = h.upload(client, request_text="make it sound nicer").json()["id"]
    run_worker(deps, job_id)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert (job["status"], job["error"]) == ("failed", NEEDS_REVIEW)
    assert job["parsed_request"]["needs_review"] is True
    assert job["parsed_request"]["raw_text"] == "make it sound nicer"


def test_unfinished_jobs_are_handed_back_to_the_provider_at_startup(tmp_path) -> None:
    deps = h.make_deps(tmp_path)
    with TestClient(create_app(deps, background=False)) as client:
        finished = h.upload(client).json()["id"]
        lost = h.upload(client).json()["id"]
        cookie = client.cookies[OWNER_COOKIE]
    run_worker(deps, finished)
    # The API restarts: a new provider that has never heard of either job.
    restarted = h.make_deps(tmp_path)
    with TestClient(create_app(restarted, background=False)) as client:
        client.cookies.set(OWNER_COOKIE, cookie)
        assert [spec["job_id"] for spec in restarted.provider.submitted] == [lost]
        assert client.get(f"/api/jobs/{finished}").json()["status"] == "ready"
        assert client.get(f"/api/jobs/{lost}").json()["status"] == "queued"


# -- one owner cannot reach another's job ------------------------------------------------


def test_every_job_route_answers_404_to_another_browser(client, stranger, deps) -> None:
    job = ready_job(client, deps)
    job_id = job["id"]
    h.upload(stranger)  # the stranger is a real owner with a job of their own
    before = h.stored_keys(deps)
    requests = [
        ("get", f"/api/jobs/{job_id}"),
        ("get", f"/api/jobs/{job_id}/events"),
        ("post", f"/api/jobs/{job_id}/links"),
        ("get", f"/api/jobs/{job_id}/stems/vocals/peaks"),
        ("delete", f"/api/jobs/{job_id}"),
    ]
    for method, path in requests:
        missing = path.replace(job_id, "00000000-0000-4000-8000-000000000000")
        theirs, nothing = stranger.request(method, path), stranger.request(method, missing)
        assert theirs.status_code == nothing.status_code == 404, (method, path)
        # Exactly what a job that does not exist gets: nothing to tell them apart.
        assert theirs.json() == nothing.json() == {"detail": "Not found."}
    assert [j["id"] for j in stranger.get("/api/jobs").json()["jobs"]] != [job_id]
    assert job_id not in json.dumps(stranger.get("/api/jobs").json())
    # Nothing of the owner's was touched, and a browser with no cookie fares no better.
    assert h.stored_keys(deps) == before
    assert client.get(f"/api/jobs/{job_id}").json() == job
    with TestClient(create_app(deps, background=False)) as nobody:
        for method, path in requests:
            assert nobody.request(method, path).status_code == 404
        assert nobody.get("/api/jobs").json() == {"jobs": []}
    for bad in ("not-a-uuid", "..%2F..%2Fetc", "' OR 1=1 --"):
        assert client.get(f"/api/jobs/{bad}").status_code == 404


def test_the_list_is_the_owners_own_newest_first(client, stranger, deps) -> None:
    first = h.upload(client).json()["id"]
    deps.clock.advance(minutes=1)
    second = h.upload(client).json()["id"]
    theirs = h.upload(stranger).json()["id"]
    assert [j["id"] for j in client.get("/api/jobs").json()["jobs"]] == [second, first]
    assert [j["id"] for j in stranger.get("/api/jobs").json()["jobs"]] == [theirs]


# -- links and downloads -----------------------------------------------------------------


def test_links_download_the_stems_and_the_zip(client, deps) -> None:
    job = ready_job(client, deps, name="Ma Chanson été.wav")
    assert client.post(f"/api/jobs/{job['id']}/links").status_code == 200
    links = client.post(f"/api/jobs/{job['id']}/links").json()
    assert links["expires_in"] == 900
    assert [stem["name"] for stem in links["stems"]] == ["vocals", "drums"]
    assert links["stems"][0]["filename"] == "Ma Chanson été (vocals).wav"
    assert links["zip"]["filename"] == "Ma Chanson été (stems).zip"
    downloaded = {}
    for stem in links["stems"]:
        assert stem["url"].startswith("http://testserver/api/files/")
        # No cookie is needed: the link is the permission.
        with TestClient(create_app(deps, background=False)) as nobody:
            response = nobody.get(stem["url"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        disposition = response.headers["content-disposition"]
        assert disposition.startswith('attachment; filename="Ma Chanson _t_ (')
        assert "filename*=UTF-8''Ma%20Chanson%20%C3%A9t%C3%A9%20%28" in disposition
        assert response.content[:4] == b"RIFF"
        downloaded[stem["name"]] = response.content
    archive = client.get(links["zip"]["url"])
    assert archive.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as unzipped:
        assert sorted(unzipped.namelist()) == ["drums.wav", "vocals.wav"]
        assert unzipped.read("vocals.wav") == downloaded["vocals"]


def test_downloads_support_range_requests(client, deps) -> None:
    job = ready_job(client, deps)
    url = client.post(f"/api/jobs/{job['id']}/links").json()["stems"][0]["url"]
    whole = client.get(url).content
    part = client.get(url, headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.content == whole[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(whole)}"
    tail = client.get(url, headers={"Range": "bytes=-50"})
    assert (tail.status_code, tail.content) == (206, whole[-50:])
    assert client.get(url, headers={"Range": f"bytes={len(whole) + 5}-"}).status_code == 416
    assert client.head(url).headers["accept-ranges"] == "bytes"


def test_links_expire_and_cannot_be_altered(client, deps) -> None:
    job = ready_job(client, deps)
    links = client.post(f"/api/jobs/{job['id']}/links").json()
    url = links["stems"][0]["url"]
    assert client.get(url).status_code == 200
    token = url.rsplit("/", 1)[1]
    flipped = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert client.get(url.replace(token, flipped)).status_code == 404
    assert client.get(url.replace(token, "nonsense")).status_code == 404
    assert client.get(url.replace(token, "a.b")).status_code == 404
    deps.clock.advance(seconds=899)
    assert client.get(url).status_code == 200
    deps.clock.advance(seconds=1)
    expired = client.get(url)
    assert expired.status_code == 410
    assert "expired" in expired.json()["detail"]
    # A fresh link works again.
    fresh = client.post(f"/api/jobs/{job['id']}/links").json()["stems"][0]["url"]
    assert fresh != url
    assert client.get(fresh).status_code == 200


def test_links_are_refused_until_the_job_is_ready(client, deps) -> None:
    job_id = h.upload(client).json()["id"]
    response = client.post(f"/api/jobs/{job_id}/links")
    assert (response.status_code, response.json()["detail"]) == (
        409,
        "This job has no files to download.",
    )


def test_peaks(client, deps) -> None:
    job = ready_job(client, deps)
    response = client.get(f"/api/jobs/{job['id']}/stems/vocals/peaks")
    assert response.status_code == 200
    peaks = response.json()
    assert peaks["buckets"] == len(peaks["min"]) == len(peaks["max"]) == 1000
    assert response.headers["cache-control"] == "private, max-age=3600"
    for name in ("bass", "nonsense", "..%2Fspec"):  # not a stem of this job
        assert client.get(f"/api/jobs/{job['id']}/stems/{name}/peaks").status_code == 404


# -- events ------------------------------------------------------------------------------


def test_events_stream_the_history_and_end_when_the_job_is_over(client, deps) -> None:
    job = ready_job(client, deps)
    response = client.get(f"/api/jobs/{job['id']}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    events = sse(response.text)
    names = [name for name, _ in events]
    assert names[-2:] == ["job", "end"]
    assert set(names[:-2]) == {"progress"}
    stages = [data["stage"] for name, data in events if name == "progress"]
    assert stages[:3] == ["uploaded", "queued", "transcoding"]
    assert stages[-1] == "ready"
    assert events[-2][1] == job
    # A reconnecting browser says what it saw last and gets only what came after.
    ids = [line[4:] for line in response.text.split("\n") if line.startswith("id: ")]
    again = client.get(f"/api/jobs/{job['id']}/events", headers={"Last-Event-ID": ids[-3]})
    resumed = sse(again.text)
    assert [name for name, _ in resumed] == ["progress", "progress", "job", "end"]


def test_events_follow_a_job_while_it_runs(tmp_path) -> None:
    """The stream stays open while the job is in progress and closes when it ends."""
    import threading
    import time

    deps = h.make_deps(tmp_path)
    with TestClient(create_app(deps, background=False)) as client:
        job_id = h.upload(client).json()["id"]
        deps.provider.states[f"manual:{job_id}"] = "running"

        def finish_later() -> None:
            time.sleep(0.5)
            run_worker(deps, job_id)

        thread = threading.Thread(target=finish_later)
        thread.start()
        started = time.perf_counter()
        text = client.get(f"/api/jobs/{job_id}/events").text
        waited = time.perf_counter() - started
        thread.join()
    events = sse(text)
    snapshots = [data["status"] for name, data in events if name == "job"]
    assert snapshots[0] == "queued"
    assert snapshots[-1] == "ready"
    assert events[-1][0] == "end"
    assert waited >= 0.4  # it waited for the worker instead of ending at once


# -- deleting ----------------------------------------------------------------------------


def test_delete_removes_every_object_and_the_links_stop_working(client, deps) -> None:
    keep = ready_job(client, deps)
    job = ready_job(client, deps)
    job_id = job["id"]
    links = client.post(f"/api/jobs/{job_id}/links").json()
    urls = [stem["url"] for stem in links["stems"]] + [links["zip"]["url"]]
    assert all(client.get(url).status_code == 200 for url in urls)
    under = [key for key in h.stored_keys(deps) if key.startswith(f"jobs/{job_id}/")]
    assert len(under) == 9  # source, spec, status, result, 2 stems, 2 peaks, zip

    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert [key for key in h.stored_keys(deps) if key.startswith(f"jobs/{job_id}/")] == []
    assert not (deps.storage.root / "objects" / "jobs" / job_id).exists()
    assert all(client.get(url).status_code == 404 for url in urls)  # still inside their 15 minutes
    assert client.post(f"/api/jobs/{job_id}/links").status_code == 409
    assert client.get(f"/api/jobs/{job_id}/stems/vocals/peaks").status_code == 404
    # The row says it was deleted and has forgotten the file's name and the stems.
    gone = client.get(f"/api/jobs/{job_id}").json()
    assert (gone["status"], gone["stage"], gone["name"], gone["stems"]) == (
        "deleted",
        "deleted",
        None,
        [],
    )
    assert count(deps, Stem, job_id) == count(deps, JobEvent, job_id) == 0
    assert count(deps, DecisionCall, job_id) == 0  # its answers were about this audio
    assert [j["id"] for j in client.get("/api/jobs").json()["jobs"]] == [keep["id"]]
    # The other job is whole, and deleting twice is a 404.
    assert len([k for k in h.stored_keys(deps) if k.startswith(f"jobs/{keep['id']}/")]) == 9
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_deleting_a_job_in_progress_stops_its_worker(client, deps) -> None:
    job_id = h.upload(client).json()["id"]
    deps.provider.states[f"manual:{job_id}"] = "running"
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert deps.provider.cancelled == [f"manual:{job_id}"]
    assert h.stored_keys(deps) == []
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "deleted"


def test_delete_me_removes_everything_of_this_browser_and_nothing_else(
    client, stranger, deps
) -> None:
    mine = [ready_job(client, deps)["id"], h.upload(client).json()["id"]]
    client.delete(f"/api/jobs/{ready_job(client, deps)['id']}")  # one already deleted
    theirs = ready_job(stranger, deps)
    response = client.delete("/api/me")
    assert response.status_code == 204
    assert f"{OWNER_COOKIE}=" in response.headers["set-cookie"]
    assert "max-age=0" in response.headers["set-cookie"].lower()
    assert OWNER_COOKIE not in client.cookies
    assert all(key.startswith(f"jobs/{theirs['id']}/") for key in h.stored_keys(deps))
    with deps.sessions() as session:
        assert len(session.scalars(select(Owner)).all()) == 1
        assert [job.id for job in session.scalars(select(Job))] == [theirs["id"]]
        assert session.scalars(select(JobEvent.job_id)).unique().all() == [theirs["id"]]
    assert deps.provider.cancelled == [f"manual:{mine[1]}"]
    assert client.get("/api/jobs").json() == {"jobs": []}
    assert stranger.get(f"/api/jobs/{theirs['id']}").json() == theirs
    # With no cookie there is nothing to delete, and saying so is not an error.
    assert client.delete("/api/me").status_code == 204


# -- parsing a request in words ----------------------------------------------------------


def test_parse_uses_the_rules_when_there_is_no_key(client, deps) -> None:
    response = client.post("/api/requests/parse", json={"text": "vocals and drums as flac"})
    assert response.status_code == 200
    body = response.json()
    assert (body["provider"], body["fallback_reason"]) == ("fallback", "no_key")
    parsed = body["parsed"]
    assert [s for s, v in parsed["stems"].items() if v["wanted"]] == ["vocals", "drums"]
    assert parsed["output_format"] == "flac"
    unclear = client.post("/api/requests/parse", json={"text": "make it nicer"}).json()
    assert unclear["parsed"]["needs_review"] is True
    for bad in ({"text": ""}, {"text": "x" * 2001}, {}, {"text": 5}):
        assert client.post("/api/requests/parse", json=bad).status_code == 422
    # Parsing makes no owner and stores nothing.
    with deps.sessions() as session:
        assert session.scalars(select(Owner)).all() == []
    assert OWNER_COOKIE not in client.cookies
