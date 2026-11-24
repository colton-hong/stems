"""Upload to download through the whole system, in one test.

The real pieces, wired as in production: the HTTP app with its maintenance thread, the
local provider starting ``python -m stems_worker.jobs`` in a subprocess, local storage on
disk, a SQLite file. The only stand-in is the separator: ``STEMS_WORKER_FAKE_MODELS``
makes the worker use band-splitting models that need no weights, so this runs anywhere.
"""

from __future__ import annotations

import io
import time
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("torch")

import api_helpers as h
import soundfile as sf
from fastapi.testclient import TestClient

from stems_api.app import create_app
from stems_api.providers import LocalProvider
from stems_api.providers.local import worker_command
from stems_api.storage import read_json
from stems_common import jobs as contract

UNCHECKED = "This stem could not be quality-checked automatically."
SECONDS = 2.0


def test_upload_to_download_with_the_local_provider(tmp_path) -> None:
    worker_env = {
        "STEMS_STORAGE_BACKEND": "local",
        "STEMS_STORAGE_LOCAL_DIR": str(tmp_path / "store"),
        "STEMS_SIGNING_SECRET": h.SECRET.decode(),
        "STEMS_WORKER_FAKE_MODELS": "1",
    }
    provider = LocalProvider(
        command=lambda job_id: [*worker_command(job_id), "--device", "cpu"],
        env=worker_env,
        log_dir=tmp_path / "logs",
    )
    deps = h.make_deps(tmp_path, provider=provider)
    # Noise, split by the stand-in models into frequency bands. The two asked for are
    # wide bands at an ordinary level; a narrow one (the band called bass) would be far
    # enough below the mix for the gate to call it empty and retry it.
    upload = h.wav_bytes(seconds=SECONDS)
    try:
        with TestClient(create_app(deps, background=True)) as client:
            # 1. Upload.
            response = h.upload(
                client, upload, name="Demo Track.wav", stems="vocals,guitar", format="flac"
            )
            assert response.status_code == 201, response.text
            job_id = response.json()["id"]

            # 2. Wait, as the web page would: poll the job until it is over.
            seen: list[tuple[str, str]] = []
            deadline = time.time() + 240
            while True:
                job = client.get(f"/api/jobs/{job_id}").json()
                if (job["status"], job["stage"]) not in seen:
                    seen.append((job["status"], job["stage"]))
                if job["status"] not in ("queued", "running"):
                    break
                assert time.time() < deadline, f"still {seen[-1]} after 240 s"
                time.sleep(0.05)
            log = (tmp_path / "logs" / f"{job_id}.log").read_text()
            assert job["status"] == "ready", f"{job['error']}\n{log}"
            assert seen[0] == ("queued", "queued")
            assert seen[-1] == ("ready", "ready")
            assert f"job {job_id}: ready" in log
            # The result is written just before the worker exits, so give it a moment.
            assert provider.wait(f"local:{job_id}", timeout=60).state == "succeeded"

            # 3. What the job says about itself.
            assert job["name"] == "Demo Track.wav"
            assert [(s["name"], s["format"], s["warning"]) for s in job["stems"]] == [
                ("vocals", "flac", UNCHECKED),
                ("guitar", "flac", UNCHECKED),
            ]
            assert job["source"]["duration_seconds"] == SECONDS
            result = contract.validate_result(
                read_json(deps.storage, contract.result_key(job_id)), job_id
            )
            assert {t["model_id"] for t in result["timings"]} == {"fake_bands"}

            # 4. The history, as the events route gives it.
            stream = client.get(f"/api/jobs/{job_id}/events").text
            stages = [
                line.split('"stage": "')[1].split('"')[0]
                for line in stream.split("\n")
                if line.startswith("data: ") and '"seq"' not in line and '"stage"' in line
            ]
            in_order = ["uploaded", "queued", "transcoding", "analysing", "planning"]
            in_order += ["separating", "assessing", "packaging", "ready"]
            positions = [stages.index(stage) for stage in in_order]
            assert positions == sorted(positions), stages
            assert stream.rstrip().endswith("event: end\ndata: {}")

            # 5. Download every stem and the zip through signed links, and decode them.
            links = client.post(f"/api/jobs/{job_id}/links").json()
            assert [s["filename"] for s in links["stems"]] == [
                "Demo Track (vocals).flac",
                "Demo Track (guitar).flac",
            ]
            audio = {}
            for stem in links["stems"]:
                got = client.get(stem["url"])
                assert got.status_code == 200
                assert got.headers["content-type"] == "audio/flac"
                samples, rate = sf.read(io.BytesIO(got.content), always_2d=True)
                assert rate == 44100
                assert samples.shape == (int(SECONDS * 44100), 2)
                assert abs(samples).max() > 0  # not silence
                audio[stem["name"]] = got.content
            # The two bands differ: each stem is its own audio, not a copy.
            assert audio["vocals"] != audio["guitar"]
            archive = client.get(links["zip"]["url"])
            assert archive.status_code == 200
            with zipfile.ZipFile(io.BytesIO(archive.content)) as unzipped:
                assert sorted(unzipped.namelist()) == ["guitar.flac", "vocals.flac"]
                assert unzipped.read("vocals.flac") == audio["vocals"]
            peaks = client.get(f"/api/jobs/{job_id}/stems/guitar/peaks").json()
            assert peaks["duration_seconds"] == SECONDS
            assert max(peaks["max"]) > 0

            # 6. Delete: every object goes and the links die.
            assert len(h.stored_keys(deps)) == 9
            assert client.delete(f"/api/jobs/{job_id}").status_code == 204
            assert h.stored_keys(deps) == []
            assert client.get(links["stems"][0]["url"]).status_code == 404
            assert client.get(f"/api/jobs/{job_id}").json()["status"] == "deleted"
    finally:
        provider.close()
