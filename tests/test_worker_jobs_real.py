"""One real checkpoint through the worker's storage entry point (slow; needs its weights).

Everything else that exercises `stems_worker.jobs` uses weightless stand-ins. This is the
one place the same entry point the API's worker uses is run with a real model, so that a
break between "the pipeline works from the command line" and "the worker works from
storage" cannot hide behind the stand-ins.

    make test-slow          # or: pytest -m slow tests/test_worker_jobs_real.py
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from stems_api.storage import LocalStorage, put_bytes, put_json, read_bytes, read_json  # noqa: E402
from stems_common import jobs as contract  # noqa: E402
from stems_common.settings import load_defaults  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.download import is_downloaded  # noqa: E402
from stems_models.registry import load_registry  # noqa: E402
from stems_worker import jobs as worker  # noqa: E402
from stems_worker.audio import load_audio, save_audio  # noqa: E402

SR = 44100
SECONDS = 6.0
SECRET = b"0123456789abcdef0123456789abcdef-test-only"
JOB = "5a1f0c7e-9b2d-4e3a-8c41-7d6e5f4a3b21"


@pytest.mark.slow
def test_a_real_vocals_model_runs_from_storage_to_storage(tmp_path) -> None:
    defaults, registry = load_defaults(), load_registry()
    model_id = defaults.ensembles.slots["vocals"].single_best
    if not is_downloaded(registry.get(model_id)):
        pytest.skip(f"{model_id} is not downloaded (python -m stems_models.download {model_id})")

    track = generate_track(seed=41, duration_seconds=SECONDS)
    upload = tmp_path / "upload.wav"
    save_audio(upload, track.mix, SR, "wav")
    storage = LocalStorage(tmp_path / "store", SECRET)
    spec = contract.build_spec(JOB, created_at=datetime.now(UTC), stems=["vocals"], fmt="flac")
    put_bytes(storage, contract.source_key(JOB), upload.read_bytes(), "application/octet-stream")
    put_json(storage, contract.spec_key(JOB), spec)

    # The real registry, the real defaults, the real model factory. No tagger and no key,
    # so the fallbacks decide and the stem is delivered with the not-checked note.
    result = worker.run_job(
        JOB, storage, defaults=defaults, device="cpu", timings_path=None, client=None, tagger=None
    )
    assert result["state"] == "ready", result.get("error")
    (stem,) = result["stems"]
    assert (stem["name"], stem["format"], stem["model_ids"]) == ("vocals", "flac", [model_id])
    assert [run["model_id"] for run in result["timings"]] == [model_id]
    assert result["timings"][0]["cached"] is False
    assert result["timings"][0]["audio_seconds"] == SECONDS

    audio_path = tmp_path / "vocals.flac"
    audio_path.write_bytes(read_bytes(storage, stem["key"]))
    samples = load_audio(audio_path).samples
    assert samples.shape == (2, int(SECONDS * SR))
    assert np.isfinite(samples).all()
    assert float(np.abs(samples).max()) > 1e-4  # the model produced sound, not silence
    peaks = read_json(storage, stem["peaks_key"])
    assert peaks["duration_seconds"] == SECONDS
    status = contract.validate_status(read_json(storage, contract.status_key(JOB)), JOB)
    assert (status["state"], status["stage"]) == ("ready", "ready")
    assert io.BytesIO(read_bytes(storage, result["zip_key"])).read(2) == b"PK"
    # Run again: the finished job is recognised and the model is not loaded a second time.
    assert worker.run_job(JOB, storage, defaults=defaults, device="cpu") == result
