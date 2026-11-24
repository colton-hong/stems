"""The worker's entry point: a job from storage to storage, with weightless models."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from stems_api.storage import LocalStorage, put_bytes, put_json, read_bytes, read_json  # noqa: E402
from stems_common import jobs as contract  # noqa: E402
from stems_common.settings import load_defaults  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_worker import jobs as worker  # noqa: E402
from stems_worker.audio import load_audio, save_audio  # noqa: E402
from stems_worker.fakes import FAKE_BANDS, fake_pipeline  # noqa: E402
from stems_worker.peaks import compute_peaks  # noqa: E402

SR = 44100
SECRET = b"0123456789abcdef0123456789abcdef-test-only"
JOB = "0b9c2f6e-58d4-4c58-9d3b-6f3d1c8a7e21"
OTHER = "7d1e9a34-2f0b-4b6c-8c59-1a2b3c4d5e6f"
START = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
UNCHECKED = "This stem could not be quality-checked automatically."


class Clock:
    """Moves on a second every time it is read, so events have distinct times in order."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class Counting:
    """A storage that counts what is done to it."""

    def __init__(self, inner) -> None:
        self.inner, self.puts, self.opens = inner, [], []

    def put(self, key, data, content_type):
        self.puts.append(key)
        return self.inner.put(key, data, content_type)

    def open(self, key):
        self.opens.append(key)
        return self.inner.open(key)

    def exists(self, key):
        return self.inner.exists(key)

    def delete_prefix(self, prefix):
        return self.inner.delete_prefix(prefix)

    def signed_url(self, key, ttl_seconds, filename=None):
        return self.inner.signed_url(key, ttl_seconds, filename)


@pytest.fixture
def storage(tmp_path) -> Counting:
    return Counting(LocalStorage(tmp_path / "store", SECRET))


@pytest.fixture
def source_bytes(tmp_path) -> bytes:
    track = generate_track(seed=41, duration_seconds=3.0)
    # The fixture has little above 5 kHz, the band the stand-in models call vocals: 21 dB
    # under the mix, which the gate would rightly call empty. A tone at 7 kHz gives that
    # band an ordinary level, so these tests see one attempt per stem.
    t = np.arange(track.mix.shape[-1]) / SR
    mix = (track.mix + 0.05 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)
    path = tmp_path / "upload.wav"
    save_audio(path, mix, SR, "wav")
    return path.read_bytes()


class Models:
    """The weightless pipeline, counting how often a model is built."""

    def __init__(self, tmp_path) -> None:
        options, self.defaults = fake_pipeline(load_defaults())
        self.built: list[str] = []
        inner = options["factory"]

        def factory(model_id, stems):
            self.built.append(model_id)
            return inner(model_id, stems)

        self.options = {**options, "factory": factory}
        self.tmp = tmp_path

    def run(self, storage, job_id: str = JOB, **overrides):
        settings = {
            "defaults": self.defaults,
            "clock": Clock(),
            "device": "cpu",
            "timings_path": None,
            **self.options,
            **overrides,
        }
        return worker.run_job(job_id, storage, **settings)


@pytest.fixture
def models(tmp_path) -> Models:
    return Models(tmp_path)


def submit(storage, source: bytes, job_id: str = JOB, **request) -> dict:
    """What the API does: store the source and the spec."""
    request.setdefault("stems", ["vocals", "drums"]) if "text" not in request else None
    spec = contract.build_spec(job_id, created_at=START, source_bytes=len(source), **request)
    put_bytes(storage, contract.source_key(job_id), source, "application/octet-stream")
    put_json(storage, contract.spec_key(job_id), spec)
    return spec


# -- the whole job -----------------------------------------------------------------------


def test_a_job_walks_every_stage_and_writes_stems_peaks_zip_and_result(
    storage, source_bytes, models
) -> None:
    submit(storage, source_bytes)
    result = models.run(storage)
    assert result["state"] == "ready"
    assert result == contract.validate_result(read_json(storage, contract.result_key(JOB)), JOB)
    assert [stem["name"] for stem in result["stems"]] == ["vocals", "drums"]
    assert result["format"] == "wav"
    assert result["device"] == "cpu"
    assert result["source"] == {"duration_seconds": 3.0, "sample_rate_hz": 44100, "channels": 2}
    assert {d["kind"] for d in result["decisions"]} == {"parse", "plan", "gate"}
    assert {d["provider"] for d in result["decisions"]} == {"selection", "fallback"}
    assert [t["model_id"] for t in result["timings"]] == ["fake_bands"]

    # Every object the result names exists, and the audio is the right length.
    for stem in result["stems"]:
        audio = read_bytes(storage, stem["key"])
        assert len(audio) == stem["bytes"]
        assert stem["key"] == f"jobs/{JOB}/stems/{stem['name']}.wav"
        assert stem["model_ids"] == ["fake_bands"]
        assert stem["ensemble_mode"] == "single_best"
        assert stem["attempts"] == 1
        # Nothing listened to it (no tagger), so it is delivered and says so.
        assert stem["warning"] == UNCHECKED
        assert stem["assessment"]["stem"] == stem["name"]
        assert stem["assessment"]["job_id"] == JOB
        peaks = read_json(storage, stem["peaks_key"])
        assert peaks["buckets"] == len(peaks["min"]) == len(peaks["max"]) == 1000
        assert peaks["duration_seconds"] == 3.0
        assert all(lo <= hi for lo, hi in zip(peaks["min"], peaks["max"], strict=True))
        assert max(peaks["max"]) > 0.001  # not silence
    with zipfile.ZipFile(io.BytesIO(read_bytes(storage, result["zip_key"]))) as archive:
        assert sorted(archive.namelist()) == ["drums.wav", "vocals.wav"]
        assert archive.read("vocals.wav") == read_bytes(storage, result["stems"][0]["key"])
        assert archive.testzip() is None

    # The status says it is over, and its history names every stage in order.
    status = contract.validate_status(read_json(storage, contract.status_key(JOB)), JOB)
    assert (status["state"], status["stage"], status["error"]) == ("ready", "ready", None)
    assert status["stems"] == {
        "vocals": {"state": "delivered_with_warning", "attempt": 0},
        "drums": {"state": "delivered_with_warning", "attempt": 0},
    }
    walked = [event["stage"] for event in status["events"] if event["stem"] is None]
    assert walked == [
        "transcoding",
        "analysing",
        "planning",
        "separating",
        "packaging",
        "ready",
    ]
    per_stem = [(e["stage"], e["stem"], e["message"]) for e in status["events"] if e["stem"]]
    assert per_stem == [
        ("assessing", "vocals", "assessing"),
        ("assessing", "vocals", "delivered_with_warning"),
        ("assessing", "drums", "assessing"),
        ("assessing", "drums", "delivered_with_warning"),
    ]
    times = [event["at"] for event in status["events"]]
    assert times == sorted(times)
    assert len(set(times)) == len(times)
    # result.json was written after everything it names, and status.json last of all.
    assert storage.puts[-2:] == [contract.result_key(JOB), contract.status_key(JOB)]
    assert storage.puts.index(result["zip_key"]) < storage.puts.index(contract.result_key(JOB))


def test_running_a_finished_job_again_does_no_work(storage, source_bytes, models) -> None:
    submit(storage, source_bytes)
    first = models.run(storage)
    puts, opens, built = len(storage.puts), len(storage.opens), len(models.built)
    again = models.run(storage)
    assert again == first
    assert len(storage.puts) == puts  # nothing written
    assert len(models.built) == built  # no model built
    assert storage.opens[opens:] == [contract.result_key(JOB)]  # one read, of the result


def test_the_stems_are_the_bands_the_stand_in_models_make(storage, source_bytes, models, tmp_path):
    submit(storage, source_bytes, stems=list(FAKE_BANDS))
    result = models.run(storage)
    assert [s["name"] for s in result["stems"]] == [
        "vocals",
        "drums",
        "bass",
        "guitar",
        "piano",
        "other",
    ]
    total = None
    for stem in result["stems"]:
        path = tmp_path / f"{stem['name']}.wav"
        path.write_bytes(read_bytes(storage, stem["key"]))
        samples = load_audio(path).samples
        total = samples if total is None else total + samples
    upload = tmp_path / "upload.wav"
    # The six bands add back up to what was uploaded (to 24-bit rounding per stem).
    assert np.abs(total - load_audio(upload).samples).max() < 6 * 2.0**-22


def test_a_text_request_is_parsed_by_the_worker(storage, source_bytes, models) -> None:
    submit(storage, source_bytes, text="just the vocals please, as flac")
    result = models.run(storage)
    assert result["state"] == "ready"
    assert [(s["name"], s["format"]) for s in result["stems"]] == [("vocals", "flac")]
    assert result["stems"][0]["key"].endswith("/stems/vocals.flac")
    assert result["parsed_request"]["raw_text"] == "just the vocals please, as flac"
    parse = next(d for d in result["decisions"] if d["kind"] == "parse")
    assert (parse["provider"], parse["fallback_reason"]) == ("fallback", "no_key")
    assert read_bytes(storage, result["stems"][0]["key"])[:4] == b"fLaC"


def test_text_that_names_no_stem_ends_in_needs_review(storage, source_bytes, models) -> None:
    submit(storage, source_bytes, text="make it sound nicer")
    result = models.run(storage)
    assert result["state"] == "needs_review"
    assert result["stems"] == []
    assert result["parsed_request"]["needs_review"] is True
    assert models.built == []  # nothing was separated
    status = read_json(storage, contract.status_key(JOB))
    assert (status["state"], status["stage"]) == ("needs_review", "ready")
    assert not storage.exists(contract.zip_key(JOB))


def test_vocal_split_and_dereverb_are_delivered_as_extra_stems(storage, source_bytes, models):
    submit(storage, source_bytes, stems=["vocals"], split_vocals=True, dereverb=True)
    result = models.run(storage)
    names = [stem["name"] for stem in result["stems"]]
    assert names == ["vocals", "lead_vocals", "backing_vocals", "vocals_dry"]
    by_name = {stem["name"]: stem for stem in result["stems"]}
    assert by_name["lead_vocals"]["model_ids"] == ["fake_split"]
    assert by_name["vocals_dry"]["model_ids"] == ["fake_dry"]
    for extra in ("lead_vocals", "backing_vocals", "vocals_dry"):
        # The extras are not gated: no assessment, no warning.
        assert (by_name[extra]["assessment"], by_name[extra]["warning"]) == (None, None)
        assert storage.exists(by_name[extra]["key"])
        assert storage.exists(by_name[extra]["peaks_key"])
    with zipfile.ZipFile(io.BytesIO(read_bytes(storage, result["zip_key"]))) as archive:
        assert len(archive.namelist()) == 4


# -- failures ----------------------------------------------------------------------------


def test_an_upload_that_is_not_audio_fails_cleanly(storage, models, capsys) -> None:
    submit(storage, b"%PDF-1.7 this is not audio at all" * 200)
    result = models.run(storage)
    assert result["state"] == "failed"
    assert result["stems"] == []
    assert "not decodable audio" in result["error"]
    assert contract.validate_result(read_json(storage, contract.result_key(JOB)), JOB) == result
    status = contract.validate_status(read_json(storage, contract.status_key(JOB)), JOB)
    assert (status["state"], status["stage"]) == ("failed", "failed")
    assert status["error"] == result["error"]
    assert status["events"][-1]["stage"] == "failed"
    assert "Traceback" in capsys.readouterr().err  # the detail goes to the worker's log
    # It is over: running it again does not try again.
    assert models.run(storage) == result


def test_audio_longer_than_the_limit_is_refused_by_the_worker_too(
    storage, source_bytes, models
) -> None:
    short = models.defaults.uploads.model_copy(update={"max_duration_seconds": 2})
    submit(storage, source_bytes)
    result = models.run(storage, defaults=models.defaults.model_copy(update={"uploads": short}))
    assert result["state"] == "failed"
    assert "exceeds 2.0 s" in result["error"]
    assert models.built == []


def test_a_failure_inside_a_model_is_reported_by_type(storage, source_bytes, models) -> None:
    def broken(model_id, stems):
        raise RuntimeError("the GPU fell over")

    submit(storage, source_bytes)
    result = models.run(storage, factory=broken)
    assert result["state"] == "failed"
    assert result["error"] == (
        "The separation failed inside the server (RuntimeError). "
        "The details are in the server's log."
    )


def test_a_failure_tells_the_user_its_kind_and_keeps_the_details_in_the_log(
    storage, source_bytes, models, capsys
) -> None:
    def broken(model_id, stems):
        raise FileNotFoundError("/srv/stems/models/weights/secret-layout/model.ckpt")

    submit(storage, source_bytes)
    result = models.run(storage, factory=broken)
    assert "FileNotFoundError" in result["error"]
    assert "/srv/stems" not in result["error"]
    assert "/srv/stems/models/weights/secret-layout/model.ckpt" in capsys.readouterr().err


def test_nothing_made_from_the_upload_is_left_on_disk(
    storage, source_bytes, models, tmp_path, monkeypatch
) -> None:
    """The source copy, the stems before upload and the model-output cache all live in the
    job's temporary folder, which is gone when the job ends: finished, failed or stopped."""
    import tempfile

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    shared = tmp_path / "shared-cache"
    monkeypatch.setattr(worker.file_job, "DEFAULT_CACHE_DIR", shared)
    seen: list[list[str]] = []
    inner = models.options["factory"]

    def watching(model_id, stems):
        (job_dir,) = scratch.iterdir()  # while it runs there is exactly one folder
        seen.append(sorted(p.name for p in job_dir.iterdir()))
        return inner(model_id, stems)

    submit(storage, source_bytes)
    assert models.run(storage, factory=watching)["state"] == "ready"
    assert seen
    assert "source" in seen[0]
    assert list(scratch.iterdir()) == []
    assert not shared.exists()  # the shared cache was never touched

    # A job that fails cleans up too.
    def broken(model_id, stems):
        raise RuntimeError("boom")

    submit(storage, source_bytes, job_id=OTHER)
    assert models.run(storage, job_id=OTHER, factory=broken)["state"] == "failed"
    assert list(scratch.iterdir()) == []


def test_a_worker_told_to_stop_cleans_up_and_writes_no_result(
    storage, source_bytes, models, tmp_path, monkeypatch
) -> None:
    import signal
    import tempfile

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def stopped(model_id, stems):
        worker._terminate(signal.SIGTERM, None)  # what the SIGTERM handler does

    submit(storage, source_bytes)
    with pytest.raises(SystemExit) as caught:
        models.run(storage, factory=stopped)
    assert caught.value.code == 128 + signal.SIGTERM
    assert list(scratch.iterdir()) == []  # the copy of the upload is gone
    assert not storage.exists(contract.result_key(JOB))  # and the job is not called finished


def test_a_job_with_no_spec_or_a_wrong_spec_is_not_run(storage, source_bytes, models) -> None:
    with pytest.raises(worker.JobNotFound):
        models.run(storage)
    with pytest.raises(contract.ContractError, match="not a job id"):
        models.run(storage, job_id="../" + JOB)
    # A spec copied from another job points at that job's source: refused.
    spec = submit(storage, source_bytes, job_id=OTHER)
    put_json(storage, contract.spec_key(JOB), spec)
    with pytest.raises(contract.ContractError, match="expected"):
        models.run(storage)
    put_json(storage, contract.spec_key(JOB), {**spec, "job_id": JOB})
    with pytest.raises(contract.ContractError, match="outside the job's own folder"):
        models.run(storage)
    assert not storage.exists(contract.result_key(JOB))
    assert models.built == []


# -- the status file ---------------------------------------------------------------------


def test_every_status_written_along_the_way_is_valid_and_progress_is_throttled(
    storage, source_bytes, models
) -> None:
    submit(storage, source_bytes)
    seen = []
    inner_put = storage.inner.put

    def recording_put(key, data, content_type):
        body = data.read()
        if key == contract.status_key(JOB):
            seen.append(contract.validate_status(json.loads(body), JOB))
        return inner_put(key, io.BytesIO(body), content_type)

    storage.inner.put = recording_put
    models.run(storage)
    stages = [status["stage"] for status in seen]
    assert stages[0] == "transcoding"
    assert stages[-1] == "ready"
    assert [s["state"] for s in seen[:-1]] == ["running"] * (len(seen) - 1)
    # Each write carries the whole history so far: it only ever grows.
    lengths = [len(status["events"]) for status in seen]
    assert lengths == sorted(lengths)
    assert any(status["progress"] is not None for status in seen)
    # A 3-second clip is a handful of chunks; progress alone did not cause a write per chunk.
    assert len(seen) < 20


def test_status_writer_throttles_progress_and_caps_the_history(tmp_path) -> None:
    store = Counting(LocalStorage(tmp_path / "s", SECRET))
    ticks = iter(float(n) for n in range(10_000))
    writer = worker.StatusWriter(store, JOB, clock=Clock(), monotonic=lambda: next(ticks) * 0.5)
    writer.stage("separating")
    writes = len(store.puts)
    for step in range(20):  # 20 updates, half a second apart: 10 seconds
        writer.progress(step / 20)
    assert len(store.puts) - writes == 5  # one every two seconds
    writer.stage("separating")  # the same stage again: nothing to say
    assert len(store.puts) - writes == 5
    for index in range(700):
        writer.stem("vocals", "assessing" if index % 2 else "retrying", index % 3)
    status = read_json(store, contract.status_key(JOB))
    assert len(status["events"]) == contract.MAX_EVENTS - 1
    writer.finish("failed", "out of memory")
    status = contract.validate_status(read_json(store, contract.status_key(JOB)), JOB)
    assert len(status["events"]) == contract.MAX_EVENTS
    assert status["events"][-1] == {
        "seq": contract.MAX_EVENTS,
        "at": status["events"][-1]["at"],
        "stage": "failed",
        "stem": None,
        "message": "out of memory",
        "progress": None,
    }


# -- peaks -------------------------------------------------------------------------------


def test_peaks_cover_every_sample_once() -> None:
    samples = np.zeros((2, 10_000), dtype=np.float32)
    samples[0, 0], samples[1, 9_999], samples[0, 5_000] = 0.5, -0.75, 2.0
    peaks = compute_peaks(samples, 1000, buckets=100)
    assert (peaks["buckets"], peaks["duration_seconds"], peaks["sample_rate_hz"]) == (
        100,
        10.0,
        1000,
    )
    assert peaks["max"][0] == 0.5
    assert peaks["min"][99] == -0.75
    assert peaks["max"][50] == 1.0  # clamped to full scale
    assert sum(1 for v in peaks["max"] if v) == 2
    assert sum(1 for v in peaks["min"] if v) == 1
    short = compute_peaks(np.ones((2, 3), dtype=np.float32) * 0.25, 44100, buckets=1000)
    assert (short["buckets"], short["max"], short["min"]) == (3, [0.25] * 3, [0.25] * 3)
    empty = compute_peaks(np.zeros((2, 0), dtype=np.float32), 44100)
    assert (empty["buckets"], empty["max"], empty["min"]) == (1, [0.0], [0.0])
    with pytest.raises(ValueError, match="channels"):
        compute_peaks(np.zeros(10, dtype=np.float32), 44100)


# -- the command -------------------------------------------------------------------------


def test_the_command_runs_a_job_from_the_environment(tmp_path, source_bytes, monkeypatch, capsys):
    root = tmp_path / "storage"
    monkeypatch.setenv("STEMS_STORAGE_BACKEND", "local")
    monkeypatch.setenv("STEMS_STORAGE_LOCAL_DIR", str(root))
    monkeypatch.setenv("STEMS_SIGNING_SECRET", SECRET.decode())
    monkeypatch.setenv("STEMS_WORKER_FAKE_MODELS", "1")
    monkeypatch.setattr(worker.file_job, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(worker.file_job, "DEFAULT_TIMINGS_PATH", tmp_path / "timings.jsonl")
    store = LocalStorage(root, SECRET)
    submit(store, source_bytes, stems=["bass"])
    assert worker.main([JOB, "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert f"job {JOB}: ready" in out
    assert f"bass: jobs/{JOB}/stems/bass.wav" in out
    assert read_json(store, contract.result_key(JOB))["state"] == "ready"
    assert worker.main([OTHER]) == 1
    assert "no spec in storage" in capsys.readouterr().err


# -- decoding the start of a file --------------------------------------------------------


def test_decode_head_decodes_a_second_or_says_the_file_is_not_audio(tmp_path, source_bytes):
    from stems_worker.audio import UnsupportedAudioError, decode_head

    upload = tmp_path / "upload.wav"
    assert decode_head(upload) == SR  # one second of a three-second file
    assert decode_head(upload, seconds=10) == 3 * SR  # no more than there is
    mp3 = tmp_path / "upload.mp3"
    save_audio(mp3, load_audio(upload).samples, SR, "mp3_192")
    assert decode_head(mp3) >= SR
    fake = tmp_path / "renamed.mp3"
    fake.write_bytes(b"%PDF-1.7 not audio at all " * 500)
    with pytest.raises(UnsupportedAudioError):
        decode_head(fake)
    # A real header with the audio cut off straight after it.
    stub = tmp_path / "header-only.wav"
    stub.write_bytes(source_bytes[:44])
    with pytest.raises(UnsupportedAudioError):
        decode_head(stub)


def test_the_tagger_is_not_loaded_when_listening_is_switched_off(tmp_path, monkeypatch) -> None:
    from stems_worker.listening import beats

    seen: dict = {}

    def fake_run(job_id, storage, **options):
        seen.update(options)
        return {"state": "ready", "stems": [], "error": None}

    def loaded():
        seen["loaded"] = True
        raise beats.BeatsError("no weights in this test")

    monkeypatch.setenv("STEMS_STORAGE_BACKEND", "local")
    monkeypatch.setenv("STEMS_STORAGE_LOCAL_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STEMS_SIGNING_SECRET", SECRET.decode())
    monkeypatch.delenv("STEMS_WORKER_FAKE_MODELS", raising=False)
    monkeypatch.setattr(worker, "run_job", fake_run)
    monkeypatch.setattr(beats.BeatsTagger, "from_registry", staticmethod(loaded))
    base = load_defaults()
    off = base.model_copy(
        update={"listening": base.listening.model_copy(update={"enabled": False})}
    )
    monkeypatch.setattr(worker, "load_defaults", lambda: off)
    assert worker.main([JOB]) == 0
    assert "loaded" not in seen
    assert seen["tagger"] is None
    # Switched on (the shipped setting), the worker does try to load it.
    assert base.listening.enabled is True
    monkeypatch.setattr(worker, "load_defaults", lambda: base)
    assert worker.main([JOB]) == 0
    assert seen["loaded"] is True
