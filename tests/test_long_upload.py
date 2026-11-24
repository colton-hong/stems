"""The long-upload check: a worker process, a job, and what is recorded about it.

Run here with the weightless stand-in models on a few seconds of audio; the real thing is
``python -m stems_bench.long_upload`` with the real model and twenty minutes.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("soundfile")

from stems_bench import long_upload as lu  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402


def test_the_signal_is_exactly_as_long_as_asked() -> None:
    audio, rate = lu.make_signal(7.5)
    assert rate == 44100
    assert audio.shape == (2, round(7.5 * 44100))
    assert audio.dtype == np.float32
    # Longer than the unit: the unit repeats.
    long, _ = lu.make_signal(lu.UNIT_SECONDS + 2.0)
    unit = round(lu.UNIT_SECONDS * rate)
    assert long.shape[-1] == unit + 2 * rate
    assert np.array_equal(long[:, :rate], long[:, unit : unit + rate])


def test_a_job_runs_in_a_worker_process_and_its_time_memory_and_lengths_are_recorded(
    tmp_path,
) -> None:
    record = lu.run(tmp_path / "run", seconds=6.0, device="cpu", fake_models=True)
    assert (record["state"], record["exit_code"], record["error"]) == ("ready", 0, None)
    assert record["audio_seconds"] == 6.0
    assert record["device"] == "cpu"
    assert record["wall_seconds"] > 0
    # A Python process that has imported torch holds well over 100 MB.
    assert record["worker_peak_rss_bytes"] > 100e6
    assert list(record["stems"]) == list(PRODUCT_STEMS)
    for stem in record["stems"].values():
        assert stem["right_length"] is True
        assert stem["frames"] == 6 * 44100
        assert stem["bytes"] > 0
    assert record["command"].startswith("-m stems_worker.jobs ")
    # The upload and the stems are gone; the worker's log stays.
    left = [p.name for p in (tmp_path / "run").rglob("*") if p.is_file()]
    assert "worker.log" in left
    assert not [
        name for name in left if name.endswith((".flac", ".wav", ".zip")) or name == "source"
    ]
    text = lu.summary(record)
    assert "ready" in text
    assert "right length" in text
    assert "peak memory" in text
    json.dumps(record)
