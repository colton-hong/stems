"""Level, silence, duration and high-band metrics, checked on hand-built signals."""

from __future__ import annotations

import json
import math

import pytest

np = pytest.importorskip("numpy")

from stems_bench.signal_metrics import (  # noqa: E402
    band_energy,
    compare,
    duration_delta_seconds,
    high_band_ratio_db,
    level_stats,
)
from stems_common.levels import FLOOR_DBFS, peak_dbfs, rms_dbfs, silence_fraction  # noqa: E402

RATE = 44100


def sine(hz: float, amplitude: float = 1.0, seconds: float = 2.0, channels: int = 2):
    t = np.arange(round(seconds * RATE)) / RATE
    return np.tile(amplitude * np.sin(2 * np.pi * hz * t), (channels, 1)).astype(np.float32)


def test_rms_and_peak_of_known_signals() -> None:
    assert rms_dbfs(sine(1000)) == pytest.approx(-3.01, abs=0.01)  # a sine is 3 dB under peak
    assert peak_dbfs(sine(1000)) == pytest.approx(0.0, abs=0.001)
    assert rms_dbfs(sine(1000, 0.1)) == pytest.approx(-23.01, abs=0.01)
    assert peak_dbfs(sine(1000, 0.1)) == pytest.approx(-20.0, abs=0.001)
    assert rms_dbfs(np.full((2, 100), 0.5, dtype=np.float32)) == pytest.approx(-6.02, abs=0.01)
    assert peak_dbfs(np.array([[0.0, -2.0, 0.5]])) == pytest.approx(6.02, abs=0.01)  # clipped


def test_silence_and_empty_audio_are_floored_not_infinite() -> None:
    silent = np.zeros((2, RATE), dtype=np.float32)
    assert rms_dbfs(silent) == FLOOR_DBFS == peak_dbfs(silent)
    empty = np.zeros((2, 0), dtype=np.float32)
    assert rms_dbfs(empty) == FLOOR_DBFS == peak_dbfs(empty)
    assert silence_fraction(empty, RATE) == 1.0
    assert rms_dbfs(np.full((1, 10), 1e-9)) == FLOOR_DBFS


def test_silence_fraction_counts_windows_under_minus_60_dbfs() -> None:
    assert silence_fraction(np.zeros((2, RATE), dtype=np.float32), RATE) == 1.0
    assert silence_fraction(sine(440, 0.5), RATE) == 0.0
    # One second of tone, then three of silence.
    padded = np.concatenate([sine(440, 0.5, 1.0), np.zeros((2, 3 * RATE), np.float32)], axis=1)
    assert silence_fraction(padded, RATE) == pytest.approx(0.75)
    # RMS of -70 dBFS counts as silent, -50 dBFS does not.
    quiet = sine(440, 10 ** (-70 / 20) * math.sqrt(2))
    audible = sine(440, 10 ** (-50 / 20) * math.sqrt(2))
    assert rms_dbfs(quiet) == pytest.approx(-70, abs=0.01)
    assert silence_fraction(quiet, RATE) == 1.0
    assert silence_fraction(audible, RATE) == 0.0
    assert silence_fraction(audible, RATE, threshold_dbfs=-40) == 1.0


def test_silence_fraction_handles_a_short_last_window_and_one_loud_channel() -> None:
    # 0.25 s: two full 0.1 s windows and one half window, only the last one sounding.
    audio = np.zeros((2, round(0.25 * RATE)), dtype=np.float32)
    audio[:, round(0.2 * RATE) :] = 0.5
    assert silence_fraction(audio, RATE) == pytest.approx(2 / 3)
    # Sound in one channel only is not silence.
    one_sided = np.zeros((2, RATE), dtype=np.float32)
    one_sided[0] = 0.1
    assert silence_fraction(one_sided, RATE) == 0.0


def test_level_stats_and_duration_delta() -> None:
    stats = level_stats(sine(1000, 0.1, seconds=1.5), RATE)
    assert stats == {
        "duration_seconds": 1.5,
        "rms_dbfs": -23.01,
        "peak_dbfs": -20.0,
        "silence_fraction": 0.0,
    }
    reference = sine(1000)
    assert duration_delta_seconds(reference, reference, RATE) == 0.0
    assert duration_delta_seconds(reference[:, :-441], reference, RATE) == pytest.approx(-0.01)
    assert duration_delta_seconds(sine(1000, seconds=2.5), reference, RATE) == pytest.approx(0.5)


def test_band_energy_measures_only_what_is_above_the_cutoff() -> None:
    low, high = sine(1000, 0.5), sine(12000, 0.2)
    # Two channels, each a sine of amplitude a: mean square a^2/2 per channel.
    assert band_energy(high, RATE) == pytest.approx(2 * 0.2**2 / 2, rel=0.01)
    assert band_energy(low + high, RATE) == pytest.approx(band_energy(high, RATE), rel=0.001)
    assert band_energy(low, RATE) < 1e-12
    assert band_energy(sine(7000, 0.5), RATE) < 1e-9  # just under the cutoff
    assert band_energy(low + high, RATE, low_hz=500) == pytest.approx(
        2 * (0.5**2 + 0.2**2) / 2, rel=0.01
    )
    assert band_energy(high[:, :1000], RATE) == 0.0  # shorter than one frame


def test_high_band_ratio_shows_lost_and_added_top_end() -> None:
    low, high = sine(1000, 0.5), sine(12000, 0.2)
    reference = low + high
    assert high_band_ratio_db(reference, reference, RATE) == pytest.approx(0.0, abs=1e-6)
    assert high_band_ratio_db(low + 0.5 * high, reference, RATE) == pytest.approx(-6.02, abs=0.01)
    assert high_band_ratio_db(low + 2.0 * high, reference, RATE) == pytest.approx(6.02, abs=0.01)
    # A stem with its top octave removed entirely.
    assert high_band_ratio_db(low, reference, RATE) < -80
    assert high_band_ratio_db(np.zeros_like(reference), reference, RATE) == FLOOR_DBFS
    # The reference has nothing up there: the ratio is undefined.
    assert high_band_ratio_db(reference, low, RATE) is None
    # Level changes below the cutoff do not move it.
    assert high_band_ratio_db(0.1 * low + high, reference, RATE) == pytest.approx(0.0, abs=0.01)


def test_compare_puts_everything_in_one_json_object() -> None:
    reference = sine(1000, 0.5) + sine(12000, 0.2)
    estimate = (0.5 * sine(1000, 0.5) + 0.5 * sine(12000, 0.2))[:, :-441]
    report = json.loads(json.dumps(compare(estimate, reference, RATE), allow_nan=False))
    assert report["duration_matches"] is False
    assert report["duration_delta_seconds"] == pytest.approx(-0.01)
    assert report["rms_delta_db"] == pytest.approx(-6.02, abs=0.01)
    assert report["high_band_ratio_db"] == pytest.approx(-6.02, abs=0.05)
    assert report["high_band_hz"] == 8000.0
    assert report["estimate"]["silence_fraction"] == 0.0
    assert report["reference"]["duration_seconds"] == 2.0
    silent = compare(np.zeros_like(reference), sine(1000), RATE)
    assert silent["estimate"]["silence_fraction"] == 1.0
    assert silent["high_band_ratio_db"] is None
    assert silent["duration_matches"] is True
