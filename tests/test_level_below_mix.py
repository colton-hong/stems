"""A stem's level against the mix, in the mix's loudest stretch: the gate's empty-stem signal."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from stems_common.levels import level_below_mix_db, loudest_window  # noqa: E402

SR = 8000


def tone(seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    wave = amplitude * np.sin(2 * np.pi * 220 * t)
    return np.stack([wave, wave]).astype(np.float32)


def test_the_window_is_the_loudest_stretch() -> None:
    quiet, loud = tone(20, 0.01), tone(10, 0.5)
    audio = np.concatenate([quiet, loud, quiet, tone(10, 0.3)], axis=1)
    assert loudest_window(audio, SR, 10.0) == (20 * SR, 10 * SR)
    start, length = loudest_window(audio, SR, 5.0)
    assert 20 * SR <= start <= 25 * SR
    assert length == 5 * SR
    # No longer than the window: the whole of it.
    assert loudest_window(loud, SR, 10.0) == (0, 10 * SR)
    assert loudest_window(loud, SR, 60.0) == (0, 10 * SR)


def test_it_is_the_rule_the_measurements_were_made_with() -> None:
    from stems_bench.clef.clips import loudest_window as measured

    rng = np.random.default_rng(3)
    for seconds, window in ((95.0, 60.0), (61.5, 60.0), (30.0, 60.0), (200.3, 60.0)):
        envelope = np.repeat(rng.uniform(0.01, 1.0, int(seconds) + 1), SR)[: round(seconds * SR)]
        audio = (envelope * rng.standard_normal((2, envelope.size))).astype(np.float32)
        expected = measured(audio, SR, window)
        assert loudest_window(audio, SR, window) == (expected.start, expected.length)


def test_a_stem_is_compared_with_the_mix_where_the_mix_is_loudest() -> None:
    # The stem plays only in the quiet first half; the mix is loudest in the second.
    stem = np.concatenate([tone(30, 0.1), np.zeros((2, 30 * SR), dtype=np.float32)], axis=1)
    mix = np.concatenate([tone(30, 0.1), tone(30, 0.5)], axis=1)
    assert level_below_mix_db(stem, mix, SR, 30.0) < -100  # silent where it counts
    # Over the whole minute it is 0.1 against the mix of both halves.
    whole = level_below_mix_db(stem, mix, SR, 60.0)
    assert whole == pytest.approx(20 * np.log10(0.1 / np.sqrt(2)) - 20 * np.log10(0.3606), abs=0.05)
    # A tenth of the mix's amplitude is 20 dB below it, wherever the window falls.
    assert level_below_mix_db(0.1 * mix, mix, SR, 30.0) == pytest.approx(-20.0, abs=1e-4)
    assert level_below_mix_db(mix, mix, SR, 30.0) == 0.0


def test_a_stem_of_another_length_is_compared_over_what_both_have() -> None:
    mix = np.concatenate([tone(10, 0.5), tone(10, 0.05)], axis=1)
    short = 0.1 * mix[:, : 10 * SR]
    assert level_below_mix_db(short, mix, SR, 5.0) == pytest.approx(-20.0, abs=1e-4)
