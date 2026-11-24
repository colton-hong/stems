"""A stem's level against the mix over the whole song: the gate's empty-stem signal."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from stems_common.levels import level_below_mix_db  # noqa: E402

SR = 8000


def tone(seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    wave = amplitude * np.sin(2 * np.pi * 220 * t)
    return np.stack([wave, wave]).astype(np.float32)


def test_a_tenth_of_the_amplitude_is_twenty_decibels_below() -> None:
    mix = np.concatenate([tone(30, 0.1), tone(30, 0.5)], axis=1)
    assert level_below_mix_db(0.1 * mix, mix) == pytest.approx(-20.0, abs=1e-4)
    assert level_below_mix_db(mix, mix) == 0.0
    assert level_below_mix_db(np.zeros_like(mix), mix) < -100  # silence is far below anything


def test_the_whole_song_counts_not_only_its_loudest_part() -> None:
    # The stem plays only in the quiet first half; the mix is loudest in the second. Over
    # the loudest half-minute the stem would be silent; over the song it is plainly there.
    stem = np.concatenate([tone(30, 0.1), np.zeros((2, 30 * SR), dtype=np.float32)], axis=1)
    mix = np.concatenate([tone(30, 0.1), tone(30, 0.5)], axis=1)
    expected = 20 * np.log10(0.1 / np.sqrt(2)) - 20 * np.log10(np.sqrt((0.01 + 0.25) / 2))
    assert level_below_mix_db(stem, mix) == pytest.approx(expected, abs=0.05)
    assert level_below_mix_db(stem, mix) > -21.68  # not empty by the gate's rule


def test_a_stem_of_another_length_is_compared_over_what_both_have() -> None:
    mix = np.concatenate([tone(10, 0.5), tone(10, 0.05)], axis=1)
    short = 0.1 * mix[:, : 10 * SR]
    assert level_below_mix_db(short, mix) == pytest.approx(-20.0, abs=1e-4)
    assert level_below_mix_db(0.1 * mix, mix[:, : 10 * SR]) == pytest.approx(-20.0, abs=1e-4)
