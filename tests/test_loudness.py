"""BS.1770 loudness, checked against the standard's own numbers."""

from __future__ import annotations

import numpy as np
import pytest

from stems_worker.listening.loudness import integrated_lufs, k_weighting


def sine(freq: float, seconds: float, sample_rate: int, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def test_coefficients_at_48k_are_the_standards():
    (shelf_b, shelf_a), (hp_b, hp_a) = k_weighting(48000)
    # ITU-R BS.1770-4, tables 1 and 2.
    assert shelf_b == pytest.approx([1.53512485958697, -2.69169618940638, 1.19839281085285])
    assert shelf_a == pytest.approx([1.0, -1.69065929318241, 0.73248077421585])
    assert hp_b == pytest.approx([1.0, -2.0, 1.0])
    assert hp_a == pytest.approx([1.0, -1.99004745483398, 0.99007225036621])


@pytest.mark.parametrize("sample_rate", [44100, 48000])
def test_full_scale_997_hz_sine_in_one_channel_reads_minus_3_01(sample_rate):
    # The standard's reference: a 0 dB FS 997 Hz sine in one channel is -3.01 LKFS.
    tone = sine(997.0, 5.0, sample_rate)
    stereo = np.stack([tone, np.zeros_like(tone)])
    assert integrated_lufs(stereo, sample_rate) == pytest.approx(-3.01, abs=0.02)
    assert integrated_lufs(tone, sample_rate) == pytest.approx(-3.01, abs=0.02)


def test_level_changes_move_the_reading_by_the_same_amount():
    tone = sine(997.0, 5.0, 44100)
    loud = integrated_lufs(tone, 44100)
    assert integrated_lufs(tone * 10 ** (-20 / 20), 44100) == pytest.approx(loud - 20.0, abs=1e-6)
    # The same tone in both channels carries twice the power.
    assert integrated_lufs(np.stack([tone, tone]), 44100) == pytest.approx(loud + 3.0103, abs=1e-3)


def test_quiet_stretches_are_gated_out():
    tone = sine(997.0, 5.0, 44100, amplitude=0.1)
    with_silence = np.concatenate([tone, np.zeros(5 * 44100)])
    with_whisper = np.concatenate([tone, tone * 10 ** (-40 / 20)])
    alone = integrated_lufs(tone, 44100)
    # Not exactly equal: the three blocks that straddle the boundary pass the gate.
    assert integrated_lufs(with_silence, 44100) == pytest.approx(alone, abs=0.2)
    assert integrated_lufs(with_whisper, 44100) == pytest.approx(alone, abs=0.2)
    # Without gating the silent half would cost 3 dB.
    assert integrated_lufs(with_silence, 44100) > alone - 0.5


def test_silence_and_too_short_audio_have_no_loudness():
    assert integrated_lufs(np.zeros((2, 44100)), 44100) is None
    assert integrated_lufs(sine(997.0, 0.2, 44100), 44100) is None
    with pytest.raises(ValueError, match="mono or stereo"):
        integrated_lufs(np.zeros((3, 44100)), 44100)
