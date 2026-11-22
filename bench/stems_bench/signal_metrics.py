"""The benchmark's metrics besides BSS Eval.

SDR is dominated by the loud, low part of the spectrum and says nothing about a stem that
is silent, the wrong length, or missing its top octave. These catch those cases:

- silence fraction, RMS and peak of the estimate (and of the reference, for comparison);
- duration match: estimate length minus reference length;
- high-band energy ratio: energy of the estimate above 8 kHz relative to the reference's.

Audio is float, shaped ``(channels, samples)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from stems_common.levels import FLOOR_DBFS, peak_dbfs, rms_dbfs, silence_fraction

HIGH_BAND_HZ = 8000.0
_FRAME = 4096
# A reference with less high-band energy than this (dB relative to full scale) has no
# top octave to lose, so the ratio is undefined rather than a huge meaningless number.
HIGH_BAND_MIN_DBFS = -90.0


def level_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Duration, RMS, peak and silence fraction of one signal."""
    return {
        "duration_seconds": audio.shape[-1] / sample_rate,
        "rms_dbfs": round(rms_dbfs(audio), 2),
        "peak_dbfs": round(peak_dbfs(audio), 2),
        "silence_fraction": round(silence_fraction(audio, sample_rate), 4),
    }


def duration_delta_seconds(estimate: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
    """Estimate length minus reference length. Zero when they match to the sample."""
    return (estimate.shape[-1] - reference.shape[-1]) / sample_rate


def band_energy(audio: np.ndarray, sample_rate: int, low_hz: float = HIGH_BAND_HZ) -> float:
    """Mean power per sample at and above ``low_hz``, summed over channels.

    Measured on Hann-windowed frames of 4096 samples with half overlap, scaled so that a
    stationary signal's result equals its mean square. Zero for audio shorter than a frame.
    """
    samples = audio.shape[-1]
    if samples < _FRAME:
        return 0.0
    window = np.hanning(_FRAME + 1)[:-1]
    first_bin = math.ceil(low_hz * _FRAME / sample_rate)
    scale = 2.0 / (_FRAME * float(np.sum(window**2)))
    hop = _FRAME // 2
    count = (samples - _FRAME) // hop + 1
    total = 0.0
    for channel in audio.reshape(-1, samples):
        frames = np.lib.stride_tricks.sliding_window_view(channel, _FRAME)[::hop][:count]
        # In blocks, so a ten-minute track never needs its whole spectrogram in memory.
        for start in range(0, count, 512):
            block = frames[start : start + 512].astype(np.float64) * window
            spectrum = np.fft.rfft(block, axis=1)[:, first_bin:]
            total += float(np.sum(spectrum.real**2 + spectrum.imag**2))
    return total * scale / count


def high_band_ratio_db(
    estimate: np.ndarray, reference: np.ndarray, sample_rate: int, low_hz: float = HIGH_BAND_HZ
) -> float | None:
    """Estimate energy over reference energy above ``low_hz``, in dB.

    0 is a match; negative means the estimate lost high end (a low-passed stem); positive
    means it has more than the reference (bleed of cymbals, hiss, ringing). ``None`` when
    the reference has no high-band energy to speak of. The result is floored at -120 dB.
    """
    floor = 10.0 ** (HIGH_BAND_MIN_DBFS / 10.0)
    reference_energy = band_energy(reference, sample_rate, low_hz)
    if reference_energy < floor:
        return None
    estimate_energy = band_energy(estimate, sample_rate, low_hz)
    if estimate_energy <= 0.0:
        return FLOOR_DBFS
    return max(10.0 * math.log10(estimate_energy / reference_energy), FLOOR_DBFS)


def compare(estimate: np.ndarray, reference: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Everything in this module for one estimated stem and its reference, JSON-ready."""
    ratio = high_band_ratio_db(estimate, reference, sample_rate)
    return {
        "estimate": level_stats(estimate, sample_rate),
        "reference": level_stats(reference, sample_rate),
        "duration_delta_seconds": duration_delta_seconds(estimate, reference, sample_rate),
        "duration_matches": estimate.shape[-1] == reference.shape[-1],
        "rms_delta_db": round(rms_dbfs(estimate) - rms_dbfs(reference), 2),
        "high_band_hz": HIGH_BAND_HZ,
        "high_band_ratio_db": None if ratio is None else round(ratio, 2),
    }
