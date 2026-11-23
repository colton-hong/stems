"""Integrated loudness in LUFS, as defined by ITU-R BS.1770-4.

K-weighting (a high shelf, then a high-pass), mean square per 400 ms block with 75 %
overlap, an absolute gate at -70 LUFS and a relative gate 10 LU below the mean of what
passed the first. Mono and stereo only: every channel has weight 1.

The filter coefficients are derived from the standard's 48 kHz biquads so that they hold
at any sample rate (the derivation libebur128 uses); at 48 kHz they are the standard's own
numbers, which a test checks.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

BLOCK_SECONDS = 0.4
STEP_SECONDS = 0.1
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
OFFSET = -0.691


def k_weighting(sample_rate: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """The two biquads as ``(b, a)`` pairs, shelf first."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = (
        np.array([(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0,
                  (vh - vb * k / q + k * k) / a0]),
        np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]),
    )  # fmt: skip
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sample_rate)
    a0 = 1.0 + k / q + k * k
    high_pass = (
        np.array([1.0, -2.0, 1.0]),
        np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]),
    )
    return shelf, high_pass


def _lufs(mean_square: float) -> float:
    return OFFSET + 10.0 * math.log10(mean_square)


def integrated_lufs(audio: np.ndarray, sample_rate: int) -> float | None:
    """Loudness of ``audio`` (``(channels, samples)`` or mono), or None if it has none.

    None when the audio is shorter than one 400 ms block or no block is louder than the
    absolute gate, which is what silence gives.
    """
    channels = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    if channels.shape[0] > 2:
        raise ValueError(f"mono or stereo only, got {channels.shape[0]} channels")
    block = round(BLOCK_SECONDS * sample_rate)
    step = round(STEP_SECONDS * sample_rate)
    if channels.shape[1] < block:
        return None
    for b, a in k_weighting(sample_rate):
        channels = lfilter(b, a, channels, axis=1)
    # Sum of squares over channels, then block means from a running total.
    total = np.concatenate([[0.0], np.cumsum(np.square(channels).sum(axis=0))])
    starts = np.arange(0, channels.shape[1] - block + 1, step)
    power = (total[starts + block] - total[starts]) / block
    power = power[power > 10.0 ** ((ABSOLUTE_GATE_LUFS - OFFSET) / 10.0)]
    if power.size == 0:
        return None
    relative = _lufs(float(power.mean())) + RELATIVE_GATE_LU
    power = power[power > 10.0 ** ((relative - OFFSET) / 10.0)]
    return _lufs(float(power.mean())) if power.size else None
