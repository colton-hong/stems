"""Level statistics of a piece of audio that need no reference.

Used by the benchmark and, later, by the quality gate, where no reference stem exists.
Audio is float, shaped ``(channels, samples)``, with full scale at 1.0.
"""

from __future__ import annotations

import math

import numpy as np

# Levels are floored here so that digital silence is a number, not minus infinity.
FLOOR_DBFS = -120.0
SILENCE_DBFS = -60.0
SILENCE_WINDOW_SECONDS = 0.1


def _db(amplitude: float) -> float:
    return max(20.0 * math.log10(amplitude), FLOOR_DBFS) if amplitude > 0 else FLOOR_DBFS


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS level over all channels and samples, in dB relative to full scale."""
    if audio.size == 0:
        return FLOOR_DBFS
    return _db(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))


def peak_dbfs(audio: np.ndarray) -> float:
    """Largest absolute sample, in dB relative to full scale. Above 0 means clipping."""
    return _db(float(np.abs(audio).max())) if audio.size else FLOOR_DBFS


def silence_fraction(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = SILENCE_DBFS,
    window_seconds: float = SILENCE_WINDOW_SECONDS,
) -> float:
    """The fraction of short windows whose RMS is below ``threshold_dbfs``.

    1.0 for empty audio. The last window may be shorter than the others; it counts as one.
    """
    samples = audio.shape[-1]
    if samples == 0:
        return 1.0
    window = max(1, round(window_seconds * sample_rate))
    starts = np.arange(0, samples, window)
    power = np.square(audio, dtype=np.float64).reshape(-1, samples).mean(axis=0)
    sums = np.add.reduceat(power, starts)
    lengths = np.diff(np.append(starts, samples))
    limit = 10.0 ** (threshold_dbfs / 10.0)
    return float(np.mean(sums / lengths < limit))


def level_below_mix_db(stem: np.ndarray, mix: np.ndarray) -> float:
    """The stem's RMS level minus the mix's, over the whole song.

    Negative when the stem is quieter than the mix, which is the usual case. If the two
    differ in length the shorter one decides how much is compared.
    """
    samples = min(stem.shape[-1], mix.shape[-1])
    return rms_dbfs(stem[..., :samples]) - rms_dbfs(mix[..., :samples])
