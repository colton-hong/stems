"""Waveform peaks for the web player.

One low and one high value per bucket of the stem, taken over both channels, so the
player can draw a waveform without downloading and decoding the audio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PEAKS_VERSION = 1
DEFAULT_BUCKETS = 1000


def compute_peaks(
    samples: np.ndarray, sample_rate: int, buckets: int = DEFAULT_BUCKETS
) -> dict[str, Any]:
    """Peaks of ``samples`` (channels, n). ``min`` and ``max`` each hold one value per bucket,
    rounded to four decimals and clamped to full scale."""
    if samples.ndim != 2:
        raise ValueError("expected audio shaped (channels, samples)")
    if buckets < 1:
        raise ValueError("at least one bucket is needed")
    length = samples.shape[1]
    count = max(1, min(buckets, length))
    lows, highs = np.zeros(count), np.zeros(count)
    if length:
        low = samples.min(axis=0)
        high = samples.max(axis=0)
        # Bucket i covers samples [edges[i], edges[i+1]); every sample is in exactly one.
        edges = np.linspace(0, length, count + 1).astype(np.int64)
        lows = np.minimum.reduceat(low, edges[:-1])
        highs = np.maximum.reduceat(high, edges[:-1])
    return {
        "version": PEAKS_VERSION,
        "sample_rate_hz": int(sample_rate),
        "duration_seconds": round(length / sample_rate, 3),
        "buckets": count,
        "min": [round(float(v), 4) for v in np.clip(lows, -1.0, 1.0)],
        "max": [round(float(v), 4) for v in np.clip(highs, -1.0, 1.0)],
    }
