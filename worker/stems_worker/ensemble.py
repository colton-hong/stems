"""Combine several models' estimates of the same stem into one.

Four methods:

- ``waveform``: the weighted mean of the waveforms. Linear, so stems that added up to the
  mix before blending still do.
- ``stft_magnitude``: the weighted mean of the STFT magnitudes, with the phase of the
  weighted sum. Where the estimates agree in phase it equals ``waveform``; where they
  disagree it keeps the energy a plain mean would cancel.
- ``stft_min``: per time-frequency bin, the estimate with the smallest magnitude. Keeps
  only what every model agrees is there: less bleed, at the cost of holes.
- ``stft_max``: the largest. Keeps what any model found: fuller, with more bleed.

The last two choose rather than average, so they take no weights.

Audio is float32, shaped ``(channels, samples)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

METHODS: tuple[str, ...] = ("waveform", "stft_magnitude", "stft_min", "stft_max")
N_FFT = 2048
HOP = 512


class EnsembleError(ValueError):
    pass


def normalise_weights(weights: Sequence[float] | None, count: int) -> np.ndarray:
    """``count`` non-negative weights that sum to 1. ``None`` means equal weights."""
    if weights is None:
        return np.full(count, 1.0 / count)
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (count,):
        raise EnsembleError(f"{count} estimates need {count} weights, got {len(weights)}")
    if not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
        raise EnsembleError(f"weights must be non-negative and not all zero, got {list(weights)}")
    return array / array.sum()


def _stft(audio: np.ndarray) -> torch.Tensor:
    window = torch.hann_window(N_FFT, dtype=torch.float64)
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float64))
    return torch.stft(x, N_FFT, HOP, window=window, center=True, return_complex=True)


def _istft(spectrum: torch.Tensor, samples: int) -> np.ndarray:
    window = torch.hann_window(N_FFT, dtype=torch.float64)
    audio = torch.istft(spectrum, N_FFT, HOP, window=window, center=True, length=samples)
    return audio.numpy().astype(np.float32)


def blend(
    estimates: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    method: str = "waveform",
) -> np.ndarray:
    """One estimate from several of the same stem.

    A single estimate is returned unchanged whatever the method. Estimates given a weight
    of zero take no part.
    """
    if method not in METHODS:
        raise EnsembleError(f"unknown blend method {method!r}; choose from {METHODS}")
    if not estimates:
        raise EnsembleError("nothing to blend")
    shape = estimates[0].shape
    if any(e.shape != shape or e.ndim != 2 for e in estimates):
        raise EnsembleError(f"estimates differ in shape: {[e.shape for e in estimates]}")
    if method in ("stft_min", "stft_max") and weights is not None and len(set(weights)) > 1:
        raise EnsembleError(f"{method} chooses per bin and takes no weights")
    w = normalise_weights(weights, len(estimates))
    used = [(e, float(wi)) for e, wi in zip(estimates, w, strict=True) if wi > 0]
    if len(used) == 1:
        return np.array(used[0][0], dtype=np.float32)
    samples = shape[1]
    if samples < N_FFT and method != "waveform":
        raise EnsembleError(f"{method} needs at least {N_FFT} samples, got {samples}")

    if method == "waveform":
        total = np.zeros(shape, dtype=np.float64)
        for estimate, weight in used:
            total += weight * estimate
        return total.astype(np.float32)

    spectra = torch.stack([_stft(e) for e, _ in used])  # (models, channels, bins, frames)
    if method == "stft_magnitude":
        weight = torch.tensor([wi for _, wi in used], dtype=torch.float64)[:, None, None, None]
        magnitude = (weight * spectra.abs()).sum(dim=0)
        combined = (weight * spectra).sum(dim=0)
        # Where the weighted sum cancels to nothing there is no phase to take: borrow the
        # first estimate's.
        direction = torch.where(combined.abs() > 0, combined, spectra[0])
        phase = torch.angle(direction)
        result = torch.polar(magnitude, phase)
    else:
        magnitudes = spectra.abs()
        index = magnitudes.argmin(dim=0) if method == "stft_min" else magnitudes.argmax(dim=0)
        result = torch.gather(spectra, 0, index[None])[0]
    return _istft(result, samples)


def blend_stems(
    per_model: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float] | None = None,
    method: str = "waveform",
) -> dict[str, np.ndarray]:
    """Blend each named output across models. Every model must offer the same names."""
    if not per_model:
        raise EnsembleError("nothing to blend")
    names = list(per_model[0])
    if any(set(m) != set(names) for m in per_model):
        raise EnsembleError(f"models offer different outputs: {[sorted(m) for m in per_model]}")
    return {name: blend([m[name] for m in per_model], weights, method) for name in names}
