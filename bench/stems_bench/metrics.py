"""Separation quality scores: museval's BSS Eval v4, per stem, on one-second windows.

The numbers come from museval's own ``metrics.bss_eval`` (the implementation behind the
SiSEC 2018 and MUSDB18 leaderboards), called the way ``museval.evaluate`` calls it: image
metrics, one distortion filter for the whole track, non-overlapping windows. A stem's score
for a track is the median over its windows, ignoring windows museval leaves undefined.

SIR, SAR and ISR come from projecting the estimate onto delayed copies of the references.
That projection is ill-conditioned when a reference is a few pure tones or its channels are
identical, which is the case for most stems of the synthetic fixture: there the three
numbers are numerical noise and only SDR, which is a plain energy ratio, means anything.
On broadband signals, and on music, all four behave.

**Every stem is scored as a pair**: (target, everything else), where "everything else" is
the mix minus the target, and the estimate of it is the mix minus the estimated target.
SDR does not depend on which other sources are in the set, so it is the standard museval
SDR. SIR and SAR do depend on it; scoring all models against the same pair makes those two
comparable between a single-stem model and a six-stem one, at the price of not being the
joint four-source SIR and SAR that MUSDB18 leaderboards print.

museval's package cannot be imported on a machine without ffmpeg (its ``musdb`` dependency
checks for the binary at import). Its ``metrics`` module needs only numpy and scipy, so it
is loaded here by file path from the installed, pinned distribution.

Audio is float, shaped ``(channels, samples)``, like everywhere else in this project.
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

WINDOW_SECONDS = 1.0
BSS_METRICS: tuple[str, ...] = ("sdr", "sir", "sar", "isr")


class MetricsError(RuntimeError):
    pass


@cache
def museval_metrics() -> ModuleType:
    """museval's ``metrics`` module, without importing the ``museval`` package."""
    spec = importlib.util.find_spec("museval")
    if spec is None or not spec.submodule_search_locations:
        raise MetricsError('museval is not installed; run `pip install -e ".[bench]"`')
    path = Path(next(iter(spec.submodule_search_locations))) / "metrics.py"
    module_spec = importlib.util.spec_from_file_location("stems_bench._museval_metrics", path)
    if module_spec is None or module_spec.loader is None:
        raise MetricsError(f"cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class BssScores:
    """One stem's scores on one track.

    ``sdr``, ``sir``, ``sar`` and ``isr`` are medians over the scored windows, in dB. They
    are ``nan`` when no window could be scored and may be ``inf`` for an exact estimate.
    ``frames`` holds every window's value, ``nan`` where museval gives none (the target,
    the rest of the mix, or either estimate is digitally silent in that window).
    """

    sdr: float
    sir: float
    sar: float
    isr: float
    windows: int
    scored_windows: int
    window_seconds: float
    frames: dict[str, np.ndarray] = field(repr=False, compare=False)

    def as_dict(self, *, frames: bool = True) -> dict[str, Any]:
        """JSON-safe form: non-finite values become ``None``."""
        out: dict[str, Any] = {name: _finite(getattr(self, name)) for name in BSS_METRICS}
        out |= {
            "windows": self.windows,
            "scored_windows": self.scored_windows,
            "window_seconds": self.window_seconds,
        }
        if frames:
            out["frames"] = {
                name: [None if not math.isfinite(v) else round(float(v), 4) for v in values]
                for name, values in self.frames.items()
            }
        return out


def fit_length(audio: np.ndarray, samples: int) -> np.ndarray:
    """``audio`` cut or zero-padded at the end to ``samples`` (what museval does)."""
    if audio.shape[-1] == samples:
        return audio
    if audio.shape[-1] > samples:
        return audio[..., :samples]
    pad = [(0, 0)] * (audio.ndim - 1) + [(0, samples - audio.shape[-1])]
    return np.pad(audio, pad)


def _median(values: np.ndarray) -> float:
    scored = values[~np.isnan(values)]
    return float(np.median(scored)) if scored.size else math.nan


def bss_eval(
    references: dict[str, np.ndarray],
    estimates: dict[str, np.ndarray],
    sample_rate: int,
    *,
    window_seconds: float = WINDOW_SECONDS,
) -> dict[str, BssScores]:
    """Score a set of sources jointly. Keys of the two dicts must match.

    Estimates are cut or padded to the references' length. A window in which any reference
    or any estimate in the set is all zeros is not scored for any source; that is museval's
    rule, and it is why ``scored_windows`` is reported next to every median.
    """
    if set(references) != set(estimates) or not references:
        raise MetricsError(
            f"references {sorted(references)} and estimates {sorted(estimates)} must match"
        )
    names = list(references)
    samples = references[names[0]].shape[-1]
    window = round(window_seconds * sample_rate)
    if samples < window:
        raise MetricsError(f"audio is shorter than one {window_seconds} s window")
    for name in names:
        if references[name].ndim != 2 or references[name].shape[-1] != samples:
            raise MetricsError(f"reference {name!r} has shape {references[name].shape}")
        if estimates[name].ndim != 2 or estimates[name].shape[0] != references[name].shape[0]:
            raise MetricsError(f"estimate {name!r} has shape {estimates[name].shape}")

    # museval wants (sources, samples, channels) and works in double precision.
    ref = np.stack([references[n].T for n in names]).astype(np.float64)
    est = np.stack([fit_length(estimates[n], samples).T for n in names]).astype(np.float64)
    windows = (samples - window) // window + 1
    if not (ref.any(axis=(1, 2)).all() and est.any(axis=(1, 2)).all()):
        # A source that is silent for the whole track. museval refuses such a set outright;
        # by its own per-window rule no window would be scored, so say exactly that.
        empty = np.full(windows, np.nan)
        return {
            name: BssScores(
                **dict.fromkeys(BSS_METRICS, math.nan),
                windows=windows,
                scored_windows=0,
                window_seconds=window_seconds,
                frames={metric: empty.copy() for metric in BSS_METRICS},
            )
            for name in names
        }
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        sdr, isr, sir, sar, _ = museval_metrics().bss_eval(
            ref,
            est,
            compute_permutation=False,
            window=window,
            hop=window,
            framewise_filters=False,
            bsseval_sources_version=False,
        )
    scores = {}
    for i, name in enumerate(names):
        frames = {"sdr": sdr[i], "sir": sir[i], "sar": sar[i], "isr": isr[i]}
        scores[name] = BssScores(
            **{metric: _median(values) for metric, values in frames.items()},
            windows=int(sdr.shape[1]),
            scored_windows=int((~np.isnan(sdr[i])).sum()),
            window_seconds=window_seconds,
            frames=frames,
        )
    return scores


def bss_eval_stem(
    reference: np.ndarray,
    estimate: np.ndarray,
    mix: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = WINDOW_SECONDS,
) -> BssScores:
    """Score one stem against the (target, rest of the mix) pair.

    This is the protocol for every stem of every model: the second source is the mix minus
    the target, and its estimate is the mix minus the estimated target.
    """
    if mix.shape != reference.shape:
        raise MetricsError(f"mix {mix.shape} and reference {reference.shape} differ in shape")
    fitted = fit_length(estimate, reference.shape[-1])
    scores = bss_eval(
        {"target": reference, "rest": mix - reference},
        {"target": fitted, "rest": mix - fitted},
        sample_rate,
        window_seconds=window_seconds,
    )
    return scores["target"]
