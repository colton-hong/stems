"""Clips and their known answers for the Clef-omni test.

Everything here is arithmetic on dataset audio, so the answers are known without anyone
listening: which 30 seconds of a track are used, which stems are present in them, and how
much of another stem was mixed into a clean one.

- The excerpt is the loudest 30-second window of the mix by RMS, searched on a one-second
  grid. The same window is used for the mix and for every stem of the track.
- A stem is present in the excerpt if its RMS there is at least -40 dBFS and it is
  non-silent for at least 5 seconds.
- A bleed clip is a clean stem plus a different stem of the same track, scaled so that
  the target is a stated number of dB above it (the SIR) over the excerpt.
- Clips are sent as 16 kHz mono 16-bit WAV, the format Clef-omni hears in.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from stems_common.levels import rms_dbfs, silence_fraction
from stems_common.stems import PRODUCT_STEMS

WINDOW_SECONDS = 30.0
GRID_SECONDS = 1.0
PRESENT_RMS_DBFS = -40.0
PRESENT_SECONDS = 5.0
SIR_LEVELS_DB = (30.0, 20.0, 12.0, 6.0)
REVERB_WET_DB = -12.0
CLIP_SAMPLE_RATE = 16000
MAX_CLIP_BYTES = 8 * 1024 * 1024
# Part B's 20 clean stems: 5 vocals, 5 drums, 5 bass, 5 guitar or piano.
PART_B_TARGETS = (*["vocals"] * 5, *["drums"] * 5, *["bass"] * 5,
                  "guitar", "piano", "guitar", "piano", "guitar")  # fmt: skip
LEAD_BACKING_ANSWERS = ("lead only", "backing only", "both", "neither")


class ClipError(ValueError):
    pass


@dataclass(frozen=True)
class Window:
    start: int  # samples
    length: int  # samples

    def cut(self, audio: np.ndarray) -> np.ndarray:
        return audio[..., self.start : self.start + self.length]


def loudest_window(
    mix: np.ndarray,
    sample_rate: int,
    seconds: float = WINDOW_SECONDS,
    grid_seconds: float = GRID_SECONDS,
) -> Window:
    """The ``seconds``-long window of ``mix`` with the highest RMS.

    Starts are tried on a grid of ``grid_seconds``; of equally loud windows the earliest
    wins. A track no longer than the window is used whole.
    """
    samples = mix.shape[-1]
    length = round(seconds * sample_rate)
    if samples == 0:
        raise ClipError("the mix is empty")
    if samples <= length:
        return Window(0, samples)
    power = np.square(mix, dtype=np.float64).reshape(-1, samples).sum(axis=0)
    total = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, samples - length + 1, max(1, round(grid_seconds * sample_rate)))
    energy = total[starts + length] - total[starts]
    return Window(int(starts[int(np.argmax(energy))]), length)


def is_present(excerpt: np.ndarray, sample_rate: int) -> bool:
    """The pre-registered presence rule, applied to one stem's excerpt."""
    if excerpt.shape[-1] == 0 or rms_dbfs(excerpt) < PRESENT_RMS_DBFS:
        return False
    sounding = (1.0 - silence_fraction(excerpt, sample_rate)) * excerpt.shape[-1] / sample_rate
    return sounding >= PRESENT_SECONDS


def presence(stems: Mapping[str, np.ndarray], window: Window, sample_rate: int) -> dict[str, bool]:
    """Which of the six stems are present in ``window``."""
    return {stem: is_present(window.cut(stems[stem]), sample_rate) for stem in PRODUCT_STEMS}


def lead_backing_answer(
    lead: np.ndarray, backing: np.ndarray, window: Window, sample_rate: int
) -> str:
    has_lead = is_present(window.cut(lead), sample_rate)
    has_backing = is_present(window.cut(backing), sample_rate)
    if has_lead and has_backing:
        return "both"
    return "lead only" if has_lead else "backing only" if has_backing else "neither"


def _power(audio: np.ndarray) -> float:
    return float(np.mean(np.square(audio, dtype=np.float64)))


def measured_sir_db(target: np.ndarray, interference: np.ndarray) -> float:
    """How far ``target`` is above ``interference``, in dB of mean power."""
    p_target, p_other = _power(target), _power(interference)
    if p_target == 0.0 or p_other == 0.0:
        raise ClipError("SIR is undefined when either signal is silent")
    return 10.0 * float(np.log10(p_target / p_other))


def mix_at_sir(
    target: np.ndarray, contaminant: np.ndarray, sir_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """``target`` plus ``contaminant`` scaled to sit ``sir_db`` below it.

    Returns the mixture and the scaled contaminant, both float64, so the SIR actually
    achieved can be measured from the two parts.
    """
    if target.shape != contaminant.shape:
        raise ClipError(f"shapes differ: {target.shape} and {contaminant.shape}")
    gain = 10.0 ** ((measured_sir_db(target, contaminant) - sir_db) / 20.0)
    scaled = contaminant.astype(np.float64) * gain
    return target.astype(np.float64) + scaled, scaled


@dataclass(frozen=True)
class BleedItem:
    """One clean stem of Part B and the stem that is mixed into it."""

    track_id: str
    target: str
    contaminant: str


def plan_part_b(
    present: Mapping[str, Mapping[str, bool]],
    seed: int,
    targets: Sequence[str] = PART_B_TARGETS,
) -> list[BleedItem]:
    """Choose Part B's clean stems and their contaminants, reproducibly.

    ``present`` is the presence table per track. Tracks are shuffled with ``seed``; each
    target takes the next unused track in which it and at least one other stem are
    present, and the contaminant is drawn from those others. No track is used twice.
    """
    rng = np.random.default_rng(seed)
    order = sorted(present)
    rng.shuffle(order)
    used: set[str] = set()
    plan: list[BleedItem] = []
    for target in targets:
        for track_id in order:
            others = [s for s in PRODUCT_STEMS if s != target and present[track_id][s]]
            if track_id in used or not present[track_id][target] or not others:
                continue
            used.add(track_id)
            plan.append(BleedItem(track_id, target, str(rng.choice(others))))
            break
        else:
            raise ClipError(f"no unused track has {target} and another stem present")
    return plan


@dataclass(frozen=True)
class EncodedClip:
    data: bytes
    seconds: float
    # Gain applied to avoid clipping; 1.0 when the clip fitted as it was.
    gain: float


def encode_clip(audio: np.ndarray, sample_rate: int) -> EncodedClip:
    """16 kHz mono 16-bit WAV. Scaled down, never up, and only if it would clip."""
    import soundfile as sf

    from stems_worker.audio import resample

    mono = np.atleast_2d(audio).astype(np.float64).mean(axis=0, keepdims=True)
    if sample_rate != CLIP_SAMPLE_RATE:
        mono = resample(mono.astype(np.float32), sample_rate, CLIP_SAMPLE_RATE)
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    gain = 0.999 / peak if peak > 0.999 else 1.0
    buffer = io.BytesIO()
    sf.write(buffer, (mono[0] * gain).astype(np.float32), CLIP_SAMPLE_RATE,
             format="WAV", subtype="PCM_16")  # fmt: skip
    data = buffer.getvalue()
    if len(data) >= MAX_CLIP_BYTES:
        raise ClipError(f"clip is {len(data)} bytes, over the {MAX_CLIP_BYTES} limit")
    return EncodedClip(data, mono.shape[1] / CLIP_SAMPLE_RATE, gain)
