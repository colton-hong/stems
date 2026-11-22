"""The interface every separation model is wrapped in, and the code that runs one.

A ``Separator`` only knows how to turn a batch of fixed-length chunks into a batch of
stems. Chunking, overlap, crossfading and timing are done here, once, for every model, so
all models are run and timed the same way.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from stems_worker.audio import PIPELINE_SAMPLE_RATE
from stems_worker.chunking import ProgressFn, apply_chunked, hop_for_overlap, plan_chunks

DEFAULT_OVERLAP = 0.25


class Separator(ABC):
    """One loaded checkpoint.

    Subclasses set ``model_id``, ``outputs`` (stem names, in the order the model emits
    them) and ``chunk_size`` (samples), and implement ``load`` and ``separate_chunks``.
    """

    model_id: str
    outputs: tuple[str, ...]
    chunk_size: int
    sample_rate: int = PIPELINE_SAMPLE_RATE
    # Set by load(); None means not loaded.
    device: str | None = None

    @abstractmethod
    def load(self, device: str) -> None:
        """Read the weights and move the model to ``device``."""

    @abstractmethod
    def separate_chunks(self, batch: np.ndarray) -> np.ndarray:
        """``(batch, channels, chunk_size)`` float32 in, ``(batch, outputs, channels,
        chunk_size)`` float32 out, as numpy arrays on the CPU."""

    def unload(self) -> None:
        """Release the model. Safe to call when nothing is loaded."""
        self.device = None


@dataclass(frozen=True)
class SeparationResult:
    model_id: str
    device: str
    stems: dict[str, np.ndarray]  # each float32 (channels, samples)
    sample_rate: int
    audio_seconds: float
    # Wall-clock time spent inside the model and the stitching, excluding load time.
    wall_seconds: float
    num_chunks: int

    @property
    def realtime_factor(self) -> float:
        """Seconds of compute per second of audio. Below 1 is faster than real time."""
        return self.wall_seconds / self.audio_seconds


def run_separator(
    separator: Separator,
    audio: np.ndarray,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
    *,
    overlap: float = DEFAULT_OVERLAP,
    batch_size: int = 1,
    progress: ProgressFn | None = None,
) -> SeparationResult:
    """Run a loaded separator over ``audio`` of any length."""
    if separator.device is None:
        raise RuntimeError(f"{separator.model_id} is not loaded; call load(device) first")
    if sample_rate != separator.sample_rate:
        raise ValueError(
            f"{separator.model_id} runs at {separator.sample_rate} Hz "
            f"but got {sample_rate} Hz audio"
        )
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"expected stereo (2, samples), got shape {audio.shape}")
    if len(set(separator.outputs)) != len(separator.outputs):
        raise ValueError(f"{separator.model_id} lists an output twice: {separator.outputs}")

    hop = hop_for_overlap(separator.chunk_size, overlap)
    plan = plan_chunks(audio.shape[1], separator.chunk_size, hop)
    started = time.perf_counter()
    stacked = apply_chunked(
        audio,
        separator.separate_chunks,
        chunk_size=separator.chunk_size,
        hop=hop,
        batch_size=batch_size,
        progress=progress,
    )
    wall = time.perf_counter() - started
    if stacked.shape[0] != len(separator.outputs):
        raise ValueError(
            f"{separator.model_id} returned {stacked.shape[0]} outputs "
            f"but declares {len(separator.outputs)}: {separator.outputs}"
        )
    return SeparationResult(
        model_id=separator.model_id,
        device=separator.device,
        stems={name: stacked[i] for i, name in enumerate(separator.outputs)},
        sample_rate=sample_rate,
        audio_seconds=audio.shape[1] / sample_rate,
        wall_seconds=wall,
        num_chunks=plan.num_chunks,
    )
