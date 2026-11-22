"""Run a fixed-length model over audio of any length, with no audible seams.

Separation models take chunks of one fixed size. Cutting a song into chunks and gluing
the outputs back together would click at every join, because a model's output near the
edge of a chunk differs from its output for the same audio in the middle of a chunk.

So chunks overlap. Each chunk's output is multiplied by a window that ramps up over the
overlap at its start and down over the overlap at its end; the windowed outputs are summed
and divided by the summed windows. Inside an overlap the result therefore crossfades from
one chunk to the next, and the overall gain is exactly one everywhere.

The ends of the audio are reflect-padded by one overlap, so the first and last samples sit
away from a chunk edge too. The padding is removed from the result.

This module is plain numpy and knows nothing about models or devices: the caller passes a
function from a batch of chunks to a batch of outputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# (batch, channels, chunk_size) -> (batch, outputs, channels, chunk_size)
ChunkFn = Callable[[np.ndarray], np.ndarray]
ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class ChunkPlan:
    """Where the chunks fall for one piece of audio."""

    num_samples: int
    chunk_size: int
    hop: int
    # Samples of reflect padding added before and after the audio.
    border: int
    # Start of each chunk in the padded signal.
    starts: tuple[int, ...]
    # Length of the padded signal, including zero fill to complete the last chunk.
    padded_length: int

    @property
    def overlap(self) -> int:
        return self.chunk_size - self.hop

    @property
    def num_chunks(self) -> int:
        return len(self.starts)


def hop_for_overlap(chunk_size: int, overlap: float) -> int:
    """Hop size for an overlap given as a fraction of the chunk (0 <= overlap < 1)."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    return max(1, round(chunk_size * (1.0 - overlap)))


def plan_chunks(num_samples: int, chunk_size: int, hop: int) -> ChunkPlan:
    if num_samples <= 0:
        raise ValueError("audio is empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 < hop <= chunk_size:
        raise ValueError(f"hop must be in (0, chunk_size], got {hop}")

    overlap = chunk_size - hop
    # Reflect padding needs more audio than padding. Very short inputs go without it.
    border = overlap if num_samples > overlap else 0
    content = num_samples + 2 * border
    count = max(1, math.ceil((content - overlap) / hop))
    starts = tuple(k * hop for k in range(count))
    return ChunkPlan(
        num_samples=num_samples,
        chunk_size=chunk_size,
        hop=hop,
        border=border,
        starts=starts,
        padded_length=starts[-1] + chunk_size,
    )


def crossfade_window(chunk_size: int, overlap: int) -> np.ndarray:
    """One in the middle, with linear ramps over ``overlap`` samples at both ends.

    The ramps never reach zero, so a sum of windows is positive wherever a chunk exists.
    When the overlap is more than half a chunk the two ramps meet; they are combined with
    a minimum, which keeps the window symmetric and positive.
    """
    window = np.ones(chunk_size, dtype=np.float32)
    if overlap > 0:
        ramp = ((np.arange(overlap) + 0.5) / overlap).astype(np.float32)
        window[:overlap] = ramp
        window[chunk_size - overlap :] = np.minimum(window[chunk_size - overlap :], ramp[::-1])
    return window


def apply_chunked(
    audio: np.ndarray,
    fn: ChunkFn,
    *,
    chunk_size: int,
    hop: int,
    batch_size: int = 1,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Apply ``fn`` chunk by chunk and stitch the outputs.

    ``audio`` is ``(channels, samples)``. The result is ``(outputs, channels, samples)``,
    where ``outputs`` is whatever ``fn`` returns along its second axis (one per stem).
    ``progress(done, total)`` is called after each batch.
    """
    if audio.ndim != 2:
        raise ValueError(f"expected (channels, samples), got shape {audio.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    channels, num_samples = audio.shape
    plan = plan_chunks(num_samples, chunk_size, hop)

    padded = np.zeros((channels, plan.padded_length), dtype=np.float32)
    body = audio.astype(np.float32, copy=False)
    if plan.border:
        body = np.pad(body, ((0, 0), (plan.border, plan.border)), mode="reflect")
    padded[:, : body.shape[1]] = body

    window = crossfade_window(chunk_size, plan.overlap)
    first_window = window.copy()
    last_window = window.copy()
    if plan.overlap:
        # Nothing precedes the first chunk or follows the last one, so there is nothing
        # to fade against at those two edges.
        first_window[: plan.overlap] = 1.0
        last_window[chunk_size - plan.overlap :] = 1.0

    def window_for(index: int) -> np.ndarray:
        if plan.num_chunks == 1:
            return np.ones(chunk_size, dtype=np.float32)
        if index == 0:
            return first_window
        if index == plan.num_chunks - 1:
            return last_window
        return window

    total: np.ndarray | None = None
    weight = np.zeros(plan.padded_length, dtype=np.float32)

    for batch_start in range(0, plan.num_chunks, batch_size):
        indices = range(batch_start, min(batch_start + batch_size, plan.num_chunks))
        batch = np.stack([padded[:, plan.starts[i] : plan.starts[i] + chunk_size] for i in indices])
        result = np.asarray(fn(batch))
        if (
            result.ndim != 4
            or result.shape[0] != len(indices)
            or result.shape[2:] != (channels, chunk_size)
        ):
            raise ValueError(
                f"model returned shape {result.shape}; expected "
                f"({len(indices)}, outputs, {channels}, {chunk_size})"
            )
        if total is None:
            total = np.zeros((result.shape[1], channels, plan.padded_length), dtype=np.float32)
        elif result.shape[1] != total.shape[0]:
            raise ValueError("model returned a different number of outputs between batches")

        for row, index in enumerate(indices):
            start = plan.starts[index]
            w = window_for(index)
            total[:, :, start : start + chunk_size] += result[row] * w
            weight[start : start + chunk_size] += w
        if progress is not None:
            progress(indices[-1] + 1, plan.num_chunks)

    assert total is not None  # plan_chunks guarantees at least one chunk
    total /= weight
    return np.ascontiguousarray(total[:, :, plan.border : plan.border + num_samples])
