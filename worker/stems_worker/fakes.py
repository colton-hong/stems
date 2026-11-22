"""Stand-in separators that need no weights.

Tests, the benchmark harness and the API's end-to-end test run these through exactly the
same code path as a real model: load onto a device, receive chunk batches, return stems.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch

from stems_worker.separator import Separator


class BandSplitSeparator(Separator):
    """Splits audio into frequency bands with a brick-wall FFT filter per chunk.

    With one cutoff it is a two-stem model: everything below the cutoff, and everything
    above. The bands add back to the input exactly. Because the filter is applied to each
    chunk on its own, its output near a chunk edge differs from its output mid-chunk, the
    same weakness real models have, so it exercises the crossfade for real.
    """

    def __init__(
        self,
        outputs: Sequence[str] = ("low", "high"),
        cutoffs_hz: Sequence[float] = (300.0,),
        chunk_size: int = 44100,
        model_id: str = "fake_band_split",
    ) -> None:
        if len(cutoffs_hz) != len(outputs) - 1:
            raise ValueError("need exactly one cutoff between each pair of neighbouring bands")
        if list(cutoffs_hz) != sorted(cutoffs_hz):
            raise ValueError("cutoffs must be in ascending order")
        self.model_id = model_id
        self.outputs = tuple(outputs)
        self.chunk_size = chunk_size
        self._cutoffs = tuple(float(c) for c in cutoffs_hz)
        self._masks: torch.Tensor | None = None

    def load(self, device: str) -> None:
        freqs = torch.fft.rfftfreq(self.chunk_size, 1.0 / self.sample_rate)
        edges = [0.0, *self._cutoffs, float("inf")]
        masks = [(freqs >= lo) & (freqs < hi) for lo, hi in itertools.pairwise(edges)]
        self._masks = torch.stack(masks).to(torch.float32).to(device)
        self.device = device

    def separate_chunks(self, batch: np.ndarray) -> np.ndarray:
        if self._masks is None or self.device is None:
            raise RuntimeError("not loaded")
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
        spectrum = torch.fft.rfft(x, dim=-1)  # (batch, channels, bins)
        bands = spectrum[:, None] * self._masks[None, :, None, :]
        stems = torch.fft.irfft(bands, n=self.chunk_size, dim=-1)
        return stems.cpu().numpy()

    def unload(self) -> None:
        self._masks = None
        super().unload()
