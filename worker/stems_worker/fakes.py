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


class OracleSeparator(Separator):
    """A "perfect model": returns stems it was given in advance, whatever the chunking.

    A separator only ever sees chunks, with no idea where in the track they came from. So
    ``prepare`` swaps the audio for a signal that encodes each sample's own position, the
    chunker cuts and pads that as usual, and ``separate_chunks`` decodes the positions and
    looks the answers up. Reflect padding mirrors the positions, which is exactly what it
    would have done to the true stems.

    ``silence`` names outputs to return as silence instead, to imitate a model that drops
    a stem. ``gains`` scales outputs, to imitate one that gets levels wrong.
    """

    _BLOCK = 4096

    def __init__(
        self,
        stems: dict[str, np.ndarray],
        *,
        model_id: str = "fake_oracle",
        chunk_size: int = 44100,
        silence: Sequence[str] = (),
        gains: dict[str, float] | None = None,
    ) -> None:
        if not stems:
            raise ValueError("an oracle needs at least one stem")
        lengths = {s.shape for s in stems.values()}
        if len(lengths) != 1:
            raise ValueError("all oracle stems must have the same shape")
        unknown = [n for n in (*silence, *(gains or {})) if n not in stems]
        if unknown:
            raise ValueError(f"unknown outputs {unknown}")
        self.model_id = model_id
        self.outputs = tuple(stems)
        self.chunk_size = chunk_size
        scale = {name: (0.0 if name in silence else (gains or {}).get(name, 1.0)) for name in stems}
        self._answers = np.stack(
            [stems[name].astype(np.float32) * np.float32(scale[name]) for name in stems]
        )
        self._length = self._answers.shape[2]
        self._blocks = -(-self._length // self._BLOCK)

    def load(self, device: str) -> None:
        self.device = device

    def prepare(self, audio: np.ndarray) -> tuple[np.ndarray, None]:
        if audio.shape[1] != self._length:
            raise ValueError(f"oracle holds {self._length} samples but was given {audio.shape[1]}")
        index = np.arange(self._length)
        # Both parts are exact in float32: a multiple of 1/4096, and a small integer ratio.
        within = (index % self._BLOCK) / self._BLOCK
        block = (index // self._BLOCK) / self._blocks
        return np.stack([within, block]).astype(np.float32), None

    def separate_chunks(self, batch: np.ndarray) -> np.ndarray:
        if self.device is None:
            raise RuntimeError("not loaded")
        within = np.rint(batch[:, 0] * self._BLOCK).astype(np.int64)
        block = np.rint(batch[:, 1] * self._blocks).astype(np.int64)
        index = np.clip(block * self._BLOCK + within, 0, self._length - 1)
        # (outputs, channels, batch, chunk) -> (batch, outputs, channels, chunk)
        return np.ascontiguousarray(self._answers[:, :, index].transpose(2, 0, 1, 3))


# --- a whole weightless pipeline ----------------------------------------------------------

# Low to high: which band of the spectrum stands in for which stem.
FAKE_BANDS: tuple[str, ...] = ("bass", "drums", "other", "piano", "guitar", "vocals")
_FAKE_CUTOFFS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "fake_bands": (FAKE_BANDS, (150.0, 400.0, 900.0, 2000.0, 5000.0)),
    "fake_bands_b": (FAKE_BANDS, (120.0, 350.0, 800.0, 1800.0, 4500.0)),
    "fake_split": (("backing_vocals", "lead_vocals"), (1000.0,)),
    "fake_split_b": (("backing_vocals", "lead_vocals"), (1200.0,)),
    "fake_dry": (("dry",), ()),
    "fake_dry_b": (("dry",), ()),
}


def fake_pipeline(defaults):  # type: ignore[no-untyped-def]
    """Everything a job needs to run with no weights, no tagger and no key.

    Returns ``(options, defaults)``: keyword arguments for the job runner (a registry of
    band-splitting stand-ins, the factory that builds them, a cache fingerprint) and the
    defaults with every slot pointed at them. The "stems" are frequency bands, so the
    output is not music; what it exercises is every step around the models. Nothing
    listens to the stems, so each is delivered with the not-checked warning.
    """
    from stems_common.settings import EnsembleSettings, SlotModels
    from stems_common.stems import PRODUCT_STEMS
    from stems_models.registry import ModelEntry, Registry

    def entry(model_id: str, stems: Sequence[str], slots: Sequence[str]):  # type: ignore[no-untyped-def]
        return ModelEntry.model_construct(
            id=model_id,
            name=f"Fake {model_id}",
            architecture="fake",
            stems=list(stems),
            candidate_for=list(slots),
            status="available",
            kind="separator",
            loader="msst",
            files=[],
        )

    def slot(first: str, second: str) -> SlotModels:
        return SlotModels(
            single_best=first, two_model_blend=[first, second], full_ensemble=[first, second]
        )

    registry = Registry.model_construct(
        schema_version=1,
        models=[
            entry("fake_bands", PRODUCT_STEMS, PRODUCT_STEMS),
            entry("fake_bands_b", PRODUCT_STEMS, PRODUCT_STEMS),
            entry("fake_split", ("lead_vocals", "backing_vocals"), ("vocal_split",)),
            entry("fake_split_b", ("lead_vocals", "backing_vocals"), ("vocal_split",)),
            entry("fake_dry", ("dry",), ("dereverb",)),
            entry("fake_dry_b", ("dry",), ("dereverb",)),
        ],
    )
    ensembles = EnsembleSettings(
        provisional=True,
        basis="Weightless stand-in models (stems_worker.fakes); not music.",
        default_mode="single_best",
        slots={
            **{stem: slot("fake_bands", "fake_bands_b") for stem in PRODUCT_STEMS},
            "vocal_split": slot("fake_split", "fake_split_b"),
            "dereverb": slot("fake_dry", "fake_dry_b"),
        },
    )

    def factory(model_id: str, stems: Sequence[str]) -> list[Separator]:
        outputs, cutoffs = _FAKE_CUTOFFS[model_id]
        return [BandSplitSeparator(outputs, cutoffs, model_id=model_id)]

    options = {
        "registry": registry,
        "factory": factory,
        "fingerprint": lambda model_id: f"fake-{model_id}",
        "rates": {},
        "client": None,
        "tagger": None,
    }
    return options, defaults.model_copy(update={"ensembles": ensembles})
