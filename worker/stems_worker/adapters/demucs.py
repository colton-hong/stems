"""Adapter for HT-Demucs checkpoints (``loader: demucs``).

The network definition comes from the official ``demucs`` package. Loading and inference
are done here instead of with its helpers:

- its checkpoints are pickled "packages" holding a class reference, constructor arguments
  and a state dict. They are read with ``weights_only=True`` and an allowlist of exactly
  the five harmless types they contain, where the package unpickles anything;
- chunking and overlap are ours (``stems_worker.separator``), as for every model family.
  The package's "shift trick" (averaging over random time shifts) is not applied;
- the input is normalised over the whole track the way the package's own separator does,
  through the ``prepare`` / ``finish`` hooks.

``htdemucs_ft`` is a bag of four models, each fine-tuned for one source. Asking for a
subset of stems loads only those models.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from demucs.htdemucs import HTDemucs
from numpy._core.multiarray import scalar as _numpy_scalar

from stems_models.download import ensure_model
from stems_models.registry import ModelEntry, Registry, load_registry
from stems_worker.separator import Separator

# Everything a released HT-Demucs package contains besides tensors and plain containers:
# the model class, the segment length, and numpy floats in its training metrics.
_SAFE_GLOBALS: list[Any] = [
    HTDemucs,
    Fraction,
    (_numpy_scalar, "numpy.core.multiarray.scalar"),  # the path numpy 1 pickles used
    np.dtype,
    np.dtypes.Float64DType,
]


class DemucsConfigError(ValueError):
    """A checkpoint does not match what the registry says about it."""


def read_package(path: Path | str) -> dict[str, Any]:
    """Read a Demucs checkpoint without unpickling anything outside the allowlist."""
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        package = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or package.get("klass") is not HTDemucs:
        raise DemucsConfigError(f"{path}: not an HT-Demucs package")
    state = package.get("state")
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise DemucsConfigError(f"{path}: state is not a plain tensor dict")
    return package


def build_model(package: dict[str, Any]) -> HTDemucs:
    accepted = inspect.signature(HTDemucs).parameters
    kwargs = {k: v for k, v in package["kwargs"].items() if k in accepted}
    model = HTDemucs(*package["args"], **kwargs)
    # Released weights are stored as float16; they are copied into a float32 model.
    model.load_state_dict({k: v.to(torch.float32) for k, v in package["state"].items()})
    return model.eval()


class DemucsSeparator(Separator):
    def __init__(
        self, entry: ModelEntry, files: dict[str, Path], stems: Sequence[str] | None = None
    ) -> None:
        if entry.loader != "demucs":
            raise DemucsConfigError(f"{entry.id} uses loader {entry.loader!r}, not 'demucs'")
        try:
            self._sources = tuple(entry.config["sources"])
            segment = float(entry.config["segment_seconds"])
        except KeyError as e:
            raise DemucsConfigError(f"{entry.id}: registry config lacks {e.args[0]!r}") from None

        self.model_id = entry.id
        self.sample_rate = int(entry.sample_rate_hz or 44100)
        self.chunk_size = round(segment * self.sample_rate)

        per_stem = {f.stem: f.filename for f in entry.weights if f.stem is not None}
        if per_stem:
            wanted = list(stems) if stems else [s for s in self._sources if s in per_stem]
            unknown = [s for s in wanted if s not in per_stem]
            if unknown:
                raise DemucsConfigError(f"{entry.id} has no model for {unknown}")
            self.outputs = tuple(wanted)
            names = [per_stem[s] for s in wanted]
            # One model per requested stem, each contributing only its own source.
            self._picks: list[int | None] = [self._sources.index(s) for s in wanted]
        else:
            if stems:
                raise DemucsConfigError(f"{entry.id} is one model; it always emits every source")
            self.outputs = self._sources
            names = [f.filename for f in entry.weights]
            self._picks = [None]
        missing = [n for n in names if n not in files]
        if missing:
            raise DemucsConfigError(f"{entry.id}: missing files {missing}")
        self._paths = [files[n] for n in names]
        self._models: list[HTDemucs] = []

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        *,
        stems: Sequence[str] | None = None,
        registry: Registry | None = None,
        directory: Path | None = None,
    ) -> DemucsSeparator:
        """Build from a registry id, downloading and verifying only the files needed."""
        registry = registry or load_registry()
        entry = registry.get(model_id)
        per_stem = any(f.stem for f in entry.weights)
        if stems and per_stem:
            files: dict[str, Path] = {}
            for stem in stems:
                files.update(
                    ensure_model(model_id, stem=stem, registry=registry, directory=directory)
                )
        else:
            files = ensure_model(model_id, registry=registry, directory=directory)
        return cls(entry, files, stems)

    def load(self, device: str) -> None:
        models = []
        for path in self._paths:
            package = read_package(path)
            model = build_model(package)
            if tuple(model.sources) != self._sources:
                raise DemucsConfigError(
                    f"{self.model_id}: checkpoint sources {model.sources} "
                    f"differ from the registry's {list(self._sources)}"
                )
            if round(float(model.segment) * model.samplerate) != self.chunk_size:
                raise DemucsConfigError(
                    f"{self.model_id}: checkpoint segment {model.segment} s "
                    f"differs from the registry's {self.chunk_size / self.sample_rate} s"
                )
            models.append(model.to(device))
        self._models = models
        self.device = device

    def prepare(self, audio: np.ndarray) -> tuple[np.ndarray, Any]:
        reference = audio.astype(np.float64).mean(axis=0)
        mean, std = float(reference.mean()), float(reference.std()) + 1e-8
        return ((audio - mean) / std).astype(np.float32), (mean, std)

    def finish(self, stems: np.ndarray, state: Any) -> np.ndarray:
        mean, std = state
        return (stems * std + mean).astype(np.float32)

    def separate_chunks(self, batch: np.ndarray) -> np.ndarray:
        if not self._models or self.device is None:
            raise RuntimeError(f"{self.model_id} is not loaded")
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
            outputs = []
            for model, pick in zip(self._models, self._picks, strict=True):
                y = model(x)  # (batch, sources, channels, samples)
                outputs.append(y if pick is None else y[:, pick : pick + 1])
            return torch.cat(outputs, dim=1).to(torch.float32).cpu().numpy()

    def unload(self) -> None:
        self._models = []
        super().unload()
