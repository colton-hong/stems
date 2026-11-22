"""Adapter for checkpoints trained with Music-Source-Separation-Training (``loader: msst``).

Upstream supplies the network definitions (fetched at a pinned commit by
``stems_models.vendor``). Everything around them is done here, deliberately not with
upstream's helpers:

- configs are parsed with a safe YAML loader that accepts the one non-standard tag these
  files use (``!!python/tuple``) and nothing else, where upstream uses ``FullLoader``;
- weights are read with ``torch.load(..., weights_only=True)``, where upstream unpickles
  arbitrary objects;
- chunking and overlap are ours (``stems_worker.separator``), so every model family is
  run the same way.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from stems_models.download import ensure_model
from stems_models.registry import ModelEntry, Registry, load_registry
from stems_models.vendor import ensure_msst
from stems_worker.separator import Separator

# model_type -> (upstream module, class name). Classes are built as cls(**config["model"]).
# P1.9 extends this to the remaining families.
_KWARGS_MODELS: dict[str, tuple[str, str]] = {
    "mel_band_roformer": ("models.bs_roformer.mel_band_roformer", "MelBandRoformer"),
    "bs_roformer": ("models.bs_roformer.bs_roformer", "BSRoformer"),
}


class ModelConfigError(ValueError):
    """A model's YAML config is missing something or does not fit its registry entry."""


# --- config -------------------------------------------------------------------------------


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader plus tuples. Any other Python tag is still refused."""


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
)


def load_model_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.load(f, Loader=_ConfigLoader)  # a SafeLoader subclass
    if not isinstance(config, dict):
        raise ModelConfigError(f"{path}: expected a mapping at the top level")
    for section in ("audio", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ModelConfigError(f"{path}: missing section {section!r}")
    return config


def resolve_outputs(
    registry_stems: list[str], instruments: list[str], target: str | None
) -> tuple[tuple[str, ...], bool]:
    """Work out what the adapter emits, in registry vocabulary.

    Returns ``(outputs, add_complement)``. A model trained with ``target_instrument``
    emits only that stem; when the registry lists two outputs the second is the rest of
    the mix (input minus target), which the adapter computes.

    The name "other" in a two-stem config means "everything else", which the registry
    calls ``instrumental``.
    """

    def canonical(name: str) -> str:
        if name == "other" and "other" not in registry_stems and "instrumental" in registry_stems:
            return "instrumental"
        return name.lower()

    if target is not None:
        primary = canonical(target)
        if primary not in registry_stems:
            raise ModelConfigError(
                f"config targets {target!r} but the registry lists {registry_stems}"
            )
        rest = [s for s in registry_stems if s != primary]
        if not rest:
            return (primary,), False
        if len(rest) == 1:
            return (primary, rest[0]), True
        raise ModelConfigError(
            f"a single-target model can emit at most two outputs, registry lists {registry_stems}"
        )

    names = tuple(canonical(i) for i in instruments)
    if set(names) != set(registry_stems) or len(names) != len(registry_stems):
        raise ModelConfigError(
            f"config instruments {instruments} do not match registry stems {registry_stems}"
        )
    return names, False


# --- weights ------------------------------------------------------------------------------


def read_state_dict(path: Path | str) -> dict[str, torch.Tensor]:
    """Load a checkpoint's tensors without unpickling arbitrary objects."""
    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    for wrapper in ("state", "state_dict", "model"):
        if isinstance(loaded, dict) and isinstance(loaded.get(wrapper), dict):
            loaded = loaded[wrapper]
    if not isinstance(loaded, dict) or not loaded:
        raise ModelConfigError(f"{path}: not a state dict")
    if not all(isinstance(v, torch.Tensor) for v in loaded.values()):
        raise ModelConfigError(f"{path}: state dict contains non-tensor values")
    return loaded


def _upstream_class(model_type: str) -> type[torch.nn.Module]:
    try:
        module_name, class_name = _KWARGS_MODELS[model_type]
    except KeyError:
        raise ModelConfigError(f"model_type {model_type!r} is not supported yet") from None
    root = str(ensure_msst())
    if root not in sys.path:
        # Upstream modules import each other as ``models.<family>...``, so its root has to
        # be importable. It is added once, at the end, so it never shadows our packages.
        sys.path.append(root)
    return getattr(importlib.import_module(module_name), class_name)


# --- the adapter --------------------------------------------------------------------------


class MsstSeparator(Separator):
    def __init__(self, entry: ModelEntry, files: dict[str, Path]) -> None:
        if entry.loader != "msst":
            raise ModelConfigError(f"{entry.id} uses loader {entry.loader!r}, not 'msst'")
        model_type = entry.config.get("model_type")
        if not model_type:
            raise ModelConfigError(f"{entry.id}: registry config has no model_type")
        config_name = entry.config.get("config_file")
        weights = [f for f in entry.weights if f.filename in files]
        if config_name not in files or len(weights) != 1:
            raise ModelConfigError(f"{entry.id}: need its config and exactly one weights file")

        self.model_id = entry.id
        self._model_type = model_type
        self._weights_path = files[weights[0].filename]
        self._config = load_model_config(files[config_name])
        audio, training = self._config["audio"], self._config["training"]
        self.chunk_size = int(audio["chunk_size"])
        self.sample_rate = int(audio.get("sample_rate", 44100))
        self.outputs, self._add_complement = resolve_outputs(
            list(entry.stems),
            [str(i) for i in training.get("instruments", [])],
            training.get("target_instrument"),
        )
        self._model: torch.nn.Module | None = None

    @classmethod
    def from_registry(
        cls, model_id: str, *, registry: Registry | None = None, directory: Path | None = None
    ) -> MsstSeparator:
        """Build from a registry id, downloading and verifying its files if needed."""
        registry = registry or load_registry()
        files = ensure_model(model_id, registry=registry, directory=directory)
        return cls(registry.get(model_id), files)

    def load(self, device: str) -> None:
        model = _upstream_class(self._model_type)(**self._config["model"])
        model.load_state_dict(read_state_dict(self._weights_path), strict=True)
        self._model = model.eval().to(device)
        self.device = device

    def separate_chunks(self, batch: np.ndarray) -> np.ndarray:
        if self._model is None or self.device is None:
            raise RuntimeError(f"{self.model_id} is not loaded")
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
            y = self._model(x)
            if y.ndim == 3:  # single-stem models drop the stem axis
                y = y[:, None]
            if self._add_complement:
                y = torch.cat([y, x[:, None] - y], dim=1)
            return y.to(torch.float32).cpu().numpy()

    def unload(self) -> None:
        self._model = None
        super().unload()
