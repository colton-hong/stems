"""BEATs (iter3+, fine-tuned on AudioSet-2M): 527 AudioSet class probabilities per window.

This is the tagger the listening layer is built around: ``class_probabilities`` returns one
row per 10-second window, in AudioSet's own class order, which is what
``stems_worker.listening.features`` and the class-to-stem table expect.

The model hears 16 kHz mono, so nothing above 8 kHz; CLAP covers that range. The model
code is Microsoft's, fetched at a pinned commit (``stems_models.vendor``); the checkpoint
is loaded with ``weights_only=True``. The checkpoint lists its classes in its own order,
by AudioSet id, and the output is rearranged to AudioSet's order by those ids.

    python -m stems_worker.listening.beats FILE     # the strongest classes of an audio file

Audio in is float, ``(channels, samples)`` or mono, at any sample rate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stems_models.download import ensure_model
from stems_models.registry import ModelEntry, Registry, load_registry
from stems_models.vendor import BEATS_PACKAGE, ensure_beats_lib
from stems_worker.audio import resample
from stems_worker.device import release_device_memory
from stems_worker.listening.audioset import AudioSetLabels, default_stem_map

SAMPLE_RATE = 16000
WINDOW_SECONDS = 10
WINDOW = SAMPLE_RATE * WINDOW_SECONDS
# A tail shorter than this after at least one full window is not scored on its own.
MIN_TAIL = SAMPLE_RATE
MODEL_ID = "beats_iter3plus_as2m_cpt2"


class BeatsError(RuntimeError):
    pass


def to_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Float32 mono at 16 kHz."""
    stereo = audio if audio.ndim == 2 else audio[None]
    mono = stereo.astype(np.float32).mean(axis=0, keepdims=True)
    if sample_rate != SAMPLE_RATE:
        mono = resample(mono, sample_rate, SAMPLE_RATE)
    return np.ascontiguousarray(mono[0])


def windows(mono: np.ndarray) -> list[np.ndarray]:
    """Consecutive 10-second windows; the last may be shorter.

    A tail of under a second after at least one full window is dropped. The model takes
    any length, so a short window is scored as it is, with no padding or repetition.
    """
    if mono.ndim != 1 or mono.size == 0:
        raise BeatsError(f"expected non-empty mono audio, got shape {mono.shape}")
    count, rest = divmod(mono.size, WINDOW)
    parts = [mono[i * WINDOW : (i + 1) * WINDOW] for i in range(count)]
    if rest and (count == 0 or rest >= MIN_TAIL):
        parts.append(mono[count * WINDOW :])
    return parts


def audioset_order(label_dict: dict[int, str], labels: AudioSetLabels) -> np.ndarray:
    """For each AudioSet class, in AudioSet's order, its position in the model's output."""
    position = {mid: index for index, mid in label_dict.items()}
    if len(position) != len(label_dict) or set(position) != set(labels.mids):
        raise BeatsError("the checkpoint's classes are not AudioSet's 527")
    return np.array([position[mid] for mid in labels.mids], dtype=np.int64)


def _beats_module() -> Any:
    root = str(ensure_beats_lib())
    if root not in sys.path:
        sys.path.append(root)
    import importlib

    return importlib.import_module(f"{BEATS_PACKAGE}.BEATs")


def read_checkpoint(path: Path) -> dict[str, Any]:
    """``cfg``, ``model`` and ``label_dict``, read without unpickling any code."""
    package = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or not {"cfg", "model", "label_dict"} <= set(package):
        raise BeatsError(f"{path.name}: not a fine-tuned BEATs checkpoint")
    return package


class BeatsTagger:
    name = MODEL_ID
    window_seconds = float(WINDOW_SECONDS)

    def __init__(self, entry: ModelEntry, files: dict[str, Path]) -> None:
        self.model_id = entry.id
        self.entry = entry
        self._files = files
        self._model: Any = None
        self._order: np.ndarray | None = None
        self.device: str | None = None

    @classmethod
    def from_registry(
        cls, model_id: str = MODEL_ID, *, registry: Registry | None = None
    ) -> BeatsTagger:
        entry = (registry or load_registry()).get(model_id)
        if entry.loader != "beats":
            raise BeatsError(f"{model_id} is not a BEATs model")
        return cls(entry, ensure_model(model_id))

    def load(self, device: str = "cpu") -> None:
        module = _beats_module()
        (weights,) = (self._files[f.filename] for f in self.entry.files if f.role == "weights")
        package = read_checkpoint(weights)
        if package["cfg"].get("predictor_class") != self.entry.config["num_classes"]:
            raise BeatsError(f"{weights.name}: expected {self.entry.config['num_classes']} classes")
        model = module.BEATs(module.BEATsConfig(package["cfg"]))
        model.load_state_dict(package["model"], strict=True)
        self._order = audioset_order(package["label_dict"], default_stem_map().labels)
        self._model = model.eval().to(device)
        self.device = device

    def unload(self) -> None:
        device = self.device
        self._model = None
        self.device = None
        if device is not None:
            release_device_memory(device)

    def class_probabilities(
        self, audio: np.ndarray, sample_rate: int, *, batch_size: int = 8
    ) -> np.ndarray:
        """``(windows, 527)`` probabilities in AudioSet's class order."""
        if self._model is None or self._order is None:
            raise BeatsError(f"{self.model_id} is not loaded; call load(device) first")
        parts = windows(to_mono(audio, sample_rate))
        full = [p for p in parts if p.size == WINDOW]
        rows: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(full), batch_size):
                batch = torch.from_numpy(np.stack(full[start : start + batch_size]))
                rows.append(self._score(batch))
            if len(full) < len(parts):
                rows.append(self._score(torch.from_numpy(parts[-1][None])))
        return np.concatenate(rows)[:, self._order]

    def _score(self, batch: torch.Tensor) -> np.ndarray:
        probabilities, _ = self._model.extract_features(batch.to(self.device), padding_mask=None)
        return probabilities.float().cpu().numpy().astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    from stems_worker.audio import read_audio

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m stems_worker.listening.beats FILE", file=sys.stderr)
        return 2
    audio, rate = read_audio(Path(args[0]))
    tagger = BeatsTagger.from_registry()
    tagger.load("cpu")
    probabilities = tagger.class_probabilities(audio, rate)
    stem_map = default_stem_map()
    mean = probabilities.mean(axis=0)
    print(f"{len(probabilities)} windows; strongest classes on average:")
    for index in np.argsort(mean)[::-1][:10]:
        print(f"  {mean[index]:.3f}  {stem_map.labels.names[index]}")
    print("stem scores:", {s: round(v, 3) for s, v in stem_map.stem_scores(mean).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
