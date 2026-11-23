"""LAION-CLAP (music checkpoint): audio and text in one embedding space, at 48 kHz.

Two uses here. Text prompts ("a recording of an acoustic guitar") are scored against
audio without any training. And because the model hears up to 14 kHz, it can tell whether
anything is going on above 8 kHz, which the 16 kHz taggers cannot: ``high_band`` compares a
clip with the same clip low-passed at 8 kHz and reports how far apart the two land.

The model code comes from the pinned ``laion_clap`` wheel (``stems_models.vendor``); the
checkpoint is loaded with ``weights_only=True``. The text tower is RoBERTa-base: its
weights are inside the checkpoint, so it is built from its config file and nothing is
fetched from a model hub at run time.

Audio in is float, ``(channels, samples)`` or mono, at any sample rate.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stems_models.download import ensure_model
from stems_models.registry import ModelEntry, Registry, load_registry
from stems_models.vendor import ensure_clap_lib
from stems_worker.audio import resample
from stems_worker.device import release_device_memory

SAMPLE_RATE = 48000
WINDOW_SECONDS = 10
WINDOW = SAMPLE_RATE * WINDOW_SECONDS
MAX_TOKENS = 77
HIGH_BAND_HZ = 8000.0
MODEL_ID = "laion_clap_music"


class ClapError(RuntimeError):
    pass


def to_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Float32 mono at 48 kHz."""
    stereo = audio if audio.ndim == 2 else audio[None]
    mono = stereo.astype(np.float32).mean(axis=0, keepdims=True)
    if sample_rate != SAMPLE_RATE:
        mono = resample(mono, sample_rate, SAMPLE_RATE)
    return np.ascontiguousarray(mono[0])


def quantise(audio: np.ndarray) -> np.ndarray:
    """Through 16-bit and back, as the model's own pipeline does before embedding."""
    as_int = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return (as_int / 32767.0).astype(np.float32)


def windows(mono: np.ndarray) -> np.ndarray:
    """Consecutive 10-second windows, ``(n, 480000)``.

    A last window shorter than 10 seconds is filled by repeating it and then padding with
    silence, which is how the model was trained on short clips. A tail shorter than one
    second after at least one full window is dropped.
    """
    if mono.ndim != 1 or mono.size == 0:
        raise ClapError(f"expected non-empty mono audio, got shape {mono.shape}")
    count, rest = divmod(mono.size, WINDOW)
    parts = [mono[i * WINDOW : (i + 1) * WINDOW] for i in range(count)]
    if rest and (count == 0 or rest >= SAMPLE_RATE):
        tail = mono[count * WINDOW :]
        repeated = np.tile(tail, WINDOW // tail.size)
        parts.append(np.pad(repeated, (0, WINDOW - repeated.size)))
    return np.stack(parts).astype(np.float32)


def low_pass(mono: np.ndarray, cutoff_hz: float = HIGH_BAND_HZ) -> np.ndarray:
    """Everything at and above ``cutoff_hz`` removed (FFT brick wall with a short ramp)."""
    spectrum = np.fft.rfft(mono.astype(np.float64))
    freqs = np.fft.rfftfreq(mono.size, 1.0 / SAMPLE_RATE)
    ramp = np.clip((cutoff_hz - freqs) / 200.0, 0.0, 1.0)  # 200 Hz roll-off below the cutoff
    return np.fft.irfft(spectrum * ramp, n=mono.size).astype(np.float32)


@dataclass(frozen=True)
class HighBand:
    """What the model makes of a clip's content above 8 kHz.

    ``distance`` is one minus the cosine similarity between the clip's embedding and that
    of its low-passed copy, averaged over windows: near 0 when nothing audible is up
    there, larger the more the top of the spectrum changes what the model hears.
    """

    distance: float
    per_window: tuple[float, ...]
    cutoff_hz: float = HIGH_BAND_HZ


def _clap_module() -> Any:
    root = str(ensure_clap_lib())
    if root not in sys.path:
        sys.path.append(root)
    from laion_clap.clap_module import model as clap_model

    return clap_model


def read_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """The model's state dict, read without unpickling anything but tensors and numbers."""
    from numpy._core.multiarray import scalar

    # The training loop saved its epoch metrics as numpy floats next to the weights.
    safe = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(safe):
        package = torch.load(str(path), map_location="cpu", weights_only=True)
    state = package.get("state_dict") if isinstance(package, dict) else None
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ClapError(f"{path.name}: no plain state_dict inside")
    return {key.removeprefix("module."): value for key, value in state.items()}


class ClapTagger:
    def __init__(self, entry: ModelEntry, files: dict[str, Path]) -> None:
        self.model_id = entry.id
        self.entry = entry
        self._files = files
        self._model: Any = None
        self._tokenizer: Any = None
        self.device: str | None = None

    @classmethod
    def from_registry(
        cls, model_id: str = MODEL_ID, *, registry: Registry | None = None
    ) -> ClapTagger:
        entry = (registry or load_registry()).get(model_id)
        if entry.loader != "laion_clap":
            raise ClapError(f"{model_id} is not a LAION-CLAP model")
        return cls(entry, ensure_model(model_id))

    def _file(self, role: str, suffix: str) -> Path:
        for f in self.entry.files:
            if f.role == role and f.filename.endswith(suffix):
                return self._files[f.filename]
        raise ClapError(f"{self.model_id}: no {role} file ending in {suffix}")

    def load(self, device: str = "cpu") -> None:
        from tokenizers import Tokenizer
        from transformers import RobertaConfig, RobertaModel

        clap_model = _clap_module()
        config_path = Path(clap_model.__file__).parent / "model_configs"
        config = json.loads((config_path / f"{self.entry.config['amodel']}.json").read_text())
        config["text_cfg"]["model_type"] = "roberta"
        roberta = json.loads(self._file("config", "config.json").read_text())

        class _LocalRoberta:
            """Stands in for the hub download: same architecture, weights from the checkpoint."""

            @staticmethod
            def from_pretrained(name: str) -> Any:
                if name != "roberta-base":
                    raise ClapError(f"unexpected text model {name!r}")
                return RobertaModel(RobertaConfig(**roberta))

        original = clap_model.RobertaModel
        clap_model.RobertaModel = _LocalRoberta
        try:
            model = clap_model.CLAP(
                **config,
                enable_fusion=bool(self.entry.config.get("enable_fusion", False)),
                fusion_type="None",
            )
        finally:
            clap_model.RobertaModel = original

        state = read_checkpoint(self._file("weights", ".pt"))
        expected = set(model.state_dict())
        # Older transformers stored the position index table in checkpoints; it is a
        # constant and newer versions rebuild it.
        state = {k: v for k, v in state.items() if k in expected or not k.endswith("position_ids")}
        model.load_state_dict(state, strict=True)
        self._model = model.eval().to(device)
        tokenizer = Tokenizer.from_file(str(self._file("config", "tokenizer.json")))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(
            length=MAX_TOKENS, pad_id=roberta["pad_token_id"], pad_token="<pad>"
        )
        self._tokenizer = tokenizer
        self.device = device

    def unload(self) -> None:
        device = self.device
        self._model = None
        self._tokenizer = None
        self.device = None
        if device is not None:
            release_device_memory(device)

    def _require(self) -> Any:
        if self._model is None:
            raise ClapError(f"{self.model_id} is not loaded; call load(device) first")
        return self._model

    def _embed_windows(self, batch: np.ndarray) -> np.ndarray:
        model = self._require()
        out = []
        with torch.no_grad():
            for start in range(0, len(batch), 4):
                chunk = torch.from_numpy(quantise(batch[start : start + 4]))
                data = [{"waveform": w} for w in chunk]
                out.append(model.get_audio_embedding(data).float().cpu().numpy())
        return np.concatenate(out)

    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """One unit-length embedding per 10-second window: ``(windows, 512)``."""
        return self._embed_windows(windows(to_mono(audio, sample_rate)))

    def embed_text(self, prompts: Sequence[str]) -> np.ndarray:
        """One unit-length embedding per prompt: ``(prompts, 512)``."""
        model = self._require()
        if not prompts:
            raise ClapError("no prompts given")
        encoded = self._tokenizer.encode_batch(list(prompts))
        data = {
            "input_ids": torch.tensor([e.ids for e in encoded], dtype=torch.long),
            "attention_mask": torch.tensor([e.attention_mask for e in encoded], dtype=torch.long),
        }
        with torch.no_grad():
            return model.get_text_embedding(data).float().cpu().numpy()

    def similarity(self, audio: np.ndarray, sample_rate: int, prompts: Sequence[str]) -> np.ndarray:
        """Cosine similarity of each window with each prompt: ``(windows, prompts)``."""
        return self.embed_audio(audio, sample_rate) @ self.embed_text(prompts).T

    def score_prompts(
        self, audio: np.ndarray, sample_rate: int, prompts: Sequence[str]
    ) -> dict[str, float]:
        """Which of ``prompts`` fits the clip best: probabilities that sum to 1.

        Similarities are averaged over windows, scaled by the model's own temperature and
        passed through a softmax, as in zero-shot classification.
        """
        model = self._require()
        mean = self.similarity(audio, sample_rate, prompts).mean(axis=0)
        logits = float(model.logit_scale_a.detach().exp()) * mean
        exp = np.exp(logits - logits.max())
        return {p: float(v) for p, v in zip(prompts, exp / exp.sum(), strict=True)}

    def high_band(self, audio: np.ndarray, sample_rate: int) -> HighBand:
        mono = to_mono(audio, sample_rate)
        full = self._embed_windows(windows(mono))
        cut = self._embed_windows(windows(low_pass(mono)))
        per_window = tuple(float(1.0 - np.dot(a, b)) for a, b in zip(full, cut, strict=True))
        return HighBand(distance=float(np.mean(per_window)), per_window=per_window)
