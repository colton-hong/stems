"""Adapter for MDX-Net ONNX models from UVR and kuielab (``loader: uvr_mdx_onnx``).

An MDX-Net model works on spectrograms: it takes the real and imaginary parts of a stereo
STFT, cropped to its lowest ``dim_f`` bins and exactly ``dim_t`` frames, and returns the
same for one stem (its "primary stem"). The other stem of a two-stem model is the input
minus that.

The STFT settings are not stored in the ONNX file. UVR keeps them in a JSON table keyed by
a hash of the model file; the registry lists that table as a ``config_index`` file and this
adapter looks the model up in it the way UVR does.

The STFT, inverse STFT and "compensate" gain follow UVR. Chunking and overlap are ours
(``stems_worker.separator``), as for every model family; where UVR discards the edge of
each chunk, our crossfade gives it almost no weight.

onnxruntime runs these on the CPU here (or CUDA where that provider is installed). There
is no MPS provider, so asking for ``mps`` runs on the CPU and the result says ``cpu``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from stems_models.download import ensure_model
from stems_models.registry import ModelEntry, Registry, load_registry
from stems_worker.separator import Separator

HOP = 1024
_HASH_TAIL_BYTES = 10_000 * 1024


class MdxConfigError(ValueError):
    """A model is missing from the model data, or does not fit its registry entry."""


@dataclass(frozen=True)
class MdxParams:
    n_fft: int
    dim_f: int
    dim_t: int
    compensate: float
    primary_stem: str

    @property
    def chunk_size(self) -> int:
        return HOP * (self.dim_t - 1)


def uvr_model_hash(path: Path | str) -> str:
    """MD5 of the last 10,240,000 bytes of the file (the whole file if it is shorter).

    This is the key UVR's model data tables use.
    """
    with Path(path).open("rb") as f:
        size = f.seek(0, 2)
        f.seek(max(0, size - _HASH_TAIL_BYTES))
        return hashlib.md5(f.read(), usedforsecurity=False).hexdigest()


def read_mdx_params(weights: Path | str, model_data: Path | str) -> MdxParams:
    table = json.loads(Path(model_data).read_text(encoding="utf-8"))
    key = uvr_model_hash(weights)
    try:
        row = table[key]
        params = MdxParams(
            n_fft=int(row["mdx_n_fft_scale_set"]),
            dim_f=int(row["mdx_dim_f_set"]),
            dim_t=2 ** int(row["mdx_dim_t_set"]),
            compensate=float(row["compensate"]),
            primary_stem=str(row["primary_stem"]),
        )
    except KeyError as e:
        raise MdxConfigError(
            f"{Path(weights).name} (hash {key}) has no usable entry in {Path(model_data).name}: "
            f"missing {e.args[0]!r}"
        ) from None
    if params.n_fft <= 0 or params.n_fft % 2 or not 0 < params.dim_f <= params.n_fft // 2 + 1:
        raise MdxConfigError(f"{Path(weights).name}: implausible STFT settings {params}")
    if not 0.5 < params.compensate < 2.0:
        raise MdxConfigError(f"{Path(weights).name}: implausible compensate {params.compensate}")
    return params


def resolve_mdx_outputs(
    registry_stems: list[str], primary_stem: str, names: dict[str, str] | None, stem: str | None
) -> tuple[tuple[str, ...], bool]:
    """Outputs in registry vocabulary, and whether the second one is input minus primary."""
    primary = (names or {}).get(primary_stem, primary_stem.lower())
    if stem is not None:  # one file per stem: the model emits exactly that stem
        if primary != stem:
            raise MdxConfigError(f"file for {stem!r} has primary stem {primary_stem!r}")
        return (stem,), False
    if primary not in registry_stems:
        raise MdxConfigError(f"primary stem {primary_stem!r} is not among {registry_stems}")
    rest = [s for s in registry_stems if s != primary]
    if len(rest) != 1:
        raise MdxConfigError(f"a two-stem MDX model needs two registry stems, got {registry_stems}")
    return (primary, rest[0]), True


def spectrogram(x: torch.Tensor, params: MdxParams, window: torch.Tensor) -> torch.Tensor:
    """``(batch, channels, samples)`` to ``(batch, channels * 2, dim_f, frames)``."""
    batch, channels, samples = x.shape
    spec = torch.stft(
        x.reshape(-1, samples), params.n_fft, HOP, window=window, center=True, return_complex=True
    )
    spec = torch.view_as_real(spec).permute(0, 3, 1, 2)  # (batch*channels, 2, bins, frames)
    spec = spec.reshape(batch, channels * 2, spec.shape[-2], spec.shape[-1])
    return spec[:, :, : params.dim_f]


def waveform(spec: torch.Tensor, params: MdxParams, window: torch.Tensor) -> torch.Tensor:
    """Inverse of ``spectrogram``; bins above ``dim_f`` are taken as zero."""
    batch, doubled, _, frames = spec.shape
    bins = params.n_fft // 2 + 1
    padded = torch.zeros(batch, doubled, bins, frames, dtype=spec.dtype)
    padded[:, :, : params.dim_f] = spec
    pairs = padded.reshape(batch * doubled // 2, 2, bins, frames).permute(0, 2, 3, 1)
    complex_spec = torch.complex(pairs[..., 0], pairs[..., 1])
    audio = torch.istft(complex_spec, params.n_fft, HOP, window=window, center=True)
    return audio.reshape(batch, doubled // 2, -1)


class MdxSeparator(Separator):
    def __init__(
        self,
        entry: ModelEntry,
        files: dict[str, Path],
        stem: str | None = None,
        *,
        denoise: bool = False,
    ) -> None:
        if entry.loader != "uvr_mdx_onnx":
            raise MdxConfigError(f"{entry.id} uses loader {entry.loader!r}, not 'uvr_mdx_onnx'")
        per_stem = {f.stem: f.filename for f in entry.weights if f.stem is not None}
        if per_stem:
            # Each stem's model has its own STFT size and chunk length, so one separator
            # handles one stem.
            if stem is None:
                raise MdxConfigError(
                    f"{entry.id} has one model per stem; pick one of {list(per_stem)}"
                )
            if stem not in per_stem:
                raise MdxConfigError(f"{entry.id} has no model for {stem!r}")
            weights_name = per_stem[stem]
        else:
            if stem is not None:
                raise MdxConfigError(f"{entry.id} is a single model; it takes no stem argument")
            weights_name = entry.weights[0].filename
        index_name = entry.config.get("params_from")
        missing = [n for n in (weights_name, index_name) if n not in files]
        if missing:
            raise MdxConfigError(f"{entry.id}: missing files {missing}")

        self.model_id = entry.id
        self.sample_rate = int(entry.sample_rate_hz or 44100)
        self._weights_path = files[weights_name]
        self.params = read_mdx_params(self._weights_path, files[index_name])
        self.chunk_size = self.params.chunk_size
        self.outputs, self._add_complement = resolve_mdx_outputs(
            list(entry.stems), self.params.primary_stem, entry.config.get("instrument_names"), stem
        )
        # UVR's "denoise" runs the model on the input and on its negation and averages,
        # which cancels a constant bias in the model's output at twice the cost.
        self.denoise = denoise
        self._session: ort.InferenceSession | None = None
        self._window = torch.hann_window(self.params.n_fft, periodic=True)

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        *,
        stem: str | None = None,
        denoise: bool = False,
        registry: Registry | None = None,
        directory: Path | None = None,
    ) -> MdxSeparator:
        registry = registry or load_registry()
        files = ensure_model(model_id, stem=stem, registry=registry, directory=directory)
        return cls(registry.get(model_id), files, stem, denoise=denoise)

    def load(self, device: str) -> None:
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        session = ort.InferenceSession(str(self._weights_path), providers=providers)
        model_input = session.get_inputs()[0]
        expected = [4, self.params.dim_f, self.params.dim_t]
        if list(model_input.shape[1:]) != expected:
            raise MdxConfigError(
                f"{self.model_id}: the ONNX input is {model_input.shape} but the model data "
                f"implies (batch, {', '.join(map(str, expected))})"
            )
        self._session = session
        self._input_name = model_input.name
        # Fixed batch size in the graph means chunks are fed one at a time.
        self._fixed_batch = model_input.shape[0] if isinstance(model_input.shape[0], int) else None
        self.device = "cuda" if session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"

    def _run(self, spec: np.ndarray) -> np.ndarray:
        assert self._session is not None
        step = self._fixed_batch or len(spec)
        parts = [
            self._session.run(None, {self._input_name: spec[i : i + step]})[0]
            for i in range(0, len(spec), step)
        ]
        return np.concatenate(parts, axis=0)

    def separate_chunks(self, batch: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError(f"{self.model_id} is not loaded")
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
        spec = spectrogram(x, self.params, self._window).numpy()
        predicted = self._run(spec)
        if self.denoise:
            predicted = 0.5 * (predicted - self._run(-spec))
        primary = waveform(torch.from_numpy(predicted), self.params, self._window)
        primary = primary * self.params.compensate
        stems = [primary]
        if self._add_complement:
            stems.append(x - primary)
        return torch.stack(stems, dim=1).to(torch.float32).numpy()

    def unload(self) -> None:
        self._session = None
        super().unload()
