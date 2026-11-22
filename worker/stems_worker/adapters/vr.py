"""Adapter for UVR's VR-architecture models (``loader: uvr_vr``): karaoke, de-echo, de-noise.

A VR model does not see audio directly. The audio is split into frequency bands, each
resampled to its own rate and turned into a spectrogram; the bands are stacked into one
tall magnitude image; the network predicts a mask over windows of that image; the masked
and the inverse-masked images are turned back into two waveforms (the model's "primary"
stem and everything else).

The band layout comes from a small JSON file named in UVR's model data, which is looked up
by UVR's hash of the weights file. Which network is built depends on the weights' size and
on two optional fields in the model data, as in UVR.

The network definitions and the multi-band spectrogram routines are UVR's, fetched at a
pinned version by ``stems_models.vendor.ensure_uvr_lib``. The weights are read with
``weights_only=True``. Each chunk handed over by ``stems_worker.separator`` is processed as
a short track of its own, and the chunk edges are crossfaded like every other model's.

Band resampling always uses scipy's polyphase filter. UVR picks a different resampler per
operating system; one fixed choice keeps results the same on every machine.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from stems_models.download import ensure_model
from stems_models.registry import ModelEntry, Registry, load_registry
from stems_models.vendor import ensure_uvr_lib
from stems_worker.adapters.mdx import uvr_model_hash
from stems_worker.separator import Separator

# Network sizes UVR recognises, in kilobytes of weights file.
_ARCH_SIZES = (31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227)
_NEW_NET_SIZES = (56817, 218409)
# Primary stems for which UVR flips the sense of its "aggression" adjustment.
_NON_ACCOMPANIMENT = (
    "Vocals", "Other", "Bass", "Drums", "Guitar", "Piano",
    "Synthesizer", "Strings", "Woodwinds", "Brass", "Wind Inst",
)  # fmt: skip
_RESAMPLER = "polyphase"

_lib: SimpleNamespace | None = None


class VrConfigError(ValueError):
    """A model is missing from the model data, or does not fit its registry entry."""


def uvr_lib() -> SimpleNamespace:
    """Import the vendored UVR modules once."""
    global _lib
    if _lib is None:
        root = str(ensure_uvr_lib())
        if root not in sys.path:
            sys.path.append(root)
        import librosa
        from audio_separator.separator.uvr_lib_v5 import spec_utils
        from audio_separator.separator.uvr_lib_v5.vr_network import nets, nets_new
        from audio_separator.separator.uvr_lib_v5.vr_network.model_param_init import ModelParameters

        spec_utils.wav_resolution = _RESAMPLER
        params_dir = Path(nets.__file__).parent / "modelparams"
        _lib = SimpleNamespace(
            librosa=librosa,
            spec_utils=spec_utils,
            nets=nets,
            nets_new=nets_new,
            ModelParameters=ModelParameters,
            params_dir=params_dir,
        )
    return _lib


@dataclass(frozen=True)
class VrParams:
    band_params: str  # name of the band-layout JSON, without extension
    primary_stem: str
    # Present only for the newer network family.
    nout: int | None
    nout_lstm: int | None


def read_vr_params(weights: Path | str, model_data: Path | str) -> VrParams:
    table = json.loads(Path(model_data).read_text(encoding="utf-8"))
    key = uvr_model_hash(weights)
    try:
        row = table[key]
        params = VrParams(
            band_params=str(row["vr_model_param"]),
            primary_stem=str(row["primary_stem"]),
            nout=int(row["nout"]) if "nout" in row else None,
            nout_lstm=int(row["nout_lstm"]) if "nout_lstm" in row else None,
        )
    except KeyError as e:
        raise VrConfigError(
            f"{Path(weights).name} (hash {key}) has no usable entry in {Path(model_data).name}: "
            f"missing {e.args[0]!r}"
        ) from None
    if not params.band_params.replace("_", "").isalnum():
        raise VrConfigError(f"{Path(weights).name}: odd band layout name {params.band_params!r}")
    return params


def arch_size_for(weights: Path | str) -> int:
    """The network size UVR would pick for this file: the nearest known size in KB."""
    kilobytes = math.ceil(Path(weights).stat().st_size / 1024)
    return min(_ARCH_SIZES, key=lambda size: abs(size - kilobytes))


class VrSeparator(Separator):
    def __init__(
        self,
        entry: ModelEntry,
        files: dict[str, Path],
        *,
        aggression: int = 5,
        window_size: int = 512,
        chunk_seconds: float = 20.0,
        inner_batch_size: int = 4,
    ) -> None:
        if entry.loader != "uvr_vr":
            raise VrConfigError(f"{entry.id} uses loader {entry.loader!r}, not 'uvr_vr'")
        weights_name = entry.weights[0].filename
        index_name = entry.config.get("params_from")
        missing = [n for n in (weights_name, index_name) if n not in files]
        if missing:
            raise VrConfigError(f"{entry.id}: missing files {missing}")

        self.model_id = entry.id
        self.sample_rate = int(entry.sample_rate_hz or 44100)
        self.chunk_size = round(chunk_seconds * self.sample_rate)
        self._weights_path = files[weights_name]
        self.params = read_vr_params(self._weights_path, files[index_name])

        expected = entry.config.get("primary_stem")
        if expected != self.params.primary_stem:
            raise VrConfigError(
                f"{entry.id}: the registry expects primary stem {expected!r} "
                f"but UVR's model data says {self.params.primary_stem!r}"
            )
        # Registry names for the model's two outputs; None means "not used".
        wanted = [entry.config.get("primary"), entry.config.get("secondary")]
        self._picks = [i for i, name in enumerate(wanted) if name is not None]
        self.outputs = tuple(wanted[i] for i in self._picks)
        if set(self.outputs) != set(entry.stems) or not self.outputs:
            raise VrConfigError(
                f"{entry.id}: config names outputs {self.outputs} but the registry "
                f"lists stems {list(entry.stems)}"
            )

        # UVR's "aggression" (0 to 100) sharpens or softens the predicted mask.
        self._aggression = aggression / 100.0
        self._window = window_size
        self._inner_batch = inner_batch_size
        self._model: torch.nn.Module | None = None
        self._bands: Any = None
        self._new_net = False

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        *,
        registry: Registry | None = None,
        directory: Path | None = None,
        **options: Any,
    ) -> VrSeparator:
        registry = registry or load_registry()
        files = ensure_model(model_id, registry=registry, directory=directory)
        return cls(registry.get(model_id), files, **options)

    def load(self, device: str) -> None:
        lib = uvr_lib()
        layout = lib.params_dir / f"{self.params.band_params}.json"
        if not layout.is_file():
            raise VrConfigError(f"{self.model_id}: no band layout {layout.name} in the UVR code")
        bands = lib.ModelParameters(str(layout))
        if bands.param["sr"] != self.sample_rate:
            raise VrConfigError(f"{self.model_id}: band layout is for {bands.param['sr']} Hz")

        size = arch_size_for(self._weights_path)
        self._new_net = size in _NEW_NET_SIZES or self.params.nout is not None
        if self._new_net:
            nout, nout_lstm = self.params.nout or 32, self.params.nout_lstm or 128
            model = lib.nets_new.CascadedNet(
                bands.param["bins"] * 2, size, nout=nout, nout_lstm=nout_lstm
            )
        else:
            model = lib.nets.determine_model_capacity(bands.param["bins"] * 2, size)
        state = torch.load(str(self._weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self._model = model.eval().to(device)
        self._bands = bands
        self.device = device

    # --- one chunk ------------------------------------------------------------------------

    def _combined_spectrogram(self, wave: np.ndarray) -> np.ndarray:
        lib, bands = uvr_lib(), self._bands
        count = len(bands.param["band"])
        waves: dict[int, np.ndarray] = {}
        specs: dict[int, np.ndarray] = {}
        for d in range(count, 0, -1):
            band = bands.param["band"][d]
            source, source_rate = (
                (wave, self.sample_rate)
                if d == count
                else (
                    waves[d + 1],
                    bands.param["band"][d + 1]["sr"],
                )
            )
            if band["sr"] == source_rate:
                waves[d] = source
            else:
                waves[d] = lib.librosa.resample(
                    source, orig_sr=source_rate, target_sr=band["sr"], res_type=_RESAMPLER
                )
            specs[d] = lib.spec_utils.wave_to_spectrogram(
                waves[d], band["hl"], band["n_fft"], bands, band=d, is_v51_model=self._new_net
            )
        return lib.spec_utils.combine_spectrograms(specs, bands, is_v51_model=self._new_net)

    def _predict_mask(self, magnitude: np.ndarray) -> np.ndarray:
        assert self._model is not None
        lib, frames = uvr_lib(), magnitude.shape[2]
        offset = self._model.offset
        pad_left, pad_right, step = lib.spec_utils.make_padding(frames, self._window, offset)
        padded = np.pad(magnitude, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")
        padded = padded / padded.max()
        patches = (padded.shape[2] - 2 * offset) // step
        windows = np.stack(
            [padded[:, :, i * step : i * step + self._window] for i in range(patches)]
        ).astype(np.float32)
        parts = []
        with torch.inference_mode():
            for start in range(0, patches, self._inner_batch):
                batch = torch.from_numpy(windows[start : start + self._inner_batch])
                predicted = self._model.predict_mask(batch.to(self.device)).cpu().numpy()
                parts.append(np.concatenate(list(predicted), axis=2))
        return np.concatenate(parts, axis=2)[:, :, :frames]

    def _separate_one(self, wave: np.ndarray) -> np.ndarray:
        """``(2, n)`` in, ``(2, 2, n)`` out: primary stem, then everything else."""
        lib, n = uvr_lib(), wave.shape[1]
        if not wave.any():
            return np.zeros((2, 2, n), dtype=np.float32)
        spectrogram = self._combined_spectrogram(np.asfortranarray(wave))
        magnitude, phase = lib.spec_utils.preprocess(spectrogram)
        mask = self._predict_mask(magnitude)
        mask = lib.spec_utils.adjust_aggr(
            mask,
            self.params.primary_stem in _NON_ACCOMPANIMENT,
            {
                "value": self._aggression,
                "split_bin": self._bands.param["band"][1]["crop_stop"],
                "aggr_correction": self._bands.param.get("aggr_correction"),
            },
        )
        rotation = np.exp(1.0j * phase)
        stems = []
        for part in (mask, 1.0 - mask):
            spec = np.nan_to_num(part * magnitude * rotation, nan=0.0, posinf=0.0, neginf=0.0)
            audio = lib.spec_utils.cmb_spectrogram_to_wave(
                spec, self._bands, is_v51_model=self._new_net
            )
            audio = np.asarray(audio, dtype=np.float32)[:, :n]
            if audio.shape[1] < n:
                audio = np.pad(audio, ((0, 0), (0, n - audio.shape[1])))
            stems.append(audio)
        return np.stack(stems)

    def separate_chunks(self, batch: np.ndarray) -> np.ndarray:
        if self._model is None or self.device is None:
            raise RuntimeError(f"{self.model_id} is not loaded")
        both = np.stack([self._separate_one(item) for item in batch.astype(np.float32)])
        return np.ascontiguousarray(both[:, self._picks])

    def unload(self) -> None:
        self._model = None
        super().unload()
