"""Typed view of config/defaults.yaml.

Environment-specific settings (storage backend, keys, provider) are added in Phase 5 and
read from environment variables; this file only holds the tunable defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from stems_common.paths import DEFAULTS_PATH

EnsembleMode = Literal["single_best", "two_model_blend", "full_ensemble"]
# How the members of a blend are combined (stems_worker.ensemble).
BlendMethod = Literal["waveform", "stft_magnitude", "stft_min", "stft_max"]
OutputFormat = Literal["wav", "flac", "mp3_320", "mp3_192"]

ENSEMBLE_SLOTS: tuple[str, ...] = (
    "vocals",
    "drums",
    "bass",
    "guitar",
    "piano",
    "other",
    "vocal_split",
    "dereverb",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioSettings(_Strict):
    sample_rate_hz: int = Field(gt=0)
    channels: Literal[1, 2]
    output_format: OutputFormat


class SeparationSettings(_Strict):
    overlap: float = Field(ge=0, lt=1)
    batch_size: int = Field(ge=1)
    other_method: Literal["auto", "residual"]
    residual_warn_db: float = Field(le=0)
    silent_stem_db: float = Field(le=0)


class UploadSettings(_Strict):
    max_bytes: int = Field(gt=0)
    max_duration_seconds: int = Field(gt=0, le=3600)
    allowed_extensions: list[str] = Field(min_length=1)


class RetentionSettings(_Strict):
    days: int = Field(gt=0)
    signed_url_ttl_seconds: int = Field(gt=0)


class DecisionSettings(_Strict):
    jev_model: str = Field(pattern=r"^jev-\d+\.\d+\.\d+$")  # a pinned version, never an alias
    min_confidence: float = Field(ge=0, le=1)
    request_timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=1)


class GateSettings(_Strict):
    pass_probability_min: float = Field(ge=0, le=1)
    score_min: float = Field(ge=0, le=4)
    confidence_min: float = Field(ge=0, le=1)
    max_retries_per_stem: int = Field(ge=0, le=2)


class PresenceSettings(_Strict):
    # A score is the mean of a class group's best `top_fraction` of audible windows.
    top_fraction: float = Field(gt=0, le=1)
    # At or above: present. At or below `absent_max`: absent. Between: unknown.
    present_min: float = Field(ge=0, le=1)
    absent_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> PresenceSettings:
        if self.absent_max >= self.present_min:
            raise ValueError("absent_max must be below present_min")
        return self


class BleedSettings(_Strict):
    # Tagger score of the strongest other source heard on a stem.
    slight_min: float = Field(ge=0, le=1)
    moderate_min: float = Field(ge=0, le=1)
    severe_min: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> BleedSettings:
        if not self.slight_min < self.moderate_min < self.severe_min:
            raise ValueError("bleed thresholds must rise: slight < moderate < severe")
        return self

    def level(self, score: float) -> Literal["none", "slight", "moderate", "severe"]:
        if score >= self.severe_min:
            return "severe"
        if score >= self.moderate_min:
            return "moderate"
        return "slight" if score >= self.slight_min else "none"


class ReverbSettings(_Strict):
    moderate_min: float = Field(ge=0, le=1)
    wet_min: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> ReverbSettings:
        if self.moderate_min >= self.wet_min:
            raise ValueError("moderate_min must be below wet_min")
        return self


class ListeningSettings(_Strict):
    # False until the thresholds below have been measured on labelled audio (P7.1).
    calibrated: bool
    presence: PresenceSettings
    bleed: BleedSettings
    genre_min: float = Field(ge=0, le=1)
    reverb: ReverbSettings


class SlotModels(_Strict):
    single_best: str
    two_model_blend: list[str] = Field(min_length=2, max_length=2)
    full_ensemble: list[str] = Field(min_length=2)
    # None: use ensembles.blend_method.
    blend_method: BlendMethod | None = None
    # Per-model weights, in the order the models are listed. None: equal weights.
    two_model_weights: list[float] | None = None
    full_ensemble_weights: list[float] | None = None

    @model_validator(mode="after")
    def _check(self) -> SlotModels:
        for name in ("two_model_blend", "full_ensemble"):
            ids = getattr(self, name)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} lists a model twice: {ids}")
        for name, ids in (
            ("two_model_weights", self.two_model_blend),
            ("full_ensemble_weights", self.full_ensemble),
        ):
            weights = getattr(self, name)
            if weights is None:
                continue
            if len(weights) != len(ids):
                raise ValueError(f"{name} needs one weight per model ({len(ids)})")
            if min(weights) < 0 or sum(weights) <= 0:
                raise ValueError(f"{name} must be non-negative and not all zero")
        return self

    def for_mode(self, mode: EnsembleMode) -> list[str]:
        if mode == "single_best":
            return [self.single_best]
        return list(getattr(self, mode))

    def weights_for(self, mode: EnsembleMode) -> list[float]:
        """One weight per model of ``mode``, summing to 1."""
        if mode == "single_best":
            return [1.0]
        key = "two_model_weights" if mode == "two_model_blend" else "full_ensemble_weights"
        raw = getattr(self, key) or [1.0] * len(self.for_mode(mode))
        total = float(sum(raw))
        return [float(w) / total for w in raw]


class EnsembleSettings(_Strict):
    # True until the Phase 2 benchmark has chosen these from measured results.
    provisional: bool
    basis: str
    default_mode: EnsembleMode
    # How blend members are combined unless a slot says otherwise.
    blend_method: BlendMethod = "waveform"
    slots: dict[str, SlotModels]

    def method_for(self, slot: str) -> BlendMethod:
        return self.slots[slot].blend_method or self.blend_method

    @model_validator(mode="after")
    def _check(self) -> EnsembleSettings:
        if set(self.slots) != set(ENSEMBLE_SLOTS):
            raise ValueError(
                f"slots must be exactly {sorted(ENSEMBLE_SLOTS)}, got {sorted(self.slots)}"
            )
        return self


class Defaults(_Strict):
    audio: AudioSettings
    separation: SeparationSettings
    uploads: UploadSettings
    retention: RetentionSettings
    decision: DecisionSettings
    gate: GateSettings
    listening: ListeningSettings
    ensembles: EnsembleSettings


def load_defaults(path: Path | str | None = None) -> Defaults:
    path = Path(path) if path is not None else DEFAULTS_PATH
    return Defaults.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
