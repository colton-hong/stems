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


class SlotModels(_Strict):
    single_best: str
    two_model_blend: list[str] = Field(min_length=2, max_length=2)
    full_ensemble: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def _check(self) -> SlotModels:
        for name in ("two_model_blend", "full_ensemble"):
            ids = getattr(self, name)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} lists a model twice: {ids}")
        return self

    def for_mode(self, mode: EnsembleMode) -> list[str]:
        if mode == "single_best":
            return [self.single_best]
        return list(getattr(self, mode))


class EnsembleSettings(_Strict):
    # True until the Phase 2 benchmark has chosen these from measured results.
    provisional: bool
    basis: str
    default_mode: EnsembleMode
    slots: dict[str, SlotModels]

    @model_validator(mode="after")
    def _check(self) -> EnsembleSettings:
        if set(self.slots) != set(ENSEMBLE_SLOTS):
            raise ValueError(
                f"slots must be exactly {sorted(ENSEMBLE_SLOTS)}, got {sorted(self.slots)}"
            )
        return self


class Defaults(_Strict):
    audio: AudioSettings
    uploads: UploadSettings
    retention: RetentionSettings
    decision: DecisionSettings
    gate: GateSettings
    ensembles: EnsembleSettings


def load_defaults(path: Path | str | None = None) -> Defaults:
    path = Path(path) if path is not None else DEFAULTS_PATH
    return Defaults.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
