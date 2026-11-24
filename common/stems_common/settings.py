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
QualityTarget = Literal["fastest", "balanced", "best_quality"]
QUALITY_TARGETS: tuple[str, ...] = ("fastest", "balanced", "best_quality")

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
    # Audio outside these is refused: decoded size grows with both.
    min_sample_rate_hz: int = Field(gt=0)
    max_sample_rate_hz: int = Field(gt=0)
    max_channels: int = Field(ge=1)
    # How long the check of an upload (header and first second) may take.
    inspect_timeout_seconds: float = Field(gt=0)


class RetentionSettings(_Strict):
    days: int = Field(gt=0)
    signed_url_ttl_seconds: int = Field(gt=0)


class DecisionSettings(_Strict):
    jev_model: str = Field(pattern=r"^jev-\d+\.\d+\.\d+$")  # a pinned version, never an alias
    min_confidence: float = Field(ge=0, le=1)
    # A stem is wanted when Jev's probability that the request asks for it is at least this.
    wanted_threshold: float = Field(ge=0.5, le=1)
    request_timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    # Wait before the first retry; it doubles each time. A reply's retry-after replaces it.
    backoff_seconds: float = Field(gt=0)
    # A retry-after longer than this ends the call and the fallback answers instead.
    max_retry_wait_seconds: float = Field(gt=0)
    # The ensemble mode planned for each quality target when Jev does not decide it.
    mode_for_quality: dict[QualityTarget, EnsembleMode]

    @model_validator(mode="after")
    def _check(self) -> DecisionSettings:
        if set(self.mode_for_quality) != set(QUALITY_TARGETS):
            raise ValueError(f"mode_for_quality needs exactly {sorted(QUALITY_TARGETS)}")
        return self


BleedLevel = Literal["none", "slight", "moderate", "severe"]


class GateRules(_Strict):
    """The rule-based scorer that assesses a stem when Jev does not. Uncalibrated (P7.1)."""

    # A stem at least this silent is "mostly_silent" and scores 0.
    silent_fraction_min: float = Field(ge=0, le=1)
    # Score on the 0-4 rubric for each bleed level the tagger estimated.
    bleed_scores: dict[BleedLevel, float]
    # Ceiling on the score when the stem's own source is not heard in it.
    not_heard_score: float = Field(ge=0, le=4)
    # Score when nothing listened to the stem, and ceiling when the tagger could not tell
    # whether the stem's own source is there.
    unchecked_score: float = Field(ge=0, le=4)
    # Confidence of a verdict that rests on tagger thresholds not yet calibrated.
    uncalibrated_confidence: float = Field(ge=0, le=1)
    # A stem this far below the mix, or further, over the whole song is treated like a
    # silent one: the instrument is probably not in the song. None switches it off.
    empty_below_mix_db: float | None = Field(le=0)

    @model_validator(mode="after")
    def _check(self) -> GateRules:
        levels = ("none", "slight", "moderate", "severe")
        if set(self.bleed_scores) != set(levels):
            raise ValueError(f"bleed_scores needs exactly {list(levels)}")
        scores = [self.bleed_scores[level] for level in levels]
        if scores != sorted(scores, reverse=True) or not 0 <= scores[-1] <= scores[0] <= 4:
            raise ValueError("bleed_scores must fall from none to severe, within 0 to 4")
        return self


class GateSettings(_Strict):
    pass_probability_min: float = Field(ge=0, le=1)
    score_min: float = Field(ge=0, le=4)
    confidence_min: float = Field(ge=0, le=1)
    max_retries_per_stem: int = Field(ge=0, le=2)
    rules: GateRules


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


# What track_features may name in a mix (the schema's detected_instruments).
MIX_INSTRUMENTS = (
    "vocals", "drums", "bass", "guitar", "piano", "strings", "synth", "brass", "other"
)  # fmt: skip


class MixThresholds(_Strict):
    """One instrument's presence thresholds in a full mix, where scores run far lower."""

    present_min: float = Field(ge=0, le=1)
    absent_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> MixThresholds:
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


class Calibrated(_Strict):
    """Which of the tagger's signals have been measured to work, one switch each.

    The gate acts on a verdict only if every signal it rests on is switched on.
    """

    # Whether a stem holds its own instrument (listening.presence).
    presence: bool
    # How much of another instrument is in it (listening.bleed).
    bleed: bool

    @property
    def all(self) -> bool:
        return self.presence and self.bleed

    @classmethod
    def both(cls, value: bool) -> Calibrated:
        return cls(presence=value, bleed=value)


class ListeningSettings(_Strict):
    # Whether the worker loads the tagger and listens to the mix and to every stem.
    enabled: bool
    # Per signal: true only once it passed its held-out check on real separator output.
    calibrated: Calibrated
    presence: PresenceSettings
    # Per instrument, the thresholds for hearing it in a full mix. Only instruments whose
    # thresholds held up on held-out mixes are listed; any other is never claimed.
    mix_presence: dict[str, MixThresholds]
    bleed: BleedSettings
    genre_min: float = Field(ge=0, le=1)
    reverb: ReverbSettings

    @model_validator(mode="after")
    def _check_mix(self) -> ListeningSettings:
        unknown = sorted(set(self.mix_presence) - set(MIX_INSTRUMENTS))
        if unknown:
            raise ValueError(f"mix_presence names must be among {MIX_INSTRUMENTS}, got {unknown}")
        return self


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


class ApiSettings(_Strict):
    # How often the API reads a job's status.json while the job is in progress.
    poll_interval_seconds: float = Field(gt=0)
    # How often the retention sweeper runs (it also runs at startup).
    sweep_interval_seconds: float = Field(gt=0)
    # How long a browser keeps its owner cookie.
    owner_cookie_days: int = Field(gt=0, le=400)


class RateLimitSettings(_Strict):
    uploads_per_owner_per_hour: int = Field(ge=0)
    uploads_per_ip_per_hour: int = Field(ge=0)
    parses_per_owner_per_minute: int = Field(ge=0)
    parses_per_ip_per_minute: int = Field(ge=0)
    # Jobs one browser may have queued or running at the same time.
    max_active_jobs_per_owner: int = Field(ge=1)
    # Jobs queued or running for everyone together; beyond it uploads are turned away.
    max_jobs_in_progress: int = Field(ge=1)


class Defaults(_Strict):
    audio: AudioSettings
    separation: SeparationSettings
    uploads: UploadSettings
    retention: RetentionSettings
    api: ApiSettings
    rate_limits: RateLimitSettings
    decision: DecisionSettings
    gate: GateSettings
    listening: ListeningSettings
    ensembles: EnsembleSettings


def load_defaults(path: Path | str | None = None) -> Defaults:
    path = Path(path) if path is not None else DEFAULTS_PATH
    return Defaults.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
