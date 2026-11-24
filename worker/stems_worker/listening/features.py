"""What the listening layer says about a mix and about each separated stem.

``build_track_features`` describes the uploaded mix: the ``track_features`` object that
the planning call sends to Jev (schema (b)). ``stem_signals`` describes one output stem:
its level statistics, whether its own instrument is heard in it, which other source bleeds
into it and how badly, as the ``metrics`` object of schema (c).

Both read a *tagger*: anything that turns audio into AudioSet's 527 class probabilities,
one row per window (BEATs, once its checkpoint is here). With no tagger everything that
needs hearing is reported as unknown, and the level statistics and loudness still hold, so
the pipeline runs without one. CLAP is optional and only adds to a stem's notes.

Every threshold comes from ``config/defaults.yaml`` (``listening``) and is uncalibrated
until measured (P7.1); the notes say so.

    python -m stems_worker.listening.features     # validate the class lists
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from stems_common import schemas
from stems_common.levels import SILENCE_DBFS, peak_dbfs, rms_dbfs, silence_fraction
from stems_common.settings import ListeningSettings, load_defaults
from stems_common.stems import PRODUCT_STEMS
from stems_worker.listening.audioset import (
    LISTENING_DIR,
    StemMap,
    StemMapError,
    default_stem_map,
)
from stems_worker.listening.loudness import integrated_lufs

FEATURE_CLASSES_PATH = LISTENING_DIR / "audioset_feature_classes.yaml"
SAMPLE_RATES = (8000, 16000, 22050, 32000, 44100, 48000, 96000)
GROUPS = ("strings", "synth", "brass")
MAX_NOTES = 1000


class Tagger(Protocol):
    """Audio in, AudioSet class probabilities out."""

    name: str
    window_seconds: float

    def class_probabilities(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """``(windows, 527)`` in AudioSet's class order, for consecutive windows."""
        ...


class PromptScorer(Protocol):
    """The part of ``ClapTagger`` used here."""

    def score_prompts(
        self, audio: np.ndarray, sample_rate: int, prompts: Sequence[str]
    ) -> dict[str, float]: ...

    def high_band(self, audio: np.ndarray, sample_rate: int) -> Any: ...


@dataclass(frozen=True)
class FeatureClasses:
    """Class groups beyond the six stems, as indices into AudioSet's label list."""

    version: int
    lead_vocals: tuple[int, ...]
    backing_vocals: tuple[int, ...]
    groups: dict[str, tuple[int, ...]]
    # The stem map's "other" without the named groups.
    other_rest: tuple[int, ...]
    genres: tuple[int, ...]
    reverb: tuple[int, ...]
    clap_prompts: dict[str, str]
    sha256: str


def load_feature_classes(
    path: Path = FEATURE_CLASSES_PATH, stem_map: StemMap | None = None
) -> FeatureClasses:
    """Read and check the class lists. Raises ``StemMapError`` on anything inconsistent."""
    stem_map = stem_map or default_stem_map()
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    index = stem_map.labels.index_of

    def indices(names: Sequence[str], what: str, within: str | None = None) -> tuple[int, ...]:
        if not names:
            raise StemMapError(f"{what} has no classes")
        if len(set(names)) != len(names):
            raise StemMapError(f"{what} lists a class twice")
        found = tuple(index(name) for name in names)
        if within is not None:
            outside = [
                n for n, i in zip(names, found, strict=True) if i not in stem_map.stems[within]
            ]
            if outside:
                raise StemMapError(f"{what}: {outside} are not in the stem map's {within!r}")
        return found

    backing = indices(data["backing_vocals"], "backing_vocals", within="vocals")
    lead = tuple(i for i in stem_map.stems["vocals"] if i not in backing)
    if not lead:
        raise StemMapError("backing_vocals leaves no lead vocal class")
    if set(data["instrument_groups"]) != set(GROUPS):
        raise StemMapError(f"instrument_groups must be exactly {GROUPS}")
    groups = {
        name: indices(data["instrument_groups"][name], f"instrument_groups.{name}", "other")
        for name in GROUPS
    }
    grouped = [i for members in groups.values() for i in members]
    if len(set(grouped)) != len(grouped):
        raise StemMapError("a class is in two instrument groups")
    prompts = dict(data["clap_prompts"])
    unknown = sorted(set(prompts) - set(PRODUCT_STEMS))
    if unknown:
        raise StemMapError(f"clap_prompts names unknown stems {unknown}")
    return FeatureClasses(
        version=int(data["version"]),
        lead_vocals=lead,
        backing_vocals=backing,
        groups=groups,
        other_rest=tuple(i for i in stem_map.stems["other"] if i not in grouped),
        genres=indices(data["genres"], "genres"),
        reverb=indices(data["reverb"], "reverb"),
        clap_prompts=prompts,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _as_channels(audio: np.ndarray) -> np.ndarray:
    channels = np.atleast_2d(np.asarray(audio))
    if channels.ndim != 2 or channels.shape[1] == 0:
        raise ValueError(f"expected non-empty (channels, samples) audio, got shape {audio.shape}")
    return channels


def audible_windows(
    audio: np.ndarray, sample_rate: int, window_seconds: float, count: int
) -> np.ndarray:
    """For each of ``count`` consecutive windows, whether it is above the silence level.

    A tagger asked about silence still answers; its answer for such a window is not used.
    """
    channels = _as_channels(audio)
    size = round(window_seconds * sample_rate)
    return np.array(
        [
            channels[:, i * size : (i + 1) * size].size > 0
            and rms_dbfs(channels[:, i * size : (i + 1) * size]) > SILENCE_DBFS
            for i in range(count)
        ],
        dtype=bool,
    )


@dataclass(frozen=True)
class Heard:
    """A tagger's answer for one piece of audio, reduced to scores on request."""

    tagger: str
    probabilities: np.ndarray  # (windows, classes)
    audible: np.ndarray  # (windows,) bool
    top_fraction: float

    def score(self, classes: Sequence[int]) -> float:
        """How clearly any of ``classes`` is heard, from 0 to 1.

        Per window, the largest probability among the classes; over the track, the mean of
        the best ``top_fraction`` of audible windows. 0 when no window is audible.
        """
        rows = self.probabilities[self.audible][:, list(classes)]
        if rows.size == 0:
            return 0.0
        per_window = np.sort(rows.max(axis=1))[::-1]
        keep = max(1, math.ceil(self.top_fraction * per_window.size))
        return float(per_window[:keep].mean())

    def stems(self, stem_map: StemMap) -> dict[str, float]:
        return {stem: self.score(stem_map.stems[stem]) for stem in PRODUCT_STEMS}


def hear(
    audio: np.ndarray,
    sample_rate: int,
    tagger: Tagger,
    settings: ListeningSettings,
    stem_map: StemMap,
) -> Heard:
    probabilities = np.asarray(tagger.class_probabilities(audio, sample_rate), dtype=np.float64)
    classes = len(stem_map.labels.names)
    if probabilities.ndim != 2 or probabilities.shape[1] != classes:
        raise ValueError(
            f"{tagger.name} returned shape {probabilities.shape}, expected (windows, {classes})"
        )
    audible = audible_windows(audio, sample_rate, tagger.window_seconds, probabilities.shape[0])
    return Heard(tagger.name, probabilities, audible, settings.presence.top_fraction)


def _settings(settings: ListeningSettings | None) -> ListeningSettings:
    return settings if settings is not None else load_defaults().listening


def build_track_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    tagger: Tagger | None = None,
    settings: ListeningSettings | None = None,
    source_sample_rate: int | None = None,
    source_channels: int | None = None,
    stem_map: StemMap | None = None,
    classes: FeatureClasses | None = None,
) -> dict[str, Any]:
    """The ``track_features`` object for a mix, valid against its schema.

    ``audio`` is the decoded mix at ``sample_rate``. ``source_sample_rate`` and
    ``source_channels`` describe the uploaded file when it differed (the pipeline works at
    44.1 kHz stereo); a source rate the schema cannot name is reported as the working rate,
    and more than two channels as two.
    """
    channels = _as_channels(audio)
    settings = _settings(settings)
    stem_map = stem_map or default_stem_map()
    classes = classes or load_feature_classes(stem_map=stem_map)
    rate = source_sample_rate if source_sample_rate in SAMPLE_RATES else sample_rate
    loudness = integrated_lufs(channels[:2], sample_rate)
    features: dict[str, Any] = {
        "duration_seconds": channels.shape[1] / sample_rate,
        "sample_rate_hz": int(rate),
        "channels": min(int(source_channels or channels.shape[0]), 2),
        "vocal_presence": "unknown",
        "detected_instruments": [],
        "genre_hint": None,
        "reverb_estimate": None,
        "loudness_lufs": None if loudness is None else round(min(max(loudness, -70.0), 0.0), 2),
    }
    if tagger is not None:
        heard = hear(channels, sample_rate, tagger, settings, stem_map)
        stems = heard.stems(stem_map)
        # An instrument playing in a full mix scores far lower than on its own stem, so a
        # mix has its own thresholds, per instrument, and only for instruments where they
        # were measured to work. Anything else is left unsaid rather than guessed.
        table = settings.mix_presence

        def there(name: str, score: float) -> bool:
            return name in table and score >= table[name].present_min

        vocals = table.get("vocals")
        if vocals is not None:
            lead = heard.score(classes.lead_vocals)
            if lead >= vocals.present_min:
                features["vocal_presence"] = "lead_vocals"
            elif (
                heard.score(classes.backing_vocals) >= vocals.present_min
                and lead <= vocals.absent_max
            ):
                features["vocal_presence"] = "backing_only"
            elif stems["vocals"] <= vocals.absent_max:
                features["vocal_presence"] = "none"
        found = [stem for stem in PRODUCT_STEMS if stem != "other" and there(stem, stems[stem])]
        found += [name for name in GROUPS if there(name, heard.score(classes.groups[name]))]
        # The whole "other" stem, groups included: the score the thresholds were measured on.
        if there("other", stems["other"]):
            found.append("other")
        features["detected_instruments"] = found
        genre = max(classes.genres, key=lambda i: heard.score([i]))
        if heard.score([genre]) >= settings.genre_min:
            features["genre_hint"] = stem_map.labels.names[genre]
        # Not hearing reverberation is not evidence of a dry recording, so "dry" is
        # never claimed here; below the first threshold the estimate stays unknown.
        reverb = heard.score(classes.reverb)
        if reverb >= settings.reverb.wet_min:
            features["reverb_estimate"] = "wet"
        elif reverb >= settings.reverb.moderate_min:
            features["reverb_estimate"] = "moderate"
    schemas.validate("track_features", features)
    return features


@dataclass(frozen=True)
class StemSignals:
    """Everything the listening layer knows about one output stem."""

    stem: str
    duration_seconds: float
    rms_dbfs: float
    peak_dbfs: float
    silence_fraction: float
    calibrated: bool
    # None when no tagger ran.
    tagger: str | None = None
    stem_scores: dict[str, float] | None = None
    # Whether the stem's own instrument is heard in it; None between the thresholds.
    present: bool | None = None
    bleed_source: str | None = None
    bleed_score: float | None = None
    bleed_level: str | None = None
    # CLAP's pick among the stem prompts (probabilities sum to 1), and its high-band
    # distance; None when CLAP did not run.
    prompt_scores: dict[str, float] | None = None
    high_band_distance: float | None = None

    @property
    def presence(self) -> float | None:
        return None if self.stem_scores is None else self.stem_scores[self.stem]

    @property
    def notes(self) -> str:
        """The signals as text for the gate, at most 1000 characters."""
        parts: list[str] = []
        if self.silence_fraction >= 1.0:
            parts.append("The stem is silent throughout.")
        elif self.stem_scores is None:
            parts.append("No tagger ran, so presence and bleed are unknown.")
        else:
            scores = ", ".join(f"{s} {self.stem_scores[s]:.2f}" for s in PRODUCT_STEMS)
            verdict = {True: "heard", False: "not heard", None: "uncertain"}[self.present]
            parts.append(f"{self.tagger} scores: {scores}.")
            parts.append(f"The stem's own source ({self.stem}) is {verdict}.")
            parts.append(
                f"Strongest other source: {self.bleed_source} at {self.bleed_score:.2f}, "
                f"bleed level {self.bleed_level}."
            )
        if self.prompt_scores:
            best = max(self.prompt_scores, key=self.prompt_scores.__getitem__)
            scores = ", ".join(f"{s} {p:.2f}" for s, p in self.prompt_scores.items())
            parts.append(f"CLAP prompt match: {scores}; best is {best}.")
        if self.high_band_distance is not None:
            parts.append(
                f"CLAP distance to the copy low-passed at 8 kHz: {self.high_band_distance:.4f} "
                "(near 0 means nothing audible above 8 kHz)."
            )
        if not self.calibrated and (self.stem_scores is not None or self.prompt_scores):
            parts.append("Thresholds are uncalibrated.")
        return " ".join(parts)[:MAX_NOTES]

    def metrics(
        self,
        *,
        sdr_db: float | None = None,
        sir_db: float | None = None,
        sar_db: float | None = None,
    ) -> dict[str, Any]:
        """The ``metrics`` object of schema (c). SDR, SIR and SAR need a reference stem,
        which user uploads never have, so they are null unless given."""
        bleed = None
        if self.bleed_level is not None:
            source = self.bleed_source if self.bleed_level != "none" else "none"
            bleed = {"suspected_source": source, "level": self.bleed_level}
        result = {
            "sdr_db": sdr_db,
            "sir_db": sir_db,
            "sar_db": sar_db,
            "duration_seconds": self.duration_seconds,
            # A float file can exceed full scale; the schema's ceiling is 0.
            "rms_dbfs": round(min(self.rms_dbfs, 0.0), 2),
            "peak_dbfs": round(min(self.peak_dbfs, 0.0), 2),
            "silence_fraction": self.silence_fraction,
            "bleed_estimate": bleed,
            "tagger_notes": self.notes,
        }
        schemas.validate("stem_metrics", result)
        return result


def stem_signals(
    stem: str,
    audio: np.ndarray,
    sample_rate: int,
    *,
    tagger: Tagger | None = None,
    clap: PromptScorer | None = None,
    settings: ListeningSettings | None = None,
    stem_map: StemMap | None = None,
    classes: FeatureClasses | None = None,
) -> StemSignals:
    """Listen to one separated stem. A silent stem is not sent to the taggers."""
    if stem not in PRODUCT_STEMS:
        raise ValueError(f"unknown stem {stem!r}")
    channels = _as_channels(audio)
    settings = _settings(settings)
    stem_map = stem_map or default_stem_map()
    levels: dict[str, Any] = {
        "stem": stem,
        "duration_seconds": channels.shape[1] / sample_rate,
        "rms_dbfs": rms_dbfs(channels),
        "peak_dbfs": peak_dbfs(channels),
        "silence_fraction": silence_fraction(channels, sample_rate),
        "calibrated": settings.calibrated.all,
    }
    if levels["silence_fraction"] >= 1.0:
        return StemSignals(**levels)
    heard: dict[str, Any] = {}
    if tagger is not None:
        scores = hear(channels, sample_rate, tagger, settings, stem_map).stems(stem_map)
        own = scores[stem]
        others = {s: v for s, v in scores.items() if s != stem}
        source = max(others, key=others.__getitem__)
        heard = {
            "tagger": tagger.name,
            "stem_scores": scores,
            "present": True
            if own >= settings.presence.present_min
            else False
            if own <= settings.presence.absent_max
            else None,
            "bleed_source": source,
            "bleed_score": others[source],
            "bleed_level": settings.bleed.level(others[source]),
        }
    if clap is not None:
        classes = classes or load_feature_classes(stem_map=stem_map)
        by_prompt = clap.score_prompts(channels, sample_rate, list(classes.clap_prompts.values()))
        heard["prompt_scores"] = {s: by_prompt[p] for s, p in classes.clap_prompts.items()}
        heard["high_band_distance"] = float(clap.high_band(channels, sample_rate).distance)
    return StemSignals(**levels, **heard)


def main() -> int:
    try:
        classes = load_feature_classes()
    except StemMapError as e:
        print(f"invalid class lists: {e}", file=sys.stderr)
        return 1
    print(f"feature classes version {classes.version}, sha256 {classes.sha256[:12]}")
    print(f"  vocals: {len(classes.lead_vocals)} lead, {len(classes.backing_vocals)} backing")
    for name in GROUPS:
        print(f"  {name}: {len(classes.groups[name])} classes")
    print(f"  other: {len(classes.other_rest)} classes outside the groups")
    print(f"  {len(classes.genres)} genres, {len(classes.reverb)} reverberation classes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
