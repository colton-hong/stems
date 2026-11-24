"""track_features for a mix and listening signals for a stem, with a scripted tagger."""

from __future__ import annotations

import numpy as np
import pytest

from stems_common import schemas
from stems_common.levels import peak_dbfs, rms_dbfs, silence_fraction
from stems_common.settings import MIX_INSTRUMENTS, MixThresholds, load_defaults
from stems_common.synthetic import generate_track
from stems_worker.listening import features
from stems_worker.listening.audioset import StemMapError, default_stem_map
from stems_worker.listening.features import (
    build_track_features,
    load_feature_classes,
    stem_signals,
)

SR = 44100
_SHIPPED = load_defaults()
# The rules under test are the same whatever the thresholds are. The presence thresholds
# are fixed here so that re-calibrating config/defaults.yaml does not change what the
# scripted scores below mean.
# In a mix an instrument is named only if it has thresholds of its own; here every
# instrument has them, so that the rules for naming can be tested.
SETTINGS = _SHIPPED.listening.model_copy(
    update={
        "presence": _SHIPPED.listening.presence.model_copy(
            update={"present_min": 0.5, "absent_max": 0.2}
        ),
        "mix_presence": {
            name: MixThresholds(present_min=0.5, absent_max=0.2) for name in MIX_INSTRUMENTS
        },
    }
)


@pytest.fixture(autouse=True)
def fixed_thresholds(monkeypatch):
    pinned = _SHIPPED.model_copy(update={"listening": SETTINGS})
    monkeypatch.setattr(features, "load_defaults", lambda: pinned)


class ScriptedTagger:
    """Answers with the class probabilities it was given, one dict per 10-second window."""

    name = "scripted"
    window_seconds = 10.0

    def __init__(self, *windows: dict[str, float]) -> None:
        self.windows = windows
        self.calls = 0

    def class_probabilities(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        names = default_stem_map().labels.names
        out = np.zeros((len(self.windows), len(names)))
        for row, window in zip(out, self.windows, strict=True):
            for name, p in window.items():
                row[names.index(name)] = p
        return out


def noise(seconds: float, channels: int = 2, level: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (level * rng.standard_normal((channels, round(seconds * SR)))).astype(np.float32)


@pytest.fixture(scope="module")
def track():
    return generate_track(seed=11, duration_seconds=6.0)


def test_without_a_tagger_the_features_are_valid_and_say_unknown(track):
    result = build_track_features(track.mix, track.sample_rate)
    assert schemas.errors("track_features", result) == []
    assert result["vocal_presence"] == "unknown"
    assert result["detected_instruments"] == []
    assert result["genre_hint"] is None
    assert result["reverb_estimate"] is None
    assert result["duration_seconds"] == pytest.approx(6.0)
    assert (result["sample_rate_hz"], result["channels"]) == (44100, 2)
    assert -70 <= result["loudness_lufs"] <= 0


def test_a_full_band_is_described():
    tagger = ScriptedTagger(
        {"Singing": 0.9, "Drum kit": 0.8, "Bass guitar": 0.7, "Synthesizer": 0.6,
         "Trumpet": 0.55, "Rock music": 0.5, "Pop music": 0.4, "Reverberation": 0.3},
        {"Singing": 0.9, "Drum kit": 0.8, "Electric guitar": 0.1, "Accordion": 0.95},
    )  # fmt: skip
    result = build_track_features(noise(20), SR, tagger=tagger)
    assert schemas.errors("track_features", result) == []
    assert result["vocal_presence"] == "lead_vocals"
    # Schema order; guitar at 0.1 and piano at 0 are not detected.
    assert result["detected_instruments"] == ["vocals", "drums", "bass", "synth", "brass", "other"]
    assert result["genre_hint"] == "Rock music"
    assert result["reverb_estimate"] == "moderate"


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ({"Female singing": 0.6}, "lead_vocals"),
        ({"Choir": 0.8}, "backing_only"),
        ({"Choir": 0.8, "Singing": 0.9}, "lead_vocals"),
        # A choir with a possible soloist: not sure enough to call it backing only.
        ({"Choir": 0.8, "Singing": 0.35}, "unknown"),
        ({"Singing": 0.35}, "unknown"),
        ({"Singing": 0.1, "Drum kit": 0.9}, "none"),
        ({}, "none"),
    ],
)
def test_vocal_presence(heard, expected):
    result = build_track_features(noise(10), SR, tagger=ScriptedTagger(heard))
    assert result["vocal_presence"] == expected


def test_an_instrument_without_mix_thresholds_is_never_named():
    loud = {"Singing": 0.95, "Drum kit": 0.9, "Piano": 0.9, "Synthesizer": 0.9, "Rock music": 0.8}
    only_piano = SETTINGS.model_copy(
        update={"mix_presence": {"piano": MixThresholds(present_min=0.3, absent_max=0.05)}}
    )
    result = build_track_features(noise(10), SR, tagger=ScriptedTagger(loud), settings=only_piano)
    assert result["detected_instruments"] == ["piano"]
    assert result["vocal_presence"] == "unknown"  # nothing was measured for vocals in a mix
    assert result["genre_hint"] == "Rock music"
    # Each instrument is judged by its own thresholds, not by the ones for stems.
    faint = build_track_features(
        noise(10), SR, tagger=ScriptedTagger({"Piano": 0.35}), settings=only_piano
    )
    assert faint["detected_instruments"] == ["piano"]


def test_as_shipped_only_instruments_whose_mix_thresholds_held_up_are_named():
    shipped = _SHIPPED.listening
    loud = {"Singing": 0.95, "Drum kit": 0.9, "Bass guitar": 0.9, "Electric guitar": 0.9,
            "Piano": 0.9, "Synthesizer": 0.9, "Trumpet": 0.9, "Violin, fiddle": 0.9}  # fmt: skip
    result = build_track_features(noise(10), SR, tagger=ScriptedTagger(loud), settings=shipped)
    assert set(result["detected_instruments"]) <= set(shipped.mix_presence)
    if "vocals" not in shipped.mix_presence:
        assert result["vocal_presence"] == "unknown"
    assert schemas.errors("track_features", result) == []


def test_reverb_is_never_called_dry_and_weak_genres_give_no_hint():
    quiet = build_track_features(
        noise(10), SR, tagger=ScriptedTagger({"Reverberation": 0.1, "Jazz": 0.2})
    )
    assert quiet["reverb_estimate"] is None
    assert quiet["genre_hint"] is None
    wet = build_track_features(noise(10), SR, tagger=ScriptedTagger({"Echo": 0.7, "Jazz": 0.3}))
    assert wet["reverb_estimate"] == "wet"
    assert wet["genre_hint"] == "Jazz"


def test_an_instrument_in_a_quarter_of_the_track_counts_and_in_an_eighth_does_not():
    solo, rest = {"Piano": 0.9}, {}
    quarter = build_track_features(noise(40), SR, tagger=ScriptedTagger(solo, rest, rest, rest))
    assert quarter["detected_instruments"] == ["piano"]
    eighth = build_track_features(noise(80), SR, tagger=ScriptedTagger(solo, *[rest] * 7))
    assert eighth["detected_instruments"] == []  # mean of the best two windows is 0.45


def test_what_a_tagger_says_about_silent_windows_is_ignored():
    audio = np.concatenate([noise(10), np.zeros((2, 10 * SR), dtype=np.float32)], axis=1)
    tagger = ScriptedTagger({"Drum kit": 0.9}, {"Singing": 0.99, "Piano": 0.99})
    result = build_track_features(audio, SR, tagger=tagger)
    assert result["detected_instruments"] == ["drums"]
    assert result["vocal_presence"] == "none"
    mask = features.audible_windows(audio, SR, 10.0, 2)
    assert mask.tolist() == [True, False]


def test_source_format_is_reported_when_the_schema_can_name_it():
    audio = noise(5, channels=2)
    assert build_track_features(audio, SR, source_sample_rate=48000)["sample_rate_hz"] == 48000
    # 88.2 kHz is not in the schema's list: the working rate is reported instead.
    assert build_track_features(audio, SR, source_sample_rate=88200)["sample_rate_hz"] == 44100
    assert build_track_features(audio, SR, source_channels=6)["channels"] == 2
    assert build_track_features(audio[0], SR)["channels"] == 1


def test_silent_and_empty_mixes():
    silent = build_track_features(np.zeros((2, 5 * SR), dtype=np.float32), SR)
    assert silent["loudness_lufs"] is None
    assert schemas.errors("track_features", silent) == []
    with pytest.raises(ValueError, match="non-empty"):
        build_track_features(np.zeros((2, 0), dtype=np.float32), SR)
    with pytest.raises(schemas.SchemaError, match="duration_seconds"):
        build_track_features(np.zeros((1, 8000 * 3601), dtype=np.float32), 8000)


def test_a_tagger_with_the_wrong_number_of_classes_is_refused():
    class Short(ScriptedTagger):
        def class_probabilities(self, audio, sample_rate):
            return np.zeros((1, 10))

    with pytest.raises(ValueError, match=r"expected \(windows, 527\)"):
        build_track_features(noise(10), SR, tagger=Short({}))


def test_stem_levels_match_the_level_functions_and_the_metrics_are_valid(track):
    for stem, audio in track.stems.items():
        signals = stem_signals(stem, audio, track.sample_rate)
        assert signals.rms_dbfs == rms_dbfs(audio)
        assert signals.peak_dbfs == peak_dbfs(audio)
        assert signals.silence_fraction == silence_fraction(audio, track.sample_rate)
        metrics = signals.metrics()
        assert schemas.errors("stem_metrics", metrics) == []
        # No tagger: no claim about bleed; no reference: no SDR.
        assert metrics["bleed_estimate"] is None
        assert metrics["sdr_db"] is None
        assert "No tagger ran" in metrics["tagger_notes"]
        assert signals.present is None
        assert signals.presence is None


@pytest.mark.parametrize(
    ("bleed", "level", "source"),
    [(0.05, "none", "none"), (0.2, "slight", "drums"), (0.6, "moderate", "drums"),
     (0.85, "severe", "drums")],
)  # fmt: skip
def test_bleed_is_the_strongest_other_source(bleed, level, source):
    tagger = ScriptedTagger({"Singing": 0.9, "Snare drum": bleed, "Piano": bleed / 2})
    signals = stem_signals("vocals", noise(10), SR, tagger=tagger)
    assert signals.present is True
    assert signals.presence == pytest.approx(0.9)
    assert signals.bleed_source == "drums"
    assert signals.bleed_score == pytest.approx(bleed)
    metrics = signals.metrics(sdr_db=7.5)
    assert metrics["bleed_estimate"] == {"suspected_source": source, "level": level}
    assert metrics["sdr_db"] == 7.5
    assert schemas.errors("stem_metrics", metrics) == []
    assert "scripted scores: vocals 0.90" in metrics["tagger_notes"]
    assert "Thresholds are uncalibrated." in metrics["tagger_notes"]


def test_a_stem_whose_own_source_is_missing_says_so():
    tagger = ScriptedTagger({"Electric guitar": 0.9, "Piano": 0.1})
    wrong = stem_signals("piano", noise(10), SR, tagger=tagger)
    assert wrong.present is False
    assert wrong.metrics()["bleed_estimate"] == {"suspected_source": "guitar", "level": "severe"}
    assert "(piano) is not heard" in wrong.notes
    unsure = stem_signals("piano", noise(10), SR, tagger=ScriptedTagger({"Piano": 0.35}))
    assert unsure.present is None
    assert "is uncertain" in unsure.notes


def test_a_silent_stem_is_not_sent_to_the_taggers():
    tagger = ScriptedTagger({"Singing": 0.99})

    class NoClap:
        def score_prompts(self, *args):
            raise AssertionError("CLAP was asked about silence")

        high_band = score_prompts

    signals = stem_signals(
        "vocals", np.zeros((2, 3 * SR), dtype=np.float32), SR, tagger=tagger, clap=NoClap()
    )
    assert tagger.calls == 0
    assert signals.silence_fraction == 1.0
    assert signals.present is None
    metrics = signals.metrics()
    assert metrics["bleed_estimate"] is None
    assert metrics["tagger_notes"] == "The stem is silent throughout."
    assert metrics["rms_dbfs"] == -120.0


def test_clap_adds_a_second_opinion_to_the_notes():
    class FakeClap:
        def score_prompts(self, audio, sample_rate, prompts):
            assert len(prompts) == 5
            return {p: (0.8 if "drums" in p else 0.05) for p in prompts}

        def high_band(self, audio, sample_rate):
            return type("HighBand", (), {"distance": 0.0321})()

    signals = stem_signals("drums", noise(10), SR, clap=FakeClap())
    assert signals.prompt_scores["drums"] == 0.8
    assert set(signals.prompt_scores) == {"vocals", "drums", "bass", "guitar", "piano"}
    assert signals.high_band_distance == pytest.approx(0.0321)
    notes = signals.metrics()["tagger_notes"]
    assert "best is drums" in notes
    assert "0.0321" in notes
    assert "No tagger ran" in notes


def test_levels_above_full_scale_and_long_notes_still_fit_the_schema():
    loud = stem_signals("bass", noise(2, level=2.0), SR)
    assert loud.peak_dbfs > 0
    metrics = loud.metrics()
    assert metrics["peak_dbfs"] == 0.0
    assert metrics["rms_dbfs"] == 0.0
    long_name = ScriptedTagger({"Bass guitar": 0.9})
    long_name.name = "x" * 2000
    notes = stem_signals("bass", noise(10), SR, tagger=long_name).metrics()["tagger_notes"]
    assert len(notes) == 1000
    with pytest.raises(ValueError, match="unknown stem"):
        stem_signals("kazoo", noise(1), SR)


def test_the_committed_class_lists_are_consistent():
    stem_map = default_stem_map()
    classes = load_feature_classes()
    names = stem_map.labels.names
    assert {names[i] for i in classes.backing_vocals} == {"Choir", "Chant", "Humming"}
    assert set(classes.lead_vocals) | set(classes.backing_vocals) == set(stem_map.stems["vocals"])
    grouped = {i for members in classes.groups.values() for i in members}
    assert grouped | set(classes.other_rest) == set(stem_map.stems["other"])
    assert not grouped & set(classes.other_rest)
    assert "other" not in classes.clap_prompts
    assert features.main() == 0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  - Humming\n", "  - Kazoo\n", "not an AudioSet class"),
        ("  - Humming\n", "  - Piano\n", "not in the stem map's 'vocals'"),
        ("    - Sampler\n", "    - Trumpet\n", "two instrument groups"),
        ("    - Sampler\n", "    - Synthesizer\n", "lists a class twice"),
        ("  brass:\n", "  horns:\n", "instrument_groups must be exactly"),
        ("  piano: the sound of a piano\n", "  kazoo: a kazoo\n", "unknown stems"),
    ],
)
def test_inconsistent_class_lists_are_refused(tmp_path, old, new, message):
    text = features.FEATURE_CLASSES_PATH.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "classes.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(StemMapError, match=message):
        load_feature_classes(path)


def test_schema_errors_name_every_problem():
    problems = schemas.errors(
        "track_features", {"duration_seconds": 0, "channels": 3, "detected_instruments": ["kazoo"]}
    )
    text = "\n".join(problems)
    assert "duration_seconds" in text
    assert "channels" in text
    assert "kazoo" in text
    assert "'sample_rate_hz' is a required property" in text
    with pytest.raises(KeyError):
        schemas.load("nothing")


def test_thresholds_must_be_ordered():
    data = SETTINGS.model_dump()
    for section, key, value in [
        ("presence", "absent_max", 0.6), ("bleed", "moderate_min", 0.1), ("reverb", "wet_min", 0.1)
    ]:  # fmt: skip
        broken = {**data, section: {**data[section], key: value}}
        with pytest.raises(ValueError, match="must"):
            type(SETTINGS).model_validate(broken)
    assert SETTINGS.calibrated.all is False
    for table in ({"kazoo": {"present_min": 0.5, "absent_max": 0.2}},
                  {"piano": {"present_min": 0.2, "absent_max": 0.2}}):  # fmt: skip
        with pytest.raises(ValueError, match="must"):
            type(SETTINGS).model_validate({**data, "mix_presence": table})
