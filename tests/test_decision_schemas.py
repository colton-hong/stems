"""The three decision schemas: valid and invalid examples, and the generated model lists."""

from __future__ import annotations

import copy
import json

import pytest

from stems_common import schemas
from stems_common.stems import PRODUCT_STEMS
from stems_models import plan_schema
from stems_models.registry import Registry, load_registry

JOB = "0d9c2f0e-6a57-4f0e-9d53-3f1f3a0f6b11"
FILE_ONLY = ("$schema", "$id", "title", "description")


def parsed_request() -> dict:
    wanted = {"wanted": True, "probability": 0.97, "confidence": 0.97}
    unwanted = {"wanted": False, "probability": 0.02}
    return {
        "request_id": JOB,
        "raw_text": "just the vocals and drums as flac",
        "stems": {s: dict(wanted if s in ("vocals", "drums") else unwanted) for s in PRODUCT_STEMS},
        "vocal_split": "single_vocals",
        "output_format": "flac",
        "quality_mode": "unspecified",
        "wanted_threshold": 0.7,
        "confidence": 0.93,
        "needs_review": False,
    }


def track_features() -> dict:
    return {
        "duration_seconds": 215.4,
        "sample_rate_hz": 44100,
        "channels": 2,
        "vocal_presence": "lead_vocals",
        "detected_instruments": ["vocals", "drums", "bass"],
        "genre_hint": None,
        "reverb_estimate": None,
        "loudness_lufs": -9.8,
    }


def ensemble_plan() -> dict:
    none = {"requested": False, "primary_model": "none", "fallback_model": "none"}
    return {
        "job_id": JOB,
        "track_features": track_features(),
        "quality_target": "balanced",
        "plan": {
            **{stem: dict(none) for stem in PRODUCT_STEMS},
            "vocals": {
                "requested": True,
                "primary_model": "mel_roformer_kim_vocals",
                "fallback_model": "bs_roformer_viperx_vocals",
                "probability": 0.81,
                "confidence": 0.84,
            },
        },
        "dereverb": {"apply": False, "model": None},
        "ensemble_mode": "single_best",
        "estimated_seconds": 45.7,
        "confidence": 0.84,
        "needs_review": False,
    }


def assessment() -> dict:
    return {
        "job_id": JOB,
        "stem": "vocals",
        "model_used": "mel_roformer_kim_vocals",
        "metrics": {
            "sdr_db": None,
            "sir_db": None,
            "sar_db": None,
            "duration_seconds": 215.4,
            "rms_dbfs": -21.3,
            "peak_dbfs": -1.2,
            "silence_fraction": 0.18,
            "bleed_estimate": {"suspected_source": "none", "level": "none"},
            "tagger_notes": "scripted",
        },
        "assessment": {
            "score": 3.2,
            "score_label": "good",
            "pass_probability": 0.93,
            "confidence": 0.88,
            "reason_code": "clean",
        },
        "verdict": {"deliver": True, "action": "deliver", "retry_model": None, "max_retries": 2},
    }


EXAMPLES = {
    "parsed_stem_request": parsed_request,
    "ensemble_plan": ensemble_plan,
    "stem_quality_assessment": assessment,
}


def _set(obj: dict, path: str, value) -> dict:
    changed = copy.deepcopy(obj)
    *parents, last = path.split("/")
    target = changed
    for key in parents:
        target = target[key]
    if value is ...:
        del target[last]
    else:
        target[last] = value
    return changed


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_valid_example_passes(name: str) -> None:
    assert schemas.errors(name, EXAMPLES[name]()) == []
    schemas.validate(name, EXAMPLES[name]())


INVALID = [
    # (schema, path, new value or ... to delete, text expected in the error)
    ("parsed_stem_request", "request_id", "not-a-uuid", "request_id"),
    ("parsed_stem_request", "raw_text", "", "raw_text"),
    ("parsed_stem_request", "stems/piano", ..., "piano"),
    ("parsed_stem_request", "stems/vocals/probability", 1.2, "stems/vocals/probability"),
    ("parsed_stem_request", "output_format", "ogg", "output_format"),
    ("parsed_stem_request", "wanted_threshold", 0.3, "wanted_threshold"),
    ("parsed_stem_request", "needs_review", ..., "needs_review"),
    ("parsed_stem_request", "extra", 1, "extra"),
    ("ensemble_plan", "job_id", "42", "job_id"),
    ("ensemble_plan", "plan/vocals/primary_model", "bs_roformer", "primary_model"),
    # A real model, but not one that can serve this stem.
    ("ensemble_plan", "plan/drums/primary_model", "mel_roformer_kim_vocals", "plan/drums"),
    ("ensemble_plan", "plan/other", ..., "other"),
    ("ensemble_plan", "dereverb/model", "mel_roformer_kim_vocals", "dereverb/model"),
    ("ensemble_plan", "ensemble_mode", "everything", "ensemble_mode"),
    ("ensemble_plan", "estimated_seconds", -1, "estimated_seconds"),
    ("ensemble_plan", "track_features/channels", 6, "track_features/channels"),
    ("ensemble_plan", "track_features/tempo", 120, "tempo"),
    ("stem_quality_assessment", "stem", "strings", "stem"),
    ("stem_quality_assessment", "model_used", "", "model_used"),
    ("stem_quality_assessment", "metrics/rms_dbfs", 1.0, "metrics/rms_dbfs"),
    ("stem_quality_assessment", "metrics/silence_fraction", ..., "silence_fraction"),
    ("stem_quality_assessment", "assessment/score", 4.5, "assessment/score"),
    ("stem_quality_assessment", "assessment/score_label", "fine", "score_label"),
    ("stem_quality_assessment", "assessment/reason_code", "noisy", "reason_code"),
    ("stem_quality_assessment", "verdict/action", "retry", "verdict/action"),
    ("stem_quality_assessment", "verdict/max_retries", 3, "max_retries"),
    ("stem_quality_assessment", "verdict/deliver", "yes", "verdict/deliver"),
]


@pytest.mark.parametrize(("name", "path", "value", "expected"), INVALID)
def test_an_invalid_example_is_rejected(name: str, path: str, value, expected: str) -> None:
    broken = _set(EXAMPLES[name](), path, value)
    problems = schemas.errors(name, broken)
    assert problems, f"{name}: {path}={value!r} was accepted"
    assert any(expected in p for p in problems), problems
    with pytest.raises(schemas.SchemaError, match=f"not a valid {name}"):
        schemas.validate(name, broken)


def test_the_embedded_parts_are_the_standalone_schemas() -> None:
    def body(name: str) -> dict:
        return {k: v for k, v in schemas.load(name).items() if k not in FILE_ONLY}

    plan = schemas.load("ensemble_plan")
    quality = schemas.load("stem_quality_assessment")
    assert plan["properties"]["track_features"] == body("track_features")
    assert quality["properties"]["metrics"] == body("stem_metrics")


def test_every_schema_file_is_itself_a_valid_schema() -> None:
    from jsonschema import Draft202012Validator

    names = sorted(p.name.removesuffix(".schema.json") for p in schemas.SCHEMA_DIR.glob("*.json"))
    assert names == [
        "ensemble_plan",
        "job_result",
        "job_spec",
        "job_status",
        "parsed_stem_request",
        "stem_metrics",
        "stem_quality_assessment",
        "track_features",
    ]
    for name in names:
        Draft202012Validator.check_schema(schemas.load(name))
    with pytest.raises(KeyError, match="no schema named"):
        schemas.load("nope")


def test_the_committed_plan_schema_matches_the_registry() -> None:
    # If this fails after a registry edit: python -m stems_models.plan_schema --write
    assert plan_schema.is_current()
    assert plan_schema.main([]) == 0
    assert plan_schema.main(["--bogus"]) == 2


def test_model_lists_are_the_loadable_candidates_plus_none() -> None:
    registry = load_registry()
    schema = plan_schema.ensemble_plan_schema(registry)
    for stem in PRODUCT_STEMS:
        candidates = [m.id for m in registry.candidates_for(stem)]
        assert candidates, stem
        for field in ("primary_model", "fallback_model"):
            allowed = schema["$defs"][f"stemPlan_{stem}"]["properties"][field]["enum"]
            assert allowed == ["none", *candidates]
        assert schema["properties"]["plan"]["properties"][stem] == {
            "$ref": f"#/$defs/stemPlan_{stem}"
        }
    dereverb = schema["properties"]["dereverb"]["properties"]["model"]["enum"]
    assert dereverb == [*(m.id for m in registry.candidates_for("dereverb")), None]
    # Nothing that is not a loadable separator is ever offered.
    offered = {m for ids in plan_schema.model_enums(registry).values() for m in ids}
    assert all(registry.get(m).is_loadable and registry.get(m).kind == "separator" for m in offered)


def test_changing_the_registry_changes_the_generated_lists() -> None:
    registry = load_registry()
    before = plan_schema.model_enums(registry)
    dropped = "mel_roformer_guitar_becruily"
    assert dropped in before["guitar"]

    # Remove a model.
    smaller = Registry(schema_version=1, models=[m for m in registry.models if m.id != dropped])
    schema = plan_schema.ensemble_plan_schema(smaller)
    guitar = schema["$defs"]["stemPlan_guitar"]["properties"]["primary_model"]["enum"]
    assert dropped not in guitar
    assert guitar == ["none", *(m for m in before["guitar"] if m != dropped)]
    assert not plan_schema.is_current(smaller)
    example = ensemble_plan()
    example["plan"]["guitar"] = {
        "requested": True,
        "primary_model": dropped,
        "fallback_model": "none",
    }
    assert schemas.errors("ensemble_plan", example) == []
    assert any("plan/guitar/primary_model" in p for p in schemas.errors_in(schema, example))

    # Make an existing model a candidate for one more stem.
    target = registry.get("bs_roformer_sw_6stem")
    assert "vocals" in target.stems
    piano_only = [
        m.model_copy(update={"candidate_for": ["piano"]}) if m.id == target.id else m
        for m in registry.models
    ]
    narrowed = plan_schema.model_enums(Registry(schema_version=1, models=piano_only))
    assert target.id in narrowed["piano"]
    assert target.id not in narrowed["guitar"]
    assert target.id in before["guitar"]

    # The committed file is untouched by any of this.
    assert json.loads(plan_schema.PATH.read_text()) == plan_schema.ensemble_plan_schema(registry)
