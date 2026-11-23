"""Ensemble planning: Jev's answers mapped to an EnsemblePlan, the rules, the config fallback."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from stems_common import schemas
from stems_common.settings import load_defaults
from stems_common.stems import PRODUCT_STEMS
from stems_models.registry import Registry, load_registry
from stems_worker.compose import Blend
from stems_worker.decision import plan as planning
from stems_worker.decision.jev import JevClient, Reply, build_request
from stems_worker.decision.mock import ReplayTransport, handwritten, save_exchange
from stems_worker.decision.parse import from_selection
from stems_worker.decision.plan import (
    PlanError,
    assignment_for,
    check_rules,
    estimate_seconds,
    models_to_run,
    plan_ensemble,
)

DEFAULTS = load_defaults()
SETTINGS = DEFAULTS.decision
REGISTRY = load_registry()
SLOTS = DEFAULTS.ensembles.slots
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "jev" / "plan.handwritten.json").read_text()
)
FEATURES = FIXTURE["track_features"]
CASES = FIXTURE["cases"]
JOB = "3f6f2f3e-52a8-4b5e-9a57-0d2d1c7a9e01"
# Seconds of work per second of audio, made up for these tests only.
RATES = {
    ("mel_roformer_kim_vocals", "cpu"): 0.8,
    ("bs_roformer_viperx_vocals", "cpu"): 1.5,
    ("htdemucs_ft", "cpu"): 1.25,
    ("scnet_xl_ihf", "cpu"): 3.0,
    ("scnet_xl_ihf", "mps"): 0.3,
}


def parsed_for(case: dict) -> dict:
    return from_selection(case["stems"], settings=SETTINGS, quality_mode=case["quality_mode"]).value


def quality_target(case: dict) -> str:
    mode = case["quality_mode"]
    return "balanced" if mode == "unspecified" else mode


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "plan.jsonl"
    for case in CASES.values():
        target = quality_target(case)
        body = build_request(
            SETTINGS.jev_model,
            planning.state(
                FEATURES,
                case["stems"],
                quality_target=target,
                dereverb=case["dereverb"],
                vocal_split=False,
            ),
            planning.questions(
                case["stems"], REGISTRY, DEFAULTS, dereverb=case["dereverb"], quality_target=target
            ),
        )
        save_exchange(path, handwritten(body, case["answers"]))
    return JevClient(ReplayTransport(path), SETTINGS, provider="mock", sleep=lambda s: None)


def run(client, name: str, **options):
    case = CASES[name]
    return plan_ensemble(
        parsed_for(case),
        FEATURES,
        client=client,
        defaults=DEFAULTS,
        registry=REGISTRY,
        job_id=JOB,
        dereverb=case["dereverb"],
        rates=options.pop("rates", RATES),
        **options,
    )


def fallback_plan(stems, **options):
    parsed = from_selection(
        stems, settings=SETTINGS, quality_mode=options.pop("quality_mode", "unspecified")
    ).value
    return plan_ensemble(
        parsed,
        FEATURES,
        client=options.pop("client", None),
        defaults=DEFAULTS,
        registry=REGISTRY,
        job_id=JOB,
        rates=options.pop("rates", RATES),
        **options,
    )


# --- Jev's answers ------------------------------------------------------------------------


def test_confident_answers_become_the_plan(client) -> None:
    decision = run(client, "confident")
    plan = decision.value
    assert schemas.errors("ensemble_plan", plan) == []
    assert check_rules(plan) == []
    assert plan["job_id"] == JOB
    assert plan["track_features"] == FEATURES
    assert plan["quality_target"] == "balanced"
    assert plan["plan"]["vocals"] == {
        "requested": True,
        "primary_model": "bs_roformer_viperx_vocals",
        "fallback_model": "mel_roformer_kim_vocals",  # Jev's second choice
        "probability": 0.86,
        "confidence": 0.86,
    }
    assert plan["plan"]["drums"]["primary_model"] == "htdemucs_ft"
    assert plan["plan"]["drums"]["fallback_model"] == "scnet_xl_ihf"
    for stem in ("bass", "guitar", "piano", "other"):
        assert plan["plan"][stem] == {
            "requested": False,
            "primary_model": "none",
            "fallback_model": "none",
        }
    assert plan["ensemble_mode"] == "single_best"  # a confident "not worth it"
    assert plan["dereverb"] == {"apply": False, "model": None}
    assert plan["confidence"] == pytest.approx(0.82)
    assert plan["needs_review"] is False
    assert decision.provider == "mock"
    assert decision.record["fallback_reason"] is None
    assert decision.record["answers"]["model.drums"]["choice"] == "htdemucs_ft"
    # 200 s of audio at 1.5 and 1.25 seconds per second, both measured.
    assert plan["estimated_seconds"] == pytest.approx(200 * 1.5 + 200 * 1.25)
    assert decision.record["notes"] == []


def test_an_unsure_stem_falls_back_to_the_config_and_says_why(client) -> None:
    decision = run(client, "unsure_drums")
    plan = decision.value
    assert check_rules(plan) == []
    # Vocals: Jev was sure, and its answer stands.
    assert plan["plan"]["vocals"]["primary_model"] == "mel_roformer_kim_vocals"
    assert plan["plan"]["vocals"]["confidence"] == 0.9
    # Drums: 0.41 is under 0.8, so the configured default and its configured fallback.
    assert plan["plan"]["drums"] == {
        "requested": True,
        "primary_model": SLOTS["drums"].single_best,
        "fallback_model": "htdemucs_ft",
    }
    assert plan["confidence"] == pytest.approx(0.41)
    assert plan["needs_review"] is True
    assert decision.provider == "mock"  # the other answers were used
    assert decision.record["fallback_reason"] is None
    notes = decision.record["notes"]
    assert notes[0] == "low_confidence: model.drums 0.41 (minimum 0.8)"
    assert "drums: Jev was unsure, so the configured default scnet_xl_ihf is used" in notes


def test_when_nothing_jev_said_can_be_used_the_rules_decide(client) -> None:
    decision = run(client, "all_unsure")
    plan = decision.value
    assert decision.provider == "fallback"
    assert decision.record["asked"] == "mock"
    assert decision.record["fallback_reason"] == (
        "low_confidence: model.bass 0.35, ensemble 0.55 (minimum 0.8)"
    )
    assert decision.record["answers"]["model.bass"]["choice"] == "scnet_xl"  # kept in the log
    assert plan["plan"]["bass"]["primary_model"] == SLOTS["bass"].single_best
    assert plan["ensemble_mode"] == SETTINGS.mode_for_quality["balanced"]
    assert plan["confidence"] == pytest.approx(0.35)
    assert plan["needs_review"] is True
    assert check_rules(plan) == []


def test_a_blend_is_planned_when_jev_says_it_is_worth_it(client) -> None:
    best = run(client, "best_quality_blend").value
    assert best["quality_target"] == "best_quality"
    assert best["ensemble_mode"] == "full_ensemble"
    assert all(best["plan"][s]["requested"] for s in PRODUCT_STEMS)
    assert check_rules(best) == []
    balanced = run(client, "balanced_blend").value
    assert balanced["ensemble_mode"] == "two_model_blend"
    # In a blend mode the configured blend serves the stem, weights and all.
    assignment = assignment_for(balanced, DEFAULTS)
    assert isinstance(assignment["drums"], Blend)
    assert assignment["drums"].models == tuple(SLOTS["drums"].two_model_blend)
    assert models_to_run(balanced, DEFAULTS, REGISTRY) == SLOTS["drums"].two_model_blend


def test_dereverb_is_planned_on_request_with_the_chosen_model(client) -> None:
    decision = run(client, "dereverb")
    plan = decision.value
    assert plan["dereverb"] == {"apply": True, "model": "mel_roformer_dereverb_echo_sucial"}
    assert plan["quality_target"] == "fastest"
    assert plan["ensemble_mode"] == "single_best"
    assert "ensemble" not in decision.record["answers"]  # not asked when speed is wanted
    assert assignment_for(plan, DEFAULTS)["dereverb"] == "mel_roformer_dereverb_echo_sucial"
    assert models_to_run(plan, DEFAULTS, REGISTRY) == [
        "mel_roformer_kim_vocals",
        "mel_roformer_dereverb_echo_sucial",
    ]
    assert check_rules(plan) == []


def test_options_are_the_registry_candidates_and_mark_the_config_default() -> None:
    asked = planning.questions(
        ["vocals", "piano"], REGISTRY, DEFAULTS, dereverb=True, quality_target="balanced"
    )
    assert list(asked) == ["model.vocals", "model.piano", "dereverb", "ensemble"]
    vocals = asked["model.vocals"]["criteria"]
    # Criteria keys are registry ids, the same values the schema's enum allows.
    assert list(vocals) == [m.id for m in REGISTRY.candidates_for("vocals")]
    assert "the configured default for vocals" in vocals[SLOTS["vocals"].single_best]
    assert "two-model blend" in vocals[SLOTS["vocals"].two_model_blend[1]]
    assert "not in the configured defaults" in vocals["scnet_small"]
    word = "provisional" if DEFAULTS.ensembles.provisional else "benchmarked"
    assert word in vocals[SLOTS["vocals"].single_best]
    assert "none" not in asked["dereverb"]["criteria"]
    fast = planning.questions(
        ["piano"], REGISTRY, DEFAULTS, dereverb=False, quality_target="fastest"
    )
    assert list(fast) == ["model.piano"]


# --- the config fallback ------------------------------------------------------------------


def test_without_jev_the_plan_is_the_configuration() -> None:
    decision = fallback_plan(PRODUCT_STEMS)
    plan = decision.value
    assert schemas.errors("ensemble_plan", plan) == []
    assert check_rules(plan) == []
    assert decision.provider == "fallback"
    assert decision.record["fallback_reason"] == "no_key"
    assert decision.record["answers"] is None
    for stem in PRODUCT_STEMS:
        entry = plan["plan"][stem]
        assert entry["primary_model"] == SLOTS[stem].single_best
        assert entry["fallback_model"] not in ("none", entry["primary_model"])
        assert "confidence" not in entry
    assert plan["plan"]["vocals"]["fallback_model"] == SLOTS["vocals"].two_model_blend[1]
    assert plan["ensemble_mode"] == SETTINGS.mode_for_quality["balanced"]
    assert plan["confidence"] == 1.0
    assert plan["needs_review"] is False
    assert assignment_for(plan, DEFAULTS) == {
        slot: models.single_best for slot, models in SLOTS.items()
    }


@pytest.mark.parametrize("quality", ["fastest", "balanced", "best_quality"])
def test_the_mode_follows_the_quality_asked_for(quality: str) -> None:
    plan = fallback_plan(["drums"], quality_mode=quality).value
    assert plan["quality_target"] == quality
    assert plan["ensemble_mode"] == SETTINGS.mode_for_quality[quality]
    assert SETTINGS.mode_for_quality["fastest"] == "single_best"


def test_a_failing_jev_falls_back_and_says_why() -> None:
    class Down:
        def __call__(self, body, timeout):
            return Reply(500, "boom")

    failing = JevClient(Down(), SETTINGS, sleep=lambda s: None)
    decision = fallback_plan(["vocals"], client=failing)
    assert decision.provider == "fallback"
    assert decision.record["asked"] == "jev"
    assert decision.record["fallback_reason"].startswith("jev_failed: HTTP 500")
    assert decision.value["plan"]["vocals"]["primary_model"] == SLOTS["vocals"].single_best
    assert decision.value["needs_review"] is False


def test_dereverb_falls_back_to_the_configured_model() -> None:
    plan = fallback_plan(["vocals"], dereverb=True).value
    assert plan["dereverb"] == {"apply": True, "model": SLOTS["dereverb"].single_best}
    assert fallback_plan(["drums"]).value["dereverb"] == {"apply": False, "model": None}


def test_nothing_requested_cannot_be_planned() -> None:
    parsed = from_selection(["vocals"], settings=SETTINGS).value
    parsed["stems"]["vocals"]["wanted"] = False
    with pytest.raises(PlanError, match="no stem is requested"):
        plan_ensemble(parsed, FEATURES, client=None, defaults=DEFAULTS, registry=REGISTRY)
    without_piano = Registry(
        schema_version=1,
        models=[
            m.model_copy(update={"candidate_for": [s for s in m.candidate_for if s != "piano"]})
            for m in REGISTRY.models
        ],
    )
    parsed = from_selection(["piano"], settings=SETTINGS).value
    with pytest.raises(PlanError, match="no loadable model in the registry can serve 'piano'"):
        plan_ensemble(parsed, FEATURES, client=None, defaults=DEFAULTS, registry=without_piano)


# --- the report's rules -------------------------------------------------------------------

RULE_BREAKS = [
    (
        {"requested": False, "primary_model": "htdemucs_ft", "fallback_model": "none"},
        "both models must be 'none'",
    ),
    (
        {"requested": False, "primary_model": "none", "fallback_model": "htdemucs_ft"},
        "both models must be 'none'",
    ),
    (
        {"requested": True, "primary_model": "none", "fallback_model": "htdemucs_ft"},
        "primary_model must not be 'none'",
    ),
    (
        {"requested": True, "primary_model": "none", "fallback_model": "none"},
        "primary_model must not be 'none'",
    ),
    (
        {"requested": True, "primary_model": "htdemucs_ft", "fallback_model": "htdemucs_ft"},
        "must differ from primary_model",
    ),
]


@pytest.mark.parametrize(("entry", "message"), RULE_BREAKS)
def test_the_requested_primary_fallback_rules_are_enforced(entry: dict, message: str) -> None:
    plan = fallback_plan(["vocals"]).value
    assert check_rules(plan) == []
    broken = copy.deepcopy(plan)
    broken["plan"]["drums"] = entry
    # The schema alone allows it; the rules do not.
    assert schemas.errors("ensemble_plan", broken) == []
    assert check_rules(broken) == [f"drums: {check_rules(broken)[0].split(': ', 1)[1]}"]
    assert message in check_rules(broken)[0]
    with pytest.raises(PlanError, match=message):
        planning.validate(broken, REGISTRY)


def test_other_rule_cases() -> None:
    plan = fallback_plan(["vocals"]).value
    # A requested stem may have no fallback at all, as long as it has a primary.
    alone = copy.deepcopy(plan)
    alone["plan"]["vocals"]["fallback_model"] = "none"
    assert check_rules(alone) == []
    for dereverb in (
        {"apply": True, "model": None},
        {"apply": False, "model": "uvr_vr_deecho_dereverb"},
    ):
        broken = copy.deepcopy(plan)
        broken["dereverb"] = dereverb
        assert check_rules(broken) == ["dereverb: a model is named exactly when apply is true"]
    wrong_model = copy.deepcopy(plan)
    wrong_model["plan"]["vocals"]["primary_model"] = "mel_roformer_guitar_becruily"
    with pytest.raises(PlanError, match="plan/vocals/primary_model"):
        planning.validate(wrong_model, REGISTRY)


# --- the time estimate --------------------------------------------------------------------


def test_the_estimate_uses_measured_rates_only() -> None:
    assert estimate_seconds(["htdemucs_ft"], 100.0, "cpu", RATES) == (125.0, [])
    assert estimate_seconds(["htdemucs_ft", "htdemucs_ft"], 100.0, "cpu", RATES) == (125.0, [])
    # No timing on this device: nothing is assumed, and the model is named.
    assert estimate_seconds(["htdemucs_ft", "scnet_large"], 100.0, "mps", RATES) == (
        0.0,
        ["htdemucs_ft", "scnet_large"],
    )
    assert estimate_seconds(["scnet_xl_ihf", "scnet_large"], 10.0, "mps", RATES) == (
        3.0,
        ["scnet_large"],
    )

    decision = fallback_plan(["drums", "piano"], device="mps")
    # scnet_xl_ihf is timed on mps here; the piano model is not.
    assert decision.value["estimated_seconds"] == pytest.approx(200 * 0.3)
    assert decision.record["notes"] == [
        "estimated_seconds is a lower bound: no timing on mps for bs_roformer_sw_6stem"
    ]
    nothing = fallback_plan(["drums"], rates={})
    assert nothing.value["estimated_seconds"] == 0.0
    assert "lower bound" in nothing.record["notes"][0]


def test_rates_come_from_the_timing_log(tmp_path) -> None:
    log = tmp_path / "timings.jsonl"
    lines = [
        {
            "model_id": "htdemucs_ft",
            "device": "cpu",
            "wall_seconds": 10,
            "audio_seconds": 10,
            "realtime_factor": 1.0,
        },
        {
            "model_id": "htdemucs_ft",
            "device": "cpu",
            "wall_seconds": 30,
            "audio_seconds": 10,
            "realtime_factor": 3.0,
        },
        {
            "model_id": "htdemucs_ft",
            "device": "cpu",
            "wall_seconds": 20,
            "audio_seconds": 10,
            "realtime_factor": 2.0,
        },
        {
            "model_id": "htdemucs_ft",
            "device": "mps",
            "wall_seconds": 2,
            "audio_seconds": 10,
            "realtime_factor": 0.2,
        },
    ]
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert planning.measured_rates(log) == {
        ("htdemucs_ft", "cpu"): 2.0,
        ("htdemucs_ft", "mps"): 0.2,
    }
    assert planning.measured_rates(tmp_path / "absent.jsonl") == {}
