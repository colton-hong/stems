"""Stem assessment: Jev's answers mapped to a StemQualityAssessment, and the rule scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stems_common import schemas
from stems_common.settings import Calibrated, load_defaults
from stems_worker.decision import assess, gate
from stems_worker.decision.assess import assess_stem, reason_code, rule_assessment
from stems_worker.decision.jev import JevClient, Reply, build_request, noul_confidence
from stems_worker.decision.mock import ReplayTransport, handwritten, save_exchange

DEFAULTS = load_defaults()
GATE = DEFAULTS.gate
ON, OFF = Calibrated.both(True), Calibrated.both(False)
FLOOR = DEFAULTS.decision.min_confidence
CALIBRATED = DEFAULTS.model_copy(
    update={"listening": DEFAULTS.listening.model_copy(update={"calibrated": ON})}
)
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "jev" / "gate.handwritten.json").read_text()
)
CASES = FIXTURE["cases"]
JOB = "5a1e6d0c-0b1f-4a76-8c52-9d0f6b7f2a33"
RETRIES = {
    "retry_fallback_model": "another_model",
    "retry_other_ensemble": None,
    "retry_with_dereverb": "a_dereverb_model",
}


def metrics(silence=0.1, bleed=None, sdr=None, rms=-20.0) -> dict:
    return {
        "sdr_db": sdr,
        "sir_db": None,
        "sar_db": None,
        "duration_seconds": 200.0,
        "rms_dbfs": rms,
        "peak_dbfs": -1.0,
        "silence_fraction": silence,
        "bleed_estimate": None
        if bleed is None
        else {"suspected_source": bleed[0], "level": bleed[1]},
        "tagger_notes": "test",
    }


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "gate.jsonl"
    for case in CASES.values():
        body = build_request(
            DEFAULTS.decision.jev_model,
            assess.state(
                case["stem"],
                case["model_used"],
                case["metrics"],
                present=case["present"],
                tagged=True,
                calibrated=DEFAULTS.listening.calibrated,
                retries_used=0,
            ),
            assess.questions(case["stem"]),
        )
        save_exchange(path, handwritten(body, case["answers"]))
    return JevClient(
        ReplayTransport(path), DEFAULTS.decision, provider="mock", sleep=lambda s: None
    )


def run(client, name: str, **options):
    case = CASES[name]
    return assess_stem(
        case["stem"],
        case["model_used"],
        case["metrics"],
        client=client,
        defaults=DEFAULTS,
        job_id=JOB,
        present=case["present"],
        available=options.pop("available", RETRIES),
        **options,
    )


# --- Jev's answers ------------------------------------------------------------------------


def test_a_good_stem_is_assembled_and_delivered(client) -> None:
    decision = run(client, "good")
    value = decision.value
    assert schemas.errors("stem_quality_assessment", value) == []
    assert value["job_id"] == JOB
    assert value["stem"] == "vocals"
    assert value["model_used"] == "mel_roformer_kim_vocals"
    assert value["metrics"] == CASES["good"]["metrics"]
    assert value["assessment"] == {
        "score": 3.4,  # Jev's fractional score, not rounded
        "score_label": "excellent",  # the most probable level (0.5 at level 4)
        "pass_probability": 0.95,
        "confidence": 0.9,  # the score answer's confidence
        "reason_code": "clean",
    }
    assert value["verdict"] == {
        "deliver": True,
        "action": "deliver",
        "retry_model": None,
        "max_retries": GATE.max_retries_per_stem,
    }
    log = decision.record
    assert (log["kind"], log["provider"], log["stem"]) == ("gate", "mock", "vocals")
    assert log["fallback_reason"] is None
    assert log["answers"]["quality"]["probabilities"]["4"] == 0.5
    assert log["answers"]["acceptable"] == {"type": "noul", "probability": 0.95, "confidence": 0.95}


def test_the_gate_thresholds_are_inclusive(client) -> None:
    value = run(client, "at_the_thresholds").value
    assert value["assessment"]["score"] == GATE.score_min == 2
    assert value["assessment"]["pass_probability"] == GATE.pass_probability_min == 0.8
    assert value["assessment"]["confidence"] == 0.8
    assert value["verdict"]["deliver"] is True
    assert value["assessment"]["reason_code"] == "none"  # passed, though slight bleed was heard


@pytest.mark.parametrize(
    ("name", "action", "model"),
    [
        ("bleed", "retry_fallback_model", "another_model"),
        ("wants_dereverb", "retry_with_dereverb", "a_dereverb_model"),
        ("confident_no_despite_a_fair_score", "retry_other_ensemble", None),
    ],
)
def test_a_failing_stem_gets_the_retry_jev_chose(client, name, action, model) -> None:
    decision = run(client, name)
    verdict = decision.value["verdict"]
    assert verdict["deliver"] is False
    assert verdict["action"] == action
    assert verdict["retry_model"] == model
    assert decision.record["suggested_action"] == action
    assert decision.provider == "mock"
    assert schemas.errors("stem_quality_assessment", decision.value) == []


def test_a_confident_no_blocks_delivery_whatever_the_score(client) -> None:
    assessment = run(client, "confident_no_despite_a_fair_score").value["assessment"]
    assert assessment["score"] == 2.6
    assert assessment["pass_probability"] == 0.15
    assert noul_confidence(0.15) >= FLOOR  # a sure "no", so it is acted on


def test_an_unsure_retry_choice_is_replaced_by_the_rules_suggestion(client) -> None:
    decision = run(client, "unsure_action")
    assert decision.value["assessment"]["score"] == 1.0  # the assessment itself is Jev's
    assert decision.provider == "mock"
    assert decision.value["verdict"]["action"] == "retry_fallback_model"
    assert any("retry_action: Jev was unsure (0.40)" in n for n in decision.record["notes"])


def test_flag_for_human_review_becomes_a_warning(client) -> None:
    verdict = run(client, "hopeless").value["verdict"]
    assert verdict["deliver"] is False
    assert verdict["action"] == "deliver_with_warning"
    assert verdict["user_message"] == "This stem may not contain the instrument it is named after."


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("unsure_score", "low_confidence: quality 0.55 (minimum 0.8)"),
        # p = 0.6: confidence max(p, 1 - p) = 0.6, inside the 0.2 to 0.8 band nobody acts on.
        ("unsure_acceptable", "low_confidence: acceptable 0.60 (minimum 0.8)"),
    ],
)
def test_an_unsure_assessment_is_replaced_by_the_rule_scorer(client, name, reason) -> None:
    decision = run(client, name)
    case = CASES[name]
    assert decision.provider == "fallback"
    assert decision.record["asked"] == "mock"
    assert decision.record["fallback_reason"] == reason
    assert decision.record["answers"]["quality"]["score"] == case["answers"]["quality"]["score"]
    expected, _ = rule_assessment(
        case["metrics"], present=case["present"], settings=GATE, calibrated=OFF
    )
    assert decision.value["assessment"] == expected
    assert schemas.errors("stem_quality_assessment", decision.value) == []


def test_noul_confidence_is_derived_not_read(client) -> None:
    # The wire answer has only the probability (docs/corrections.md, C1).
    assert set(CASES["good"]["answers"]["acceptable"]) == {"type", "noul"}
    for p, confident in [
        (0.0, True),
        (0.2, True),
        (0.21, False),
        (0.5, False),
        (0.79, False),
        (0.8, True),
        (1.0, True),
    ]:
        assert (noul_confidence(p) >= FLOOR) is confident, p
    assert noul_confidence(0.3) == pytest.approx(max(0.3, 1 - 0.3))


def test_the_state_fills_in_nothing(client) -> None:
    case = CASES["bleed"]
    sent = assess.state(
        "drums", "m", case["metrics"], present=True, tagged=True, calibrated=OFF, retries_used=1
    )
    assert sent["metrics"]["sdr_db"] is None  # no reference, so no SDR, and none is invented
    assert sent["metrics"] == case["metrics"]
    assert sent["own_source_heard"] == "yes"
    assert sent["retries_so_far"] == 1
    assert sent["tagger_thresholds_calibrated"] == {"presence": False, "bleed": False}
    untagged = assess.state(
        "drums", "m", metrics(), present=None, tagged=False, calibrated=OFF, retries_used=0
    )
    assert untagged["own_source_heard"] == "not checked"
    asked = assess.questions("drums")
    assert asked["quality"]["criteria"] == ["unusable", "poor", "acceptable", "good", "excellent"]
    schema = schemas.load("stem_quality_assessment")["properties"]
    assert asked["quality"]["criteria"] == schema["assessment"]["properties"]["score_label"]["enum"]
    assert set(asked["retry_action"]["criteria"]) < set(
        schema["verdict"]["properties"]["action"]["enum"]
    )


# --- the rule-based scorer ----------------------------------------------------------------

RULES = [
    # (metrics, own source heard, reason, score, label, pass, suggested retry)
    (metrics(bleed=("none", "none")), True, "clean", 3, "good", 1.0, None),
    (metrics(bleed=("drums", "slight")), True, "bleed", 2, "acceptable", 1.0, None),
    (metrics(bleed=("drums", "moderate")), True, "bleed", 1, "poor", 0.0, "retry_fallback_model"),
    (metrics(bleed=("drums", "severe")), True, "bleed", 0, "unusable", 0.0, "retry_fallback_model"),
    # Its own instrument is not heard in it: wrong content, however clean it is.
    (
        metrics(bleed=("none", "none")),
        False,
        "wrong_content",
        1,
        "poor",
        0.0,
        "retry_fallback_model",
    ),
    (
        metrics(bleed=("bass", "severe")),
        False,
        "wrong_content",
        0,
        "unusable",
        0.0,
        "retry_fallback_model",
    ),
    # The tagger could not tell whether its own instrument is there: no better than acceptable.
    (metrics(bleed=("none", "none")), None, "none", 2, "acceptable", 1.0, None),
    (metrics(bleed=("drums", "moderate")), None, "bleed", 1, "poor", 0.0, "retry_fallback_model"),
]


@pytest.mark.parametrize(("m", "present", "reason", "score", "label", "passed", "suggested"), RULES)
def test_rule_scorer_on_tagger_signals(m, present, reason, score, label, passed, suggested) -> None:
    assessment, action = rule_assessment(m, present=present, settings=GATE, calibrated=ON)
    assert assessment == {
        "score": float(score),
        "score_label": label,
        "pass_probability": passed,
        "confidence": 1.0,
        "reason_code": reason,
    }
    assert action == suggested
    assert reason_code(m, present, GATE) == reason
    # The same signals with uncalibrated thresholds: same reading, low confidence.
    unsure, _ = rule_assessment(m, present=present, settings=GATE, calibrated=OFF)
    assert unsure == {**assessment, "confidence": GATE.rules.uncalibrated_confidence}
    assert GATE.rules.uncalibrated_confidence < GATE.confidence_min


def test_a_silent_stem_is_certain_without_any_tagger() -> None:
    for calibrated in (ON, OFF):
        assessment, action = rule_assessment(
            metrics(silence=1.0), present=None, settings=GATE, calibrated=calibrated
        )
        assert assessment == {
            "score": 0.0,
            "score_label": "unusable",
            "pass_probability": 0.0,
            "confidence": 1.0,
            "reason_code": "mostly_silent",
        }
        assert action == "retry_fallback_model"
    edge = GATE.rules.silent_fraction_min
    assert reason_code(metrics(silence=edge), None, GATE) == "mostly_silent"
    assert reason_code(metrics(silence=edge - 0.01), None, GATE) == "low_confidence"


def test_a_stem_far_below_the_mix_is_treated_like_a_silent_one() -> None:
    limit = GATE.rules.empty_below_mix_db
    assert limit == -21.68  # measured: docs/gate-on-real-output.md
    # Not silent at all (10% of it is), but leftovers 25 dB under the song.
    for calibrated in (ON, OFF):
        assessment, action = rule_assessment(
            metrics(), present=None, settings=GATE, calibrated=calibrated, below_mix_db=-25.0
        )
        assert assessment["reason_code"] == "mostly_silent"
        assert (assessment["score"], assessment["pass_probability"]) == (0.0, 0.0)
        assert assessment["confidence"] == 1.0  # a level measurement, whatever the tagger's state
        assert action == "retry_fallback_model"
    # The threshold is inclusive, and a tagger that hears the instrument does not overrule it.
    assert reason_code(metrics(), True, GATE, limit) == "mostly_silent"
    assert reason_code(metrics(), True, GATE, limit + 0.01) == "clean"
    assert reason_code(metrics(), None, GATE, -5.0) == "low_confidence"
    assert reason_code(metrics(), None, GATE, None) == "low_confidence"  # not measured
    assert gate.warning({**assessment}, GATE) == (
        "This stem is silent or very quiet compared with the song. The instrument may not "
        "be in the song, or it was not found."
    )


def test_the_empty_stem_rule_can_be_switched_off() -> None:
    off = GATE.model_copy(
        update={"rules": GATE.rules.model_copy(update={"empty_below_mix_db": None})}
    )
    assert reason_code(metrics(), None, off, -60.0) == "low_confidence"
    assert assess.is_empty(-60.0, off) is False
    assert assess.is_empty(-60.0, GATE) is True
    assert assess.is_empty(None, GATE) is False


def test_each_tagger_signal_has_its_own_switch() -> None:
    confident, unsure = 1.0, GATE.rules.uncalibrated_confidence
    absent = (metrics(bleed=("none", "none")), False)  # rests on presence
    bled = (metrics(bleed=("drums", "moderate")), True)  # rests on bleed
    clean = (metrics(bleed=("none", "none")), True)  # rests on both
    for switches, expected in [
        (Calibrated(presence=True, bleed=False), (confident, unsure, unsure)),
        (Calibrated(presence=False, bleed=True), (unsure, confident, unsure)),
        (ON, (confident, confident, confident)),
        (OFF, (unsure, unsure, unsure)),
    ]:
        got = tuple(
            rule_assessment(m, present=present, settings=GATE, calibrated=switches)[0]["confidence"]
            for m, present in (absent, bled, clean)
        )
        assert got == expected, switches


def test_jev_is_told_the_level_and_which_tagger_signals_are_switched_on() -> None:
    sent = assess.state(
        "piano",
        "m",
        metrics(),
        present=None,
        tagged=False,
        calibrated=Calibrated(presence=True, bleed=False),
        retries_used=0,
        below_mix_db=-23.456,
    )
    assert sent["level_below_mix_db"] == -23.46
    assert sent["tagger_thresholds_calibrated"] == {"presence": True, "bleed": False}
    unmeasured = assess.state(
        "piano", "m", metrics(), present=None, tagged=False, calibrated=OFF, retries_used=0
    )
    assert unmeasured["level_below_mix_db"] is None


def test_with_no_tagger_the_scorer_says_it_cannot_tell() -> None:
    assessment, action = rule_assessment(metrics(), present=None, settings=GATE, calibrated=ON)
    assert assessment == {
        "score": float(GATE.rules.unchecked_score),
        "score_label": "acceptable",
        "pass_probability": 0.5,
        "confidence": 0.0,
        "reason_code": "low_confidence",
    }
    assert action is None


def test_a_stem_with_no_reference_sdr_is_gated_on_tagger_signals() -> None:
    def gated(m, present=True):
        return assess_stem(
            "bass",
            "htdemucs_ft",
            m,
            client=None,
            defaults=CALIBRATED,
            present=present,
            available=RETRIES,
        ).value

    clean = gated(metrics(bleed=("none", "none")))
    assert clean["metrics"]["sdr_db"] is None
    assert clean["verdict"]["deliver"] is True
    assert clean["assessment"]["reason_code"] == "clean"
    bled = gated(metrics(bleed=("guitar", "severe")))
    assert bled["metrics"]["sdr_db"] is None
    assert bled["verdict"] == {
        "deliver": False,
        "action": "retry_fallback_model",
        "retry_model": "another_model",
        "max_retries": GATE.max_retries_per_stem,
    }
    absent = gated(metrics(bleed=("none", "none")), present=False)
    assert absent["verdict"]["deliver"] is False
    assert absent["assessment"]["reason_code"] == "wrong_content"
    # An SDR, where a benchmark has one, changes nothing: the rules never read it.
    for sdr in (-5.0, 3.0, 14.0):
        with_reference = gated(metrics(bleed=("guitar", "severe"), sdr=sdr))
        assert with_reference["assessment"] == bled["assessment"]
        assert with_reference["verdict"] == bled["verdict"]
        assert with_reference["metrics"]["sdr_db"] == sdr


def test_without_jev_the_rules_assess_and_the_record_says_so() -> None:
    decision = assess_stem(
        "drums",
        "scnet_xl_ihf",
        metrics(bleed=("bass", "moderate")),
        client=None,
        defaults=CALIBRATED,
        job_id=JOB,
        present=True,
        available=RETRIES,
    )
    assert decision.provider == "fallback"
    assert decision.record["fallback_reason"] == "no_key"
    assert decision.record["answers"] is None
    assert decision.record["suggested_action"] == "retry_fallback_model"
    assert decision.value["verdict"]["action"] == "retry_fallback_model"
    assert schemas.errors("stem_quality_assessment", decision.value) == []


def test_uncalibrated_or_unchecked_stems_are_warned_about_not_retried() -> None:
    # The shipped configuration: listening.calibrated is false.
    assert DEFAULTS.listening.calibrated == OFF
    heard_bleed = assess_stem(
        "drums",
        "scnet_xl_ihf",
        metrics(bleed=("bass", "severe")),
        client=None,
        defaults=DEFAULTS,
        present=True,
        available=RETRIES,
    )
    assert heard_bleed.value["assessment"]["confidence"] == GATE.rules.uncalibrated_confidence
    assert heard_bleed.value["verdict"]["action"] == "deliver_with_warning"
    # The suspicion is logged, but the user is not told something the gate does not trust.
    assert heard_bleed.value["assessment"]["reason_code"] == "bleed"
    assert (
        heard_bleed.value["verdict"]["user_message"]
        == "This stem could not be quality-checked automatically."
    )
    assert "the tagger's thresholds are uncalibrated" in heard_bleed.record["notes"]

    unchecked = assess_stem(
        "drums", "scnet_xl_ihf", metrics(), client=None, defaults=DEFAULTS, available=RETRIES
    )
    assert unchecked.value["assessment"]["reason_code"] == "low_confidence"
    assert unchecked.value["verdict"] == {
        "deliver": False,
        "action": "deliver_with_warning",
        "retry_model": None,
        "max_retries": GATE.max_retries_per_stem,
        "user_message": "This stem could not be quality-checked automatically.",
    }
    assert unchecked.record["notes"] == []

    # A silent stem is a measurement, so it is acted on even now.
    silent = assess_stem(
        "piano",
        "bs_roformer_sw_6stem",
        metrics(silence=1.0, rms=-120.0),
        client=None,
        defaults=DEFAULTS,
        available=RETRIES,
    )
    assert silent.value["verdict"]["action"] == "retry_fallback_model"


def test_a_failing_jev_hands_over_to_the_rules() -> None:
    class Down:
        def __call__(self, body, timeout):
            return Reply(429, "slow down", {"retry-after": "600"})

    failing = JevClient(Down(), DEFAULTS.decision, sleep=lambda s: None)
    decision = assess_stem(
        "vocals",
        "m",
        metrics(bleed=("none", "none")),
        client=failing,
        defaults=CALIBRATED,
        present=True,
    )
    assert decision.provider == "fallback"
    assert decision.record["fallback_reason"].startswith(
        "jev_failed: HTTP 429: asked to wait 600 s"
    )
    assert decision.value["verdict"]["deliver"] is True


def test_bad_input_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown stem"):
        assess_stem("strings", "m", metrics(), client=None, defaults=DEFAULTS)
    broken = {**metrics(), "rms_dbfs": 3.0}
    with pytest.raises(schemas.SchemaError, match="rms_dbfs"):
        assess_stem("drums", "m", broken, client=None, defaults=DEFAULTS)


# --- the gate rule ------------------------------------------------------------------------

GATE_CASES = [
    # (pass probability, score, confidence, delivered)
    (0.8, 2.0, 0.7, True),  # exactly at all three thresholds
    (0.95, 3.6, 0.9, True),
    (0.79, 4.0, 1.0, False),  # pass probability just under
    (1.0, 1.99, 1.0, False),  # score just under
    (1.0, 4.0, 0.69, False),  # confidence just under
    (0.0, 0.0, 1.0, False),
]


@pytest.mark.parametrize(("p", "score", "confidence", "delivered"), GATE_CASES)
def test_the_gate_rule(p: float, score: float, confidence: float, delivered: bool) -> None:
    assessment = {"pass_probability": p, "score": score, "confidence": confidence}
    assert gate.passes(assessment, GATE) is delivered
    assert gate.verdict(assessment, settings=GATE)["deliver"] is delivered
    assert (GATE.pass_probability_min, GATE.score_min, GATE.confidence_min) == (0.8, 2, 0.7)


def test_the_gate_thresholds_come_from_config() -> None:
    assessment = {"pass_probability": 0.75, "score": 1.5, "confidence": 0.6}
    assert gate.passes(assessment, GATE) is False
    looser = GATE.model_copy(
        update={"pass_probability_min": 0.75, "score_min": 1.5, "confidence_min": 0.6}
    )
    assert gate.passes(assessment, looser) is True
    for name in ("pass_probability_min", "score_min", "confidence_min"):
        one_tighter = looser.model_copy(update={name: getattr(looser, name) + 0.01})
        assert gate.passes(assessment, one_tighter) is False
