"""Writes tests/fixtures/jev/gate.handwritten.json: hand-written gate answers for tests."""

import json
import pathlib
import sys

LEVELS = 5
ACTIONS = (
    "retry_fallback_model",
    "retry_other_ensemble",
    "retry_with_dereverb",
    "flag_for_human_review",
)


def score(value, confidence, probabilities):
    assert len(probabilities) == LEVELS
    assert abs(sum(probabilities) - 1) < 1e-9
    return {
        "type": "score",
        "score": value,
        "confidence": confidence,
        "legend": dict(enumerate(("unusable", "poor", "acceptable", "good", "excellent"))),
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
    }


def action(picked, confidence):
    rest = round((1 - confidence) / (len(ACTIONS) - 1), 6)
    probabilities = {a: (confidence if a == picked else rest) for a in ACTIONS}
    return {
        "type": "choice",
        "choice": picked,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def metrics(rms, silence, bleed=None):
    return {
        "sdr_db": None,
        "sir_db": None,
        "sar_db": None,
        "duration_seconds": 200.0,
        "rms_dbfs": rms,
        "peak_dbfs": -1.0,
        "silence_fraction": silence,
        "bleed_estimate": None
        if bleed is None
        else {"suspected_source": bleed[0], "level": bleed[1]},
        "tagger_notes": "hand-written for a test",
    }


def case(stem, model, m, present, quality, acceptable, retry):
    return {
        "stem": stem,
        "model_used": model,
        "metrics": m,
        "present": present,
        "answers": {
            "quality": quality,
            "acceptable": {"type": "noul", "noul": acceptable},
            "retry_action": retry,
        },
    }


CASES = {
    "good": case(
        "vocals",
        "mel_roformer_kim_vocals",
        metrics(-20.5, 0.2, ("none", "none")),
        True,
        score(3.4, 0.9, [0.0, 0.0, 0.1, 0.4, 0.5]),
        0.95,
        action("flag_for_human_review", 0.5),
    ),
    "at_the_thresholds": case(
        "bass",
        "htdemucs_ft",
        metrics(-24.0, 0.1, ("drums", "slight")),
        True,
        score(2.0, 0.8, [0.0, 0.1, 0.8, 0.1, 0.0]),
        0.8,
        action("retry_fallback_model", 0.6),
    ),
    "bleed": case(
        "drums",
        "scnet_xl_ihf",
        metrics(-18.0, 0.05, ("bass", "severe")),
        True,
        score(1.2, 0.85, [0.1, 0.65, 0.2, 0.05, 0.0]),
        0.06,
        action("retry_fallback_model", 0.9),
    ),
    "wants_dereverb": case(
        "vocals",
        "mel_roformer_kim_vocals",
        metrics(-22.0, 0.15, ("other", "moderate")),
        True,
        score(1.8, 0.82, [0.0, 0.3, 0.6, 0.1, 0.0]),
        0.1,
        action("retry_with_dereverb", 0.88),
    ),
    "unsure_action": case(
        "guitar",
        "bs_roformer_sw_6stem",
        metrics(-26.0, 0.3, ("piano", "moderate")),
        True,
        score(1.0, 0.9, [0.1, 0.8, 0.1, 0.0, 0.0]),
        0.05,
        action("retry_other_ensemble", 0.4),
    ),
    "hopeless": case(
        "piano",
        "bs_roformer_sw_6stem",
        metrics(-30.0, 0.4, ("guitar", "severe")),
        False,
        score(0.4, 0.92, [0.65, 0.3, 0.05, 0.0, 0.0]),
        0.03,
        action("flag_for_human_review", 0.9),
    ),
    "unsure_score": case(
        "other",
        "scnet_xl_ihf",
        metrics(-21.0, 0.1, ("guitar", "slight")),
        True,
        score(2.1, 0.55, [0.1, 0.2, 0.3, 0.3, 0.1]),
        0.9,
        action("retry_fallback_model", 0.9),
    ),
    "unsure_acceptable": case(
        "vocals",
        "bs_roformer_viperx_vocals",
        metrics(-19.0, 0.25, ("none", "none")),
        True,
        score(2.8, 0.9, [0.0, 0.0, 0.3, 0.6, 0.1]),
        0.6,
        action("retry_fallback_model", 0.9),
    ),
    "confident_no_despite_a_fair_score": case(
        "drums",
        "htdemucs_ft",
        metrics(-17.0, 0.02, ("none", "none")),
        True,
        score(2.6, 0.88, [0.0, 0.05, 0.4, 0.45, 0.1]),
        0.15,
        action("retry_other_ensemble", 0.85),
    ),
}
out = {
    "about": "Hand-written answers in Jev's wire format, for tests of how answers are mapped "
    "to a StemQualityAssessment. They are not recordings of Jev and say nothing about how "
    "Jev would rate a stem; real recordings come with P4.9. The request body is built by "
    "the test from the current question wording.",
    "origin": "handwritten",
    "cases": CASES,
}
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(path, len(CASES))
