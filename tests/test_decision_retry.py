"""The gate and retry state machine, driven by scripted assessments."""

from __future__ import annotations

import pytest

from stems_common.settings import load_defaults
from stems_worker.decision import gate
from stems_worker.decision.retry import DELIVERED, RUNNING, WARNED, GateError, Spec, StemGate

GATE = load_defaults().gate
BLENDS = {
    "two_model_blend": ("primary", "second"),
    "full_ensemble": ("primary", "second", "third"),
}
PASS = (3.0, 0.95, 0.9)
FAIL = (1.0, 0.05, 0.9)


def machine(
    stem="vocals", *, settings=GATE, fallback="fallback", blends=BLENDS, dereverb="dry", first=None
):
    return StemGate(
        first or Spec(stem, ("primary",)),
        settings=settings,
        fallback_model=fallback,
        blends=blends,
        dereverb_model=dereverb,
    )


def assessed(m: StemGate, score, p, confidence, suggested=None, reason="bleed") -> dict:
    """An assessment of the machine's current attempt, with the gate's verdict on it."""
    assessment = {
        "score": score,
        "pass_probability": p,
        "confidence": confidence,
        "reason_code": reason,
    }
    verdict = gate.verdict(
        assessment,
        settings=m.settings,
        suggested=suggested,
        available=m.available(),
        retries_used=m.retries_used,
    )
    return {
        "stem": m.stem,
        "model_used": m.current.model_used,
        "assessment": assessment,
        "verdict": verdict,
    }


def drive(m: StemGate, script) -> list:
    """Feed scripted (score, p, confidence[, suggested]) results until the stem is finished."""
    steps = []
    for result in script:
        steps.append(m.step(assessed(m, *result)))
        if m.state != RUNNING:
            break
    return steps


def test_a_passing_stem_is_delivered_at_once() -> None:
    m = machine()
    assert m.state == RUNNING
    assert m.retries_used == 0
    (step,) = drive(m, [PASS])
    assert (step.action, step.next, step.chosen) == ("deliver", None, 0)
    assert m.state == DELIVERED
    assert m.delivered.spec == Spec("vocals", ("primary",))
    assert m.warning is None
    assert m.retries_used == 0
    with pytest.raises(GateError, match="already delivered"):
        m.step(assessed(machine(), *PASS))


RETRIES = [
    # (stem, suggested by the assessor, action taken, the attempt it leads to)
    ("vocals", "retry_fallback_model", "retry_fallback_model", Spec("vocals", ("fallback",))),
    (
        "vocals",
        "retry_other_ensemble",
        "retry_other_ensemble",
        Spec("vocals", BLENDS["two_model_blend"], "two_model_blend"),
    ),
    (
        "vocals",
        "retry_with_dereverb",
        "retry_with_dereverb",
        Spec("vocals", ("primary",), dereverb="dry"),
    ),
    # No suggestion: the first possible action, in the fixed order.
    ("vocals", None, "retry_fallback_model", Spec("vocals", ("fallback",))),
    # De-reverb is for vocals only; another stem gets the next possible action.
    ("drums", "retry_with_dereverb", "retry_fallback_model", Spec("drums", ("fallback",))),
]


@pytest.mark.parametrize(("stem", "suggested", "action", "expected"), RETRIES)
def test_each_retry_action_leads_to_its_attempt(stem, suggested, action, expected) -> None:
    m = machine(stem)
    step = m.step(assessed(m, *FAIL, suggested))
    assert step.action == action
    assert step.next == expected == m.current
    assert step.chosen is None
    assert m.state == RUNNING
    assert m.retries_used == 1
    assert m.attempts[0].assessment["verdict"]["action"] == action
    # The retry passes: it is the one delivered.
    (done,) = drive(m, [PASS])
    assert (done.action, done.chosen) == ("deliver", 1)
    assert m.delivered.spec == expected
    assert m.state == DELIVERED


def test_the_verdict_names_the_model_a_retry_brings_in() -> None:
    m = machine()
    assert m.available() == {
        "retry_fallback_model": "fallback",
        "retry_other_ensemble": None,
        "retry_with_dereverb": "dry",
    }
    verdict = assessed(m, *FAIL, "retry_fallback_model")["verdict"]
    assert verdict["retry_model"] == "fallback"
    assert verdict["max_retries"] == GATE.max_retries_per_stem == 2
    assert assessed(m, *FAIL, "retry_with_dereverb")["verdict"]["retry_model"] == "dry"
    assert assessed(m, *FAIL, "retry_other_ensemble")["verdict"]["retry_model"] is None


def test_an_action_that_is_not_possible_gives_way_to_the_next() -> None:
    no_fallback = machine(fallback="none")
    assert "retry_fallback_model" not in no_fallback.available()
    step = no_fallback.step(assessed(no_fallback, *FAIL, "retry_fallback_model"))
    assert step.action == "retry_other_ensemble"
    assert step.next == Spec("vocals", BLENDS["two_model_blend"], "two_model_blend")

    # A fallback that is the model already running is not a different attempt.
    same = machine(fallback="primary", blends={}, dereverb=None)
    assert same.available() == {}
    (step,) = drive(same, [FAIL])
    assert (step.action, same.state) == ("deliver_with_warning", WARNED)

    # A "blend" of one model, or the blend already tried, is not offered.
    single = machine(
        "drums", blends={"two_model_blend": ("primary",), "full_ensemble": ("primary", "second")}
    )
    assert single.options()["retry_other_ensemble"].mode == "full_ensemble"
    twins = machine(
        "piano",
        fallback=None,
        blends={"two_model_blend": ("a", "b"), "full_ensemble": ("a", "b")},
        first=Spec("piano", ("a", "b"), "two_model_blend"),
    )
    assert twins.available() == {}


def test_the_retry_cap_is_two_and_then_the_stem_is_delivered_with_a_warning() -> None:
    m = machine()
    steps = drive(m, [FAIL, FAIL, FAIL, FAIL, FAIL])
    assert [s.action for s in steps] == [
        "retry_fallback_model",
        "retry_other_ensemble",
        "deliver_with_warning",
    ]
    assert len(m.attempts) == 3  # the first attempt and two retries, never a third
    assert m.retries_used == GATE.max_retries_per_stem == 2
    assert m.state == WARNED
    assert [a.spec.model_used for a in m.attempts] == ["primary", "fallback", "primary+second"]
    last = m.attempts[-1].assessment["verdict"]
    assert (last["deliver"], last["action"]) == (False, "deliver_with_warning")
    assert last["user_message"] == "Other instruments may be audible in this stem."
    assert m.warning == "Other instruments may be audible in this stem."
    with pytest.raises(GateError, match="already delivered_with_warning"):
        m.step(assessed(machine(), *FAIL))


@pytest.mark.parametrize(
    ("scores", "chosen"),
    [
        ([(1.0, 0.1), (1.8, 0.1), (1.2, 0.1)], 1),  # the best score, not the last attempt
        ([(1.9, 0.1), (1.0, 0.1), (0.5, 0.1)], 0),
        ([(1.0, 0.1), (1.0, 0.1), (1.5, 0.1)], 2),
        ([(1.5, 0.1), (1.5, 0.6), (1.5, 0.3)], 1),  # equal scores: the higher pass probability
        ([(1.5, 0.1), (1.5, 0.1), (1.5, 0.1)], 0),  # all equal: the earliest
    ],
)
def test_deliver_with_warning_chooses_the_best_attempt(scores, chosen) -> None:
    m = machine()
    steps = drive(m, [(score, p, 0.9) for score, p in scores])
    assert steps[-1].action == "deliver_with_warning"
    assert steps[-1].chosen == chosen == m.best()
    assert m.delivered is m.attempts[chosen]
    assert m.delivered.assessment["assessment"]["score"] == scores[chosen][0]


def test_a_later_attempt_that_passes_is_delivered_without_a_warning() -> None:
    m = machine()
    steps = drive(m, [FAIL, FAIL, PASS])
    assert [s.action for s in steps] == ["retry_fallback_model", "retry_other_ensemble", "deliver"]
    assert m.state == DELIVERED
    assert m.chosen == 2
    assert m.warning is None


def test_an_unverifiable_stem_is_warned_about_without_a_retry() -> None:
    m = machine()
    (step,) = drive(m, [(2.0, 0.5, 0.0, None, "low_confidence")])
    assert step.action == "deliver_with_warning"
    assert len(m.attempts) == 1
    assert m.warning == "This stem could not be quality-checked automatically."
    # Low confidence about a suspected problem: the problem is named, still no retry.
    unsure = machine()
    (step,) = drive(unsure, [(0.0, 0.0, 0.5, "retry_fallback_model", "bleed")])
    assert (step.action, len(unsure.attempts)) == ("deliver_with_warning", 1)
    assert unsure.warning == "Other instruments may be audible in this stem."


def test_flag_for_human_review_ends_in_a_warning() -> None:
    m = machine()
    (step,) = drive(m, [(*FAIL, "flag_for_human_review")])
    assert step.action == "deliver_with_warning"
    assert m.state == WARNED


@pytest.mark.parametrize(("cap", "attempts"), [(0, 1), (1, 2), (2, 3)])
def test_the_cap_comes_from_config(cap: int, attempts: int) -> None:
    m = machine(settings=GATE.model_copy(update={"max_retries_per_stem": cap}))
    steps = drive(m, [FAIL] * 5)
    assert len(m.attempts) == attempts
    assert steps[-1].action == "deliver_with_warning"
    assert m.attempts[0].assessment["verdict"]["max_retries"] == cap


def test_thresholds_come_from_config() -> None:
    result = (1.5, 0.75, 0.6)
    strict = machine()
    assert drive(strict, [result])[0].action == "deliver_with_warning"  # confidence under 0.7
    loose = machine(
        settings=GATE.model_copy(
            update={"pass_probability_min": 0.7, "score_min": 1.5, "confidence_min": 0.6}
        )
    )
    assert drive(loose, [result])[0].action == "deliver"


def test_running_out_of_things_to_try_before_the_cap() -> None:
    m = machine("drums", fallback="fallback", blends={}, dereverb=None)
    steps = drive(m, [FAIL, FAIL, FAIL])
    assert [s.action for s in steps] == ["retry_fallback_model", "deliver_with_warning"]
    assert len(m.attempts) == 2


def test_dereverb_stays_on_through_later_retries() -> None:
    m = machine()
    steps = drive(m, [(*FAIL, "retry_with_dereverb"), (*FAIL, "retry_fallback_model"), PASS])
    assert steps[1].next == Spec("vocals", ("fallback",), dereverb="dry")
    assert m.delivered.spec.model_used == "fallback>dry"
    assert Spec("vocals", ("a", "b"), "two_model_blend").model_used == "a+b"


def test_the_machine_refuses_what_its_states_do_not_allow() -> None:
    m = machine(settings=GATE.model_copy(update={"max_retries_per_stem": 0}))
    forged = assessed(m, *FAIL)
    forged["verdict"] = {**forged["verdict"], "action": "retry_fallback_model"}
    with pytest.raises(GateError, match="beyond the cap"):
        m.step(forged)
    m = machine("drums", fallback=None, blends={}, dereverb=None)
    forged = assessed(m, *FAIL)
    forged["verdict"] = {**forged["verdict"], "action": "retry_with_dereverb"}
    with pytest.raises(GateError, match="not possible from here"):
        m.step(forged)
    with pytest.raises(GateError, match="still being assessed"):
        _ = machine().delivered
    with pytest.raises(GateError, match="bad attempt"):
        Spec("drums", ())
    with pytest.raises(GateError, match="bad attempt"):
        Spec("drums", ("a",), "everything")
