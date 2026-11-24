"""The quality gate: whether an assessed stem is delivered, retried or delivered with a warning.

The rule is project decision 8, with its numbers in ``config/defaults.yaml`` (``gate``):

    deliver = pass_probability >= 0.8 and score >= 2 and confidence >= 0.7

A stem that does not pass is retried only when two things hold: the assessment is
confident that something is wrong (``confidence`` at least ``confidence_min``), and there
is still something different to try within the retry cap. Otherwise it is delivered with
a warning. In particular a stem that could not be checked at all (no tagger ran, or the
tagger's thresholds are uncalibrated) is delivered with a warning at once: running another
model would produce a stem that is just as unverifiable.

There is no human review queue in v1, so ``flag_for_human_review`` from Jev becomes
``deliver_with_warning`` too.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stems_common.settings import GateSettings

# In the order they are tried when the suggested one is not possible.
RETRY_ACTIONS: tuple[str, ...] = (
    "retry_fallback_model",
    "retry_other_ensemble",
    "retry_with_dereverb",
)
FLAG = "flag_for_human_review"
DELIVER = "deliver"
WARN = "deliver_with_warning"

_WARNINGS = {
    "mostly_silent": (
        "This stem is silent or very quiet compared with the song. The instrument may not "
        "be in the song, or it was not found."
    ),
    "bleed": "Other instruments may be audible in this stem.",
    "wrong_content": "This stem may not contain the instrument it is named after.",
    "artifacts": "This stem may contain audible artifacts.",
    "muffled_or_dull": "This stem may sound muffled.",
    "low_confidence": "This stem could not be quality-checked automatically.",
}
_GENERIC = "This stem did not pass the automatic quality check."


def passes(assessment: Mapping[str, Any], settings: GateSettings) -> bool:
    """The gate rule. All three thresholds are inclusive."""
    return (
        assessment["pass_probability"] >= settings.pass_probability_min
        and assessment["score"] >= settings.score_min
        and assessment["confidence"] >= settings.confidence_min
    )


def warning(assessment: Mapping[str, Any], settings: GateSettings) -> str:
    """What the user is told about a stem that is delivered without having passed."""
    if assessment["confidence"] < settings.confidence_min:
        # The gate does not trust this assessment enough to act on it, so the user is not
        # told what it suspects either. The suspicion stays in the log as reason_code.
        return _WARNINGS["low_confidence"]
    return _WARNINGS.get(assessment.get("reason_code", "none"), _GENERIC)


def verdict(
    assessment: Mapping[str, Any],
    *,
    settings: GateSettings,
    suggested: str | None = None,
    available: Mapping[str, str | None] | None = None,
    retries_used: int = 0,
) -> dict[str, Any]:
    """The ``verdict`` object of schema (c) for one assessed attempt.

    ``suggested`` is the retry action the assessor prefers. ``available`` maps each retry
    action that would produce a different attempt to the model it would use (None when
    the action does not change the model). ``retries_used`` counts retries already made
    for this stem.
    """
    available = dict(available or {})
    result: dict[str, Any] = {
        "deliver": False,
        "action": WARN,
        "retry_model": None,
        "max_retries": settings.max_retries_per_stem,
    }
    if passes(assessment, settings):
        return {**result, "deliver": True, "action": DELIVER}
    message = warning(assessment, settings)
    if assessment["confidence"] < settings.confidence_min:
        # Not sure enough that anything is wrong to spend a retry on it.
        return {**result, "user_message": message}
    if retries_used >= settings.max_retries_per_stem or suggested == FLAG:
        return {**result, "user_message": message}
    order = [a for a in (suggested, *RETRY_ACTIONS) if a in RETRY_ACTIONS]
    for action in dict.fromkeys(order):
        if action in available:
            return {**result, "action": action, "retry_model": available[action]}
    return {**result, "user_message": message}
