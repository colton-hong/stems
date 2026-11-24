"""Decision 3 of 3: is this output stem good enough (``StemQualityAssessment``, schema (c)).

Jev never hears the stem. It is sent the stem's measured levels and the tagger's signals
(the ``metrics`` object, plus whether the stem's own instrument was heard in it) and
answers three questions: a score on the 0 to 4 rubric, whether the stem is acceptable to
deliver, and which retry it would pick.

- ``score`` is Jev's score, which may be fractional; ``score_label`` is its most probable
  level; ``confidence`` is the score answer's confidence.
- ``pass_probability`` is the yes/no answer. That answer has no confidence of its own
  (docs/corrections.md, C1), so ``max(p, 1 - p)`` stands in for one: an answer between 0.2
  and 0.8 is not acted on.
- If either is under ``decision.min_confidence``, the rule-based scorer decides instead.

The rule-based scorer reads the same signals and nothing else: how much of the stem is
silent, how far its level is below the mix, the bleed level the tagger estimated, and
whether the stem's own source was heard. It never uses SDR, SIR or SAR, which exist only
where there is a reference stem (never for an upload). Its numbers are in
``config/defaults.yaml`` (``gate.rules``).

Two of those signals are level measurements and are acted on as they are: a silent stem,
and a stem far below the mix (measured on real separator output,
docs/gate-on-real-output.md). The other two are the tagger's, each with a switch in
``listening.calibrated``; a verdict that rests on a signal whose switch is off carries the
low ``uncalibrated_confidence``, which the gate does not act on. With no tagger at all
the scorer says so (``low_confidence``) instead of inventing a judgement.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from stems_common import schemas
from stems_common.settings import Calibrated, Defaults, GateSettings
from stems_common.stems import PRODUCT_STEMS
from stems_worker.decision import gate, jev
from stems_worker.decision.jev import JevClient
from stems_worker.decision.record import LOW_CONFIDENCE, Decision, ask, record

KIND = "gate"
# Schema (c): "these five strings are the Jev score criteria, in order".
LABELS: tuple[str, ...] = ("unusable", "poor", "acceptable", "good", "excellent")
HEARD = {True: "yes", False: "no", None: "uncertain"}


def questions(stem: str) -> dict[str, dict[str, Any]]:
    return {
        "quality": jev.score(
            f"Rate this separated {stem} stem from the measurements in the state. 0 unusable: "
            "silent, or mostly something else. 1 poor: heavy bleed or clearly damaged. "
            "2 acceptable: usable, with minor bleed or artifacts. 3 good: clean, small "
            "flaws only. 4 excellent: nothing wrong that the measurements show.",
            LABELS,
        ),
        "acceptable": jev.noul(
            f"Is this {stem} stem acceptable to deliver to the user as it is?",
            yes="It contains its own instrument and little else.",
            no="It is silent, mostly another source, or heavily contaminated.",
        ),
        "retry_action": jev.choice(
            "If the stem were not delivered as it is, what should be tried next?",
            {
                "retry_fallback_model": "Separate it again with a different model",
                "retry_other_ensemble": "Separate it again with a blend of several models",
                "retry_with_dereverb": "Remove reverberation from it (vocals only)",
                gate.FLAG: "Nothing automatic is likely to help",
            },
        ),
    }


def state(
    stem: str,
    model_used: str,
    metrics: Mapping[str, Any],
    *,
    present: bool | None,
    tagged: bool,
    calibrated: Calibrated,
    retries_used: int,
    below_mix_db: float | None = None,
) -> dict[str, Any]:
    """What Jev is told about the stem. Missing measurements stay null; none is filled in."""
    return {
        "stem": stem,
        "model_used": model_used,
        "retries_so_far": retries_used,
        "metrics": dict(metrics),
        # The stem's level against the mix, over the whole song. Far below it
        # (see gate.rules.empty_below_mix_db) the instrument is probably not in the song.
        "level_below_mix_db": None if below_mix_db is None else round(below_mix_db, 2),
        "own_source_heard": HEARD[present] if tagged else "not checked",
        "tagger_thresholds_calibrated": calibrated.model_dump(),
    }


# --- the rule-based scorer ----------------------------------------------------------------


def is_empty(below_mix_db: float | None, settings: GateSettings) -> bool:
    """Is the stem so far below the mix that the instrument is probably not in the song?"""
    limit = settings.rules.empty_below_mix_db
    return limit is not None and below_mix_db is not None and below_mix_db <= limit


def reason_code(
    metrics: Mapping[str, Any],
    present: bool | None,
    settings: GateSettings,
    below_mix_db: float | None = None,
) -> str:
    """The main thing the signals say about a stem, as a schema ``reason_code``."""
    bleed = metrics.get("bleed_estimate")
    if metrics["silence_fraction"] >= settings.rules.silent_fraction_min:
        return "mostly_silent"
    if is_empty(below_mix_db, settings):
        # Leftovers, not silence, but the same finding: nothing of the instrument is there.
        return "mostly_silent"
    if bleed is None and present is None:
        return "low_confidence"  # nothing listened to it
    if present is False:
        return "wrong_content"
    if bleed is not None and bleed["level"] != "none":
        return "bleed"
    return "clean" if present else "none"


def rule_assessment(
    metrics: Mapping[str, Any],
    *,
    present: bool | None,
    settings: GateSettings,
    calibrated: Calibrated,
    below_mix_db: float | None = None,
) -> tuple[dict[str, Any], str | None]:
    """The ``assessment`` object from the signals alone, and the retry the rules suggest.

    SDR, SIR and SAR are not read: a stem is judged the same with or without a reference.
    ``below_mix_db`` is the stem's level against the mix (``level_below_mix_db``).
    """
    rules = settings.rules
    reason = reason_code(metrics, present, settings, below_mix_db)
    if reason == "mostly_silent":
        # A level measurement, not a tagger's opinion: certain whatever is calibrated.
        score, passed, confidence = 0.0, 0.0, 1.0
    elif reason == "low_confidence":
        score, passed, confidence = rules.unchecked_score, 0.5, 0.0
    else:
        bleed = metrics.get("bleed_estimate")
        score = rules.bleed_scores["none" if bleed is None else bleed["level"]]
        if present is False:
            score = min(score, rules.not_heard_score)
        elif present is None:
            score = min(score, rules.unchecked_score)
        passed = 1.0 if score >= settings.score_min and present is not False else 0.0
        # The verdict is trusted only if every tagger signal it rests on is switched on:
        # "the instrument is not here" rests on presence, "something else is here" on
        # bleed, and a clean bill of health on both.
        trusted = {"wrong_content": calibrated.presence, "bleed": calibrated.bleed}
        confidence = 1.0 if trusted.get(reason, calibrated.all) else rules.uncalibrated_confidence
    assessment = {
        "score": float(score),
        "score_label": LABELS[min(max(round(score), 0), len(LABELS) - 1)],
        "pass_probability": passed,
        "confidence": confidence,
        "reason_code": reason,
    }
    suggested = "retry_fallback_model" if passed == 0.0 else None
    if reason == "mostly_silent" and not rules.retry_empty:
        # Another model's answer to "there is nothing here" is leftovers more often than
        # the instrument (docs/gate-on-real-output.md): deliver it with its note instead.
        suggested = gate.FLAG
    return assessment, suggested


# --- assembling ---------------------------------------------------------------------------


def assess_stem(
    stem: str,
    model_used: str,
    metrics: Mapping[str, Any],
    *,
    client: JevClient | None,
    defaults: Defaults,
    job_id: str | None = None,
    present: bool | None = None,
    below_mix_db: float | None = None,
    retries_used: int = 0,
    available: Mapping[str, str | None] | None = None,
) -> Decision:
    """Assess one attempt at one stem and say what the gate does with it.

    ``metrics`` is the ``metrics`` object of schema (c) (``StemSignals.metrics()``);
    ``present`` is whether the tagger heard the stem's own source in it; ``below_mix_db``
    is the stem's level against the mix over the whole song. ``retries_used``
    and ``available`` are what the gate needs to choose between a retry and a warning
    (``gate.verdict``).
    """
    if stem not in PRODUCT_STEMS:
        raise ValueError(f"unknown stem {stem!r}")
    schemas.validate("stem_metrics", dict(metrics))
    settings, floor = defaults.gate, defaults.decision.min_confidence
    calibrated = defaults.listening.calibrated
    tagged = metrics.get("bleed_estimate") is not None or present is not None
    by_rules, rule_action = rule_assessment(
        metrics,
        present=present,
        settings=settings,
        calibrated=calibrated,
        below_mix_db=below_mix_db,
    )
    asked = ask(
        client,
        KIND,
        state(
            stem,
            model_used,
            metrics,
            present=present,
            tagged=tagged,
            calibrated=calibrated,
            retries_used=retries_used,
            below_mix_db=below_mix_db,
        ),
        questions(stem),
    )
    notes: list[str] = []
    answers = asked.answers
    if answers is None:
        reason, assessment, suggested = asked.failure, by_rules, rule_action
    else:
        quality, acceptable, action = (
            answers["quality"],
            answers["acceptable"],
            answers["retry_action"],
        )
        unsure = [
            f"{name} {answer['confidence']:.2f}"
            for name, answer in (("quality", quality), ("acceptable", acceptable))
            if answer["confidence"] < floor
        ]
        if unsure:
            reason = f"{LOW_CONFIDENCE}: {', '.join(unsure)} (minimum {floor:g})"
            assessment, suggested = by_rules, rule_action
        else:
            reason = None
            assessment = {
                "score": quality["score"],
                "score_label": LABELS[quality["top_level"]],
                "pass_probability": acceptable["probability"],
                "confidence": quality["confidence"],
            }
            # What is wrong is read from the signals; Jev is not asked to name it.
            code = by_rules["reason_code"]
            if gate.passes(assessment, settings):
                code = "clean" if code == "clean" else "none"
            elif code == "low_confidence":
                code = "none"
            assessment["reason_code"] = code
            if action["confidence"] >= floor:
                suggested = action["choice"]
            else:
                suggested = rule_action
                notes.append(
                    f"retry_action: Jev was unsure ({action['confidence']:.2f}), so the "
                    f"rules' suggestion is used ({rule_action})"
                )
    if by_rules["reason_code"] == "mostly_silent" and not settings.rules.retry_empty:
        if suggested in gate.RETRY_ACTIONS:
            notes.append(f"{suggested} was suggested for an empty stem and is not done")
        suggested = gate.FLAG
    if not calibrated.all and tagged:
        notes.append("the tagger's thresholds are uncalibrated")
    value = {
        "job_id": job_id or str(uuid.uuid4()),
        "stem": stem,
        "model_used": model_used,
        "metrics": dict(metrics),
        "assessment": assessment,
        "verdict": gate.verdict(
            assessment,
            settings=settings,
            suggested=suggested,
            available=available,
            retries_used=retries_used,
        ),
    }
    schemas.validate("stem_quality_assessment", value)
    log = record(KIND, asked, fallback_reason=reason, notes=notes)
    return Decision(KIND, value, {**log, "stem": stem, "suggested_action": suggested})
