"""Decision 2 of 3: which models produce each requested stem (``EnsemblePlan``, schema (b)).

With Jev: for each requested stem, a choice among the registry models that can produce
it; when de-reverb was asked for, a choice of de-reverb model; and a yes/no on whether
blending several models is worth the extra time for this track. Each option is described
by its registry name and by its place in ``config/defaults.yaml``, which is where the
benchmark's result lives, so the choice is put to Jev with the measured default marked.

Without Jev, or for any answer Jev is not sure of, the plan is the configured default:
the slot's ``single_best`` model, with the next configured model as its fallback. An
unsure answer never fails the plan; it is replaced, ``needs_review`` is set, and the
record says which answer and how unsure.

The report's rules are enforced on every plan (``check_rules``): a stem that was not
requested has no models; a requested stem has a primary model, and a fallback that differs
from it.

What is not decided here: nothing the user did not ask for is added. De-reverb is planned
only on request (the gate may still add it as a retry), and the lead/backing split is
taken from the parsed request, since schema (b) has no place for it.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from stems_common import schemas
from stems_common.settings import Defaults
from stems_common.stems import PRODUCT_STEMS
from stems_models import plan_schema
from stems_models.registry import ModelEntry, Registry
from stems_worker.compose import Blend, as_blend, plan_composition
from stems_worker.decision import jev
from stems_worker.decision.jev import JevClient
from stems_worker.decision.parse import wanted_stems
from stems_worker.decision.record import (
    FALLBACK,
    LOW_CONFIDENCE,
    NOT_NEEDED,
    Asked,
    Decision,
    ask,
    record,
)
from stems_worker.pipeline import default_assignment
from stems_worker.timing import DEFAULT_TIMINGS_PATH, read_timings, summarise

KIND = "plan"
NONE = "none"


class PlanError(ValueError):
    """The request cannot be planned, or a plan breaks the schema or the report's rules."""


# --- rules --------------------------------------------------------------------------------


def check_rules(plan: Mapping[str, Any]) -> list[str]:
    """Every way ``plan`` breaks the requested/primary/fallback rules; empty when it does not."""
    problems: list[str] = []
    for stem in PRODUCT_STEMS:
        entry = plan["plan"][stem]
        primary, fallback = entry["primary_model"], entry["fallback_model"]
        if not entry["requested"]:
            if primary != NONE or fallback != NONE:
                problems.append(f"{stem}: not requested, so both models must be 'none'")
        elif primary == NONE:
            problems.append(f"{stem}: requested, so primary_model must not be 'none'")
        elif fallback == primary:
            problems.append(f"{stem}: fallback_model must differ from primary_model")
    dereverb = plan.get("dereverb")
    if dereverb is not None and dereverb["apply"] != (dereverb["model"] is not None):
        problems.append("dereverb: a model is named exactly when apply is true")
    return problems


def validate(plan: Mapping[str, Any], registry: Registry) -> None:
    """Raise ``PlanError`` unless ``plan`` fits the schema for ``registry`` and the rules."""
    problems = schemas.errors_in(plan_schema.ensemble_plan_schema(registry), plan)
    problems += check_rules(plan) if not problems else []
    if problems:
        raise PlanError("not a valid EnsemblePlan: " + "; ".join(problems))


# --- questions ----------------------------------------------------------------------------


def _describe(entry: ModelEntry, slot: str, defaults: Defaults) -> str:
    models = defaults.ensembles.slots[slot]
    measured = (
        "provisional, not yet benchmarked" if defaults.ensembles.provisional else "benchmarked"
    )
    if entry.id == models.single_best:
        role = f"the configured default for {slot} ({measured})"
    elif entry.id in models.two_model_blend:
        role = f"in the configured two-model blend for {slot} ({measured})"
    elif entry.id in models.full_ensemble:
        role = f"in the configured full ensemble for {slot} ({measured})"
    else:
        role = "a registered candidate, not in the configured defaults"
    return f"{entry.name}; {entry.architecture}; {role}"


def questions(
    requested: Sequence[str],
    registry: Registry,
    defaults: Defaults,
    *,
    dereverb: bool,
    quality_target: str,
) -> dict[str, dict[str, Any]]:
    """The questions for one plan. A stem with a single candidate needs no question."""
    asked: dict[str, dict[str, Any]] = {}
    for stem in requested:
        candidates = registry.candidates_for(stem)
        if len(candidates) > 1:
            asked[f"model.{stem}"] = jev.choice(
                f"Which model should separate the {stem} stem of this track?",
                {m.id: _describe(m, stem, defaults) for m in candidates},
            )
    candidates = registry.candidates_for("dereverb")
    if dereverb and len(candidates) > 1:
        asked["dereverb"] = jev.choice(
            "Which model should remove the reverberation from the vocals of this track?",
            {m.id: _describe(m, "dereverb", defaults) for m in candidates},
        )
    if quality_target != "fastest":
        asked["ensemble"] = jev.noul(
            "Is blending the outputs of several models worth the extra processing time "
            "for this track?",
            yes="The track is dense or difficult enough that a blend should be clearly better.",
            no="One good model should do about as well.",
        )
    return asked


def state(
    track_features: Mapping[str, Any],
    requested: Sequence[str],
    *,
    quality_target: str,
    dereverb: bool,
    vocal_split: bool,
) -> dict[str, Any]:
    return {
        "track_features": dict(track_features),
        "requested_stems": list(requested),
        "quality_target": quality_target,
        "dereverb_requested": dereverb,
        "vocal_split_requested": vocal_split,
    }


# --- time estimate ------------------------------------------------------------------------


def measured_rates(path: Any = DEFAULT_TIMINGS_PATH) -> dict[tuple[str, str], float]:
    """Median seconds of work per second of audio, per (model, device), from the timing log."""
    return {
        (row["model_id"], row["device"]): row["median_realtime_factor"]
        for row in summarise(read_timings(path))
    }


def estimate_seconds(
    model_ids: Sequence[str],
    duration_seconds: float,
    device: str,
    rates: Mapping[tuple[str, str], float],
) -> tuple[float, list[str]]:
    """Separation time from measured rates, and the models that have no measurement.

    Unmeasured models add nothing, so with any of them the figure is a lower bound, and
    the plan's record says so. Nothing is assumed about a model that was never timed.
    """
    total, unmeasured = 0.0, []
    for model_id in dict.fromkeys(model_ids):
        rate = rates.get((model_id, device))
        if rate is None:
            unmeasured.append(model_id)
        else:
            total += rate * duration_seconds
    return round(total, 1), unmeasured


# --- the plan in the pipeline's terms -----------------------------------------------------


def assignment_for(plan: Mapping[str, Any], defaults: Defaults) -> dict[str, str | Blend]:
    """Who serves each slot under ``plan``, as ``plan_composition`` takes it.

    In ``single_best`` mode a requested stem is served by the plan's primary model. In a
    blend mode it is served by the configured blend for that mode, because the members and
    weights of a blend are what the benchmark tunes. Everything the plan does not name
    (stems needed only as inputs, the vocal split) keeps its configured default.
    """
    mode = plan.get("ensemble_mode") or defaults.ensembles.default_mode
    assignment = default_assignment(defaults, mode)
    if mode == "single_best":
        for stem in PRODUCT_STEMS:
            if plan["plan"][stem]["requested"]:
                assignment[stem] = plan["plan"][stem]["primary_model"]
    dereverb = plan.get("dereverb") or {}
    if dereverb.get("apply"):
        assignment["dereverb"] = dereverb["model"]
    return assignment


def models_to_run(
    plan: Mapping[str, Any],
    defaults: Defaults,
    registry: Registry,
    *,
    vocal_split: bool = False,
) -> list[str]:
    """Every model the plan runs, once each, in the order they run."""
    assignment = assignment_for(plan, defaults)
    requested = [s for s in PRODUCT_STEMS if plan["plan"][s]["requested"]]
    dereverb = bool((plan.get("dereverb") or {}).get("apply"))
    stages = [s for s, on in (("vocal_split", vocal_split), ("dereverb", dereverb)) if on]
    composition = plan_composition(
        requested,
        assignment,
        registry,
        defaults.separation.other_method,
        also_compute=("vocals",) if stages else (),
    )
    ids = [run.model_id for run in composition.runs]
    for stage in stages:
        ids += as_blend(assignment[stage]).models
    return list(dict.fromkeys(ids))


# --- assembling ---------------------------------------------------------------------------


def _config_fallback(stem: str, primary: str, defaults: Defaults, registry: Registry) -> str:
    """The next model to try after ``primary``: configured ones first, then any candidate."""
    slot = defaults.ensembles.slots[stem]
    candidates = [m.id for m in registry.candidates_for(stem)]
    for model_id in (slot.single_best, *slot.two_model_blend, *slot.full_ensemble, *candidates):
        if model_id != primary and model_id in candidates:
            return model_id
    return NONE


def config_fallback(stem: str, defaults: Defaults, registry: Registry) -> str:
    """The model the rules name as the fallback for ``stem``'s configured default."""
    return _config_fallback(stem, defaults.ensembles.slots[stem].single_best, defaults, registry)


def plan_ensemble(
    parsed: Mapping[str, Any],
    track_features: Mapping[str, Any],
    *,
    client: JevClient | None,
    defaults: Defaults,
    registry: Registry,
    job_id: str | None = None,
    dereverb: bool = False,
    device: str = "cpu",
    rates: Mapping[tuple[str, str], float] | None = None,
) -> Decision:
    """Plan the separation of one track for one parsed request.

    ``device`` and ``rates`` (default: the timing log) are only used for
    ``estimated_seconds``.
    """
    requested = wanted_stems(dict(parsed))
    if not requested:
        raise PlanError("no stem is requested, so there is nothing to plan")
    floor = defaults.decision.min_confidence
    quality = parsed.get("quality_mode", "unspecified")
    quality_target = "balanced" if quality == "unspecified" else quality
    vocal_split = parsed.get("vocal_split") == "lead_and_backing"
    for slot in (*requested, *(("dereverb",) if dereverb else ())):
        if not registry.candidates_for(slot):
            raise PlanError(f"no loadable model in the registry can serve {slot!r}")

    asked_questions = questions(
        requested, registry, defaults, dereverb=dereverb, quality_target=quality_target
    )
    request_state = state(
        track_features,
        requested,
        quality_target=quality_target,
        dereverb=dereverb,
        vocal_split=vocal_split,
    )
    # With one candidate per stem and no mode to choose, there is nothing to ask.
    asked = (
        ask(client, KIND, request_state, asked_questions)
        if asked_questions
        else Asked(None, NOT_NEEDED, FALLBACK)
    )
    answers = asked.answers or {}
    notes: list[str] = []
    seen: list[float] = []  # confidence of every answer Jev gave
    unsure: list[str] = []
    used = 0

    def trusted(qid: str) -> dict[str, Any] | None:
        nonlocal used
        answer = answers.get(qid)
        if answer is None:
            return None
        seen.append(answer["confidence"])
        if answer["confidence"] < floor:
            unsure.append(f"{qid} {answer['confidence']:.2f}")
            return None
        used += 1
        return answer

    entries: dict[str, dict[str, Any]] = {}
    for stem in PRODUCT_STEMS:
        if stem not in requested:
            entries[stem] = {"requested": False, "primary_model": NONE, "fallback_model": NONE}
            continue
        default = defaults.ensembles.slots[stem].single_best
        answer = trusted(f"model.{stem}")
        if answer is None:
            if f"model.{stem}" in answers:
                notes.append(f"{stem}: Jev was unsure, so the configured default {default} is used")
            entries[stem] = {
                "requested": True,
                "primary_model": default,
                "fallback_model": _config_fallback(stem, default, defaults, registry),
            }
            continue
        primary = answer["choice"]
        others = {m: p for m, p in answer["probabilities"].items() if m != primary}
        entries[stem] = {
            "requested": True,
            "primary_model": primary,
            # Jev's second choice.
            "fallback_model": max(others, key=others.__getitem__) if others else NONE,
            "probability": answer["probabilities"][primary],
            "confidence": answer["confidence"],
        }

    mode = defaults.decision.mode_for_quality[quality_target]
    answer = trusted("ensemble")
    if answer is not None:
        if answer["probability"] < 0.5:
            mode = "single_best"
        else:
            mode = "full_ensemble" if quality_target == "best_quality" else "two_model_blend"
    elif "ensemble" in answers:
        notes.append(f"ensemble: Jev was unsure, so the configured mode {mode} is used")

    dereverb_plan: dict[str, Any] = {"apply": False, "model": None}
    if dereverb:
        default = defaults.ensembles.slots["dereverb"].single_best
        answer = trusted("dereverb")
        if answer is None and "dereverb" in answers:
            notes.append(f"dereverb: Jev was unsure, so the configured default {default} is used")
        dereverb_plan = {"apply": True, "model": default if answer is None else answer["choice"]}

    plan: dict[str, Any] = {
        "job_id": job_id or str(uuid.uuid4()),
        "track_features": dict(track_features),
        "quality_target": quality_target,
        "plan": entries,
        "dereverb": dereverb_plan,
        "ensemble_mode": mode,
        "estimated_seconds": 0.0,
        # The lowest confidence Jev gave, whether or not that answer was used. A plan made
        # by the rules alone is the configuration, applied exactly.
        "confidence": min(seen) if seen else 1.0,
        "needs_review": bool(unsure),
    }
    ids = models_to_run(plan, defaults, registry, vocal_split=vocal_split)
    seconds, unmeasured = estimate_seconds(
        ids,
        float(track_features["duration_seconds"]),
        device,
        measured_rates() if rates is None else rates,
    )
    plan["estimated_seconds"] = seconds
    if unmeasured:
        notes.append(
            f"estimated_seconds is a lower bound: no timing on {device} for "
            + ", ".join(unmeasured)
        )
    validate(plan, registry)

    if asked.answers is None:
        reason = asked.failure
    elif used == 0 and unsure:
        reason = f"{LOW_CONFIDENCE}: {', '.join(unsure)} (minimum {floor:g})"
    else:
        reason = None
        if unsure:
            notes.insert(0, f"{LOW_CONFIDENCE}: {', '.join(unsure)} (minimum {floor:g})")
    return Decision(KIND, plan, record(KIND, asked, fallback_reason=reason, notes=notes))
