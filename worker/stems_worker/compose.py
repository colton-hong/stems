"""Turn "these stems, from these models" into audio.

Given the stems a job asks for and which model serves each one, the composer:

1. plans the model runs, so a model that serves several stems runs once;
2. runs them one at a time, unloading each before the next;
3. builds the "other" stem;
4. checks how well the six stems add back up to the mix.

"Other" always means the same thing here: what is left of the mix once vocals, drums,
bass, guitar and piano are taken out. Models disagree about it. A 4-stem model's "other"
still holds guitar and piano, so the composer subtracts the guitar and piano stems from
it, and it will run the guitar and piano models to get them even if the job did not ask
for those stems. A 6-stem model's "other" is used as it is. The alternative, "residual",
ignores the model and takes the mix minus the other five stems.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from stems_common.stems import PRODUCT_STEMS
from stems_models.registry import Registry
from stems_worker.separator import DEFAULT_OVERLAP, Separator, run_separator

OtherMethod = Literal["auto", "direct", "subtract", "residual"]
_NAMED = tuple(s for s in PRODUCT_STEMS if s != "other")  # the five stems that are not "other"


class CompositionError(ValueError):
    """The request cannot be served by the models assigned to it."""


class SeparatorFactory(Protocol):
    def __call__(self, model_id: str, stems: tuple[str, ...]) -> Sequence[Separator]:
        """Separators that together emit ``stems`` for ``model_id`` (usually just one)."""


@dataclass(frozen=True)
class ModelRun:
    model_id: str
    # Outputs taken from this model, in registry vocabulary.
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class ComposePlan:
    requested: tuple[str, ...]
    runs: tuple[ModelRun, ...]
    # How "other" is built; None when "other" is neither requested nor needed.
    other_method: Literal["direct", "subtract", "residual"] | None
    # Stems computed only because something else needs them. Not delivered.
    dependencies: tuple[str, ...]
    # Which model supplies each stem that comes straight from a model.
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    model_id: str
    device: str
    outputs: tuple[str, ...]
    audio_seconds: float
    wall_seconds: float
    num_chunks: int


@dataclass(frozen=True)
class ResidualReport:
    """How well the six stems add back up to the mix.

    ``level_db`` is the RMS of ``mix - sum(stems)`` relative to the RMS of the mix: very
    negative is good, near zero means as much is missing or duplicated as the mix holds.
    """

    computed: bool
    level_db: float | None
    # Stems that came back as (near) silence. With a poor level this is the usual culprit.
    silent_stems: tuple[str, ...]
    ok: bool | None
    note: str


@dataclass(frozen=True)
class Composition:
    stems: dict[str, np.ndarray]  # the requested stems, in request order
    dependencies: dict[str, np.ndarray]
    runs: tuple[RunRecord, ...]
    residual: ResidualReport
    plan: ComposePlan


# --- planning -----------------------------------------------------------------------------


def plan_composition(
    requested: Sequence[str],
    assignment: dict[str, str],
    registry: Registry,
    other_method: OtherMethod = "auto",
) -> ComposePlan:
    """Decide which models run and what is taken from each.

    ``assignment`` maps a stem name to the registry id of the model that serves it.
    """
    requested = tuple(dict.fromkeys(requested))
    unknown = [s for s in requested if s not in PRODUCT_STEMS]
    if unknown or not requested:
        raise CompositionError(f"requested stems must be among {PRODUCT_STEMS}, got {requested}")

    needed: list[str] = [s for s in requested if s != "other"]
    method: Literal["direct", "subtract", "residual"] | None = None
    if "other" in requested:
        if other_method == "residual":
            method = "residual"
        else:
            model = _model_for("other", assignment, registry)
            natural = "subtract" if model.other_contains_guitar_piano else "direct"
            if other_method in ("direct", "subtract") and other_method != natural:
                raise CompositionError(
                    f"{model.id} emits a {'4' if natural == 'subtract' else '6'}-stem 'other'; "
                    f"other_method={other_method!r} does not fit it"
                )
            method = natural
        extra = {"direct": (), "subtract": ("guitar", "piano"), "residual": _NAMED}[method]
        needed += [s for s in extra if s not in needed]

    dependencies = tuple(s for s in needed if s not in requested)
    from_models = list(needed) + (["other"] if method in ("direct", "subtract") else [])

    sources: dict[str, str] = {}
    by_model: dict[str, list[str]] = {}
    for stem in from_models:
        model = _model_for(stem, assignment, registry)
        sources[stem] = model.id
        by_model.setdefault(model.id, []).append(stem)
    runs = tuple(ModelRun(model_id, tuple(outputs)) for model_id, outputs in by_model.items())
    return ComposePlan(requested, runs, method, dependencies, sources)


def _model_for(stem: str, assignment: dict[str, str], registry: Registry):
    try:
        model_id = assignment[stem]
    except KeyError:
        raise CompositionError(f"no model is assigned to {stem!r}") from None
    if model_id not in registry:
        raise CompositionError(f"{stem}: {model_id!r} is not in the registry")
    model = registry.get(model_id)
    if not model.is_loadable:
        raise CompositionError(f"{stem}: {model_id} has status {model.status!r}")
    if stem not in model.stems:
        raise CompositionError(f"{stem}: {model_id} does not emit it (emits {model.stems})")
    return model


# --- running ------------------------------------------------------------------------------


def _level_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def residual_report(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    *,
    other_method: str | None,
    warn_db: float,
    silent_db: float,
) -> ResidualReport:
    missing = [s for s in PRODUCT_STEMS if s not in stems]
    if missing:
        return ResidualReport(False, None, (), None, f"needs all six stems; missing {missing}")
    mix_db = _level_db(mix)
    if mix_db < -100:
        return ResidualReport(False, None, (), None, "the mix is silent")
    silent = tuple(s for s in PRODUCT_STEMS if _level_db(stems[s]) - mix_db < silent_db)
    if other_method == "residual":
        note = "'other' is the residual by construction, so the sum says nothing"
        return ResidualReport(False, None, silent, None, note)
    total = np.zeros_like(mix, dtype=np.float64)
    for name in PRODUCT_STEMS:
        total += stems[name]
    level = _level_db(mix - total) - mix_db
    ok = level <= warn_db
    if ok:
        note = "stems add up to the mix"
    elif silent:
        note = f"stems do not add up to the mix; silent: {', '.join(silent)}"
    else:
        note = "stems do not add up to the mix: content is missing or duplicated"
    return ResidualReport(True, level, silent, ok, note)


def compose(
    mix: np.ndarray,
    sample_rate: int,
    plan: ComposePlan,
    factory: SeparatorFactory,
    *,
    device: str,
    overlap: float = DEFAULT_OVERLAP,
    batch_size: int = 1,
    residual_warn_db: float = -10.0,
    silent_stem_db: float = -60.0,
    progress: Callable[[str, int, int], None] | None = None,
) -> Composition:
    """Run the plan on ``mix`` (stereo, ``(2, samples)``) and assemble the stems.

    ``progress(model_id, chunks_done, chunks_total)`` is called as each model works.
    """
    produced: dict[str, np.ndarray] = {}
    records: list[RunRecord] = []
    for run in plan.runs:
        separators = list(factory(run.model_id, run.outputs))
        offered = [name for s in separators for name in s.outputs]
        lacking = [name for name in run.outputs if name not in offered]
        if lacking:
            raise CompositionError(f"{run.model_id} did not offer {lacking} (offered {offered})")
        for separator in separators:
            wanted = tuple(name for name in run.outputs if name in separator.outputs)
            if not wanted:
                continue
            separator.load(device)
            try:
                result = run_separator(
                    separator,
                    mix,
                    sample_rate,
                    overlap=overlap,
                    batch_size=batch_size,
                    progress=(
                        None
                        if progress is None
                        else lambda done, total, _id=run.model_id: progress(_id, done, total)
                    ),
                )
            finally:
                separator.unload()
            for name in wanted:
                produced[name] = result.stems[name]
            records.append(
                RunRecord(
                    run.model_id,
                    result.device,
                    wanted,
                    result.audio_seconds,
                    result.wall_seconds,
                    result.num_chunks,
                )
            )

    if plan.other_method == "subtract":
        produced["other"] = produced["other"] - produced["guitar"] - produced["piano"]
    elif plan.other_method == "residual":
        rest = np.zeros_like(mix, dtype=np.float32)
        for name in _NAMED:
            rest += produced[name]
        produced["other"] = mix.astype(np.float32) - rest

    report = residual_report(
        mix,
        produced,
        other_method=plan.other_method,
        warn_db=residual_warn_db,
        silent_db=silent_stem_db,
    )
    return Composition(
        stems={name: produced[name] for name in plan.requested},
        dependencies={name: produced[name] for name in plan.dependencies},
        runs=tuple(records),
        residual=report,
        plan=plan,
    )
