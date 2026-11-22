"""Cross-checks between config/defaults.yaml and the model registry."""

from __future__ import annotations

from stems_common.settings import Defaults
from stems_models.registry import Registry


def check_defaults(defaults: Defaults, registry: Registry) -> list[str]:
    """Return a list of problems (empty if every configured model can fill its slot)."""
    problems: list[str] = []
    for slot, models in defaults.ensembles.slots.items():
        ids = {models.single_best, *models.two_model_blend, *models.full_ensemble}
        for model_id in sorted(ids):
            if model_id not in registry:
                problems.append(f"{slot}: {model_id!r} is not in the registry")
                continue
            entry = registry.get(model_id)
            if not entry.is_loadable:
                problems.append(f"{slot}: {model_id!r} has status {entry.status!r}")
            if slot not in entry.candidate_for:
                problems.append(f"{slot}: {model_id!r} is not a registered candidate for it")
        if models.single_best not in models.two_model_blend:
            problems.append(f"{slot}: two_model_blend should include the single_best model")
        if not set(models.two_model_blend) <= set(models.full_ensemble):
            problems.append(f"{slot}: full_ensemble should include both blend models")
        if slot == "other":
            # Averaging a four-stem "other" (guitar and piano still in it) with a six-stem
            # one would average two different things.
            for mode in ("two_model_blend", "full_ensemble"):
                kinds = {
                    registry.get(m).other_contains_guitar_piano
                    for m in getattr(models, mode)
                    if m in registry
                }
                if len(kinds) > 1:
                    problems.append(f"other: {mode} mixes four-stem and six-stem 'other' models")
    return problems
