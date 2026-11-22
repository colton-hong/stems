"""config/defaults.yaml is valid and only names models the registry can supply."""

from __future__ import annotations

import pytest
import yaml

from stems_common.paths import DEFAULTS_PATH
from stems_common.settings import ENSEMBLE_SLOTS, Defaults, load_defaults
from stems_models.registry import load_registry
from stems_models.selection import check_defaults


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))


def test_defaults_load_and_match_the_registry() -> None:
    assert check_defaults(load_defaults(), load_registry()) == []


def test_gate_thresholds_match_project_decision_8() -> None:
    gate = load_defaults().gate
    assert (gate.pass_probability_min, gate.score_min, gate.confidence_min) == (0.8, 2, 0.7)
    assert gate.max_retries_per_stem == 2


def test_jev_model_is_pinned_and_retention_is_seven_days() -> None:
    defaults = load_defaults()
    assert defaults.decision.jev_model == "jev-1.14.0"
    assert defaults.retention.days == 7


def test_models_are_marked_provisional_until_benchmarked() -> None:
    # Phase 2 flips this once defaults come from measured results.
    assert load_defaults().ensembles.provisional is True


def test_every_slot_supports_all_three_modes() -> None:
    slots = load_defaults().ensembles.slots
    assert set(slots) == set(ENSEMBLE_SLOTS)
    for models in slots.values():
        assert models.for_mode("single_best") == [models.single_best]
        assert len(models.for_mode("two_model_blend")) == 2
        assert len(models.for_mode("full_ensemble")) >= 2


def test_rejects_an_alias_as_jev_model(raw: dict) -> None:
    raw["decision"]["jev_model"] = "jev-latest"
    with pytest.raises(ValueError, match="jev_model"):
        Defaults.model_validate(raw)


def test_rejects_more_than_two_retries(raw: dict) -> None:
    raw["gate"]["max_retries_per_stem"] = 3
    with pytest.raises(ValueError, match="max_retries_per_stem"):
        Defaults.model_validate(raw)


def test_rejects_a_missing_slot(raw: dict) -> None:
    del raw["ensembles"]["slots"]["piano"]
    with pytest.raises(ValueError, match="slots must be exactly"):
        Defaults.model_validate(raw)


def test_check_defaults_reports_unknown_and_wrong_slot_models(raw: dict) -> None:
    raw["ensembles"]["slots"]["drums"]["single_best"] = "does_not_exist"
    raw["ensembles"]["slots"]["piano"]["two_model_blend"] = [
        "bs_roformer_sw_6stem",
        "mel_roformer_kim_vocals",
    ]
    problems = check_defaults(Defaults.model_validate(raw), load_registry())
    assert any("does_not_exist" in p and "not in the registry" in p for p in problems)
    assert any(
        "mel_roformer_kim_vocals" in p and "not a registered candidate" in p for p in problems
    )
