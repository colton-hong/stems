"""The committed registry is valid and covers everything the project brief requires."""

from __future__ import annotations

import copy

import pytest
import yaml

from stems_common.paths import REGISTRY_PATH
from stems_common.stems import PRODUCT_STEMS
from stems_models.registry import Registry, RegistryError, load_registry, summary_table


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_loads_and_ids_are_unique(registry: Registry) -> None:
    ids = registry.ids()
    assert len(ids) == len(set(ids))
    assert len(ids) >= 40


# Starting candidates named in the project brief (decision 2), by registry id.
BRIEF_CANDIDATES = {
    "vocals": ["mel_roformer_kim_vocals", "bs_roformer_viperx_vocals", "bs_polarformer_vocals"],
    "drums": ["scnet_xl_ihf", "scnet_large", "htdemucs_ft"],
    "other": ["scnet_xl_ihf", "scnet_large", "htdemucs_ft"],
    "bass": ["htdemucs", "htdemucs_ft", "bs_roformer_sw_6stem", "scnet_xl_ihf"],
    "guitar": ["bs_roformer_sw_6stem", "mel_roformer_guitar_becruily", "htdemucs_6s"],
    "piano": ["bs_roformer_sw_6stem", "htdemucs_6s"],
    "vocal_split": ["mel_roformer_karaoke_aufr33_viperx", "uvr_vr_5hp_karaoke"],
    "dereverb": ["mel_roformer_dereverb_anvuew", "uvr_vr_deecho_dereverb"],
    "tagging": ["beats_iter3plus_as2m_cpt2", "laion_clap_music", "mert_v1_95m"],
}


@pytest.mark.parametrize("slot", sorted(BRIEF_CANDIDATES))
def test_brief_candidates_are_registered_for_their_slot(registry: Registry, slot: str) -> None:
    candidate_ids = {m.id for m in registry.candidates_for(slot)}
    assert set(BRIEF_CANDIDATES[slot]) <= candidate_ids


@pytest.mark.parametrize("stem", PRODUCT_STEMS)
def test_every_product_stem_has_at_least_two_loadable_candidates(
    registry: Registry, stem: str
) -> None:
    assert len(registry.candidates_for(stem)) >= 2


def test_report_sections_are_both_represented(registry: Registry) -> None:
    origins = {m.origin for m in registry.models}
    assert {"report_s1", "report_s5"} <= origins
    assert "clef_omni" in registry
    assert registry.get("clef_omni").status == "hosted"


def test_every_entry_has_the_fields_the_brief_asks_for(registry: Registry) -> None:
    for m in registry.models:
        assert m.architecture
        assert m.source_page.startswith("https://")
        assert m.licence_note
        if m.is_loadable:
            assert m.weights, m.id
            assert all(f.url.startswith("https://") for f in m.files), m.id


def test_per_stem_bags_only_need_one_file_per_stem(registry: Registry) -> None:
    ft = registry.get("htdemucs_ft")
    assert [f.stem for f in ft.files_for("bass")] == ["bass"]
    assert len(ft.files_for()) == 4


def test_other_semantics_are_derived_from_outputs(registry: Registry) -> None:
    assert registry.get("scnet_xl_ihf").other_contains_guitar_piano
    assert not registry.get("bs_roformer_sw_6stem").other_contains_guitar_piano
    assert not registry.get("mel_roformer_kim_vocals").other_contains_guitar_piano


def test_summary_table_lists_every_model(registry: Registry) -> None:
    table = summary_table(registry)
    assert all(f"`{model_id}`" in table for model_id in registry.ids())


def test_unknown_id_and_slot_raise(registry: Registry) -> None:
    with pytest.raises(KeyError):
        registry.get("nope")
    with pytest.raises(KeyError):
        registry.candidates_for("kazoo")


# --- validation rejects broken registries -------------------------------------------------


def _first(raw: dict, model_id: str) -> dict:
    return next(m for m in raw["models"] if m["id"] == model_id)


def _validate(raw: dict) -> None:
    Registry.model_validate(raw)


def test_rejects_duplicate_ids(raw: dict) -> None:
    raw["models"].append(copy.deepcopy(raw["models"][0]))
    with pytest.raises(ValueError, match="duplicate model id"):
        _validate(raw)


def test_rejects_candidate_for_a_stem_the_model_does_not_emit(raw: dict) -> None:
    _first(raw, "mel_roformer_kim_vocals")["candidate_for"] = ["drums"]
    with pytest.raises(ValueError, match="candidate for 'drums'"):
        _validate(raw)


def test_rejects_available_model_without_weights(raw: dict) -> None:
    _first(raw, "htdemucs")["files"] = []
    with pytest.raises(ValueError, match="no weights file"):
        _validate(raw)


def test_rejects_non_https_url(raw: dict) -> None:
    _first(raw, "htdemucs")["files"][0]["url"] = "http://example.invalid/x.th"
    with pytest.raises(ValueError, match="url"):
        _validate(raw)


def test_rejects_path_in_filename(raw: dict) -> None:
    _first(raw, "htdemucs")["files"][0]["filename"] = "../evil.th"
    with pytest.raises(ValueError, match="bare file name"):
        _validate(raw)


def test_rejects_unknown_fields(raw: dict) -> None:
    _first(raw, "htdemucs")["sdr"] = 9.0
    with pytest.raises(ValueError, match="sdr"):
        _validate(raw)


def test_rejects_config_file_that_is_not_listed(raw: dict) -> None:
    _first(raw, "scnet_large")["config"]["config_file"] = "missing.yaml"
    with pytest.raises(ValueError, match="config_file"):
        _validate(raw)


def test_load_registry_reports_missing_and_malformed_files(tmp_path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nmodels: [{id: x}]\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="failed validation"):
        load_registry(bad)
