"""The composer plans shared model runs, builds "other", and reports the residual."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stems_common.settings import load_defaults  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_common.synthetic import SyntheticTrack, generate_track  # noqa: E402
from stems_models.registry import ModelEntry, Registry, load_registry  # noqa: E402
from stems_worker.compose import (  # noqa: E402
    CompositionError,
    compose,
    plan_composition,
)
from stems_worker.factory import UnsupportedModelError, build_separators  # noqa: E402
from stems_worker.fakes import OracleSeparator  # noqa: E402

SR = 44100


def _entry(model_id: str, stems: list[str], status: str = "available") -> ModelEntry:
    return ModelEntry.model_construct(
        id=model_id, stems=stems, status=status, kind="separator", loader="msst", files=[]
    )


REGISTRY = Registry.model_construct(
    schema_version=1,
    models=[
        _entry("fake_vox", ["vocals", "instrumental"]),
        _entry("fake4", ["vocals", "drums", "bass", "other"]),
        _entry("fake6", list(PRODUCT_STEMS)),
        _entry("fake_hosted", ["vocals"], status="hosted"),
    ],
)
# vocals from a vocal model, the rhythm section and "other" from a 4-stem model,
# guitar and piano from a 6-stem model: the shape of the real default assignment.
ASSIGNMENT = {
    "vocals": "fake_vox",
    "drums": "fake4",
    "bass": "fake4",
    "other": "fake4",
    "guitar": "fake6",
    "piano": "fake6",
}


@pytest.fixture(scope="module")
def track() -> SyntheticTrack:
    return generate_track(seed=31, duration_seconds=4.0)


class Factory:
    """Hands out perfect models built from the fixture's own stems, and counts calls."""

    def __init__(self, track: SyntheticTrack, **faults) -> None:
        s = track.stems
        self.truth = {
            "fake_vox": {"vocals": s["vocals"], "instrumental": track.mix - s["vocals"]},
            # A 4-stem model's "other" still holds the guitar and the piano.
            "fake4": {
                "vocals": s["vocals"],
                "drums": s["drums"],
                "bass": s["bass"],
                "other": s["other"] + s["guitar"] + s["piano"],
            },
            "fake6": dict(s),
        }
        self.faults = faults
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.made: list[OracleSeparator] = []

    def __call__(self, model_id, stems):
        self.calls.append((model_id, tuple(stems)))
        oracle = OracleSeparator(
            self.truth[model_id], model_id=model_id, **self.faults.get(model_id, {})
        )
        self.made.append(oracle)
        return [oracle]


def _run(track, requested=PRODUCT_STEMS, assignment=ASSIGNMENT, factory=None, **kw):
    other_method = kw.pop("other_method", "auto")
    plan = plan_composition(requested, assignment, REGISTRY, other_method)
    factory = factory or Factory(track)
    return compose(track.mix, SR, plan, factory, device="cpu", **kw), factory


# --- planning ------------------------------------------------------------------------------


def test_plan_shares_one_run_per_model() -> None:
    plan = plan_composition(PRODUCT_STEMS, ASSIGNMENT, REGISTRY)
    assert {r.model_id: r.outputs for r in plan.runs} == {
        "fake_vox": ("vocals",),
        "fake4": ("drums", "bass", "other"),
        "fake6": ("guitar", "piano"),
    }
    assert plan.other_method == "subtract"
    assert plan.dependencies == ()
    assert plan.sources["other"] == "fake4"


def test_other_from_a_4_stem_model_pulls_in_guitar_and_piano() -> None:
    plan = plan_composition(["vocals", "other"], ASSIGNMENT, REGISTRY)
    assert plan.requested == ("vocals", "other")
    assert plan.other_method == "subtract"
    assert plan.dependencies == ("guitar", "piano")
    assert {r.model_id: r.outputs for r in plan.runs}["fake6"] == ("guitar", "piano")


def test_other_from_a_6_stem_model_is_used_directly() -> None:
    plan = plan_composition(["other"], {**ASSIGNMENT, "other": "fake6"}, REGISTRY)
    assert plan.other_method == "direct"
    assert plan.dependencies == ()
    assert [(r.model_id, r.outputs) for r in plan.runs] == [("fake6", ("other",))]


def test_residual_other_needs_the_five_named_stems_and_no_other_model() -> None:
    plan = plan_composition(["other"], ASSIGNMENT, REGISTRY, "residual")
    assert plan.other_method == "residual"
    assert plan.dependencies == ("vocals", "drums", "bass", "guitar", "piano")
    assert "other" not in plan.sources


def test_requests_without_other_plan_nothing_extra() -> None:
    plan = plan_composition(["bass", "vocals", "bass"], ASSIGNMENT, REGISTRY)
    assert plan.requested == ("bass", "vocals")
    assert plan.other_method is None
    assert plan.dependencies == ()


@pytest.mark.parametrize(
    ("requested", "assignment", "method", "message"),
    [
        (["kazoo"], ASSIGNMENT, "auto", "must be among"),
        ([], ASSIGNMENT, "auto", "must be among"),
        (["vocals"], {}, "auto", "no model is assigned to 'vocals'"),
        (["vocals"], {"vocals": "nope"}, "auto", "not in the registry"),
        (["vocals"], {"vocals": "fake_hosted"}, "auto", "status 'hosted'"),
        (["guitar"], {"guitar": "fake4"}, "auto", "does not emit it"),
        (["other"], ASSIGNMENT, "direct", "4-stem 'other'"),
        (["other"], {**ASSIGNMENT, "other": "fake6"}, "subtract", "6-stem 'other'"),
        (["other"], {"other": "fake4"}, "auto", "no model is assigned to 'guitar'"),
    ],
)
def test_unservable_requests_are_rejected(requested, assignment, method, message) -> None:
    with pytest.raises(CompositionError, match=message):
        plan_composition(requested, assignment, REGISTRY, method)


# --- acceptance ----------------------------------------------------------------------------


def test_perfect_models_give_exact_stems_and_a_residual_below_minus_80_db(track) -> None:
    result, factory = _run(track)

    assert list(result.stems) == list(PRODUCT_STEMS)
    for name in PRODUCT_STEMS:
        assert float(np.abs(result.stems[name] - track.stems[name]).max()) < 1e-6, name
    # The composed "other" is the fixture's, although no model emitted it as such.
    assert float(np.abs(result.stems["other"] - track.stems["other"]).max()) < 1e-6

    assert result.residual.computed is True
    assert result.residual.level_db < -80
    assert result.residual.ok is True
    assert result.residual.silent_stems == ()

    assert sorted(model_id for model_id, _ in factory.calls) == ["fake4", "fake6", "fake_vox"]
    assert {r.model_id: r.outputs for r in result.runs}["fake4"] == ("drums", "bass", "other")
    assert all(r.device == "cpu" and r.wall_seconds >= 0 for r in result.runs)


def test_a_model_that_drops_a_stem_is_reported(track) -> None:
    factory = Factory(track, fake4={"silence": ("bass",)})
    result, _ = _run(track, factory=factory)

    assert result.residual.computed is True
    assert result.residual.ok is False
    assert result.residual.silent_stems == ("bass",)
    assert "bass" in result.residual.note
    # What is missing is exactly the bass, so the residual sits at the bass's own level.
    bass_db = 20 * np.log10(
        float(np.sqrt(np.mean(track.stems["bass"] ** 2))) / float(np.sqrt(np.mean(track.mix**2)))
    )
    assert result.residual.level_db == pytest.approx(bass_db, abs=0.1)
    assert not result.stems["bass"].any()


def test_duplicated_content_is_reported_without_blaming_a_silent_stem(track) -> None:
    factory = Factory(track, fake4={"gains": {"drums": 2.0}})
    result, _ = _run(track, factory=factory)
    assert result.residual.ok is False
    assert result.residual.silent_stems == ()
    assert "missing or duplicated" in result.residual.note


# --- "other" -------------------------------------------------------------------------------


def test_dependencies_are_computed_but_not_delivered(track) -> None:
    result, _ = _run(track, requested=["vocals", "other"])
    assert list(result.stems) == ["vocals", "other"]
    assert sorted(result.dependencies) == ["guitar", "piano"]
    assert float(np.abs(result.stems["other"] - track.stems["other"]).max()) < 1e-6
    assert result.residual.computed is False
    assert "missing" in result.residual.note


def test_residual_other_is_the_mix_minus_the_five_named_stems(track) -> None:
    result, factory = _run(track, other_method="residual")
    assert float(np.abs(result.stems["other"] - track.stems["other"]).max()) < 1e-6
    assert ("fake4", ("drums", "bass")) in factory.calls  # "other" was not asked of the model
    assert result.residual.computed is False
    assert "by construction" in result.residual.note


def test_residual_other_absorbs_what_a_model_drops(track) -> None:
    factory = Factory(track, fake4={"silence": ("bass",)})
    result, _ = _run(track, factory=factory, other_method="residual")
    expected = track.stems["other"] + track.stems["bass"]
    assert float(np.abs(result.stems["other"] - expected).max()) < 1e-6
    assert result.residual.silent_stems == ("bass",)


def test_a_silent_mix_is_not_scored(track) -> None:
    silent = SyntheticTrack(
        track_id="silent",
        seed=0,
        sample_rate=SR,
        stems={k: np.zeros_like(v) for k, v in track.stems.items()},
        lead_vocals=np.zeros_like(track.mix),
        backing_vocals=np.zeros_like(track.mix),
        vocals_wet=np.zeros_like(track.mix),
    )
    result, _ = _run(silent, factory=Factory(silent))
    assert result.residual.computed is False
    assert "silent" in result.residual.note


# --- running -------------------------------------------------------------------------------


def test_a_bag_served_by_several_separators_is_assembled(track) -> None:
    class Bag(Factory):
        def __call__(self, model_id, stems):
            if model_id != "fake4":
                return super().__call__(model_id, stems)
            self.calls.append((model_id, tuple(stems)))
            return [
                OracleSeparator({name: self.truth["fake4"][name]}, model_id="fake4")
                for name in stems
            ]

    result, _ = _run(track, factory=Bag(track))
    assert result.residual.level_db < -80
    assert [r.outputs for r in result.runs if r.model_id == "fake4"] == [
        ("drums",),
        ("bass",),
        ("other",),
    ]


def test_a_factory_that_cannot_supply_a_stem_is_an_error(track) -> None:
    def factory(model_id, stems):
        return [OracleSeparator({"vocals": track.stems["vocals"]}, model_id=model_id)]

    with pytest.raises(CompositionError, match="did not offer"):
        _run(track, factory=factory)


def test_models_are_unloaded_even_when_one_fails(track) -> None:
    class Exploding(OracleSeparator):
        def separate_chunks(self, batch):
            raise RuntimeError("boom")

    made = []

    def factory(model_id, stems):
        made.append(Exploding({s: track.stems[s] for s in stems}, model_id=model_id))
        return [made[-1]]

    with pytest.raises(RuntimeError, match="boom"):
        _run(track, requested=["vocals"], factory=factory)
    assert made[0].device is None


def test_every_model_is_unloaded_after_its_run_and_progress_is_reported(track) -> None:
    seen: list[tuple[str, int, int]] = []
    result, factory = _run(track, progress=lambda model_id, d, t: seen.append((model_id, d, t)))
    assert all(oracle.device is None for oracle in factory.made)
    assert {model_id for model_id, _, _ in seen} == {"fake_vox", "fake4", "fake6"}
    assert all(done <= total for _, done, total in seen)
    assert len(result.runs) == 3


# --- the real factory ----------------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["laion_clap_music", "clef_omni", "bs_roformer_paper"])
def test_factory_refuses_entries_that_are_not_loadable_separators(model_id) -> None:
    with pytest.raises(UnsupportedModelError, match="not a loadable separator"):
        build_separators(model_id)


def test_default_assignment_can_be_planned_against_the_real_registry() -> None:
    defaults = load_defaults()
    assignment = {s: defaults.ensembles.slots[s].single_best for s in PRODUCT_STEMS}
    plan = plan_composition(PRODUCT_STEMS, assignment, load_registry())
    by_model = {r.model_id: r.outputs for r in plan.runs}
    # Provisional defaults: four models serve the six stems.
    assert by_model == {
        "mel_roformer_kim_vocals": ("vocals",),
        "scnet_xl_ihf": ("drums", "other"),
        "htdemucs_ft": ("bass",),
        "bs_roformer_sw_6stem": ("guitar", "piano"),
    }
    assert plan.other_method == "subtract"


def _fraction(output, stem) -> float:
    s = stem.astype(np.float64).ravel()
    return float(np.dot(output.astype(np.float64).ravel(), s) / np.dot(s, s))


@pytest.mark.slow
def test_default_models_compose_six_stems_from_the_fixture() -> None:
    from stems_models.download import is_downloaded
    from stems_worker.device import pick_device

    registry, defaults = load_registry(), load_defaults()
    assignment = {s: defaults.ensembles.slots[s].single_best for s in PRODUCT_STEMS}
    if not all(is_downloaded(registry.get(m)) for m in set(assignment.values())):
        pytest.skip("default models are not all downloaded")

    track = generate_track(seed=1, duration_seconds=10.0)
    plan = plan_composition(PRODUCT_STEMS, assignment, registry)
    result = compose(
        track.mix,
        SR,
        plan,
        lambda model_id, stems: build_separators(model_id, stems, registry=registry),
        device=pick_device(),
        overlap=defaults.separation.overlap,
    )

    assert list(result.stems) == list(PRODUCT_STEMS)
    for stem in result.stems.values():
        assert stem.shape == track.mix.shape
        assert np.isfinite(stem).all()
    assert [r.model_id for r in result.runs] == [
        "mel_roformer_kim_vocals",
        "scnet_xl_ihf",
        "htdemucs_ft",
        "bs_roformer_sw_6stem",
    ]
    assert result.residual.computed is True
    assert result.residual.level_db is not None
    assert _fraction(result.stems["drums"], track.stems["drums"]) > 0.6
    assert _fraction(result.stems["bass"], track.stems["bass"]) > 0.6
