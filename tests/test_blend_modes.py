"""``separate --mode``: two-model blends and full ensembles in the product pipeline."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from stems_common.settings import Defaults, load_defaults  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.registry import ModelEntry, Registry  # noqa: E402
from stems_worker import cli  # noqa: E402
from stems_worker.audio import load_audio, save_audio  # noqa: E402
from stems_worker.compose import Blend, CompositionError, plan_composition  # noqa: E402
from stems_worker.fakes import OracleSeparator  # noqa: E402
from stems_worker.pipeline import (  # noqa: E402
    PipelineError,
    SeparateOptions,
    default_assignment,
    separate_file,
)

SR = 44100
LSB_24 = 2.0**-23
FOUR = ["vocals", "drums", "bass", "other"]

# model id -> (outputs, gain). Every fake returns the true stems times its gain, so a
# blend's output is the truth times the weighted mean of its members' gains.
MODELS = {
    "a4": (FOUR, 1.0),
    "b4": (FOUR, 0.5),
    "c6": (list(PRODUCT_STEMS), 1.0),
    "d6": (list(PRODUCT_STEMS), 0.5),
    "k1": (["lead_vocals", "backing_vocals"], 1.0),
    "k2": (["lead_vocals", "backing_vocals"], 0.5),
    "r1": (["dry"], 1.0),
    "r2": (["dry"], 0.5),
}
REGISTRY = Registry.model_construct(
    schema_version=1,
    models=[
        ModelEntry.model_construct(
            id=model_id,
            stems=outputs,
            status="available",
            kind="separator",
            loader="msst",
            files=[],
            config={},
        )
        for model_id, (outputs, _) in MODELS.items()
    ],
)
SLOTS = {
    "vocals": ["a4", "b4", "c6"],
    "drums": ["a4", "b4", "c6"],
    "bass": ["a4", "b4", "c6"],
    "guitar": ["c6", "d6"],
    "piano": ["c6", "d6"],
    "other": ["a4", "b4"],
    "vocal_split": ["k1", "k2"],
    "dereverb": ["r1", "r2"],
}


def defaults(**changes) -> Defaults:
    raw = load_defaults().model_dump()
    raw["ensembles"]["slots"] = {
        slot: {"single_best": ids[0], "two_model_blend": ids[:2], "full_ensemble": ids}
        for slot, ids in SLOTS.items()
    }
    for slot, fields in changes.items():
        if slot == "blend_method":
            raw["ensembles"]["blend_method"] = fields
        else:
            raw["ensembles"]["slots"][slot].update(fields)
    return Defaults.model_validate(raw)


class Song:
    def __init__(self, tmp_path) -> None:
        track = generate_track(seed=43, duration_seconds=3.0)
        q = lambda x: (np.round(x.astype(np.float64) / LSB_24) * LSB_24).astype(np.float32)  # noqa: E731
        self.stems = {name: q(stem) for name, stem in track.stems.items()}
        self.mix = sum(self.stems.values()).astype(np.float32)
        self.path = tmp_path / "song.wav"
        save_audio(self.path, self.mix, SR, "wav")
        four = {k: self.stems[k] for k in ("vocals", "drums", "bass")}
        four["other"] = self.stems["other"] + self.stems["guitar"] + self.stems["piano"]
        self.lead = q(track.lead_vocals)
        split = {"lead_vocals": self.lead, "backing_vocals": self.stems["vocals"] - self.lead}
        self.truth = {
            "a4": four,
            "b4": four,
            "c6": self.stems,
            "d6": self.stems,
            "k1": split,
            "k2": split,
            "r1": {"dry": self.stems["vocals"]},
            "r2": {"dry": self.stems["vocals"]},
        }
        self.loads: list[str] = []
        self.tmp_path = tmp_path

    def factory(self, model_id, stems):
        outer = self

        class Counting(OracleSeparator):
            def load(self, device: str) -> None:
                outer.loads.append(self.model_id)
                super().load(device)

        gain = MODELS[model_id][1]
        gains = dict.fromkeys(self.truth[model_id], gain) if gain != 1.0 else None
        return [Counting(self.truth[model_id], model_id=model_id, gains=gains)]

    def separate(self, out, mode, stems=PRODUCT_STEMS, config=None, **options):
        options.setdefault("cache_dir", None)
        options.setdefault("timings_path", None)
        return separate_file(
            self.path,
            self.tmp_path / out,
            SeparateOptions(stems=tuple(stems), mode=mode, device="cpu", **options),
            defaults=config or defaults(),
            factory=self.factory,
            registry=REGISTRY,
            fingerprint=lambda model_id: f"test-{model_id}",
        )

    def written(self, out, name):
        return load_audio(self.tmp_path / out / f"{name}.wav").samples


@pytest.fixture
def song(tmp_path) -> Song:
    return Song(tmp_path)


def _close(got, want, tolerance=4 * LSB_24) -> bool:
    return float(np.abs(got - want).max()) < tolerance


def test_two_model_blend_runs_each_model_once_and_blends_every_stem(song) -> None:
    manifest = song.separate("out", "two_model_blend")
    assert sorted(song.loads) == ["a4", "b4", "c6", "d6"]  # once each, shared across stems
    assert manifest["request"]["mode"] == "two_model_blend"
    # Members at gains 1.0 and 0.5, equally weighted: three quarters of the truth.
    for stem in PRODUCT_STEMS:
        assert _close(song.written("out", stem), 0.75 * song.stems[stem]), stem
    assert manifest["stems"]["vocals"] == {
        "file": "vocals.wav",
        "source": "blend of a4 (0.50), b4 (0.50); waveform",
        "rms_dbfs": manifest["stems"]["vocals"]["rms_dbfs"],
        "peak": manifest["stems"]["vocals"]["peak"],
        "models": ["a4", "b4"],
        "weights": [0.5, 0.5],
        "blend_method": "waveform",
    }
    assert manifest["stems"]["guitar"]["models"] == ["c6", "d6"]
    # "Other" is the four-stem blend with the blended guitar and piano taken out.
    assert manifest["stems"]["other"]["source"] == (
        "blend of a4 (0.50), b4 (0.50); waveform minus guitar and piano"
    )
    assert manifest["other_method"] == "subtract"
    runs = {m["model_id"]: m["outputs"] for m in manifest["models"]}
    assert runs == {
        "a4": ["vocals", "drums", "bass", "other"],
        "b4": ["vocals", "drums", "bass", "other"],
        "c6": ["guitar", "piano"],
        "d6": ["guitar", "piano"],
    }
    assert json.loads((song.tmp_path / "out" / "manifest.json").read_text()) == manifest


def test_full_ensemble_uses_every_listed_model(song) -> None:
    manifest = song.separate("out", "full_ensemble", stems=["vocals", "guitar"])
    assert sorted(song.loads) == ["a4", "b4", "c6", "d6"]
    assert manifest["stems"]["vocals"]["models"] == ["a4", "b4", "c6"]
    assert manifest["stems"]["vocals"]["weights"] == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert _close(song.written("out", "vocals"), (2.5 / 3) * song.stems["vocals"])
    assert _close(song.written("out", "guitar"), 0.75 * song.stems["guitar"])
    # c6 serves vocals in one blend and guitar in another, in a single run.
    assert {m["model_id"]: m["outputs"] for m in manifest["models"]}["c6"] == ["vocals", "guitar"]


def test_single_best_is_unchanged(song) -> None:
    manifest = song.separate("out", "single_best")
    assert sorted(song.loads) == ["a4", "c6"]
    for stem in PRODUCT_STEMS:
        assert _close(song.written("out", stem), song.stems[stem], LSB_24), stem
    assert manifest["stems"]["vocals"] == {
        "file": "vocals.wav",
        "source": "a4",
        "rms_dbfs": manifest["stems"]["vocals"]["rms_dbfs"],
        "peak": manifest["stems"]["vocals"]["peak"],
    }
    assert manifest["stems"]["other"]["source"] == "a4 minus guitar and piano"
    assert manifest["residual"]["level_db"] < -80


def test_weights_and_method_come_from_config(song) -> None:
    config = defaults(
        vocals={"two_model_weights": [3, 1]},
        drums={"blend_method": "stft_magnitude", "two_model_weights": [1, 3]},
    )
    manifest = song.separate("out", "two_model_blend", stems=["vocals", "drums"], config=config)
    assert manifest["stems"]["vocals"]["weights"] == [0.75, 0.25]
    assert manifest["stems"]["vocals"]["source"] == "blend of a4 (0.75), b4 (0.25); waveform"
    assert _close(song.written("out", "vocals"), 0.875 * song.stems["vocals"])
    assert manifest["stems"]["drums"]["blend_method"] == "stft_magnitude"
    # In phase, the magnitude blend is the weighted mean too: 0.25 * 1.0 + 0.75 * 0.5.
    assert _close(song.written("out", "drums"), 0.625 * song.stems["drums"], 1e-4)
    everywhere = defaults(blend_method="stft_max")
    manifest = song.separate("max", "two_model_blend", stems=["bass"], config=everywhere)
    assert manifest["stems"]["bass"]["blend_method"] == "stft_max"
    assert _close(song.written("max", "bass"), song.stems["bass"], 1e-4)  # the louder member


def test_a_model_override_replaces_the_blend_for_that_stem(song) -> None:
    manifest = song.separate(
        "out", "two_model_blend", stems=["vocals", "drums"], models={"vocals": "c6"}
    )
    assert manifest["stems"]["vocals"]["source"] == "c6"
    assert "models" not in manifest["stems"]["vocals"]
    assert _close(song.written("out", "vocals"), song.stems["vocals"], LSB_24)
    assert manifest["stems"]["drums"]["models"] == ["a4", "b4"]


def test_stages_are_blended_too(song) -> None:
    manifest = song.separate(
        "out", "two_model_blend", stems=["vocals"], split_vocals=True, dereverb=True
    )
    assert _close(song.written("out", "lead_vocals"), 0.75 * song.lead)
    assert _close(song.written("out", "vocals_dry"), 0.75 * song.stems["vocals"])
    lead = manifest["extras"]["lead_vocals"]
    assert lead["models"] == ["k1", "k2"]
    assert lead["weights"] == [0.5, 0.5]
    assert lead["source"] == "blend of k1 (0.50), k2 (0.50); waveform on the vocals stem"
    assert manifest["extras"]["vocals_dry"]["models"] == ["r1", "r2"]
    stage_runs = [(m["model_id"], m["stage"]) for m in manifest["models"]]
    stage_runs = [run for run in stage_runs if run[1] != "separation"]
    assert stage_runs == [
        ("k1", "vocal_split"),
        ("k2", "vocal_split"),
        ("r1", "dereverb"),
        ("r2", "dereverb"),
    ]


def test_a_blend_reuses_what_single_best_already_cached(song) -> None:
    cache = song.tmp_path / "cache"
    song.separate("one", "single_best", stems=["vocals"], cache_dir=cache)
    song.loads.clear()
    manifest = song.separate("two", "two_model_blend", stems=["vocals"], cache_dir=cache)
    assert song.loads == ["b4"]
    assert {m["model_id"]: m["cached"] for m in manifest["models"]} == {"a4": True, "b4": False}


def test_bad_modes_and_inconsistent_blends_are_refused(song) -> None:
    with pytest.raises(PipelineError, match="unknown mode 'everything'"):
        song.separate("out", "everything")
    mixed = defaults(other={"two_model_blend": ["a4", "c6"], "full_ensemble": ["a4", "c6"]})
    with pytest.raises(CompositionError, match="mixes 4-stem and 6-stem 'other' models"):
        song.separate("out", "two_model_blend", config=mixed)
    with pytest.raises(CompositionError, match="guitar: a4 does not emit it"):
        plan_composition(["guitar"], {"guitar": Blend(("c6", "a4"))}, REGISTRY)
    with pytest.raises(CompositionError, match="distinct models"):
        Blend(("a4", "a4"))
    with pytest.raises(CompositionError, match="bad weights"):
        Blend(("a4", "b4"), (1.0,))
    assert song.loads == []


def test_blend_weights_are_normalised_and_described() -> None:
    blend = Blend(("a4", "b4"), (3.0, 1.0), "stft_magnitude")
    assert blend.weights == (0.75, 0.25)
    assert blend.describe() == "blend of a4 (0.75), b4 (0.25); stft_magnitude"
    assert Blend(("a4",)).describe() == "a4"
    assert Blend(("a4", "b4", "c6")).weights == pytest.approx((1 / 3,) * 3)


def test_default_assignment_reads_members_weights_and_method_from_config() -> None:
    config = defaults(vocals={"full_ensemble_weights": [2, 1, 1], "blend_method": "stft_min"})
    assert default_assignment(config, "single_best")["vocals"] == "a4"
    full = default_assignment(config, "full_ensemble")
    assert full["vocals"] == Blend(("a4", "b4", "c6"), (0.5, 0.25, 0.25), "stft_min")
    assert full["other"] == Blend(("a4", "b4"), (0.5, 0.5), "waveform")
    # The real config works in every mode.
    for mode in ("single_best", "two_model_blend", "full_ensemble"):
        assert set(default_assignment(load_defaults(), mode)) == set(SLOTS)


def test_command_takes_the_mode(song, capsys) -> None:
    out = song.tmp_path / "cli"
    argv = [str(song.path), "--stems", "vocals", "--out", str(out), "--device", "cpu"]
    argv += ["--no-cache", "--no-timings", "--quiet", "--mode", "two_model_blend"]
    inject = {"factory": song.factory, "registry": REGISTRY, "defaults": defaults()}
    assert cli.main(argv, fingerprint=lambda model_id: model_id, **inject) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stems"]["vocals"]["models"] == ["a4", "b4"]
    assert "a4: vocals" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*argv[:-1], "biggest"], **inject)
    assert exit_info.value.code == 2
