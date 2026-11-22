"""The separate command writes stems and a manifest, and reuses cached model outputs."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.download import ChecksumLock  # noqa: E402
from stems_models.registry import ModelEntry, Registry, load_registry  # noqa: E402
from stems_worker import cli  # noqa: E402
from stems_worker.audio import load_audio, save_audio  # noqa: E402
from stems_worker.cache import OutputCache, audio_sha256, registry_fingerprint  # noqa: E402
from stems_worker.fakes import OracleSeparator  # noqa: E402
from stems_worker.pipeline import PipelineError, SeparateOptions, separate_file  # noqa: E402

SR = 44100
LSB_24 = 2.0**-23


def _entry(model_id: str, stems: list[str]) -> ModelEntry:
    return ModelEntry.model_construct(
        id=model_id, stems=stems, status="available", kind="separator", loader="msst", files=[]
    )


REGISTRY = Registry.model_construct(
    schema_version=1,
    models=[
        _entry("fake4", ["vocals", "drums", "bass", "other"]),
        _entry("fake6", list(PRODUCT_STEMS)),
        _entry("fake_kara", ["lead_vocals", "backing_vocals"]),
        _entry("fake_dry", ["dry"]),
    ],
)
MODELS = {
    "vocals": "fake4",
    "drums": "fake4",
    "bass": "fake4",
    "other": "fake4",
    "guitar": "fake6",
    "piano": "fake6",
    "vocal_split": "fake_kara",
    "dereverb": "fake_dry",
}
OVERRIDES = [arg for stem, model in MODELS.items() for arg in ("--model", f"{stem}={model}")]


class Song:
    """A fixture track on disk, with perfect fake models that know its stems."""

    def __init__(self, tmp_path, seed: int = 41, gain: float = 1.0) -> None:
        track = generate_track(seed=seed, duration_seconds=3.0)
        # Quantise like the 24-bit file the models will be given, so "perfect" is exact.
        q = lambda x: (np.round(x.astype(np.float64) / LSB_24) * LSB_24).astype(np.float32)  # noqa: E731
        stems = {name: q(stem) for name, stem in track.stems.items()}
        self.stems = stems
        self.mix = sum(stems.values()).astype(np.float32)
        self.path = tmp_path / f"song{seed}.wav"
        save_audio(self.path, self.mix, SR, "wav")
        four = {k: stems[k] for k in ("vocals", "drums", "bass")}
        four["other"] = stems["other"] + stems["guitar"] + stems["piano"]
        self.truth = {"fake4": four, "fake6": stems}
        # Stage models run on the vocals stem: a lead/backing split that adds back up to
        # it exactly, and a "de-reverb" that is recognisable as half the level.
        self.lead = q(track.lead_vocals)
        self.backing = stems["vocals"] - self.lead
        self.truth["fake_kara"] = {"lead_vocals": self.lead, "backing_vocals": self.backing}
        self.truth["fake_dry"] = {"dry": stems["vocals"] * np.float32(0.5)}
        self.gain = gain
        self.loads = 0
        self.cache_dir = tmp_path / "cache"

    def factory(self, model_id, stems):
        outer = self

        class Counting(OracleSeparator):
            def load(self, device: str) -> None:
                outer.loads += 1
                super().load(device)

        gains = dict.fromkeys(self.truth[model_id], self.gain) if self.gain != 1.0 else None
        return [Counting(self.truth[model_id], model_id=model_id, gains=gains)]

    def inject(self) -> dict:
        return {
            "factory": self.factory,
            "registry": REGISTRY,
            "fingerprint": lambda model_id: f"test-{model_id}",
        }

    def separate(self, out, stems=PRODUCT_STEMS, **options):
        options.setdefault("cache_dir", self.cache_dir)
        options.setdefault("device", "cpu")
        return separate_file(
            self.path,
            out,
            SeparateOptions(stems=tuple(stems), models=MODELS, **options),
            **self.inject(),
        )


@pytest.fixture
def song(tmp_path) -> Song:
    return Song(tmp_path)


# --- acceptance: six files and a manifest --------------------------------------------------


def test_six_stems_and_a_manifest_are_written(song, tmp_path) -> None:
    out = tmp_path / "out"
    manifest = song.separate(out)

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"{stem}.wav" for stem in PRODUCT_STEMS] + ["manifest.json"]
    )
    assert json.loads((out / "manifest.json").read_text()) == manifest
    for stem in PRODUCT_STEMS:
        written = load_audio(out / f"{stem}.wav").samples
        assert float(np.abs(written - song.stems[stem]).max()) < 1e-6, stem

    # Models, device and seconds per model.
    assert manifest["device"] == "cpu"
    assert [(m["model_id"], m["outputs"]) for m in manifest["models"]] == [
        ("fake4", ["vocals", "drums", "bass", "other"]),
        ("fake6", ["guitar", "piano"]),
    ]
    for model in manifest["models"]:
        assert model["device"] == "cpu"
        assert model["wall_seconds"] >= 0
        assert model["audio_seconds"] == pytest.approx(3.0)
        assert model["cached"] is False
    assert manifest["stems"]["vocals"] == {
        "file": "vocals.wav",
        "source": "fake4",
        "rms_dbfs": manifest["stems"]["vocals"]["rms_dbfs"],
        "peak": manifest["stems"]["vocals"]["peak"],
    }
    assert manifest["stems"]["other"]["source"] == "fake4 minus guitar and piano"
    assert manifest["residual"]["computed"] is True
    assert manifest["residual"]["level_db"] < -80
    assert manifest["output_gain"] == 1.0
    assert manifest["input"]["name"] == song.path.name
    assert manifest["input"]["duration_seconds"] == pytest.approx(3.0)
    assert manifest["input"]["audio_sha256"] == audio_sha256(song.mix, SR)
    assert manifest["request"]["stems"] == list(PRODUCT_STEMS)


@pytest.mark.parametrize(("fmt", "extension"), [("flac", ".flac"), ("mp3_320", ".mp3")])
def test_other_formats(song, tmp_path, fmt, extension) -> None:
    out = tmp_path / fmt
    manifest = song.separate(out, stems=["vocals", "drums"], fmt=fmt)
    assert manifest["stems"]["drums"]["file"] == f"drums{extension}"
    decoded = load_audio(out / f"drums{extension}")
    assert abs(decoded.samples.shape[1] - song.mix.shape[1]) <= 4096
    if fmt == "flac":
        assert float(np.abs(decoded.samples - song.stems["drums"]).max()) < 1e-6


def test_a_subset_writes_only_what_was_asked_for(song, tmp_path) -> None:
    out = tmp_path / "out"
    manifest = song.separate(out, stems=["other", "bass"])
    assert sorted(p.name for p in out.iterdir()) == ["bass.wav", "manifest.json", "other.wav"]
    assert manifest["dependencies"] == ["guitar", "piano"]
    assert manifest["residual"]["computed"] is False


def test_stems_that_would_clip_are_all_lowered_together(tmp_path) -> None:
    hot = Song(tmp_path, gain=12.0)
    out = tmp_path / "out"
    manifest = hot.separate(out)
    assert 0 < manifest["output_gain"] < 1.0
    loaded = {s: load_audio(out / f"{s}.wav").samples for s in PRODUCT_STEMS}
    assert max(float(np.abs(x).max()) for x in loaded.values()) <= 1.0
    expected = hot.stems["bass"] * 12.0 * manifest["output_gain"]
    assert float(np.abs(loaded["bass"] - expected).max()) < 1e-5


# --- acceptance: the cache -----------------------------------------------------------------


def test_a_second_run_reuses_cached_model_outputs(song, tmp_path) -> None:
    first = song.separate(tmp_path / "a")
    assert song.loads == 2
    second = song.separate(tmp_path / "b")

    assert song.loads == 2  # no model was loaded the second time
    assert [m["cached"] for m in second["models"]] == [True, True]
    assert [m["wall_seconds"] for m in second["models"]] == [
        m["wall_seconds"] for m in first["models"]
    ]
    for stem in PRODUCT_STEMS:
        a = load_audio(tmp_path / "a" / f"{stem}.wav").samples
        b = load_audio(tmp_path / "b" / f"{stem}.wav").samples
        assert np.array_equal(a, b), stem


def test_cached_outputs_serve_a_different_request_for_the_same_audio(song, tmp_path) -> None:
    song.separate(tmp_path / "a")
    manifest = song.separate(tmp_path / "b", stems=["drums"], fmt="flac")
    assert song.loads == 2
    assert manifest["models"][0]["cached"] is True


def test_the_cache_is_not_used_when_something_that_matters_changes(song, tmp_path) -> None:
    song.separate(tmp_path / "a")
    song.separate(tmp_path / "b", overlap=0.5)
    assert song.loads == 4  # a different overlap is a different result

    other = Song(tmp_path, seed=42)
    other.cache_dir = song.cache_dir
    other.separate(tmp_path / "c")
    assert other.loads == 2  # different audio


def test_no_cache_always_runs_the_models(song, tmp_path) -> None:
    song.separate(tmp_path / "a", cache_dir=None)
    manifest = song.separate(tmp_path / "b", cache_dir=None)
    assert song.loads == 4
    assert not any(m["cached"] for m in manifest["models"])
    assert not song.cache_dir.exists()


def test_cache_keys_and_partial_entries(tmp_path) -> None:
    def cache(**changes):
        settings = {"audio_hash": "a", "overlap": 0.25, "device": "cpu"}
        settings.update(changes)
        return OutputCache(tmp_path, fingerprint=lambda m: f"f-{m}", **settings)

    base = cache()
    assert base.key("m") == cache().key("m")
    for different in (cache(audio_hash="b"), cache(overlap=0.5), cache(device="mps")):
        assert different.key("m") != base.key("m")
    assert base.key("m") != base.key("n")
    changed_weights = OutputCache(
        tmp_path, audio_hash="a", overlap=0.25, device="cpu", fingerprint=lambda m: "other"
    )
    assert changed_weights.key("m") != base.key("m")

    drums = np.ones((2, 8), dtype=np.float32)
    assert base.get("m", ("drums",)) is None
    base.put("m", {"drums": drums}, {"wall_seconds": 1.5, "device": "cpu"})
    assert np.array_equal(base.get("m", ("drums",)).stems["drums"], drums)
    assert base.get("m", ("drums", "bass")) is None  # only part of what is wanted
    base.put("m", {"bass": drums * 2}, {"wall_seconds": 0.5, "device": "cpu"})
    both = base.get("m", ("drums", "bass"))
    assert sorted(both.meta["outputs"]) == ["bass", "drums"]

    (base._directory("m") / "meta.json").write_text("not json")
    assert base.get("m", ("drums",)) is None


def test_fingerprint_follows_the_pinned_weights_and_the_registry_config(tmp_path) -> None:
    registry = load_registry()
    entry = registry.get("htdemucs")
    empty = registry_fingerprint(registry, ChecksumLock(tmp_path / "empty.json"))("htdemucs")

    lock = ChecksumLock(tmp_path / "lock.json")
    lock.check_or_record(entry.files[0].url, entry.files[0].filename, "a" * 64, 1)
    pinned = registry_fingerprint(registry, lock)("htdemucs")
    assert pinned != empty
    assert pinned == registry_fingerprint(registry, lock)("htdemucs")

    edited = entry.model_copy(update={"config": {**entry.config, "segment_seconds": 5.0}})
    other_registry = Registry.model_construct(schema_version=1, models=[edited])
    assert registry_fingerprint(other_registry, lock)("htdemucs") != pinned


# --- acceptance: vocal split and de-reverb -------------------------------------------------


def test_vocal_split_and_dereverb_write_extra_files_and_are_recorded(song, tmp_path) -> None:
    out = tmp_path / "out"
    manifest = song.separate(out, stems=["vocals", "drums"], split_vocals=True, dereverb=True)

    assert sorted(p.name for p in out.iterdir()) == [
        "backing_vocals.wav",
        "drums.wav",
        "lead_vocals.wav",
        "manifest.json",
        "vocals.wav",
        "vocals_dry.wav",
    ]
    lead = load_audio(out / "lead_vocals.wav").samples
    backing = load_audio(out / "backing_vocals.wav").samples
    dry = load_audio(out / "vocals_dry.wav").samples
    vocals = load_audio(out / "vocals.wav").samples
    assert float(np.abs(lead - song.lead).max()) < 1e-6
    assert float(np.abs(lead + backing - vocals).max()) < 1e-6  # the split adds back up
    assert float(np.abs(dry - 0.5 * vocals).max()) < 1e-6

    assert manifest["request"]["split_vocals"] is True
    assert manifest["request"]["dereverb"] is True
    stages = {m["stage"]: m for m in manifest["models"] if m["stage"] != "separation"}
    assert stages["vocal_split"]["model_id"] == "fake_kara"
    assert stages["vocal_split"]["input"] == "vocals stem"
    assert stages["vocal_split"]["outputs"] == ["lead_vocals", "backing_vocals"]
    assert stages["dereverb"]["model_id"] == "fake_dry"
    assert stages["dereverb"]["wall_seconds"] >= 0
    assert manifest["extras"]["vocals_dry"]["file"] == "vocals_dry.wav"
    assert manifest["extras"]["vocals_dry"]["stage"] == "dereverb"
    assert manifest["extras"]["lead_vocals"]["source"] == "fake_kara on the vocals stem"
    assert set(manifest["extras"]) == {"lead_vocals", "backing_vocals", "vocals_dry"}


def test_stages_compute_the_vocals_even_when_they_were_not_requested(song, tmp_path) -> None:
    out = tmp_path / "out"
    manifest = song.separate(out, stems=["drums"], split_vocals=True)
    assert sorted(p.name for p in out.iterdir()) == [
        "backing_vocals.wav",
        "drums.wav",
        "lead_vocals.wav",
        "manifest.json",
    ]
    assert manifest["dependencies"] == ["vocals"]
    assert "vocals" not in manifest["stems"]
    assert float(np.abs(load_audio(out / "lead_vocals.wav").samples - song.lead).max()) < 1e-6


def test_without_the_flags_no_stage_runs(song, tmp_path) -> None:
    manifest = song.separate(tmp_path / "out", stems=["vocals"])
    assert manifest["extras"] == {}
    assert {m["stage"] for m in manifest["models"]} == {"separation"}
    assert song.loads == 1


def test_stage_outputs_are_cached_too(song, tmp_path) -> None:
    song.separate(tmp_path / "a", stems=["vocals"], split_vocals=True, dereverb=True)
    assert song.loads == 3  # the vocal model, the splitter, the de-reverb
    again = song.separate(tmp_path / "b", stems=["vocals"], split_vocals=True, dereverb=True)
    assert song.loads == 3
    assert all(m["cached"] for m in again["models"])
    # Asking for less reuses everything already done.
    song.separate(tmp_path / "c", stems=["vocals"], split_vocals=True)
    assert song.loads == 3
    # Asking the vocal model for a stem it has not produced yet runs that model again,
    # and only that model.
    song.separate(tmp_path / "d", stems=["vocals", "drums"], split_vocals=True)
    assert song.loads == 4


def test_a_stage_model_that_cannot_serve_the_stage_is_rejected(song, tmp_path) -> None:
    from stems_worker.compose import CompositionError

    wrong = {**MODELS, "vocal_split": "fake_dry"}
    options = SeparateOptions(stems=("vocals",), models=wrong, split_vocals=True, device="cpu")
    with pytest.raises(CompositionError, match="vocal_split: fake_dry cannot serve it"):
        separate_file(song.path, tmp_path / "x", options, **song.inject())
    missing = {**MODELS, "dereverb": "nope"}
    options = SeparateOptions(stems=("vocals",), models=missing, dereverb=True, device="cpu")
    with pytest.raises(CompositionError, match="dereverb: 'nope' is not in the registry"):
        separate_file(song.path, tmp_path / "x", options, **song.inject())
    assert song.loads == 0


def test_command_flags_for_the_stages(song, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    code = _main(
        song, "--stems", "vocals", "--split-vocals", "--dereverb", "--no-cache", "--out", str(out)
    )
    assert code == 0
    printed = capsys.readouterr().out
    for name in ("vocals", "lead_vocals", "backing_vocals", "vocals_dry"):
        assert (out / f"{name}.wav").exists()
        assert f"[{name}]" in printed
    assert "fake_kara: lead_vocals, backing_vocals" in printed


# --- the command line ----------------------------------------------------------------------


def _main(song, *args):
    return cli.main(
        [str(song.path), *OVERRIDES, "--device", "cpu", "--quiet", *args], **song.inject()
    )


def test_command_writes_next_to_the_input_by_default(song, capsys) -> None:
    code = _main(
        song, "--stems", "vocals,drums,bass,guitar,piano,other", "--cache-dir", str(song.cache_dir)
    )
    out_dir = song.path.with_name("song41_stems")
    assert code == 0
    assert sorted(p.name for p in out_dir.iterdir())[:2] == ["bass.wav", "drums.wav"]
    assert (out_dir / "manifest.json").exists()
    printed = capsys.readouterr().out
    assert "fake4: vocals, drums, bass, other" in printed
    assert str(out_dir / "piano.wav") in printed
    assert "residual:" in printed


def test_command_accepts_all_and_other_options(song, tmp_path, capsys) -> None:
    out = tmp_path / "chosen"
    code = _main(
        song,
        "--stems",
        "all",
        "--out",
        str(out),
        "--format",
        "flac",
        "--no-cache",
        "--other-method",
        "residual",
    )
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["request"]["format"] == "flac"
    assert manifest["other_method"] == "residual"
    assert manifest["stems"]["other"]["source"] == "mix minus the other five stems"
    assert (out / "other.flac").exists()


def test_command_reports_cached_runs(song, tmp_path, capsys) -> None:
    args = ("--stems", "drums", "--cache-dir", str(song.cache_dir))
    _main(song, *args, "--out", str(tmp_path / "a"))
    capsys.readouterr()
    _main(song, *args, "--out", str(tmp_path / "b"))
    assert "fake4: drums (cached)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "bad", [["--stems", "kazoo"], ["--stems", ""], ["--stems", "vocals", "--model", "kazoo=x"], []]
)
def test_usage_errors_exit_with_2(song, bad, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        _main(song, *bad)
    assert error.value.code == 2


def test_failures_exit_with_1_and_a_message(song, tmp_path, capsys) -> None:
    def run(*argv):
        code = cli.main([*argv, "--quiet", "--no-cache"], **song.inject())
        return code, capsys.readouterr().err

    code, err = run(str(tmp_path / "absent.wav"), "--stems", "vocals", *OVERRIDES)
    assert (code, "file not found" in err) == (1, True)

    text = tmp_path / "notes.wav"
    text.write_text("this is not audio\n" * 100)
    code, err = run(str(text), "--stems", "vocals", *OVERRIDES)
    assert (code, "not decodable audio" in err) == (1, True)

    code, err = run(str(song.path), "--stems", "vocals", *OVERRIDES, "--mode", "full_ensemble")
    assert (code, "not available yet" in err) == (1, True)

    code, err = run(str(song.path), "--stems", "vocals", "--model", "vocals=nope")
    assert (code, "not in the registry" in err) == (1, True)

    code, err = run(str(song.path), "--stems", "vocals", *OVERRIDES, "--device", "quantum")
    assert (code, "not usable" in err) == (1, True)
    assert not (tmp_path / "absent_stems").exists()


def test_unknown_format_is_rejected_before_any_work(song, tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown output format"):
        song.separate(tmp_path / "x", fmt="ogg")
    assert song.loads == 0
    with pytest.raises(PipelineError, match="not available yet"):
        song.separate(tmp_path / "x", mode="two_model_blend")


# --- acceptance: real models (need downloaded weights) -------------------------------------


@pytest.mark.slow
def test_command_with_the_default_models(tmp_path, capsys) -> None:
    from stems_common.settings import load_defaults
    from stems_models.download import is_downloaded

    registry, defaults = load_registry(), load_defaults()
    needed = {defaults.ensembles.slots[s].single_best for s in PRODUCT_STEMS}
    if not all(is_downloaded(registry.get(m)) for m in needed):
        pytest.skip("default models are not all downloaded")

    path = tmp_path / "fixture.wav"
    save_audio(path, generate_track(seed=1, duration_seconds=10.0).mix, SR, "wav")
    argv = [str(path), "--stems", "vocals,drums,bass,guitar,piano,other", "--quiet"]
    argv += ["--cache-dir", str(tmp_path / "cache"), "--out", str(tmp_path / "out")]

    assert cli.main(argv) == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        [f"{s}.wav" for s in PRODUCT_STEMS] + ["manifest.json"]
    )
    assert len(manifest["models"]) == 4
    assert all(m["wall_seconds"] > 0 and not m["cached"] for m in manifest["models"])

    assert cli.main([*argv[:-1], str(tmp_path / "again")]) == 0
    again = json.loads((tmp_path / "again" / "manifest.json").read_text())
    assert all(m["cached"] for m in again["models"])
    for stem in PRODUCT_STEMS:
        a = load_audio(tmp_path / "out" / f"{stem}.wav").samples
        b = load_audio(tmp_path / "again" / f"{stem}.wav").samples
        assert np.array_equal(a, b)


@pytest.mark.slow
def test_command_stages_with_the_default_models(tmp_path) -> None:
    from stems_common.settings import load_defaults
    from stems_models.download import is_downloaded

    registry, slots = load_registry(), load_defaults().ensembles.slots
    needed = {slots[s].single_best for s in ("vocals", "vocal_split", "dereverb")}
    if not all(is_downloaded(registry.get(m)) for m in needed):
        pytest.skip("default vocal, split and de-reverb models are not all downloaded")

    path = tmp_path / "fixture.wav"
    save_audio(path, generate_track(seed=1, duration_seconds=10.0).mix, SR, "wav")
    out = tmp_path / "out"
    argv = [str(path), "--stems", "vocals", "--split-vocals", "--dereverb", "--quiet"]
    assert cli.main([*argv, "--cache-dir", str(tmp_path / "cache"), "--out", str(out)]) == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert [m["stage"] for m in manifest["models"]] == ["separation", "vocal_split", "dereverb"]
    vocals = load_audio(out / "vocals.wav").samples
    lead = load_audio(out / "lead_vocals.wav").samples
    backing = load_audio(out / "backing_vocals.wav").samples
    dry = load_audio(out / "vocals_dry.wav").samples
    for audio in (lead, backing, dry):
        assert audio.shape == vocals.shape
        assert np.isfinite(audio).all()
    # The default splitter emits the lead and the rest of its input, so the two add back
    # up to the vocals stem (to within 24-bit rounding of three files).
    assert float(np.abs(lead + backing - vocals).max()) < 1e-5
