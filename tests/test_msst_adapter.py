"""The MSST adapter parses configs and weights safely and runs a real checkpoint."""

from __future__ import annotations

import io
import tarfile

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.download import is_downloaded  # noqa: E402
from stems_models.registry import load_registry  # noqa: E402
from stems_models.vendor import MSST_COMMIT, extract_subset, msst_root  # noqa: E402
from stems_worker.adapters.msst import (  # noqa: E402
    ModelConfigError,
    MsstSeparator,
    load_model_config,
    read_state_dict,
    resolve_outputs,
)
from stems_worker.device import usable_devices  # noqa: E402
from stems_worker.separator import run_separator  # noqa: E402

CONFIG = """\
audio:
  chunk_size: 352800
  sample_rate: 44100
model:
  dim: 8
  freqs_per_bands: !!python/tuple
    - 2
    - 4
  nested:
    sizes: !!python/tuple [1, 2, 3]
training:
  instruments: [vocals, other]
  target_instrument: vocals
"""


# --- config parsing ------------------------------------------------------------------------


def test_config_loader_accepts_tuples(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    config = load_model_config(path)
    assert config["model"]["freqs_per_bands"] == (2, 4)
    assert config["model"]["nested"]["sizes"] == (1, 2, 3)
    assert config["audio"]["chunk_size"] == 352800


@pytest.mark.parametrize(
    "payload",
    [
        "!!python/object/apply:os.system ['echo unsafe']",
        "!!python/object/new:subprocess.Popen [['true']]",
        "!!python/name:os.system",
        "!!python/module:os",
    ],
)
def test_config_loader_refuses_every_other_python_tag(tmp_path, payload) -> None:
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(CONFIG + f"evil: {payload}\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_model_config(path)


def test_config_must_have_the_expected_sections(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("audio: {chunk_size: 1}\nmodel: {}\n", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="training"):
        load_model_config(path)
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="mapping"):
        load_model_config(path)


# --- output naming -------------------------------------------------------------------------


def test_single_target_model_gains_the_complement() -> None:
    outputs, complement = resolve_outputs(["vocals", "instrumental"], ["vocals", "other"], "vocals")
    assert (outputs, complement) == (("vocals", "instrumental"), True)


def test_single_target_model_with_one_registry_stem_emits_only_that() -> None:
    assert resolve_outputs(["guitar"], ["guitar", "other"], "guitar") == (("guitar",), False)


def test_multi_stem_model_keeps_the_config_order() -> None:
    outputs, complement = resolve_outputs(
        ["vocals", "drums", "bass", "other"], ["bass", "drums", "other", "vocals"], None
    )
    assert outputs == ("bass", "drums", "other", "vocals")
    assert complement is False


def test_two_stem_other_is_called_instrumental() -> None:
    outputs, _ = resolve_outputs(["vocals", "instrumental"], ["vocals", "other"], None)
    assert outputs == ("vocals", "instrumental")


def test_registry_name_map_translates_config_names() -> None:
    karaoke = {"Vocals": "lead_vocals", "Instrumental": "backing_vocals"}
    stems = ["lead_vocals", "backing_vocals"]
    # Single target: the lead comes from the model, the rest is the complement.
    assert resolve_outputs(stems, ["Vocals", "Instrumental"], "Vocals", karaoke) == (
        ("lead_vocals", "backing_vocals"),
        True,
    )
    # Two heads: both come from the model, in config order.
    assert resolve_outputs(stems, ["Vocals", "Instrumental"], None, karaoke) == (
        ("lead_vocals", "backing_vocals"),
        False,
    )
    assert resolve_outputs(["dry"], ["noreverb", "reverb"], "noreverb", {"noreverb": "dry"}) == (
        ("dry",),
        False,
    )


def test_heads_the_pipeline_does_not_use_can_be_ignored() -> None:
    # A two-head de-reverb model emits the dry signal and the reverb it removed.
    assert resolve_outputs(["dry"], ["dry", "other"], None, None, ("other",)) == (("dry",), False)
    with pytest.raises(ModelConfigError, match="do not match"):
        resolve_outputs(["dry"], ["dry", "other"], None)


def test_config_names_are_lower_cased_when_no_map_is_given() -> None:
    assert resolve_outputs(["guitar"], ["Guitar", "Other"], "Guitar") == (("guitar",), False)
    outputs, _ = resolve_outputs(["vocals", "instrumental"], ["Vocals", "Instrumental"], None)
    assert outputs == ("vocals", "instrumental")


def test_mismatched_config_and_registry_are_reported() -> None:
    with pytest.raises(ModelConfigError, match="do not match"):
        resolve_outputs(["vocals", "drums"], ["vocals", "bass"], None)
    with pytest.raises(ModelConfigError, match="targets 'piano'"):
        resolve_outputs(["vocals", "instrumental"], ["piano", "other"], "piano")
    with pytest.raises(ModelConfigError, match="at most two"):
        resolve_outputs(["vocals", "drums", "bass"], ["vocals"], "vocals")


# --- weights -------------------------------------------------------------------------------


def test_state_dict_is_read_from_bare_and_wrapped_checkpoints(tmp_path) -> None:
    tensors = {"layer.weight": torch.ones(2, 2), "layer.bias": torch.zeros(2)}
    for name, payload in {
        "bare.ckpt": tensors,
        "state.ckpt": {"state": tensors, "epoch": 3},
        "state_dict.ckpt": {"state_dict": tensors},
    }.items():
        torch.save(payload, tmp_path / name)
        loaded = read_state_dict(tmp_path / name)
        assert set(loaded) == set(tensors)
        assert torch.equal(loaded["layer.weight"], tensors["layer.weight"])


class _Payload:
    """Stands in for the arbitrary object a malicious checkpoint would carry."""


def test_checkpoints_carrying_arbitrary_objects_are_refused(tmp_path) -> None:
    import pickle

    path = tmp_path / "evil.ckpt"
    torch.save({"layer.weight": torch.ones(1), "extra": _Payload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        read_state_dict(path)


def test_non_tensor_state_dicts_are_refused(tmp_path) -> None:
    path = tmp_path / "odd.ckpt"
    torch.save({"layer.weight": torch.ones(1), "note": "hello"}, path)
    with pytest.raises(ModelConfigError, match="non-tensor"):
        read_state_dict(path)
    torch.save({}, path)
    with pytest.raises(ModelConfigError, match="not a state dict"):
        read_state_dict(path)


# --- vendored source -----------------------------------------------------------------------


def _archive(path, members: dict[str, bytes], links: dict[str, str] | None = None) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)


def test_only_the_wanted_files_are_unpacked_and_nothing_escapes(tmp_path) -> None:
    archive = tmp_path / "src.tar.gz"
    _archive(
        archive,
        {
            "repo-abc/models/net.py": b"x = 1\n",
            "repo-abc/models/sub/deep.py": b"y = 2\n",
            "repo-abc/LICENSE": b"MIT\n",
            "repo-abc/train.py": b"not wanted\n",
            "repo-abc/modelsX/sneaky.py": b"not wanted\n",
            "repo-abc/models/../../escape.py": b"escape\n",
            "repo-abc/models/../outside.py": b"escape\n",
        },
        links={"repo-abc/models/link.py": "/etc/passwd"},
    )
    out = tmp_path / "out"
    written = extract_subset(archive, out, ("models/", "LICENSE"))

    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["LICENSE", "models/net.py", "models/sub/deep.py"]
    assert written == 3
    assert not (tmp_path / "escape.py").exists()
    assert not (out / "outside.py").exists()
    assert not (out / "models" / "link.py").exists()


def test_vendored_directory_is_named_after_the_pinned_commit(tmp_path) -> None:
    assert len(MSST_COMMIT) == 40
    assert msst_root(tmp_path).name == f"msst-{MSST_COMMIT[:12]}"


# --- adapter construction ------------------------------------------------------------------


def test_adapter_rejects_entries_it_cannot_load(tmp_path) -> None:
    registry = load_registry()
    with pytest.raises(ModelConfigError, match="not 'msst'"):
        MsstSeparator(registry.get("htdemucs"), {})
    untyped = registry.get("scnet_large").model_copy(update={"config": {"config_file": "c.yaml"}})
    with pytest.raises(ModelConfigError, match="no model_type"):
        MsstSeparator(untyped, {})
    with pytest.raises(ModelConfigError, match="need its config"):
        MsstSeparator(registry.get("mel_roformer_kim_vocals"), {})


def test_every_msst_entry_names_a_supported_model_type() -> None:
    from stems_worker.adapters.msst import _BUILDERS

    for entry in load_registry().models:
        if entry.loader == "msst":
            assert entry.config.get("model_type") in _BUILDERS, entry.id


def test_unsupported_model_type_is_reported() -> None:
    from stems_worker.adapters.msst import build_model

    with pytest.raises(ModelConfigError, match="'kazoo_net' is not supported"):
        build_model("kazoo_net", {"model": {}})


def test_config_namespace_reads_keys_as_attributes() -> None:
    from stems_worker.adapters.msst import _as_namespace

    config = _as_namespace({"audio": {"dim_f": 1024}, "training": {"instruments": ["a", "b"]}})
    assert config.audio.dim_f == 1024
    assert config.training.get("target_instrument") is None
    assert config.training.instruments == ["a", "b"]
    with pytest.raises(AttributeError):
        _ = config.audio.missing


MSST_IDS = [m.id for m in load_registry().models if m.loader == "msst"]


@pytest.mark.parametrize("model_id", MSST_IDS)
def test_downloaded_configs_agree_with_the_registry(model_id) -> None:
    """Cheap consistency check that needs only the small config file, not the weights."""
    from stems_models.download import model_dir

    entry = load_registry().get(model_id)
    path = model_dir(model_id) / entry.config["config_file"]
    if not path.exists():
        pytest.skip(f"{path.name} is not downloaded")
    config = load_model_config(path)
    outputs, _ = resolve_outputs(
        list(entry.stems),
        [str(i) for i in config["training"].get("instruments", [])],
        config["training"].get("target_instrument"),
        entry.config.get("instrument_names"),
        tuple(entry.config.get("ignore_instruments", ())),
    )
    assert set(outputs) == set(entry.stems)
    assert config["audio"].get("sample_rate", 44100) == entry.sample_rate_hz


# --- acceptance: real checkpoints (need downloaded weights) --------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_id", MSST_IDS)
def test_every_msst_model_loads_and_emits_its_registry_outputs(model_id) -> None:
    from stems_worker.device import pick_device

    entry = load_registry().get(model_id)
    if not is_downloaded(entry):
        pytest.skip(f"{model_id} is not downloaded")

    track = generate_track(seed=1, duration_seconds=6.0)
    if "vocal_split" in entry.candidate_for:
        audio = track.stems["vocals"]
    elif "dereverb" in entry.candidate_for:
        audio = track.vocals_wet
    else:
        audio = track.mix

    separator = MsstSeparator.from_registry(model_id)
    separator.load(pick_device())
    result = run_separator(separator, audio, 44100)
    separator.unload()

    assert set(result.stems) == set(entry.stems)
    for stem in result.stems.values():
        assert stem.shape == audio.shape
        assert np.isfinite(stem).all()
    if len(result.stems) > 1:
        # Models that split the input into parts must account for all of it, to within
        # what a network leaves over (complement models are exact by construction).
        residual = audio - sum(result.stems.values())
        assert float(np.sqrt(np.mean(residual**2))) < 0.5 * float(np.sqrt(np.mean(audio**2)))
    # Behavioural check where the fixture can give one: drums and bass are recognisable
    # even as synthetic tones, so a model that separates them must route them correctly.
    # Measured here: drums 0.82 to 0.95, bass 0.99 to 1.02.
    for name in ("drums", "bass"):
        if name in result.stems and audio is track.mix:
            assert _fraction(result.stems[name], track.stems[name]) > 0.5, name
            other = "bass" if name == "drums" else "drums"
            assert _fraction(result.stems[name], track.stems[other]) < 0.4, name


KIM = "mel_roformer_kim_vocals"


def _fraction(output, stem) -> float:
    """How much of ``stem`` ended up in ``output`` (1.0 = all of it)."""
    s = stem.astype(np.float64).ravel()
    return float(np.dot(output.astype(np.float64).ravel(), s) / np.dot(s, s))


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_kim_vocals_separates_a_ten_second_fixture_clip(device) -> None:
    if not is_downloaded(load_registry().get(KIM)):
        pytest.skip(f"{KIM} is not downloaded (python -m stems_models.download {KIM})")
    if device not in usable_devices():
        pytest.skip(f"{device} is not usable on this machine")

    track = generate_track(seed=1, duration_seconds=10.0)
    separator = MsstSeparator.from_registry(KIM)
    assert separator.outputs == ("vocals", "instrumental")
    assert separator.chunk_size == 352800
    separator.load(device)
    result = run_separator(separator, track.mix, 44100, overlap=0.25)
    separator.unload()

    vocals, instrumental = result.stems["vocals"], result.stems["instrumental"]
    assert result.device == device
    assert result.num_chunks == 2
    assert vocals.shape == instrumental.shape == track.mix.shape
    assert np.isfinite(vocals).all()
    assert float(np.abs(vocals + instrumental - track.mix).max()) < 1e-6

    # Sanity, not quality: the fixture is sine tones, not singing. A vocal model should
    # still keep drums and bass out of the vocal output and pull in some of the vibrato
    # "voice". Measured on this machine: drums 0.000, bass 0.001, voice 0.51.
    assert abs(_fraction(vocals, track.stems["drums"])) < 0.05
    assert abs(_fraction(vocals, track.stems["bass"])) < 0.05
    assert _fraction(vocals, track.stems["vocals"]) > 0.2
