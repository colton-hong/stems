"""The VR adapter reads UVR's model data, maps its two outputs by name, and runs."""

from __future__ import annotations

import json
import os
import zipfile

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.download import is_downloaded  # noqa: E402
from stems_models.registry import load_registry  # noqa: E402
from stems_models.vendor import AUDIO_SEPARATOR_VERSION, extract_zip_subset, uvr_lib_root  # noqa: E402
from stems_worker.adapters.mdx import uvr_model_hash  # noqa: E402
from stems_worker.adapters.vr import (  # noqa: E402
    VrConfigError,
    VrParams,
    VrSeparator,
    arch_size_for,
    read_vr_params,
)
from stems_worker.device import pick_device  # noqa: E402
from stems_worker.separator import run_separator  # noqa: E402

REGISTRY = load_registry()
VR_IDS = [m.id for m in REGISTRY.models if m.loader == "uvr_vr"]


def _fake_model(tmp_path, row, name="model.pth"):
    weights = tmp_path / name
    weights.write_bytes(b"not really weights " * 50)
    table = tmp_path / "vr_model_data.json"
    table.write_text(json.dumps({uvr_model_hash(weights): row}), encoding="utf-8")
    return weights, table


# --- model data ----------------------------------------------------------------------------


def test_params_are_read_from_the_table(tmp_path) -> None:
    weights, table = _fake_model(
        tmp_path, {"vr_model_param": "4band_v3", "primary_stem": "No Echo"}
    )
    assert read_vr_params(weights, table) == VrParams("4band_v3", "No Echo", None, None)

    row = {"vr_model_param": "4band_v3", "primary_stem": "Noise", "nout": 48, "nout_lstm": 128}
    weights, table = _fake_model(tmp_path, row)
    assert read_vr_params(weights, table) == VrParams("4band_v3", "Noise", 48, 128)


def test_unknown_or_malformed_entries_are_reported(tmp_path) -> None:
    weights, table = _fake_model(tmp_path, {"primary_stem": "No Echo"})
    with pytest.raises(VrConfigError, match="vr_model_param"):
        read_vr_params(weights, table)
    other = tmp_path / "other.pth"
    other.write_bytes(b"different")
    with pytest.raises(VrConfigError, match="no usable entry"):
        read_vr_params(other, table)
    # The layout name becomes a file name, so it must not be able to leave its folder.
    weights, table = _fake_model(tmp_path, {"vr_model_param": "../../x", "primary_stem": "A"})
    with pytest.raises(VrConfigError, match="odd band layout name"):
        read_vr_params(weights, table)


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [(126_782_699, 123812), (127_139_365, 123821), (223_650_277, 218409), (58_000_000, 56817)],
)
def test_network_size_is_the_nearest_known_one(tmp_path, size_bytes, expected) -> None:
    path = tmp_path / "sized.pth"
    path.touch()
    os.truncate(path, size_bytes)  # sparse: no real disk use
    assert arch_size_for(path) == expected


# --- output naming -------------------------------------------------------------------------


def _separator(tmp_path, model_id, primary_stem):
    entry = REGISTRY.get(model_id)
    name = entry.weights[0].filename
    weights, table = _fake_model(
        tmp_path, {"vr_model_param": "4band_v3", "primary_stem": primary_stem}, name
    )
    return VrSeparator(entry, {name: weights, "vr_model_data.json": table})


def test_karaoke_models_emit_backing_then_lead(tmp_path) -> None:
    separator = _separator(tmp_path, "uvr_vr_5hp_karaoke", "Instrumental")
    assert separator.outputs == ("backing_vocals", "lead_vocals")
    assert separator._picks == [0, 1]
    assert separator.chunk_size == 20 * 44100


def test_cleanup_models_keep_only_the_dry_output(tmp_path) -> None:
    # De-echo: the model's primary output is the clean signal.
    assert _separator(tmp_path, "uvr_vr_deecho_normal", "No Echo")._picks == [0]
    # De-noise: the primary output is the noise, so the clean signal is the second one.
    denoise = _separator(tmp_path, "uvr_vr_denoise", "Noise")
    assert (denoise.outputs, denoise._picks) == (("dry",), [1])


def test_disagreement_with_uvr_model_data_is_an_error(tmp_path) -> None:
    with pytest.raises(VrConfigError, match="expects primary stem 'No Echo'"):
        _separator(tmp_path, "uvr_vr_deecho_normal", "Vocals")
    with pytest.raises(VrConfigError, match="missing files"):
        VrSeparator(REGISTRY.get("uvr_vr_denoise"), {})
    with pytest.raises(VrConfigError, match="not 'uvr_vr'"):
        VrSeparator(REGISTRY.get("htdemucs"), {})


@pytest.mark.parametrize("model_id", VR_IDS)
def test_registry_entries_name_their_outputs(model_id) -> None:
    config = REGISTRY.get(model_id).config
    named = {config.get("primary"), config.get("secondary")} - {None}
    assert config["primary_stem"]
    assert named == set(REGISTRY.get(model_id).stems)
    assert config["params_from"] == "vr_model_data.json"


# --- vendored source -----------------------------------------------------------------------


def test_only_the_wanted_wheel_files_are_unpacked(tmp_path) -> None:
    wheel = tmp_path / "pkg.whl"
    with zipfile.ZipFile(wheel, "w") as bundle:
        bundle.writestr("pkg/lib/nets.py", "x = 1\n")
        bundle.writestr("pkg/lib/params/a.json", "{}")
        bundle.writestr("pkg/__init__.py", "import everything\n")
        bundle.writestr("pkg/lib/../../escape.py", "escape\n")
        bundle.writestr("/absolute.py", "escape\n")
        bundle.writestr("pkg/libX/sneaky.py", "no\n")
    out = tmp_path / "out"
    written = extract_zip_subset(wheel, out, ("pkg/lib/",))
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["pkg/lib/nets.py", "pkg/lib/params/a.json"]
    assert written == 2
    assert not (tmp_path / "escape.py").exists()


def test_vendored_directory_is_named_after_the_pinned_version(tmp_path) -> None:
    assert uvr_lib_root(tmp_path).name == f"audio_separator-{AUDIO_SEPARATOR_VERSION}"


# --- acceptance: real models (need downloaded weights) -------------------------------------


def _fraction(output, stem) -> float:
    s = stem.astype(np.float64).ravel()
    return float(np.dot(output.astype(np.float64).ravel(), s) / np.dot(s, s))


def _level_db(x) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-12))


@pytest.mark.slow
@pytest.mark.parametrize("model_id", VR_IDS)
def test_vr_models_run_on_the_fixture(model_id) -> None:
    entry = REGISTRY.get(model_id)
    if not is_downloaded(entry):
        pytest.skip(f"{model_id} is not downloaded")
    track = generate_track(seed=1, duration_seconds=10.0)
    karaoke = "vocal_split" in entry.candidate_for
    audio = track.stems["vocals"] if karaoke else track.vocals_wet

    separator = VrSeparator.from_registry(model_id)
    separator.load(pick_device())
    result = run_separator(separator, audio, 44100)
    separator.unload()

    assert set(result.stems) == set(entry.stems)
    for output in result.stems.values():
        assert output.shape == audio.shape
        assert np.isfinite(output).all()

    if karaoke:
        lead, backing = result.stems["lead_vocals"], result.stems["backing_vocals"]
        # These models do split the fixture's centred lead from its panned backing.
        # Measured: lead output 0.67 and 0.90 of the lead, 0.11 and 0.27 of the backing;
        # backing output 0.95 and 0.80 of the backing.
        assert _fraction(lead, track.lead_vocals) > _fraction(lead, track.backing_vocals) + 0.2
        assert _fraction(backing, track.backing_vocals) > 0.5
        # The two outputs come from a mask and its inverse, so they add back to the input.
        assert _level_db(lead + backing - audio) - _level_db(audio) < -30  # measured -47, -51
    else:
        dry = result.stems["dry"]
        # Sanity, not quality: synthetic reverb on synthetic tones is not what these were
        # trained on. The clean output must still be mostly the dry signal, and no louder
        # than the input. Measured: 0.83 to 0.96 of the dry signal kept.
        assert _fraction(dry, track.stems["vocals"]) > 0.5
        assert _level_db(dry) < _level_db(audio) + 1.0
