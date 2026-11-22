"""The MDX-Net adapter finds its settings by hash, maps outputs correctly, and runs."""

from __future__ import annotations

import hashlib
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.download import is_downloaded  # noqa: E402
from stems_models.registry import load_registry  # noqa: E402
from stems_worker.adapters.mdx import (  # noqa: E402
    HOP,
    MdxConfigError,
    MdxParams,
    MdxSeparator,
    read_mdx_params,
    resolve_mdx_outputs,
    spectrogram,
    uvr_model_hash,
    waveform,
)
from stems_worker.separator import run_separator  # noqa: E402

REGISTRY = load_registry()
ROW = {
    "compensate": 1.035,
    "mdx_dim_f_set": 1025,
    "mdx_dim_t_set": 6,
    "mdx_n_fft_scale_set": 2048,
    "primary_stem": "Vocals",
}


def _fake_model(tmp_path, row=ROW, name="model.onnx"):
    """A stand-in weights file plus a model-data table that knows its hash."""
    weights = tmp_path / name
    weights.write_bytes(b"not really onnx " * 100)
    table = tmp_path / "mdx_model_data.json"
    existing = json.loads(table.read_text()) if table.exists() else {}
    existing[uvr_model_hash(weights)] = row
    table.write_text(json.dumps(existing), encoding="utf-8")
    return weights, table


# --- model data lookup ---------------------------------------------------------------------


def test_hash_covers_the_whole_of_a_small_file(tmp_path) -> None:
    path = tmp_path / "small.onnx"
    path.write_bytes(b"abc" * 1000)
    assert uvr_model_hash(path) == hashlib.md5(b"abc" * 1000).hexdigest()


def test_hash_covers_only_the_tail_of_a_large_file(tmp_path) -> None:
    tail = bytes(range(256)) * 40_000  # exactly 10,240,000 bytes
    path = tmp_path / "large.onnx"
    path.write_bytes(b"HEADER-THAT-IS-NOT-HASHED" + tail)
    assert uvr_model_hash(path) == hashlib.md5(tail).hexdigest()


def test_params_are_read_from_the_table(tmp_path) -> None:
    weights, table = _fake_model(tmp_path)
    params = read_mdx_params(weights, table)
    assert params == MdxParams(
        n_fft=2048, dim_f=1025, dim_t=64, compensate=1.035, primary_stem="Vocals"
    )
    assert params.chunk_size == HOP * 63


def test_unknown_or_implausible_models_are_reported(tmp_path) -> None:
    weights, table = _fake_model(tmp_path)
    other = tmp_path / "other.onnx"
    other.write_bytes(b"different")
    with pytest.raises(MdxConfigError, match="no usable entry"):
        read_mdx_params(other, table)
    for bad in ({"mdx_dim_f_set": 5000}, {"mdx_n_fft_scale_set": 2047}, {"compensate": 40.0}):
        weights, table = _fake_model(tmp_path, {**ROW, **bad})
        with pytest.raises(MdxConfigError, match="implausible"):
            read_mdx_params(weights, table)
    incomplete = {k: v for k, v in ROW.items() if k != "primary_stem"}
    weights, table = _fake_model(tmp_path, incomplete)
    with pytest.raises(MdxConfigError, match="primary_stem"):
        read_mdx_params(weights, table)


# --- output naming -------------------------------------------------------------------------


def test_primary_stem_decides_which_output_comes_from_the_model() -> None:
    stems = ["vocals", "instrumental"]
    assert resolve_mdx_outputs(stems, "Vocals", None, None) == (("vocals", "instrumental"), True)
    assert resolve_mdx_outputs(stems, "Instrumental", None, None) == (
        ("instrumental", "vocals"),
        True,
    )


def test_karaoke_names_come_from_the_registry_map() -> None:
    stems = ["lead_vocals", "backing_vocals"]
    assert resolve_mdx_outputs(stems, "Vocals", {"Vocals": "lead_vocals"}, None) == (
        ("lead_vocals", "backing_vocals"),
        True,
    )
    assert resolve_mdx_outputs(stems, "Instrumental", {"Instrumental": "backing_vocals"}, None) == (
        ("backing_vocals", "lead_vocals"),
        True,
    )


def test_per_stem_files_emit_only_their_stem_and_must_match() -> None:
    four = ["vocals", "drums", "bass", "other"]
    assert resolve_mdx_outputs(four, "Drums", None, "drums") == (("drums",), False)
    with pytest.raises(MdxConfigError, match="primary stem 'Bass'"):
        resolve_mdx_outputs(four, "Bass", None, "drums")
    with pytest.raises(MdxConfigError, match="not among"):
        resolve_mdx_outputs(["vocals", "instrumental"], "Drums", None, None)


# --- STFT ----------------------------------------------------------------------------------


def test_spectrogram_has_the_shape_the_model_expects_and_inverts() -> None:
    params = MdxParams(n_fft=2048, dim_f=1025, dim_t=64, compensate=1.0, primary_stem="Vocals")
    window = torch.hann_window(params.n_fft, periodic=True)
    x = torch.from_numpy(
        generate_track(seed=3, duration_seconds=2.0).mix[None, :, : params.chunk_size]
    )
    spec = spectrogram(x, params, window)
    assert spec.shape == (1, 4, params.dim_f, params.dim_t)
    restored = waveform(spec, params, window)
    assert restored.shape == x.shape
    assert float((restored - x).abs().max()) < 1e-5


def test_bins_above_dim_f_are_dropped() -> None:
    full = MdxParams(n_fft=2048, dim_f=1025, dim_t=64, compensate=1.0, primary_stem="Vocals")
    half = MdxParams(n_fft=2048, dim_f=512, dim_t=64, compensate=1.0, primary_stem="Vocals")
    window = torch.hann_window(2048, periodic=True)
    t = torch.arange(full.chunk_size) / 44100
    low, high = torch.sin(2 * torch.pi * 1000 * t), torch.sin(2 * torch.pi * 15000 * t)
    x = (low + high).repeat(1, 2, 1) * 0.25
    kept = waveform(spectrogram(x, half, window), half, window)
    inner = slice(4096, -4096)
    # 512 bins of a 2048-point FFT reach about 11 kHz: the 1 kHz tone survives alone
    # (to within the leakage of a brick-wall cut), and the 15 kHz tone is gone.
    assert float((kept - 0.25 * low)[..., inner].abs().max()) < 5e-3
    assert float((kept - x)[..., inner].abs().max()) > 0.2


# --- the adapter, with a stand-in for the ONNX session -------------------------------------


class _FakeSession:
    """Returns the input scaled, plus a fixed offset, like a slightly-off model.

    The offset varies across frequency bins. (An offset that is the same in every bin
    would be an impulse at the start of each frame, where the synthesis window is zero,
    and so would vanish from the audio.)
    """

    def __init__(self, scale: float = 0.5, bias: float = 0.0) -> None:
        self.scale, self.bias = scale, bias
        self.calls = 0

    def run(self, _outputs, feed):
        self.calls += 1
        (spec,) = feed.values()
        shape = np.cos(np.arange(spec.shape[2], dtype=np.float32) * 0.37)[None, None, :, None]
        return [spec * self.scale + self.bias * shape]


def _separator(tmp_path, model_id="uvr_mdxnet_voc_ft", stem=None, row=ROW, **kw):
    entry = REGISTRY.get(model_id)
    weights_name = next(f.filename for f in entry.weights if f.stem == stem)
    weights, table = _fake_model(tmp_path, row, weights_name)
    separator = MdxSeparator(
        entry, {weights_name: weights, "mdx_model_data.json": table}, stem, **kw
    )
    separator._session = _FakeSession()
    separator._input_name, separator._fixed_batch = "input", None
    separator.device = "cpu"
    return separator


def test_primary_gets_the_compensation_gain_and_the_rest_is_the_complement(tmp_path) -> None:
    separator = _separator(tmp_path)
    assert separator.outputs == ("vocals", "instrumental")
    mix = generate_track(seed=8, duration_seconds=3.0).mix
    result = run_separator(separator, mix, 44100)
    expected = mix * 0.5 * 1.035
    assert float(np.abs(result.stems["vocals"] - expected).max()) < 1e-4
    assert float(np.abs(result.stems["vocals"] + result.stems["instrumental"] - mix).max()) < 1e-6


def test_denoise_cancels_a_constant_model_bias(tmp_path) -> None:
    mix = generate_track(seed=8, duration_seconds=2.0).mix
    plain = _separator(tmp_path)
    plain._session = _FakeSession(scale=0.5, bias=0.3)
    biased = run_separator(plain, mix, 44100).stems["vocals"]

    denoised = _separator(tmp_path, denoise=True)
    denoised._session = _FakeSession(scale=0.5, bias=0.3)
    clean = run_separator(denoised, mix, 44100).stems["vocals"]

    expected = mix * 0.5 * 1.035
    assert float(np.abs(clean - expected).max()) < 1e-4
    assert float(np.abs(biased - expected).max()) > 1e-3  # measured 0.0074
    assert denoised._session.calls == 2 * plain._session.calls


def test_fixed_batch_models_are_fed_one_chunk_at_a_time(tmp_path) -> None:
    separator = _separator(tmp_path)
    separator._fixed_batch = 1
    mix = generate_track(seed=8, duration_seconds=4.0).mix
    result = run_separator(separator, mix, 44100, batch_size=4)
    assert separator._session.calls == result.num_chunks
    assert float(np.abs(result.stems["vocals"] - mix * 0.5 * 1.035).max()) < 1e-4


def test_per_stem_bags_need_a_stem_and_single_models_refuse_one(tmp_path) -> None:
    drums_row = {**ROW, "primary_stem": "Drums"}
    separator = _separator(tmp_path, "mdxnet_kuielab_a", stem="drums", row=drums_row)
    assert separator.outputs == ("drums",)

    bag = REGISTRY.get("mdxnet_kuielab_a")
    with pytest.raises(MdxConfigError, match="pick one of"):
        MdxSeparator(bag, {})
    with pytest.raises(MdxConfigError, match="no model for 'piano'"):
        MdxSeparator(bag, {}, "piano")
    with pytest.raises(MdxConfigError, match="takes no stem"):
        MdxSeparator(REGISTRY.get("uvr_mdxnet_voc_ft"), {}, "vocals")
    with pytest.raises(MdxConfigError, match="missing files"):
        MdxSeparator(REGISTRY.get("uvr_mdxnet_voc_ft"), {})
    with pytest.raises(MdxConfigError, match="not 'uvr_mdx_onnx'"):
        MdxSeparator(REGISTRY.get("htdemucs"), {})


def test_every_mdx_entry_lists_its_model_data_file() -> None:
    for entry in REGISTRY.models:
        if entry.loader == "uvr_mdx_onnx":
            assert entry.config["params_from"] in {f.filename for f in entry.files}, entry.id


# --- acceptance: real models (need downloaded weights) -------------------------------------

MDX_CASES = [
    (entry.id, stem)
    for entry in REGISTRY.models
    if entry.loader == "uvr_mdx_onnx"
    for stem in ([f.stem for f in entry.weights if f.stem] or [None])
]


def _fraction(output, stem) -> float:
    s = stem.astype(np.float64).ravel()
    return float(np.dot(output.astype(np.float64).ravel(), s) / np.dot(s, s))


@pytest.mark.slow
@pytest.mark.parametrize(("model_id", "stem"), MDX_CASES)
def test_mdx_models_run_and_map_their_outputs_correctly(model_id, stem) -> None:
    entry = REGISTRY.get(model_id)
    if not is_downloaded(entry, stem=stem):
        pytest.skip(f"{model_id} is not downloaded")
    track = generate_track(seed=1, duration_seconds=10.0)
    karaoke = "vocal_split" in entry.candidate_for
    audio = track.stems["vocals"] if karaoke else track.mix

    separator = MdxSeparator.from_registry(model_id, stem=stem)
    separator.load("cpu")
    result = run_separator(separator, audio, 44100)
    separator.unload()

    assert set(result.stems) == ({stem} if stem else set(entry.stems))
    assert result.device == "cpu"
    for output in result.stems.values():
        assert output.shape == audio.shape
        assert np.isfinite(output).all()

    drums, bass = track.stems["drums"], track.stems["bass"]
    if "instrumental" in result.stems:
        # The decisive check for primary versus secondary: whichever output the model
        # emits, drums and bass belong in the instrumental and not in the vocals.
        # Measured: 1.03 in the instrumental, 0.00 in the vocals, for all seven models.
        assert _fraction(result.stems["instrumental"], drums) > 0.8
        assert _fraction(result.stems["instrumental"], bass) > 0.8
        assert abs(_fraction(result.stems["vocals"], drums)) < 0.2
        assert abs(_fraction(result.stems["vocals"], bass)) < 0.2
    if stem == "drums":
        assert _fraction(result.stems["drums"], drums) > 0.6  # measured 0.81 and 0.89
        assert _fraction(result.stems["drums"], bass) < 0.4
    if stem == "bass":
        assert _fraction(result.stems["bass"], bass) > 0.6  # measured 0.92 and 0.93
        assert _fraction(result.stems["bass"], drums) < 0.4
    if karaoke:
        # Direction only: the lead output takes more of the lead than of the backing.
        # Measured: 0.38 against 0.02, and 0.13 against 0.07.
        lead = result.stems["lead_vocals"]
        assert _fraction(lead, track.lead_vocals) > _fraction(lead, track.backing_vocals)
        total = result.stems["lead_vocals"] + result.stems["backing_vocals"]
        assert float(np.abs(total - audio).max()) < 1e-5
