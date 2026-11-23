"""The CLAP wrapper: windowing and filtering without weights, the model itself with them."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("soxr")

from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.registry import load_registry  # noqa: E402
from stems_models.vendor import CLAP_KEEP, CLAP_STUBS, LAION_CLAP_WHEEL  # noqa: E402
from stems_worker.listening import clap  # noqa: E402
from stems_worker.listening.clap import (  # noqa: E402
    SAMPLE_RATE,
    WINDOW,
    ClapError,
    ClapTagger,
    low_pass,
    quantise,
    to_mono,
    windows,
)


def _tone(hz, seconds, amplitude=0.3):
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# --- no weights needed --------------------------------------------------------------------


def test_to_mono_averages_channels_and_resamples_to_48k() -> None:
    stereo = np.stack([np.ones(44100, np.float32), np.zeros(44100, np.float32)])
    mono = to_mono(stereo, 44100)
    assert mono.shape == (48000,)
    assert mono.dtype == np.float32
    assert float(np.median(mono)) == pytest.approx(0.5, abs=1e-3)
    assert to_mono(np.ones(48000, np.float32), 48000).shape == (48000,)


def test_windows_are_ten_seconds_with_a_repeated_padded_tail() -> None:
    assert windows(np.ones(2 * WINDOW, np.float32)).shape == (2, WINDOW)
    # 24 s: two full windows and a 4 s tail, repeated twice and padded with silence.
    ramp = np.arange(24 * SAMPLE_RATE, dtype=np.float32)
    w = windows(ramp)
    assert w.shape == (3, WINDOW)
    tail = ramp[2 * WINDOW :]
    np.testing.assert_array_equal(w[2, : tail.size], tail)
    np.testing.assert_array_equal(w[2, tail.size : 2 * tail.size], tail)
    assert not w[2, 2 * tail.size :].any()
    # A clip shorter than a window is still one window; a sub-second scrap after a full
    # window is dropped.
    assert windows(np.ones(SAMPLE_RATE // 2, np.float32)).shape == (1, WINDOW)
    assert windows(np.ones(WINDOW + 100, np.float32)).shape == (1, WINDOW)
    with pytest.raises(ClapError, match="non-empty mono"):
        windows(np.zeros(0, np.float32))
    with pytest.raises(ClapError, match="non-empty mono"):
        windows(np.zeros((2, 100), np.float32))


def test_quantise_matches_sixteen_bit_and_clips() -> None:
    x = np.array([0.0, 0.5, -0.5, 1.5, -2.0, 1e-6], dtype=np.float32)
    q = quantise(x)
    assert q.dtype == np.float32
    assert q.tolist() == pytest.approx([0.0, 16383 / 32767, -16383 / 32767, 1.0, -1.0, 0.0])


def test_low_pass_keeps_what_is_below_8_khz_and_removes_what_is_above() -> None:
    low, high = _tone(1000, 2.0), _tone(12000, 2.0)
    out = low_pass(low + high)
    assert out.shape == low.shape
    assert float(np.abs(out - low).max()) < 1e-3
    assert float(np.sqrt(np.mean(low_pass(high) ** 2))) < 1e-4
    assert float(np.abs(low_pass(_tone(7000, 2.0)) - _tone(7000, 2.0)).max()) < 1e-3


def test_vendored_code_is_pinned_and_small() -> None:
    assert "laion_clap-1.1.7-py3-none-any.whl" in LAION_CLAP_WHEEL
    assert LAION_CLAP_WHEEL.startswith("https://files.pythonhosted.org/")
    kept = [name for name in CLAP_KEEP if name.endswith(".py")]
    assert sorted(kept) == [
        "laion_clap/clap_module/feature_fusion.py",
        "laion_clap/clap_module/htsat.py",
        "laion_clap/clap_module/model.py",
    ]
    for source in CLAP_STUBS.values():
        compile(source, "<stub>", "exec")
    namespace: dict = {}
    exec(CLAP_STUBS["laion_clap/clap_module/utils.py"], namespace)
    x = torch.arange(6.0).reshape(1, 3, 2)
    up = namespace["interpolate"](x, 2)
    assert up.shape == (1, 6, 2)
    assert up[0, :, 0].tolist() == [0.0, 0.0, 2.0, 2.0, 4.0, 4.0]


def test_registry_entry_names_the_checkpoint_tokenizer_and_text_config() -> None:
    entry = load_registry().get("laion_clap_music")
    assert entry.loader == "laion_clap"
    names = {f.filename: f.role for f in entry.files}
    assert names == {
        "music_audioset_epoch_15_esc_90.14.pt": "weights",
        "roberta_base_tokenizer.json": "config",
        "roberta_base_config.json": "config",
    }
    # Hub files are pinned to a commit, not to a moving branch.
    hub = [f.url for f in entry.files if "roberta-base" in f.url]
    assert all("/resolve/e2da8e2f811d1448a5b465c236feacd80ffbac7b/" in url for url in hub)


class _FakeModel:
    """Embeds a window as its energy below and above 8 kHz, so tests can reason about it."""

    logit_scale_a = torch.tensor(0.0)

    def parameters(self):
        yield torch.zeros(1)

    def get_audio_embedding(self, data):
        rows = []
        for item in data:
            spectrum = torch.fft.rfft(item["waveform"].double()).abs() ** 2
            split = round(8000 * WINDOW / SAMPLE_RATE)
            v = torch.stack([spectrum[:split].sum(), spectrum[split:].sum()]) + 1e-12
            rows.append(v / v.norm())
        return torch.stack(rows)


def test_high_band_is_zero_without_top_end_and_grows_with_it() -> None:
    tagger = ClapTagger.__new__(ClapTagger)
    tagger.model_id, tagger._model = "fake", _FakeModel()
    low = _tone(1000, 10.0)
    quiet_top = low + _tone(12000, 10.0, 0.05)
    loud_top = low + _tone(12000, 10.0, 0.3)
    none = tagger.high_band(low, SAMPLE_RATE)
    some = tagger.high_band(quiet_top, SAMPLE_RATE)
    much = tagger.high_band(loud_top, SAMPLE_RATE)
    assert none.distance == pytest.approx(0.0, abs=1e-6)
    assert none.distance < some.distance < much.distance
    assert len(much.per_window) == 1
    assert much.cutoff_hz == 8000.0
    two = tagger.high_band(np.concatenate([low, loud_top]), SAMPLE_RATE)
    assert two.per_window[0] == pytest.approx(0.0, abs=1e-4)
    assert two.per_window[1] > 0.1
    assert two.distance == pytest.approx(np.mean(two.per_window))


def test_an_unloaded_tagger_says_so() -> None:
    tagger = ClapTagger(load_registry().get("laion_clap_music"), {})
    with pytest.raises(ClapError, match="is not loaded"):
        tagger.embed_text(["drums"])
    with pytest.raises(ClapError, match="is not a LAION-CLAP model"):
        ClapTagger.from_registry("htdemucs")
    assert clap.MODEL_ID == "laion_clap_music"


# --- with the real checkpoint -------------------------------------------------------------


@pytest.fixture(scope="module")
def tagger():
    tagger = ClapTagger.from_registry()
    tagger.load("cpu")
    yield tagger
    tagger.unload()


@pytest.fixture(scope="module")
def track():
    return generate_track(seed=1, duration_seconds=12.0)


@pytest.mark.slow
def test_embeddings_are_unit_vectors_in_one_space(tagger, track) -> None:
    audio = tagger.embed_audio(track.mix, track.sample_rate)
    assert audio.shape == (2, 512)  # 12 s: one full window and a padded tail
    np.testing.assert_allclose(np.linalg.norm(audio, axis=1), 1.0, atol=1e-5)
    text = tagger.embed_text(["drums", "a quiet piano piece"])
    assert text.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-5)
    assert tagger.similarity(track.mix, track.sample_rate, ["drums"]).shape == (2, 1)
    # The same input gives the same embedding.
    np.testing.assert_allclose(audio, tagger.embed_audio(track.mix, track.sample_rate), atol=1e-6)
    with pytest.raises(ClapError, match="no prompts"):
        tagger.embed_text([])


@pytest.mark.slow
def test_text_prompts_pick_out_the_fixture_stems(tagger, track) -> None:
    prompts = ["the sound of drums", "the sound of a bass guitar", "a person singing", "silence"]
    drums = tagger.score_prompts(track.stems["drums"], track.sample_rate, prompts)
    bass = tagger.score_prompts(track.stems["bass"], track.sample_rate, prompts)
    assert sum(drums.values()) == pytest.approx(1.0)
    assert max(drums, key=drums.get) == "the sound of drums"
    assert max(bass, key=bass.get) == "the sound of a bass guitar"


@pytest.mark.slow
def test_high_band_differs_in_the_right_direction_on_the_fixture(tagger, track) -> None:
    # The fixture's drums carry nearly all of its energy above 8 kHz; its bass has none.
    with_top = tagger.high_band(track.mix, track.sample_rate)
    without_top = tagger.high_band(low_pass(to_mono(track.mix, track.sample_rate)), SAMPLE_RATE)
    assert with_top.distance > 10 * max(without_top.distance, 1e-4)
    assert without_top.distance < 0.005
    drums = tagger.high_band(track.stems["drums"], track.sample_rate).distance
    bass = tagger.high_band(track.stems["bass"], track.sample_rate).distance
    assert drums > 0.02
    assert bass < 0.005
    assert drums > 10 * max(bass, 1e-4)


@pytest.mark.slow
def test_the_checkpoint_loads_without_unpickling_code(tagger) -> None:
    state = clap.read_checkpoint(tagger._file("weights", ".pt"))
    assert all(isinstance(v, torch.Tensor) for v in state.values())
    assert not any(key.startswith("module.") for key in state)
    assert "text_branch.pooler.dense.weight" in state  # the text tower is in the file
    assert tagger.device == "cpu"
