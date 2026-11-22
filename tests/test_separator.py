"""A fake model runs through device selection, chunking, stitching and the writer."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stems_common.synthetic import generate_track  # noqa: E402
from stems_worker import device as device_module  # noqa: E402
from stems_worker.audio import load_audio, save_stems  # noqa: E402
from stems_worker.chunking import hop_for_overlap, plan_chunks  # noqa: E402
from stems_worker.device import DeviceError, pick_device, usable_devices  # noqa: E402
from stems_worker.fakes import BandSplitSeparator  # noqa: E402
from stems_worker.separator import Separator, run_separator  # noqa: E402

SR = 44100


@pytest.fixture(scope="module")
def mix():
    return generate_track(seed=21, duration_seconds=5.0).mix


def _band_fraction(x, lo: float, hi: float) -> float:
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / SR)
    return float(power[:, (freqs >= lo) & (freqs < hi)].sum() / power.sum())


# --- acceptance: end to end on CPU ---------------------------------------------------------


def test_fake_two_stem_model_runs_end_to_end_on_cpu(tmp_path, mix) -> None:
    separator = BandSplitSeparator(outputs=("low", "high"), cutoffs_hz=(300.0,))
    separator.load("cpu")
    result = run_separator(separator, mix, SR, overlap=0.25, batch_size=2)

    assert result.model_id == "fake_band_split"
    assert result.device == "cpu"
    assert set(result.stems) == {"low", "high"}
    assert result.audio_seconds == pytest.approx(5.0)
    assert result.wall_seconds > 0
    assert result.realtime_factor == pytest.approx(result.wall_seconds / 5.0)
    assert result.num_chunks == plan_chunks(mix.shape[1], SR, hop_for_overlap(SR, 0.25)).num_chunks
    for stem in result.stems.values():
        assert stem.shape == mix.shape
        assert stem.dtype == np.float32

    # The model did its job: each band holds the right part of the spectrum.
    assert _band_fraction(result.stems["low"], 0, 400) > 0.99
    assert _band_fraction(result.stems["high"], 0, 200) < 0.01
    # And the stems still add up to the mix after chunking and crossfading.
    assert float(np.abs(result.stems["low"] + result.stems["high"] - mix).max()) < 2e-6

    written = save_stems(tmp_path / "out", result.stems, SR, "wav")
    assert written.gain == 1.0
    assert sorted(p.name for p in written.paths.values()) == ["high.wav", "low.wav"]
    low = load_audio(written.paths["low"]).samples
    high = load_audio(written.paths["high"]).samples
    assert low.shape == mix.shape
    assert float(np.abs(low + high - mix).max()) < 2e-6


def test_crossfading_improves_on_gluing_for_this_model(mix) -> None:
    """The fake has real chunk-edge error, so overlap must measurably help."""
    whole = BandSplitSeparator(chunk_size=mix.shape[1])
    whole.load("cpu")
    reference = whole.separate_chunks(mix[None])[0, 0]  # the model applied to the whole clip

    separator = BandSplitSeparator(chunk_size=8192)
    separator.load("cpu")
    glued = run_separator(separator, mix, SR, overlap=0.0).stems["low"]
    crossfaded = run_separator(separator, mix, SR, overlap=0.5).stems["low"]

    inner = slice(8192, -8192)  # compare away from the clip's own ends
    error_glued = float(np.abs(glued[:, inner] - reference[:, inner]).max())
    error_crossfaded = float(np.abs(crossfaded[:, inner] - reference[:, inner]).max())
    assert error_crossfaded < 0.5 * error_glued


def test_three_band_model(mix) -> None:
    separator = BandSplitSeparator(
        outputs=("bass", "mid", "high"), cutoffs_hz=(250.0, 4000.0), model_id="fake_three"
    )
    separator.load("cpu")
    result = run_separator(separator, mix, SR)
    assert list(result.stems) == ["bass", "mid", "high"]
    total = sum(result.stems.values())
    assert float(np.abs(total - mix).max()) < 3e-6
    assert _band_fraction(result.stems["high"], 3800, SR / 2) > 0.95
    assert _band_fraction(result.stems["bass"], 0, 300) > 0.95


@pytest.mark.gpu
def test_accelerated_device_matches_cpu(mix) -> None:
    accelerated = [d for d in usable_devices() if d != "cpu"]
    if not accelerated:
        pytest.skip("no accelerated device on this machine")
    on_cpu = BandSplitSeparator()
    on_cpu.load("cpu")
    on_gpu = BandSplitSeparator()
    on_gpu.load(accelerated[0])
    a = run_separator(on_cpu, mix, SR)
    b = run_separator(on_gpu, mix, SR)
    assert b.device == accelerated[0]
    for name in a.stems:
        assert float(np.abs(a.stems[name] - b.stems[name]).max()) < 1e-4, name


def test_progress_is_reported(mix) -> None:
    separator = BandSplitSeparator()
    separator.load("cpu")
    seen: list[tuple[int, int]] = []
    result = run_separator(
        separator, mix, SR, progress=lambda done, total: seen.append((done, total))
    )
    assert seen[-1] == (result.num_chunks, result.num_chunks)


# --- misuse --------------------------------------------------------------------------------


def test_unloaded_separator_and_bad_audio_are_rejected(mix) -> None:
    separator = BandSplitSeparator()
    with pytest.raises(RuntimeError, match="not loaded"):
        run_separator(separator, mix, SR)
    separator.load("cpu")
    with pytest.raises(ValueError, match="48000 Hz"):
        run_separator(separator, mix, 48000)
    with pytest.raises(ValueError, match="stereo"):
        run_separator(separator, mix[:1], SR)
    separator.unload()
    assert separator.device is None
    with pytest.raises(RuntimeError, match="not loaded"):
        run_separator(separator, mix, SR)


def test_a_model_that_emits_the_wrong_number_of_stems_is_caught(mix) -> None:
    class Liar(Separator):
        model_id = "liar"
        outputs = ("a", "b", "c")
        chunk_size = 4096

        def load(self, device: str) -> None:
            self.device = device

        def separate_chunks(self, batch):
            return np.stack([batch, batch], axis=1)

    liar = Liar()
    liar.load("cpu")
    with pytest.raises(ValueError, match="returned 2 outputs but declares 3"):
        run_separator(liar, mix, SR)


def test_whole_track_hooks_wrap_the_chunked_run(mix) -> None:
    class Scaled(BandSplitSeparator):
        """Halves the track before the model and doubles the stems after it."""

        calls: list[str] = []  # noqa: RUF012 - shared on purpose, one instance per test

        def prepare(self, audio):
            self.calls.append("prepare")
            return audio * np.float32(0.5), 2.0

        def finish(self, stems, state):
            self.calls.append("finish")
            return stems * np.float32(state)

    separator = Scaled()
    separator.load("cpu")
    result = run_separator(separator, mix, SR)
    assert Scaled.calls == ["prepare", "finish"]
    assert float(np.abs(result.stems["low"] + result.stems["high"] - mix).max()) < 2e-6


def test_fake_rejects_inconsistent_configuration() -> None:
    with pytest.raises(ValueError, match="one cutoff"):
        BandSplitSeparator(outputs=("a", "b", "c"), cutoffs_hz=(100.0,))
    with pytest.raises(ValueError, match="ascending"):
        BandSplitSeparator(outputs=("a", "b", "c"), cutoffs_hz=(500.0, 100.0))


def test_unloading_returns_accelerator_memory() -> None:
    if "mps" not in usable_devices():
        pytest.skip("needs MPS to observe the allocator")
    separator = BandSplitSeparator(chunk_size=1 << 22)
    separator.load("mps")
    separator.separate_chunks(np.zeros((1, 2, 1 << 22), dtype=np.float32))
    loaded = torch.mps.driver_allocated_memory()

    separator.unload()
    # Without the cache flush in unload() nothing is returned at all. The driver keeps
    # some working buffers, so the bar is "most of it", not "all of it".
    assert torch.mps.driver_allocated_memory() < 0.5 * loaded


# --- device selection ----------------------------------------------------------------------


def test_pick_device_prefers_argument_then_environment_then_best(monkeypatch) -> None:
    monkeypatch.delenv("STEMS_DEVICE", raising=False)
    assert pick_device() == usable_devices()[0]
    assert pick_device("cpu") == "cpu"
    monkeypatch.setenv("STEMS_DEVICE", "cpu")
    assert pick_device() == "cpu"
    assert pick_device("auto") == usable_devices()[0]


def test_pick_device_refuses_a_device_that_is_not_usable(monkeypatch) -> None:
    monkeypatch.setattr(device_module, "usable_devices", lambda: ("cpu",))
    monkeypatch.delenv("STEMS_DEVICE", raising=False)
    assert pick_device() == "cpu"
    with pytest.raises(DeviceError, match="'cuda' is not usable"):
        pick_device("cuda")
    monkeypatch.setenv("STEMS_DEVICE", "mps")
    with pytest.raises(DeviceError, match="'mps' is not usable"):
        pick_device()


# --- writing stems -------------------------------------------------------------------------


def test_hot_stems_get_one_shared_gain_so_they_still_sum_to_the_mix(tmp_path) -> None:
    track = generate_track(seed=2, duration_seconds=1.0)
    loud = {name: stem * np.float32(6.0) for name, stem in track.stems.items()}
    assert max(float(np.abs(s).max()) for s in loud.values()) > 1.0

    written = save_stems(tmp_path, loud, SR, "flac")
    assert written.peak > 1.0
    assert written.gain == pytest.approx(1.0 / written.peak, rel=1e-5)

    loaded = {name: load_audio(path).samples for name, path in written.paths.items()}
    assert max(float(np.abs(s).max()) for s in loaded.values()) <= 1.0
    expected = sum(loud.values()) * written.gain
    assert float(np.abs(sum(loaded.values()) - expected).max()) < 1e-6
    # Every stem was scaled by the same amount.
    for name in loud:
        assert float(np.abs(loaded[name] - loud[name] * written.gain).max()) < 1e-6


def test_save_stems_rejects_unsafe_names_and_empty_input(tmp_path) -> None:
    audio = np.zeros((2, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="not usable as a file name"):
        save_stems(tmp_path, {"../evil": audio})
    with pytest.raises(ValueError, match="no stems"):
        save_stems(tmp_path, {})
    assert list(tmp_path.iterdir()) == []
