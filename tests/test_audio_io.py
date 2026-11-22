"""Decoding, resampling and encoding keep length, level, channels and precision."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
av = pytest.importorskip("av")

from stems_common.synthetic import generate_track  # noqa: E402
from stems_worker.audio import (  # noqa: E402
    AudioTooLongError,
    UnsupportedAudioError,
    extension_for,
    load_audio,
    probe,
    resample,
    resampled_length,
    save_audio,
    to_stereo,
)

LSB_24 = 2.0**-23


def _sine(freq: float, seconds: float, rate: int, amplitude: float = 0.5):
    t = np.arange(round(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _quantise_24(x):
    return (np.round(x.astype(np.float64) / LSB_24) * LSB_24).astype(np.float32)


def _db(x) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-12))


def _best_correlation(a, b, max_lag: int = 4000) -> float:
    """Highest normalised correlation between two mono signals over small delays of b."""
    n = min(len(a), len(b)) - max_lag
    a = a[:n].astype(np.float64)
    best = 0.0
    for lag in range(0, max_lag, 7):
        seg = b[lag : lag + n].astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(seg)
        if denom > 0:
            best = max(best, float(np.dot(a, seg) / denom))
    return best


def _write_aac(path: Path, samples, rate: int) -> None:
    """Make an M4A test file, which only the PyAV decoder can read."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("aac", rate=rate)
        stream.codec_context.layout = "stereo"
        stream.codec_context.bit_rate = 192_000
        for start in range(0, samples.shape[1], 1024):
            block = np.ascontiguousarray(samples[:, start : start + 1024])
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


@pytest.fixture(scope="module")
def mix():
    return generate_track(seed=11, duration_seconds=3.0).mix


# --- lossless round trips (acceptance: exact at 24-bit precision) --------------------------


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_lossless_round_trip_is_exact_at_24_bits(tmp_path, mix, fmt) -> None:
    quantised = _quantise_24(mix)
    path = tmp_path / f"mix{extension_for(fmt)}"
    result = save_audio(path, quantised, 44100, fmt)
    loaded = load_audio(path)

    assert result.path == path
    assert result.clipped_samples == 0
    assert loaded.sample_rate == 44100
    assert loaded.samples.dtype == np.float32
    assert np.array_equal(loaded.samples, quantised)


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_unquantised_audio_is_within_half_a_24_bit_step(tmp_path, mix, fmt) -> None:
    path = tmp_path / f"mix{extension_for(fmt)}"
    save_audio(path, mix, 44100, fmt)
    error = np.abs(load_audio(path).samples.astype(np.float64) - mix)
    assert float(error.max()) <= LSB_24 / 2 + 1e-9


def test_source_info_describes_the_file(tmp_path, mix) -> None:
    path = tmp_path / "mix.wav"
    save_audio(path, mix, 44100, "wav")
    info = probe(path)
    assert (info.sample_rate, info.channels, info.num_samples) == (44100, 2, mix.shape[1])
    assert info.duration_seconds == pytest.approx(3.0)
    assert info.decoder == "soundfile"
    assert "PCM_24" in info.codec


# --- channels ------------------------------------------------------------------------------


def test_mono_is_duplicated_to_both_channels_at_the_same_level(tmp_path) -> None:
    mono = _quantise_24(_sine(440, 1.0, 44100))
    path = tmp_path / "mono.wav"
    sf.write(str(path), mono, 44100, subtype="PCM_24")
    loaded = load_audio(path)
    assert loaded.source.channels == 1
    assert loaded.samples.shape == (2, 44100)
    assert np.array_equal(loaded.samples[0], mono)
    assert np.array_equal(loaded.samples[1], mono)


def test_left_and_right_are_not_swapped_or_mixed(tmp_path) -> None:
    left_only = np.stack([_sine(440, 1.0, 48000), np.zeros(48000, dtype=np.float32)])
    path = tmp_path / "left.wav"
    sf.write(str(path), left_only.T, 48000, subtype="PCM_24")
    loaded = load_audio(path)
    assert _db(loaded.samples[0]) > -10
    assert not loaded.samples[1].any()  # the silent channel stays exactly silent


def test_more_than_two_channels_are_downmixed_to_stereo(tmp_path) -> None:
    surround = np.zeros((6, 44100), dtype=np.float32)
    surround[0] = _sine(440, 1.0, 44100)  # front left only
    path = tmp_path / "surround.wav"
    sf.write(str(path), surround.T, 44100, subtype="PCM_24")
    loaded = load_audio(path)
    assert loaded.source.channels == 6
    assert loaded.source.decoder == "pyav"
    assert loaded.samples.shape == (2, 44100)
    assert _db(loaded.samples[0]) > _db(loaded.samples[1]) + 20


def test_to_stereo_rejects_other_shapes() -> None:
    with pytest.raises(ValueError, match="channels, samples"):
        to_stereo(np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="3 channels"):
        to_stereo(np.zeros((3, 10), dtype=np.float32))


# --- sample rates (acceptance: 48 kHz and 22.05 kHz, length within one sample) -------------


@pytest.mark.parametrize("rate", [48000, 22050, 96000, 32000, 8000])
def test_other_sample_rates_are_resampled_to_44k1_with_exact_length(tmp_path, rate) -> None:
    seconds = 1.37
    source = np.stack([_sine(1000, seconds, rate), _sine(1500, seconds, rate)])
    path = tmp_path / f"in_{rate}.wav"
    sf.write(str(path), source.T, rate, subtype="FLOAT")
    loaded = load_audio(path)

    assert loaded.source.sample_rate == rate
    assert loaded.sample_rate == 44100
    n = loaded.samples.shape[1]
    assert n == resampled_length(source.shape[1], rate, 44100)
    assert abs(n - source.shape[1] * 44100 / rate) <= 1

    # Away from the edges the result matches a sine generated directly at 44.1 kHz.
    edge = 2000
    for channel, freq in enumerate((1000, 1500)):
        t = np.arange(n) / 44100
        ideal = 0.5 * np.sin(2 * np.pi * freq * t)
        error = loaded.samples[channel, edge:-edge] - ideal[edge:-edge]
        assert _db(error) < -80, (rate, channel)


def test_resample_is_a_no_op_at_the_same_rate_and_keeps_channels_independent() -> None:
    x = np.stack([_sine(300, 0.5, 44100), np.zeros(22050, dtype=np.float32)])
    assert np.array_equal(resample(x, 44100, 44100), x)
    up = resample(x, 44100, 48000)
    assert up.shape == (2, 24000)
    assert up.dtype == np.float32
    assert not up[1].any()


# --- lossy formats -------------------------------------------------------------------------


def test_m4a_is_decoded_by_the_pyav_fallback(tmp_path, mix) -> None:
    path = tmp_path / "mix.m4a"
    _write_aac(path, mix, 44100)
    assert probe(path).decoder == "pyav"
    loaded = load_audio(path)
    assert loaded.source.decoder == "pyav"
    assert loaded.source.codec == "aac"
    assert loaded.samples.shape[0] == 2
    assert abs(loaded.samples.shape[1] - mix.shape[1]) <= 4096  # codec padding only
    for channel in range(2):
        assert _best_correlation(mix[channel], loaded.samples[channel]) > 0.97


@pytest.mark.parametrize(("fmt", "bit_rate"), [("mp3_320", 320_000), ("mp3_192", 192_000)])
def test_mp3_is_written_at_the_requested_bitrate_and_sounds_like_the_input(
    tmp_path, mix, fmt, bit_rate
) -> None:
    path = tmp_path / "mix.mp3"
    save_audio(path, mix, 44100, fmt)
    expected_bytes = bit_rate / 8 * mix.shape[1] / 44100
    assert os.path.getsize(path) == pytest.approx(expected_bytes, rel=0.08)

    loaded = load_audio(path)
    assert loaded.samples.shape[0] == 2
    assert abs(loaded.samples.shape[1] - mix.shape[1]) <= 4096
    for channel in range(2):
        assert _best_correlation(mix[channel], loaded.samples[channel]) > 0.98


def test_mono_can_be_saved_in_every_format(tmp_path) -> None:
    mono = _sine(440, 0.5, 44100)[None, :]
    for fmt in ("wav", "flac", "mp3_320"):
        path = tmp_path / f"mono{extension_for(fmt)}"
        save_audio(path, mono, 44100, fmt)
        assert probe(path).channels == 1


# --- saving --------------------------------------------------------------------------------


def test_save_reports_peak_and_clipping(tmp_path) -> None:
    hot = np.stack([_sine(100, 0.2, 44100, amplitude=1.5)] * 2)
    result = save_audio(tmp_path / "hot.wav", hot, 44100, "wav")
    assert result.peak == pytest.approx(1.5, rel=1e-3)
    assert result.clipped_samples > 0
    assert float(np.abs(load_audio(result.path).samples).max()) <= 1.0


def test_save_is_atomic_and_overwrites(tmp_path, mix) -> None:
    path = tmp_path / "nested" / "out.wav"
    save_audio(path, mix, 44100, "wav")
    save_audio(path, mix[:, :1000], 44100, "wav")
    assert probe(path).num_samples == 1000
    assert [p.name for p in path.parent.iterdir()] == ["out.wav"]


def test_save_rejects_bad_input(tmp_path) -> None:
    good = np.zeros((2, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown output format"):
        save_audio(tmp_path / "x.ogg", good, 44100, "ogg")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        save_audio(tmp_path / "x.wav", np.zeros((3, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        save_audio(tmp_path / "x.wav", np.zeros((2, 0), dtype=np.float32))
    bad = good.copy()
    bad[0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_audio(tmp_path / "x.wav", bad)
    assert list(tmp_path.iterdir()) == []


def test_extension_for() -> None:
    assert [extension_for(f) for f in ("wav", "flac", "mp3_320", "mp3_192")] == [
        ".wav",
        ".flac",
        ".mp3",
        ".mp3",
    ]
    with pytest.raises(ValueError, match="unknown output format"):
        extension_for("ogg")


# --- bad input -----------------------------------------------------------------------------


def test_a_file_that_is_not_audio_is_rejected_whatever_its_name(tmp_path) -> None:
    for name in ("notes.wav", "notes.mp3", "notes.m4a"):
        path = tmp_path / name
        path.write_bytes(b"This is a text file, not audio.\n" * 200)
        with pytest.raises(UnsupportedAudioError):
            load_audio(path)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/file.wav")


def test_empty_and_non_finite_files_are_rejected(tmp_path) -> None:
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros((0, 2), dtype=np.float32), 44100)
    with pytest.raises(UnsupportedAudioError, match="no audio samples"):
        load_audio(empty)

    nan = np.zeros((100, 2), dtype=np.float32)
    nan[5, 0] = np.nan
    bad = tmp_path / "nan.wav"
    sf.write(str(bad), nan, 44100, subtype="FLOAT")
    with pytest.raises(UnsupportedAudioError, match="non-finite"):
        load_audio(bad)


def test_duration_limit_applies_to_both_decoders(tmp_path, mix) -> None:
    wav = tmp_path / "mix.wav"
    save_audio(wav, mix, 44100, "wav")
    m4a = tmp_path / "mix.m4a"
    _write_aac(m4a, mix, 44100)

    for path in (wav, m4a):
        assert load_audio(path, max_duration_seconds=10).samples.shape[0] == 2
        with pytest.raises(AudioTooLongError):
            load_audio(path, max_duration_seconds=1.0)


def test_duration_limit_holds_while_decoding_not_just_from_the_header(tmp_path, mix) -> None:
    from stems_worker import audio

    m4a = tmp_path / "mix.m4a"
    _write_aac(m4a, mix, 44100)
    with pytest.raises(AudioTooLongError):
        audio._decode_pyav(m4a, max_duration_seconds=0.5)
