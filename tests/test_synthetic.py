"""The synthetic fixture is deterministic, sums exactly, and has the character tests rely on."""

from __future__ import annotations

import pytest

from stems_common.stems import PRODUCT_STEMS

np = pytest.importorskip("numpy")

from stems_common.synthetic import (  # noqa: E402  (after the numpy skip guard)
    DEFAULT_SAMPLE_RATE,
    SyntheticTrack,
    add_reverb,
    generate_track,
)


@pytest.fixture(scope="module")
def track() -> SyntheticTrack:
    return generate_track(seed=0)


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _band_energy(x, sample_rate: int, lo: float, hi: float) -> float:
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / sample_rate)
    return float(power[:, (freqs >= lo) & (freqs < hi)].sum())


# --- acceptance: determinism and exact mixing ----------------------------------------------


def test_same_seed_gives_identical_samples() -> None:
    a, b = generate_track(seed=7), generate_track(seed=7)
    for name in PRODUCT_STEMS:
        assert np.array_equal(a.stems[name], b.stems[name]), name
    assert np.array_equal(a.mix, b.mix)
    assert np.array_equal(a.lead_vocals, b.lead_vocals)
    assert np.array_equal(a.backing_vocals, b.backing_vocals)
    assert np.array_equal(a.vocals_wet, b.vocals_wet)


def test_different_seeds_give_different_tracks() -> None:
    a, b = generate_track(seed=1), generate_track(seed=2)
    assert a.track_id != b.track_id
    for name in PRODUCT_STEMS:
        assert not np.array_equal(a.stems[name], b.stems[name]), name


def test_mix_equals_the_sum_of_its_stems(track: SyntheticTrack) -> None:
    total = np.zeros_like(track.mix, dtype=np.float64)
    for name in PRODUCT_STEMS:
        total += track.stems[name]
    assert float(np.abs(track.mix - total).max()) < 1e-6


def test_vocals_equal_lead_plus_backing_exactly(track: SyntheticTrack) -> None:
    assert np.array_equal(track.stems["vocals"], track.lead_vocals + track.backing_vocals)


# --- format --------------------------------------------------------------------------------


def test_format_is_44k1_stereo_float32(track: SyntheticTrack) -> None:
    assert track.sample_rate == DEFAULT_SAMPLE_RATE == 44100
    assert set(track.stems) == set(PRODUCT_STEMS)
    expected = (2, round(6.0 * 44100))
    for array in (*track.stems.values(), track.mix, track.vocals_wet, track.lead_vocals):
        assert array.shape == expected
        assert array.dtype == np.float32
    assert track.num_samples == expected[1]
    assert track.duration_seconds == pytest.approx(6.0)


def test_duration_and_sample_rate_are_respected() -> None:
    short = generate_track(seed=3, duration_seconds=0.2)
    assert short.mix.shape == (2, 8820)
    other_rate = generate_track(seed=3, duration_seconds=1.0, sample_rate=48000)
    assert other_rate.mix.shape == (2, 48000)
    assert other_rate.sample_rate == 48000


def test_no_seed_clips() -> None:
    for seed in range(25):
        t = generate_track(seed=seed, duration_seconds=4.0)
        assert float(np.abs(t.mix).max()) < 0.95, seed
        assert float(np.abs(t.mix_wet).max()) < 0.95, seed


def test_every_stem_is_audible_and_finite(track: SyntheticTrack) -> None:
    for name, stem in track.stems.items():
        assert np.isfinite(stem).all(), name
        assert _rms(stem) > 0.01, name


# --- character that later tests depend on --------------------------------------------------


def test_bass_is_low_and_drums_own_the_band_above_8k(track: SyntheticTrack) -> None:
    sr = track.sample_rate
    bass = track.stems["bass"]
    assert _band_energy(bass, sr, 0, 300) / _band_energy(bass, sr, 0, sr / 2) > 0.95

    high = {n: _band_energy(s, sr, 8000, sr / 2) for n, s in track.stems.items()}
    assert high["drums"] / sum(high.values()) > 0.95
    drums = track.stems["drums"]
    assert high["drums"] / _band_energy(drums, sr, 0, sr / 2) > 0.01


def test_stereo_content_differs_between_channels(track: SyntheticTrack) -> None:
    # Panned stems are louder on one side; "other" is decorrelated between channels.
    guitar, piano, other = (track.stems[n] for n in ("guitar", "piano", "other"))
    assert _rms(guitar[0]) > 1.5 * _rms(guitar[1])
    assert _rms(piano[1]) > 1.5 * _rms(piano[0])
    assert abs(float(np.corrcoef(other[0], other[1])[0, 1])) < 0.3
    assert not np.array_equal(track.mix[0], track.mix[1])


def test_backing_vocals_enter_late_and_lead_has_rests(track: SyntheticTrack) -> None:
    n = track.num_samples
    assert not track.backing_vocals[:, : int(0.4 * n)].any()
    assert _rms(track.backing_vocals[:, n // 2 :]) > 0.01
    lead_silent = np.abs(track.lead_vocals).max(axis=0) < 1e-4
    assert 0.05 < float(lead_silent.mean()) < 0.6


# --- switching stems off -------------------------------------------------------------------


def test_inactive_stems_are_silent_and_do_not_change_the_others() -> None:
    full = generate_track(seed=5)
    partial = generate_track(seed=5, active_stems=("drums", "bass", "guitar"))
    for name in ("vocals", "piano", "other"):
        assert not partial.stems[name].any(), name
    for name in ("drums", "bass", "guitar"):
        assert np.array_equal(partial.stems[name], full.stems[name]), name
    assert not partial.lead_vocals.any()
    assert not partial.backing_vocals.any()
    assert not partial.vocals_wet.any()


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stems"):
        generate_track(active_stems=("vocals", "kazoo"))
    with pytest.raises(ValueError, match="duration_seconds"):
        generate_track(duration_seconds=0)


# --- reverberant variant -------------------------------------------------------------------


def test_reverb_adds_a_tail_at_the_requested_level(track: SyntheticTrack) -> None:
    dry = track.stems["vocals"]
    added = track.vocals_wet.astype(np.float64) - dry
    assert 20 * np.log10(_rms(added) / _rms(dry)) == pytest.approx(-12.0, abs=0.05)

    louder = add_reverb(dry, track.sample_rate, seed=0, wet_db=-6.0)
    added = louder.astype(np.float64) - dry
    assert 20 * np.log10(_rms(added) / _rms(dry)) == pytest.approx(-6.0, abs=0.05)


def test_reverb_rings_on_after_the_dry_signal_stops() -> None:
    sr = 44100
    click = np.zeros((2, sr), dtype=np.float32)
    click[:, 100] = 0.5
    wet = add_reverb(click, sr, seed=1, rt60_seconds=0.5)
    assert wet.shape == click.shape
    assert wet.dtype == np.float32
    early = _rms(wet[:, 200 : 200 + sr // 10])
    late = _rms(wet[:, sr // 2 : sr // 2 + sr // 10])
    assert early > 0
    assert 0 < late < 0.1 * early  # decaying, but not gone


def test_mix_wet_swaps_only_the_vocals(track: SyntheticTrack) -> None:
    difference = track.mix_wet.astype(np.float64) - track.mix
    expected = track.vocals_wet.astype(np.float64) - track.stems["vocals"]
    assert float(np.abs(difference - expected).max()) < 1e-6
