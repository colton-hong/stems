"""BSS Eval scores come from museval and behave as the benchmark assumes."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
if importlib.util.find_spec("museval") is None:
    pytest.skip("museval is not installed", allow_module_level=True)

from stems_bench.metrics import (  # noqa: E402
    BssScores,
    MetricsError,
    bss_eval,
    bss_eval_stem,
    fit_length,
    museval_metrics,
)
from stems_common.synthetic import generate_track  # noqa: E402

SECONDS = 5.0


@pytest.fixture(scope="module")
def track():
    return generate_track(seed=1, duration_seconds=SECONDS)


def _noise(shape, level, seed=0):
    return (level * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32)


def test_metrics_module_is_museval_own_code() -> None:
    module = museval_metrics()
    assert module.__file__.endswith("museval/metrics.py")
    assert callable(module.bss_eval)


@pytest.mark.parametrize("stem", ["vocals", "drums", "bass", "guitar", "piano", "other"])
def test_a_perfect_estimate_scores_above_60_db(track, stem) -> None:
    scores = bss_eval_stem(track.stems[stem], track.stems[stem], track.mix, track.sample_rate)
    assert scores.sdr > 60
    assert scores.windows == int(SECONDS)
    assert scores.scored_windows == scores.windows


def test_more_noise_means_lower_sdr_by_the_expected_amount(track) -> None:
    reference = track.stems["vocals"]
    levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sdrs = [
        bss_eval_stem(
            reference, reference + _noise(reference.shape, level), track.mix, track.sample_rate
        ).sdr
        for level in levels
    ]
    assert all(a > b for a, b in itertools.pairwise(sdrs))
    # Ten times the noise amplitude is 20 dB less SDR.
    for a, b in itertools.pairwise(sdrs):
        assert a - b == pytest.approx(20.0, abs=0.5)


def test_sdr_matches_its_definition(track) -> None:
    reference = track.stems["drums"].astype(np.float64)
    estimate = reference + _noise(reference.shape, 0.01)
    scores = bss_eval_stem(reference, estimate, track.mix, track.sample_rate)
    rate = track.sample_rate
    expected = []
    for w in range(int(SECONDS)):
        ref = reference[:, w * rate : (w + 1) * rate]
        err = (estimate - reference)[:, w * rate : (w + 1) * rate]
        expected.append(10 * np.log10((ref**2).sum() / (err**2).sum()))
    assert scores.frames["sdr"] == pytest.approx(expected, abs=0.01)
    assert scores.sdr == pytest.approx(float(np.median(expected)), abs=0.01)


def test_a_single_stem_is_scored_as_target_and_rest_of_mix(track) -> None:
    reference = track.stems["vocals"]
    estimate = reference + _noise(reference.shape, 0.01)
    single = bss_eval_stem(reference, estimate, track.mix, track.sample_rate)
    pair = bss_eval(
        {"target": reference, "rest": track.mix - reference},
        {"target": estimate, "rest": track.mix - estimate},
        track.sample_rate,
    )["target"]
    assert single == pair
    for metric in ("sdr", "sir", "sar", "isr"):
        np.testing.assert_array_equal(single.frames[metric], pair.frames[metric])


def test_sdr_does_not_depend_on_the_other_sources_in_the_set(track) -> None:
    names = ["vocals", "drums", "bass", "other"]
    noisy = {
        n: track.stems[n] + _noise(track.stems[n].shape, 0.01, seed=i) for i, n in enumerate(names)
    }
    joint = bss_eval({n: track.stems[n] for n in names}, noisy, track.sample_rate)
    for name in names:
        alone = bss_eval_stem(track.stems[name], noisy[name], track.mix, track.sample_rate)
        assert alone.sdr == pytest.approx(joint[name].sdr, abs=1e-6)


def _broadband(seed, colour, samples):
    """Coloured noise with partly correlated channels: a well-conditioned test source."""
    x = np.random.default_rng(seed).standard_normal((2, samples))
    kernel = np.exp(-np.arange(64) / colour)
    x = np.stack([np.convolve(channel, kernel, mode="same") for channel in x])
    x[1] = 0.6 * x[0] + 0.8 * x[1]
    return (0.1 * x / x.std()).astype(np.float32)


def test_leakage_lowers_sir_and_noise_lowers_sar() -> None:
    rate, samples = 44100, 4 * 44100
    target, other, third = (_broadband(s, c, samples) for s, c in ((1, 3.0), (2, 10.0), (3, 1.0)))
    mix = target + other + third
    with_leak = bss_eval_stem(target, target + 0.3 * other, mix, rate)
    with_noise = bss_eval_stem(target, target + _noise(target.shape, 0.03), mix, rate)
    assert with_leak.sir < with_noise.sir - 10  # the leak is interference
    assert with_noise.sar < with_leak.sar - 3  # the noise is an artefact
    # A wrong level is neither: it only shows in SDR and ISR.
    quiet = bss_eval_stem(target, 0.5 * target, mix, rate)
    assert quiet.sdr == pytest.approx(6.02, abs=0.01)
    assert quiet.isr == pytest.approx(6.02, abs=0.01)
    assert quiet.sir > 100
    assert quiet.sar > 100


def test_sir_and_sar_are_not_meaningful_on_the_tonal_fixture(track) -> None:
    # Recorded so nobody reads them off a synthetic report: the fixture's bass is a few
    # pure tones with identical channels, and museval's projection cannot be solved for it.
    reference = track.stems["bass"]
    scores = bss_eval_stem(
        reference, reference + _noise(reference.shape, 0.02), track.mix, track.sample_rate
    )
    assert scores.sdr == pytest.approx(9.56, abs=0.5)  # SDR is still right
    assert scores.isr < -20  # nonsense: the estimate has exactly the right level


def test_windows_with_a_silent_source_are_not_scored(track) -> None:
    rate = track.sample_rate
    reference = track.stems["vocals"].copy()
    reference[:, : 2 * rate] = 0.0  # the singer comes in after two seconds
    mix = track.mix - track.stems["vocals"] + reference
    estimate = reference + _noise(reference.shape, 0.01)
    scores = bss_eval_stem(reference, estimate, mix, rate)
    assert scores.windows == 5
    assert scores.scored_windows == 3
    assert np.isnan(scores.frames["sdr"][:2]).all()
    assert math.isfinite(scores.sdr)


def test_a_stem_that_is_absent_from_the_track_gives_nan_not_an_error(track) -> None:
    silent = np.zeros_like(track.mix)
    estimate = _noise(silent.shape, 0.01)
    for reference, guess in ((silent, estimate), (track.stems["piano"], silent)):
        scores = bss_eval_stem(reference, guess, track.mix, track.sample_rate)
        assert (scores.windows, scores.scored_windows) == (5, 0)
        assert math.isnan(scores.sdr)
        assert scores.as_dict()["sdr"] is None
        assert scores.as_dict()["frames"]["sir"] == [None] * 5


def test_estimate_length_is_fitted_to_the_reference(track) -> None:
    reference = track.stems["drums"]
    exact = bss_eval_stem(reference, reference, track.mix, track.sample_rate)
    longer = np.concatenate([reference, np.ones((2, 100), dtype=np.float32)], axis=1)
    assert bss_eval_stem(reference, longer, track.mix, track.sample_rate).sdr == exact.sdr
    shorter = bss_eval_stem(reference, reference[:, :-1000], track.mix, track.sample_rate)
    assert shorter.sdr < exact.sdr or math.isinf(exact.sdr)
    assert fit_length(np.ones((2, 3)), 5).tolist() == [[1, 1, 1, 0, 0]] * 2
    assert fit_length(np.ones((2, 5)), 3).shape == (2, 3)


def test_bad_input_is_refused(track) -> None:
    rate = track.sample_rate
    with pytest.raises(MetricsError, match="must match"):
        bss_eval({"a": track.mix}, {"b": track.mix}, rate)
    with pytest.raises(MetricsError, match="shorter than one"):
        bss_eval({"a": track.mix[:, :1000]}, {"a": track.mix[:, :1000]}, rate)
    with pytest.raises(MetricsError, match="differ in shape"):
        bss_eval_stem(track.mix[:, :-1], track.mix, track.mix, rate)
    with pytest.raises(MetricsError, match="estimate 'a' has shape"):
        bss_eval({"a": track.mix}, {"a": track.mix[:1]}, rate)


def test_scores_serialise_to_plain_json(track) -> None:
    reference = track.stems["vocals"]
    scores = bss_eval_stem(
        reference, reference + _noise(reference.shape, 0.01), track.mix, track.sample_rate
    )
    assert isinstance(scores, BssScores)
    data = json.loads(json.dumps(scores.as_dict(), allow_nan=False))
    assert data["sdr"] == pytest.approx(scores.sdr)
    assert len(data["frames"]["sdr"]) == 5
    assert "frames" not in scores.as_dict(frames=False)
