"""Blending estimates: the invariants, and that a blend lands between its members."""

from __future__ import annotations

import copy
import importlib.util

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from stems_common.settings import Defaults, load_defaults  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_worker.ensemble import (  # noqa: E402
    METHODS,
    EnsembleError,
    blend,
    blend_stems,
    normalise_weights,
)

RATE = 44100


@pytest.fixture(scope="module")
def track():
    return generate_track(seed=3, duration_seconds=3.0)


def _noise(shape, level, seed):
    return (level * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32)


def _sdr(reference, estimate) -> float:
    error = estimate.astype(np.float64) - reference
    return float(10 * np.log10((reference.astype(np.float64) ** 2).sum() / (error**2).sum()))


@pytest.mark.parametrize("method", METHODS)
def test_blending_copies_of_one_estimate_returns_it(track, method) -> None:
    estimate = track.stems["drums"] + _noise(track.mix.shape, 0.01, 0)
    for count in (2, 3):
        out = blend([estimate] * count, method=method)
        assert out.dtype == np.float32
        assert out.shape == estimate.shape
        np.testing.assert_allclose(out, estimate, atol=2e-6)


@pytest.mark.parametrize("method", METHODS)
def test_one_estimate_is_returned_as_it_is(track, method) -> None:
    estimate = track.stems["bass"]
    np.testing.assert_array_equal(blend([estimate], method=method), estimate)


def test_waveform_blend_is_the_weighted_mean(track) -> None:
    a, b = track.stems["vocals"], track.stems["guitar"]
    np.testing.assert_allclose(blend([a, b]), 0.5 * a + 0.5 * b, atol=1e-7)
    np.testing.assert_allclose(blend([a, b], [3, 1]), 0.75 * a + 0.25 * b, atol=1e-7)
    # Weights are relative: only their ratio matters.
    np.testing.assert_array_equal(blend([a, b], [3, 1]), blend([a, b], [0.75, 0.25]))
    # A zero weight removes a member entirely, for every method.
    for method in ("waveform", "stft_magnitude"):
        np.testing.assert_array_equal(blend([a, b], [1, 0], method), a)


def test_waveform_blends_of_complete_sets_still_sum_to_the_mix(track) -> None:
    rng = np.random.default_rng(1)
    models = []
    for _ in range(3):
        # Each "model" moves some energy between vocals and drums but loses none.
        leak = (0.2 * rng.standard_normal() * track.stems["drums"]).astype(np.float32)
        stems = dict(track.stems)
        stems["vocals"] = stems["vocals"] + leak
        stems["drums"] = stems["drums"] - leak
        models.append(stems)
    blended = blend_stems(models, [2, 1, 1])
    total = sum(blended[name].astype(np.float64) for name in blended)
    np.testing.assert_allclose(total, track.mix, atol=1e-5)


@pytest.mark.parametrize("method", ["waveform", "stft_magnitude"])
def test_a_weighted_mix_of_good_and_bad_scores_between_them(track, method) -> None:
    reference = track.stems["vocals"]
    good = reference + _noise(reference.shape, 0.003, 1)
    bad = reference + _noise(reference.shape, 0.05, 2)
    assert _sdr(reference, good) > _sdr(reference, bad) + 20
    scores = [_sdr(reference, blend([good, bad], [w, 1 - w], method)) for w in (0.2, 0.5, 0.8)]
    for score in scores:
        assert _sdr(reference, bad) < score < _sdr(reference, good)
    assert scores[0] < scores[1] < scores[2]  # more weight on the good one, better score


def test_averaging_two_equally_noisy_estimates_beats_either(track) -> None:
    reference = track.stems["other"]
    a = reference + _noise(reference.shape, 0.02, 1)
    b = reference + _noise(reference.shape, 0.02, 2)
    # Independent errors of equal size: the mean has half the error power, 3 dB better.
    gain = _sdr(reference, blend([a, b])) - _sdr(reference, a)
    assert gain == pytest.approx(3.0, abs=0.2)


def test_stft_magnitude_keeps_energy_that_a_plain_mean_cancels(track) -> None:
    estimate = track.stems["drums"]
    opposite = -estimate
    assert np.abs(blend([estimate, opposite])).max() < 1e-7  # the plain mean is silence
    kept = blend([estimate, opposite], method="stft_magnitude")
    np.testing.assert_allclose(kept, estimate, atol=2e-6)
    # In phase, the two methods agree.
    quieter = 0.5 * estimate
    np.testing.assert_allclose(
        blend([estimate, quieter], method="stft_magnitude"), blend([estimate, quieter]), atol=2e-6
    )


def test_min_keeps_what_all_have_and_max_what_any_has() -> None:
    t = np.arange(2 * RATE) / RATE
    low = np.tile(0.3 * np.sin(2 * np.pi * 220 * t), (2, 1)).astype(np.float32)
    high = np.tile(0.3 * np.sin(2 * np.pi * 5000 * t), (2, 1)).astype(np.float32)
    with_bleed = low + high  # one model let the high tone through
    clean = low
    inside = slice(4096, -4096)
    least = blend([with_bleed, clean], method="stft_min")
    most = blend([with_bleed, clean], method="stft_max")
    np.testing.assert_allclose(least[:, inside], clean[:, inside], atol=1e-4)
    np.testing.assert_allclose(most[:, inside], with_bleed[:, inside], atol=1e-4)


def test_bad_requests_are_refused(track) -> None:
    a = track.stems["vocals"]
    with pytest.raises(EnsembleError, match="unknown blend method"):
        blend([a, a], method="median")
    with pytest.raises(EnsembleError, match="nothing to blend"):
        blend([])
    with pytest.raises(EnsembleError, match="differ in shape"):
        blend([a, a[:, :-1]])
    with pytest.raises(EnsembleError, match="2 estimates need 2 weights, got 3"):
        blend([a, a], [1, 1, 1])
    with pytest.raises(EnsembleError, match="non-negative and not all zero"):
        blend([a, a], [1, -1])
    with pytest.raises(EnsembleError, match="non-negative and not all zero"):
        blend([a, a], [0, 0])
    with pytest.raises(EnsembleError, match="takes no weights"):
        blend([a, a], [2, 1], "stft_min")
    with pytest.raises(EnsembleError, match="needs at least 2048 samples"):
        blend([a[:, :100], 2 * a[:, :100]], method="stft_max")
    with pytest.raises(EnsembleError, match="different outputs"):
        blend_stems([{"vocals": a}, {"drums": a}])
    assert normalise_weights(None, 4).tolist() == [0.25] * 4
    assert normalise_weights([1, 3], 2).tolist() == [0.25, 0.75]
    assert blend([a, a], [1, 1], "stft_min").shape == a.shape  # equal weights are fine


# --- weights and method come from config ---------------------------------------------------


def test_defaults_give_equal_weights_and_the_waveform_method() -> None:
    ensembles = load_defaults().ensembles
    assert ensembles.blend_method == "waveform"
    for slot, models in ensembles.slots.items():
        assert ensembles.method_for(slot) == "waveform"
        assert models.weights_for("single_best") == [1.0]
        assert models.weights_for("two_model_blend") == [0.5, 0.5]
        full = models.weights_for("full_ensemble")
        assert len(full) == len(models.full_ensemble)
        assert sum(full) == pytest.approx(1.0)
        assert len(set(full)) == 1


def _raw_defaults() -> dict:
    return copy.deepcopy(load_defaults().model_dump())


def test_weights_and_method_are_read_from_config(track) -> None:
    raw = _raw_defaults()
    vocals = raw["ensembles"]["slots"]["vocals"]
    vocals["two_model_weights"] = [3, 1]
    vocals["full_ensemble_weights"] = [2, 1, 1]
    vocals["blend_method"] = "stft_magnitude"
    ensembles = Defaults.model_validate(raw).ensembles
    slot = ensembles.slots["vocals"]
    assert slot.weights_for("two_model_blend") == [0.75, 0.25]
    assert slot.weights_for("full_ensemble") == [0.5, 0.25, 0.25]
    assert ensembles.method_for("vocals") == "stft_magnitude"
    assert ensembles.method_for("drums") == "waveform"
    # ...and they are what the combiner is given.
    reference = track.stems["vocals"]
    good = reference + _noise(reference.shape, 0.003, 1)
    bad = reference + _noise(reference.shape, 0.05, 2)
    weighted = blend([good, bad], slot.weights_for("two_model_blend"), "waveform")
    np.testing.assert_allclose(weighted, 0.75 * good + 0.25 * bad, atol=1e-7)


def test_config_with_bad_weights_or_method_is_rejected() -> None:
    for field, value, message in [
        ("two_model_weights", [1, 1, 1], "needs one weight per model"),
        ("full_ensemble_weights", [1, -1, 1], "non-negative and not all zero"),
        ("two_model_weights", [0, 0], "non-negative and not all zero"),
        ("blend_method", "median", "blend_method"),
    ]:
        raw = _raw_defaults()
        raw["ensembles"]["slots"]["vocals"][field] = value
        with pytest.raises(ValueError, match=message):
            Defaults.model_validate(raw)
    raw = _raw_defaults()
    raw["ensembles"]["blend_method"] = "loudest"
    with pytest.raises(ValueError, match="blend_method"):
        Defaults.model_validate(raw)


@pytest.mark.skipif(importlib.util.find_spec("museval") is None, reason="museval not installed")
def test_museval_agrees_that_the_blend_is_between_its_members() -> None:
    from stems_bench.metrics import bss_eval_stem

    rng = np.random.default_rng(4)
    reference = (0.1 * rng.standard_normal((2, 3 * RATE))).astype(np.float32)
    mix = reference + (0.1 * rng.standard_normal(reference.shape)).astype(np.float32)
    good = reference + _noise(reference.shape, 0.003, 1)
    bad = reference + _noise(reference.shape, 0.05, 2)
    scores = [
        bss_eval_stem(reference, estimate, mix, RATE).sdr
        for estimate in (bad, blend([good, bad], [0.5, 0.5]), good)
    ]
    assert scores[0] < scores[1] < scores[2]
