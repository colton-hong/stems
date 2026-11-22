"""Chunked inference reconstructs exactly, crossfades without seams, and keeps channels apart."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from stems_common.synthetic import generate_track  # noqa: E402
from stems_worker.chunking import (  # noqa: E402
    apply_chunked,
    crossfade_window,
    hop_for_overlap,
    plan_chunks,
)

CHUNK = 4096
HOP = 3072  # 25% overlap
OVERLAP = CHUNK - HOP


def identity(batch):
    return batch[:, None]


def _noise(channels: int, n: int, seed: int = 0):
    return np.random.default_rng(seed).uniform(-1, 1, (channels, n)).astype(np.float32)


# --- acceptance: identity reconstruction ---------------------------------------------------


@pytest.mark.parametrize(
    "n",
    [
        1,  # a single sample
        500,  # shorter than the overlap: no reflect padding possible
        OVERLAP,  # exactly the overlap
        OVERLAP + 1,
        1000,  # shorter than one chunk
        CHUNK,  # exactly one chunk
        CHUNK + 1,
        2 * HOP,  # a multiple of the hop
        10_007,  # not a multiple of the hop
        5 * CHUNK + 123,
    ],
)
def test_identity_model_reconstructs_the_input(n: int) -> None:
    audio = _noise(2, n, seed=n)
    out = apply_chunked(audio, identity, chunk_size=CHUNK, hop=HOP)
    assert out.shape == (1, 2, n)
    assert out.dtype == np.float32
    assert float(np.abs(out[0] - audio).max()) < 1e-6


@pytest.mark.parametrize("hop", [CHUNK, CHUNK - 1, CHUNK // 2, CHUNK // 4, 1000, 7])
def test_identity_holds_for_any_overlap(hop: int) -> None:
    audio = _noise(2, 9001, seed=hop)
    chunk = CHUNK if hop >= 1000 else 64
    hop = min(hop, chunk)
    out = apply_chunked(audio, identity, chunk_size=chunk, hop=hop)
    assert float(np.abs(out[0] - audio).max()) < 1e-6


def test_identity_on_the_fixture_at_a_realistic_chunk_size() -> None:
    mix = generate_track(seed=4, duration_seconds=5.0).mix
    chunk = 2 * 44100
    out = apply_chunked(mix, identity, chunk_size=chunk, hop=hop_for_overlap(chunk, 0.25))
    assert float(np.abs(out[0] - mix).max()) < 1e-6


def test_batch_size_does_not_change_the_result() -> None:
    audio = _noise(2, 20_000)
    reference = apply_chunked(audio, identity, chunk_size=CHUNK, hop=HOP, batch_size=1)
    for batch_size in (2, 3, 64):
        out = apply_chunked(audio, identity, chunk_size=CHUNK, hop=HOP, batch_size=batch_size)
        assert np.array_equal(out, reference)


# --- acceptance: no seams ------------------------------------------------------------------


def _ramp_model(amplitude: float):
    """Adds an offset that grows across each chunk: 0 at its start, ``amplitude`` at its end.

    A real model's edge error behaves like this. Glued without overlap, the output jumps
    by ``amplitude`` at every chunk boundary.
    """

    def fn(batch):
        ramp = np.linspace(0.0, amplitude, batch.shape[-1], dtype=np.float32)
        return (batch + ramp)[:, None]

    return fn


def test_position_dependent_error_leaves_no_step_at_chunk_boundaries() -> None:
    amplitude = 1.0
    silence = np.zeros((2, 6 * CHUNK), dtype=np.float32)

    glued = apply_chunked(silence, _ramp_model(amplitude), chunk_size=CHUNK, hop=CHUNK)
    crossfaded = apply_chunked(silence, _ramp_model(amplitude), chunk_size=CHUNK, hop=HOP)

    step_glued = float(np.abs(np.diff(glued[0], axis=1)).max())
    step_crossfaded = float(np.abs(np.diff(crossfaded[0], axis=1)).max())
    assert step_glued > 0.99 * amplitude  # the test can see a seam when there is one
    assert step_crossfaded < 0.002 * amplitude


def test_crossfade_is_smooth_for_larger_overlaps_too() -> None:
    silence = np.zeros((1, 8 * CHUNK), dtype=np.float32)
    for hop in (CHUNK // 2, CHUNK // 4):
        out = apply_chunked(silence, _ramp_model(1.0), chunk_size=CHUNK, hop=hop)
        assert float(np.abs(np.diff(out[0], axis=1)).max()) < 0.002, hop


def test_edges_of_the_audio_are_kept_away_from_chunk_edges() -> None:
    """A model that ruins the first 64 samples of every chunk must not ruin the song's start."""

    def fn(batch):
        out = batch.copy()
        out[..., :64] = 0.0
        return out[:, None]

    audio = _noise(2, 3 * CHUNK)
    out = apply_chunked(audio, fn, chunk_size=CHUNK, hop=HOP)
    assert float(np.abs(out[0, :, :64] - audio[:, :64]).max()) < 1e-6


# --- acceptance: stereo channels stay independent ------------------------------------------


def test_a_silent_channel_stays_exactly_silent() -> None:
    audio = _noise(2, 12_345)
    audio[1] = 0.0
    out = apply_chunked(audio, identity, chunk_size=CHUNK, hop=HOP)
    assert not out[0, 1].any()
    assert float(np.abs(out[0, 0] - audio[0]).max()) < 1e-6


def test_channels_are_not_swapped_or_mixed() -> None:
    def fn(batch):
        scaled = batch * np.array([2.0, -3.0], dtype=np.float32)[None, :, None]
        return scaled[:, None]

    audio = _noise(2, 12_345) * 0.2
    out = apply_chunked(audio, fn, chunk_size=CHUNK, hop=HOP)
    assert float(np.abs(out[0, 0] - 2.0 * audio[0]).max()) < 1e-6
    assert float(np.abs(out[0, 1] + 3.0 * audio[1]).max()) < 1e-6


def test_mono_and_multi_output_models() -> None:
    def two_stems(batch):
        return np.stack([0.25 * batch, 0.75 * batch], axis=1)

    for channels in (1, 2):
        audio = _noise(channels, 10_007)
        out = apply_chunked(audio, two_stems, chunk_size=CHUNK, hop=HOP, batch_size=2)
        assert out.shape == (2, channels, 10_007)
        assert float(np.abs(out[0] - 0.25 * audio).max()) < 1e-6
        assert float(np.abs(out.sum(axis=0) - audio).max()) < 1e-6


# --- planning ------------------------------------------------------------------------------


def test_plan_covers_the_padded_audio_with_the_expected_chunks() -> None:
    plan = plan_chunks(10_007, CHUNK, HOP)
    assert plan.border == OVERLAP
    assert plan.overlap == OVERLAP
    assert plan.starts[0] == 0
    assert all(b - a == HOP for a, b in zip(plan.starts, plan.starts[1:], strict=False))
    assert plan.padded_length == plan.starts[-1] + CHUNK
    assert plan.padded_length >= 10_007 + 2 * OVERLAP
    assert plan.num_chunks == 4

    assert plan_chunks(CHUNK, CHUNK, CHUNK).num_chunks == 1
    assert plan_chunks(CHUNK + 1, CHUNK, CHUNK).num_chunks == 2
    short = plan_chunks(500, CHUNK, HOP)
    assert (short.border, short.num_chunks) == (0, 1)


def test_progress_reports_every_batch() -> None:
    calls: list[tuple[int, int]] = []
    audio = _noise(2, 20_000)
    apply_chunked(
        audio,
        identity,
        chunk_size=CHUNK,
        hop=HOP,
        batch_size=2,
        progress=lambda d, t: calls.append((d, t)),
    )
    total = plan_chunks(20_000, CHUNK, HOP).num_chunks
    assert calls[-1] == (total, total)
    assert [d for d, _ in calls] == sorted({d for d, _ in calls})
    assert len(calls) == -(-total // 2)


def test_window_is_positive_symmetric_and_flat_in_the_middle() -> None:
    window = crossfade_window(CHUNK, OVERLAP)
    assert window.dtype == np.float32
    assert float(window.min()) > 0
    assert np.allclose(window, window[::-1])
    assert np.all(window[OVERLAP : CHUNK - OVERLAP] == 1.0)
    # A chunk's fade-out and the next chunk's fade-in add up to one.
    assert np.allclose(window[:OVERLAP] + window[CHUNK - OVERLAP :], 1.0)
    assert np.all(crossfade_window(CHUNK, 0) == 1.0)
    assert float(crossfade_window(64, 57).min()) > 0


def test_hop_for_overlap() -> None:
    assert hop_for_overlap(4096, 0.25) == 3072
    assert hop_for_overlap(4096, 0.0) == 4096
    assert hop_for_overlap(10, 0.99) == 1
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="overlap"):
            hop_for_overlap(4096, bad)


# --- errors --------------------------------------------------------------------------------


def test_bad_arguments_are_rejected() -> None:
    audio = _noise(2, 1000)
    with pytest.raises(ValueError, match="channels, samples"):
        apply_chunked(audio[0], identity, chunk_size=CHUNK, hop=HOP)
    with pytest.raises(ValueError, match="empty"):
        apply_chunked(audio[:, :0], identity, chunk_size=CHUNK, hop=HOP)
    with pytest.raises(ValueError, match="hop"):
        apply_chunked(audio, identity, chunk_size=CHUNK, hop=0)
    with pytest.raises(ValueError, match="hop"):
        apply_chunked(audio, identity, chunk_size=CHUNK, hop=CHUNK + 1)
    with pytest.raises(ValueError, match="batch_size"):
        apply_chunked(audio, identity, chunk_size=CHUNK, hop=HOP, batch_size=0)


def test_a_model_returning_the_wrong_shape_is_reported() -> None:
    audio = _noise(2, 1000)
    with pytest.raises(ValueError, match="model returned shape"):
        apply_chunked(audio, lambda b: b, chunk_size=CHUNK, hop=HOP)
    with pytest.raises(ValueError, match="model returned shape"):
        apply_chunked(audio, lambda b: b[:, None, :, :-1], chunk_size=CHUNK, hop=HOP)
