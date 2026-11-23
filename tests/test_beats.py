"""The BEATs wrapper: windowing, class order and vendoring without weights; the model with them."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from stems_common import schemas
from stems_common.paths import REPO_ROOT
from stems_common.synthetic import generate_track
from stems_models.registry import load_registry
from stems_models.vendor import (
    BEATS_COMMIT,
    BEATS_SOURCES,
    TORCHAUDIO_TAG,
    beats_lib_root,
    rewrite_imports,
)
from stems_worker.listening import beats
from stems_worker.listening.audioset import default_stem_map
from stems_worker.listening.beats import (
    SAMPLE_RATE,
    WINDOW,
    BeatsError,
    BeatsTagger,
    audioset_order,
    to_mono,
    windows,
)
from stems_worker.listening.features import build_track_features, stem_signals

# --- no weights needed --------------------------------------------------------------------


def test_to_mono_averages_channels_and_resamples_to_16k() -> None:
    stereo = np.stack([np.ones(44100, np.float32), np.zeros(44100, np.float32)])
    mono = to_mono(stereo, 44100)
    assert mono.shape == (16000,)
    assert mono.dtype == np.float32
    assert float(np.median(mono)) == pytest.approx(0.5, abs=1e-3)
    assert to_mono(np.ones(16000, np.float32), 16000).shape == (16000,)


def test_windows_are_ten_seconds_and_the_tail_is_kept_as_it_is() -> None:
    ramp = np.arange(24 * SAMPLE_RATE, dtype=np.float32)
    parts = windows(ramp)
    assert [p.size for p in parts] == [WINDOW, WINDOW, 4 * SAMPLE_RATE]
    np.testing.assert_array_equal(np.concatenate(parts), ramp)
    # A clip shorter than a window is one short window; under a second after a full
    # window is dropped; exactly thirty seconds is three windows.
    assert [p.size for p in windows(np.ones(SAMPLE_RATE // 2, np.float32))] == [SAMPLE_RATE // 2]
    assert [p.size for p in windows(np.ones(WINDOW + 100, np.float32))] == [WINDOW]
    assert len(windows(np.ones(3 * WINDOW, np.float32))) == 3
    with pytest.raises(BeatsError, match="non-empty mono"):
        windows(np.zeros(0, np.float32))
    with pytest.raises(BeatsError, match="non-empty mono"):
        windows(np.zeros((2, 100), np.float32))


def test_output_is_rearranged_to_audiosets_order_by_class_id() -> None:
    labels = default_stem_map().labels
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(labels.mids))
    # The model's output position i holds the class labels.mids[shuffled[i]].
    label_dict = {i: labels.mids[int(j)] for i, j in enumerate(shuffled)}
    order = audioset_order(label_dict, labels)
    model_output = shuffled.astype(float)  # each position holds its class's AudioSet index
    np.testing.assert_array_equal(model_output[order], np.arange(len(labels.mids)))
    with pytest.raises(BeatsError, match="not AudioSet's 527"):
        audioset_order({0: "/m/09x0r"}, labels)
    with pytest.raises(BeatsError, match="not AudioSet's 527"):
        audioset_order({**label_dict, 0: label_dict[1]}, labels)


def test_vendored_code_is_pinned_by_commit_and_hash() -> None:
    assert len(BEATS_COMMIT) == 40
    assert beats_lib_root().name == f"beats-{BEATS_COMMIT[:12]}"
    lock = json.loads((REPO_ROOT / "models" / "checksums.lock.json").read_text())
    for name, (url, _) in BEATS_SOURCES.items():
        assert BEATS_COMMIT in url or f"/{TORCHAUDIO_TAG}/" in url
        assert url.startswith("https://raw.githubusercontent.com/")
        assert lock[url]["filename"] == name
        assert len(lock[url]["sha256"]) == 64
    assert {n for n in BEATS_SOURCES if n.endswith(".py")} == {
        "BEATs.py",
        "backbone.py",
        "modules.py",
        "kaldi.py",
    }


def test_imports_are_rewritten_exactly_once_or_not_at_all() -> None:
    source = "import torch\nfrom backbone import (\n    Thing,\n)\n"
    fixed = rewrite_imports("x.py", source, {"from backbone import (": "from .backbone import ("})
    assert "from .backbone import (" in fixed
    assert rewrite_imports("x.py", source, {}) == source
    with pytest.raises(RuntimeError, match="expected one"):
        rewrite_imports("x.py", source, {"from modules import (": "from .modules import ("})
    with pytest.raises(RuntimeError, match="found 2"):
        rewrite_imports("x.py", source + source, {"from backbone import (": "x"})


def test_a_file_that_is_not_a_beats_checkpoint_is_refused(tmp_path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(BeatsError, match="not a fine-tuned BEATs checkpoint"):
        beats.read_checkpoint(path)


def test_an_unloaded_tagger_says_so() -> None:
    tagger = BeatsTagger(load_registry().get(beats.MODEL_ID), {})
    with pytest.raises(BeatsError, match="is not loaded"):
        tagger.class_probabilities(np.zeros(16000, np.float32), 16000)
    with pytest.raises(BeatsError, match="is not a BEATs model"):
        BeatsTagger.from_registry("htdemucs")
    assert (tagger.name, tagger.window_seconds) == ("beats_iter3plus_as2m_cpt2", 10.0)


# --- with the real checkpoint -------------------------------------------------------------


@pytest.fixture(scope="module")
def tagger():
    tagger = BeatsTagger.from_registry()
    tagger.load("cpu")
    yield tagger
    tagger.unload()


@pytest.fixture(scope="module")
def track():
    return generate_track(seed=11, duration_seconds=25.0)


@pytest.mark.slow
def test_527_probabilities_per_ten_second_window(tagger, track) -> None:
    probabilities = tagger.class_probabilities(track.mix, track.sample_rate)
    assert probabilities.shape == (3, 527)  # 25 s: two full windows and a 5 s tail
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    # The same input gives the same answer, whatever the batch size.
    again = tagger.class_probabilities(track.mix, track.sample_rate, batch_size=1)
    np.testing.assert_allclose(probabilities, again, atol=1e-5)
    mono = tagger.class_probabilities(track.mix.mean(axis=0), track.sample_rate)
    np.testing.assert_allclose(probabilities, mono, atol=1e-5)


@pytest.mark.slow
def test_classes_come_out_in_audiosets_order(tagger, track) -> None:
    names = default_stem_map().labels.names
    mix = tagger.class_probabilities(track.mix, track.sample_rate).mean(axis=0)
    assert names[int(np.argmax(mix))] == "Music"
    drums = tagger.class_probabilities(track.stems["drums"], track.sample_rate).mean(axis=0)
    scores = default_stem_map().stem_scores(drums)
    assert max(scores, key=scores.get) == "drums"
    assert scores["drums"] > 0.5
    piano = tagger.class_probabilities(track.stems["piano"], track.sample_rate).mean(axis=0)
    scores = default_stem_map().stem_scores(piano)
    assert max(scores, key=scores.get) == "piano"


@pytest.mark.slow
def test_it_feeds_the_feature_builder(tagger, track) -> None:
    features = build_track_features(track.mix, track.sample_rate, tagger=tagger)
    assert schemas.errors("track_features", features) == []
    signals = stem_signals("drums", track.stems["drums"], track.sample_rate, tagger=tagger)
    assert signals.tagger == "beats_iter3plus_as2m_cpt2"
    assert signals.present is True
    metrics = signals.metrics()
    assert schemas.errors("stem_metrics", metrics) == []
    assert metrics["bleed_estimate"] is not None


@pytest.mark.slow
@pytest.mark.gpu
def test_mps_agrees_with_cpu(tagger, track) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    on_cpu = tagger.class_probabilities(track.mix, track.sample_rate)
    other = BeatsTagger.from_registry()
    other.load("mps")
    try:
        on_mps = other.class_probabilities(track.mix, track.sample_rate)
    finally:
        other.unload()
    np.testing.assert_allclose(on_cpu, on_mps, atol=1e-3)
