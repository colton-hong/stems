"""The Demucs adapter loads safely, picks only the files a request needs, and runs."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("demucs")

from demucs.htdemucs import HTDemucs  # noqa: E402

from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.download import is_downloaded  # noqa: E402
from stems_models.registry import load_registry  # noqa: E402
from stems_worker.adapters.demucs import (  # noqa: E402
    DemucsConfigError,
    DemucsSeparator,
    read_package,
)
from stems_worker.device import pick_device  # noqa: E402
from stems_worker.separator import run_separator  # noqa: E402

REGISTRY = load_registry()
FT = REGISTRY.get("htdemucs_ft")
FT_FILES = {f.filename: Path("/weights") / f.filename for f in FT.weights}


# --- safe loading --------------------------------------------------------------------------


class _Payload:
    """Stands in for the arbitrary object a malicious checkpoint would carry."""


def _package(**overrides):
    package = {
        "klass": HTDemucs,
        "args": (),
        "kwargs": {"sources": ["drums", "bass"]},
        "state": {"layer.weight": torch.ones(2, dtype=torch.float16)},
        "training_args": {},
        "metrics": {},
    }
    package.update(overrides)
    return package


def test_a_well_formed_package_is_read(tmp_path) -> None:
    path = tmp_path / "ok.th"
    torch.save(_package(), path)
    package = read_package(path)
    assert package["klass"] is HTDemucs
    assert package["kwargs"]["sources"] == ["drums", "bass"]


def test_packages_carrying_other_objects_are_refused(tmp_path) -> None:
    path = tmp_path / "evil.th"
    torch.save(_package(training_args={"hook": _Payload()}), path)
    with pytest.raises(pickle.UnpicklingError):
        read_package(path)


def test_packages_that_are_not_htdemucs_are_refused(tmp_path) -> None:
    path = tmp_path / "other.th"
    torch.save(_package(klass="HTDemucs"), path)
    with pytest.raises(DemucsConfigError, match="not an HT-Demucs package"):
        read_package(path)
    torch.save(_package(state={"layer.weight": "not a tensor"}), path)
    with pytest.raises(DemucsConfigError, match="plain tensor dict"):
        read_package(path)


# --- choosing files and outputs ------------------------------------------------------------


def test_single_stem_request_uses_only_that_stems_file() -> None:
    separator = DemucsSeparator(FT, FT_FILES, stems=["bass"])
    assert separator.outputs == ("bass",)
    assert [p.name for p in separator._paths] == ["d12395a8-e57c48e6.th"]
    # The bass model emits all four sources; only its bass output is kept.
    assert separator._picks == [1]


def test_single_stem_request_needs_only_that_stems_file_on_disk() -> None:
    only_bass = {"d12395a8-e57c48e6.th": Path("/weights/d12395a8-e57c48e6.th")}
    assert DemucsSeparator(FT, only_bass, stems=["bass"]).outputs == ("bass",)
    with pytest.raises(DemucsConfigError, match="missing files"):
        DemucsSeparator(FT, only_bass, stems=["bass", "drums"])


def test_bag_without_a_stem_request_uses_every_model_in_source_order() -> None:
    separator = DemucsSeparator(FT, FT_FILES)
    assert separator.outputs == ("drums", "bass", "other", "vocals")
    assert [p.name[:8] for p in separator._paths] == [
        "f7e0c4bc",
        "d12395a8",
        "92cfc3b6",
        "04573f0d",
    ]
    assert separator.chunk_size == round(7.8 * 44100)


def test_requested_stem_order_is_kept() -> None:
    separator = DemucsSeparator(FT, FT_FILES, stems=["vocals", "drums"])
    assert separator.outputs == ("vocals", "drums")
    assert separator._picks == [3, 0]


def test_single_model_entries_emit_every_source() -> None:
    entry = REGISTRY.get("htdemucs_6s")
    files = {f.filename: Path("/weights") / f.filename for f in entry.weights}
    separator = DemucsSeparator(entry, files)
    assert separator.outputs == ("drums", "bass", "other", "vocals", "guitar", "piano")
    with pytest.raises(DemucsConfigError, match="always emits every source"):
        DemucsSeparator(entry, files, stems=["guitar"])


def test_bad_requests_are_rejected() -> None:
    with pytest.raises(DemucsConfigError, match="no model for"):
        DemucsSeparator(FT, FT_FILES, stems=["piano"])
    with pytest.raises(DemucsConfigError, match="not 'demucs'"):
        DemucsSeparator(REGISTRY.get("scnet_large"), {})
    bare = FT.model_copy(update={"config": {"bag": "htdemucs_ft"}})
    with pytest.raises(DemucsConfigError, match="lacks 'sources'"):
        DemucsSeparator(bare, FT_FILES)


def test_every_demucs_entry_declares_sources_and_segment() -> None:
    for entry in REGISTRY.models:
        if entry.loader == "demucs":
            assert set(entry.config["sources"]) == set(entry.stems), entry.id
            assert entry.config["segment_seconds"] > 0, entry.id


# --- whole-track normalisation -------------------------------------------------------------


def test_normalisation_is_undone_exactly_enough() -> None:
    separator = DemucsSeparator(FT, FT_FILES, stems=["bass"])
    audio = generate_track(seed=6, duration_seconds=1.0).mix + np.float32(0.01)
    prepared, state = separator.prepare(audio)
    reference = prepared.mean(axis=0)
    assert abs(float(reference.mean())) < 1e-6
    assert float(reference.std()) == pytest.approx(1.0, abs=1e-4)
    restored = separator.finish(prepared[None], state)[0]
    assert restored.dtype == np.float32
    assert float(np.abs(restored - audio).max()) < 1e-6


def test_silence_survives_normalisation() -> None:
    separator = DemucsSeparator(FT, FT_FILES, stems=["bass"])
    silence = np.zeros((2, 1000), dtype=np.float32)
    prepared, state = separator.prepare(silence)
    assert np.isfinite(prepared).all()
    assert not separator.finish(prepared[None], state).any()


# --- acceptance: real checkpoints (need downloaded weights) --------------------------------


def _fraction(output, stem) -> float:
    s = stem.astype(np.float64).ravel()
    return float(np.dot(output.astype(np.float64).ravel(), s) / np.dot(s, s))


def _level_db(x) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-12))


@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["htdemucs", "htdemucs_ft", "htdemucs_6s"])
def test_demucs_models_load_and_separate_the_fixture(model_id) -> None:
    entry = REGISTRY.get(model_id)
    if not is_downloaded(entry):
        pytest.skip(f"{model_id} is not downloaded")
    track = generate_track(seed=1, duration_seconds=10.0)

    separator = DemucsSeparator.from_registry(model_id)
    separator.load(pick_device())
    result = run_separator(separator, track.mix, 44100)
    separator.unload()

    assert tuple(result.stems) == tuple(entry.config["sources"])
    for stem in result.stems.values():
        assert stem.shape == track.mix.shape
        assert np.isfinite(stem).all()
    # The sources add back to roughly the mix. Measured: -33 dB (htdemucs), -15 dB (ft,
    # four independent models), -24 dB (6s), relative to the mix.
    residual = track.mix - sum(result.stems.values())
    assert _level_db(residual) - _level_db(track.mix) < -10
    # Drums and bass are recognisable even as synthetic tones. Measured: 0.93 to 1.00.
    assert _fraction(result.stems["drums"], track.stems["drums"]) > 0.6
    assert _fraction(result.stems["bass"], track.stems["bass"]) > 0.6
    assert _fraction(result.stems["drums"], track.stems["bass"]) < 0.3
    assert _fraction(result.stems["bass"], track.stems["drums"]) < 0.3


@pytest.mark.slow
def test_single_stem_request_loads_one_model_of_the_bag() -> None:
    if not is_downloaded(FT, stem="bass"):
        pytest.skip("htdemucs_ft bass model is not downloaded")
    track = generate_track(seed=1, duration_seconds=10.0)
    separator = DemucsSeparator.from_registry("htdemucs_ft", stems=["bass"])
    separator.load(pick_device())
    assert len(separator._models) == 1
    result = run_separator(separator, track.mix, 44100)
    separator.unload()
    assert list(result.stems) == ["bass"]
    assert _fraction(result.stems["bass"], track.stems["bass"]) > 0.6
