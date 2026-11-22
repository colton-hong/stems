"""The MUSDB18-HQ loader, against a fake tree of synthetic tracks laid out the same way."""

from __future__ import annotations

import json
import subprocess

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from stems_bench import datasets  # noqa: E402
from stems_bench.datasets import (  # noqa: E402
    DatasetError,
    DatasetUnavailableError,
    get_dataset,
    musdb,
)
from stems_bench.datasets.musdb import Musdb18HqDataset  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402

RATE = 44100
NAMES = ("Artist A - First", "Artist B - Second", "Artist C - Third")


def _write_track(folder, seed, seconds=2.0):
    track = generate_track(seed=seed, duration_seconds=seconds)
    parts = {
        "vocals": track.stems["vocals"],
        "drums": track.stems["drums"],
        "bass": track.stems["bass"],
        "other": track.stems["guitar"] + track.stems["piano"] + track.stems["other"],
    }
    folder.mkdir(parents=True)
    for name, audio in parts.items():
        sf.write(folder / f"{name}.wav", audio.T, RATE, subtype="PCM_16")
    sf.write(folder / "mixture.wav", sum(parts.values()).T, RATE, subtype="PCM_16")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("musdb18hq")
    for i, name in enumerate(NAMES):
        _write_track(root / "test" / name, seed=10 + i)
    _write_track(root / "train" / "Artist D - Tuning", seed=20)
    (root / "test" / ".DS_Store").write_bytes(b"junk")  # stray files are not tracks
    return root


def _dataset(tree, tmp_path, split="test", expected=3):
    return Musdb18HqDataset(
        split, root=tree, frozen_path=tmp_path / f"musdb18hq_{split}.json", expected_tracks=expected
    )


def test_first_contact_freezes_the_list_with_a_hash_of_every_file(tree, tmp_path) -> None:
    dataset = _dataset(tree, tmp_path)
    assert not (tmp_path / "musdb18hq_test.json").exists()
    assert dataset.track_ids() == sorted(NAMES)
    data = json.loads((tmp_path / "musdb18hq_test.json").read_text())
    assert data["dataset"] == "musdb18hq_test"
    assert data["split"] == "test"
    assert data["protocol_version"] == 1
    files = data["tracks"][0]["files"]
    assert sorted(files) == ["bass.wav", "drums.wav", "mixture.wav", "other.wav", "vocals.wav"]
    assert all(len(digest) == 64 for digest in files.values())
    # A second dataset object reads the same list instead of rebuilding it.
    before = (tmp_path / "musdb18hq_test.json").read_bytes()
    assert _dataset(tree, tmp_path).frozen.sha256 == dataset.frozen.sha256
    assert (tmp_path / "musdb18hq_test.json").read_bytes() == before


def test_benchmarking_is_refused_until_the_list_is_committed(tree, tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    dataset = Musdb18HqDataset(
        "test", root=tree, frozen_path=repo / "musdb18hq_test.json", expected_tracks=3
    )
    dataset.track_ids()  # writes the list
    with pytest.raises(DatasetError, match="is not committed"):
        dataset.check_ready()
    subprocess.run([*git, "add", "musdb18hq_test.json"], check=True)
    with pytest.raises(DatasetError, match="is not committed"):
        dataset.check_ready()  # staged is not committed
    subprocess.run([*git, "commit", "-q", "-m", "freeze"], check=True)
    dataset.check_ready()
    (repo / "musdb18hq_test.json").write_text((repo / "musdb18hq_test.json").read_text() + "\n")
    with pytest.raises(DatasetError, match="uncommitted changes"):
        dataset.check_ready()


def test_the_committed_synthetic_list_is_ready() -> None:
    get_dataset("synthetic").check_ready()


def test_a_track_loads_with_musdb_stems_mapped_to_our_targets(tree, tmp_path) -> None:
    dataset = _dataset(tree, tmp_path)
    track = dataset.load(NAMES[0])
    assert track.sample_rate == RATE
    assert track.duration_seconds == 2.0
    assert set(track.inputs) == {"mix"}
    assert set(track.references) == set(dataset.targets)
    assert "other" not in track.references  # MUSDB's other is the four-stem one
    original = generate_track(seed=10, duration_seconds=2.0)
    step = 1.0 / 32768
    np.testing.assert_allclose(track.references["vocals"], original.stems["vocals"], atol=step)
    np.testing.assert_allclose(
        track.references["other_4stem"],
        original.stems["guitar"] + original.stems["piano"] + original.stems["other"],
        atol=step,
    )
    np.testing.assert_allclose(
        track.references["instrumental"] + track.references["vocals"], track.mix, atol=4 * step
    )
    for audio in (track.mix, *track.references.values()):
        assert audio.dtype == np.float32
        assert audio.shape == (2, 2 * RATE)


def test_changed_missing_or_odd_audio_is_refused(tree, tmp_path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(tree, copy)
    dataset = _dataset(copy, tmp_path)
    dataset.track_ids()

    drums = copy / "test" / NAMES[0] / "drums.wav"
    audio, _ = sf.read(drums, dtype="float32", always_2d=True)
    sf.write(drums, audio * 0.5, RATE, subtype="PCM_16")
    with pytest.raises(DatasetError, match=r"drums\.wav does not match"):
        dataset.load(NAMES[0])

    (copy / "test" / NAMES[1] / "bass.wav").unlink()
    with pytest.raises(DatasetError, match=r"bass\.wav is missing"):
        dataset.load(NAMES[1])

    with pytest.raises(DatasetError, match=r"is not in musdb18hq_test\.json"):
        dataset.load("Artist Z - Not There")
    dataset.load(NAMES[2])  # the untouched track still loads


def test_freezing_refuses_a_tree_that_is_not_the_released_dataset(tree, tmp_path) -> None:
    with pytest.raises(DatasetError, match="found 3 tracks in test/, expected 50"):
        Musdb18HqDataset("test", root=tree, frozen_path=tmp_path / "a.json").track_ids()
    assert not (tmp_path / "a.json").exists()

    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(tree, broken)
    (broken / "test" / NAMES[0] / "other.wav").unlink()
    with pytest.raises(DatasetError, match=r"other\.wav is missing"):
        _dataset(broken, tmp_path / "b").track_ids()

    mono = tmp_path / "mono"
    shutil.copytree(tree, mono)
    path = mono / "test" / NAMES[0] / "vocals.wav"
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    sf.write(path, audio[:, 0], 48000, subtype="PCM_16")
    dataset = _dataset(mono, tmp_path / "c")
    with pytest.raises(DatasetError, match="expected 44100 Hz stereo, found 48000 Hz, 1 channel"):
        dataset.load(NAMES[0])


def test_the_dataset_is_reported_unavailable_without_a_path(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MUSDB18HQ_PATH", raising=False)
    monkeypatch.setattr(musdb, "get_env", lambda name: None)
    with pytest.raises(DatasetUnavailableError, match="MUSDB18HQ_PATH is not set"):
        Musdb18HqDataset("test", frozen_path=tmp_path / "x.json").track_ids()
    assert not (tmp_path / "x.json").exists()  # nothing is frozen from nothing
    monkeypatch.setattr(musdb, "get_env", lambda name: str(tmp_path))
    with pytest.raises(DatasetUnavailableError, match="has no test/ folder"):
        Musdb18HqDataset("test", frozen_path=tmp_path / "x.json").track_ids()
    with pytest.raises(DatasetError, match="unknown MUSDB18-HQ split"):
        Musdb18HqDataset("validation")


def test_path_comes_from_the_environment_and_train_is_its_own_dataset(
    tree, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MUSDB18HQ_PATH", str(tree))
    train = Musdb18HqDataset("train", frozen_path=tmp_path / "train.json", expected_tracks=1)
    assert train.name == "musdb18hq_train"
    assert train.track_ids() == ["Artist D - Tuning"]
    assert train.load("Artist D - Tuning").meta == {"split": "train"}
    assert isinstance(get_dataset("musdb18hq_train"), Musdb18HqDataset)
    assert get_dataset("musdb18hq_test").split == "test"
    assert datasets.is_committed(tmp_path / "train.json") is False


def test_command_reports_what_is_missing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("MUSDB18HQ_PATH", raising=False)
    monkeypatch.setattr(musdb, "get_env", lambda name: None)
    monkeypatch.setattr(musdb, "FROZEN_DIR", tmp_path)
    assert musdb.main([]) == 1
    assert "MUSDB18HQ_PATH is not set" in capsys.readouterr().err
