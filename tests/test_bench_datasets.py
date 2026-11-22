"""Datasets load only the tracks of a committed, matching frozen list."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

np = pytest.importorskip("numpy")

from stems_bench import protocol  # noqa: E402
from stems_bench.datasets import (  # noqa: E402
    Dataset,
    DatasetError,
    get_dataset,
    load_frozen,
    synthetic,
    write_frozen,
)
from stems_bench.datasets.synthetic import SyntheticDataset, build_frozen, fingerprint  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return SyntheticDataset()


def _copy_of_frozen(tmp_path, change):
    data = copy.deepcopy(SyntheticDataset().frozen.data)
    change(data)
    path = tmp_path / "synthetic.json"
    write_frozen(path, data)
    return path


def test_the_committed_list_names_the_tracks_seeds_and_protocol(dataset) -> None:
    frozen = dataset.frozen
    assert frozen.data["dataset"] == "synthetic"
    assert frozen.data["protocol_version"] == protocol.PROTOCOL_VERSION == 1
    assert frozen.track_ids == [f"synth-{i:02d}" for i in range(1, 9)]
    assert [t["seed"] for t in frozen.tracks] == list(range(101, 109))
    assert frozen.sha256 == hashlib.sha256(synthetic.FROZEN_PATH.read_bytes()).hexdigest()
    assert get_dataset("synthetic").track_ids() == frozen.track_ids
    assert isinstance(dataset, Dataset)


def test_the_committed_list_is_what_the_spec_generates_today() -> None:
    # Exact for the spec; within tolerance for the fingerprints, which may differ in the
    # last digits between platforms.
    committed = SyntheticDataset().frozen.data
    built = build_frozen()
    strip = lambda d: [{k: v for k, v in t.items() if k != "fingerprint"} for t in d["tracks"]]  # noqa: E731
    assert strip(committed) == strip(built)
    assert {k: v for k, v in committed.items() if k != "tracks"} == {
        k: v for k, v in built.items() if k != "tracks"
    }
    for a, b in zip(committed["tracks"], built["tracks"], strict=True):
        assert a["fingerprint"].keys() == b["fingerprint"].keys()
        for name in a["fingerprint"]:
            assert a["fingerprint"][name] == pytest.approx(b["fingerprint"][name], abs=1e-6)


def test_a_track_has_every_input_and_reference_and_they_add_up(dataset) -> None:
    track = dataset.load("synth-01")
    assert track.sample_rate == 44100
    assert track.duration_seconds == 12.0
    assert set(track.inputs) == set(dataset.inputs) == {"mix", "vocals", "vocals_wet"}
    assert set(track.references) == set(dataset.targets) == set(protocol.TARGETS)
    for audio in (*track.inputs.values(), *track.references.values()):
        assert audio.dtype == np.float32
        assert audio.shape == track.mix.shape
    ref = track.references
    total = sum(ref[s].astype(np.float64) for s in PRODUCT_STEMS)
    np.testing.assert_allclose(total, track.mix, atol=1e-6)
    np.testing.assert_allclose(ref["instrumental"] + ref["vocals"], track.mix, atol=1e-6)
    np.testing.assert_allclose(
        ref["other_4stem"], ref["guitar"] + ref["piano"] + ref["other"], atol=1e-6
    )
    np.testing.assert_allclose(ref["lead_vocals"] + ref["backing_vocals"], ref["vocals"], atol=1e-6)
    np.testing.assert_array_equal(ref["dry"], track.inputs["vocals"])
    assert np.abs(track.inputs["vocals_wet"] - ref["dry"]).max() > 1e-3  # there is a reverb tail
    assert dataset.reliable_metrics == ("sdr",)


def test_incomplete_tracks_have_silent_references_not_missing_ones(dataset) -> None:
    no_keys = dataset.load("synth-07")
    assert not no_keys.references["guitar"].any()
    assert not no_keys.references["piano"].any()
    assert no_keys.references["vocals"].any()
    instrumental = dataset.load("synth-08")
    for name in ("vocals", "lead_vocals", "backing_vocals", "dry"):
        assert not instrumental.references[name].any()
    assert instrumental.meta["active_stems"] == ["drums", "bass", "guitar", "piano", "other"]


def test_loading_is_refused_when_the_audio_no_longer_matches_the_list(tmp_path) -> None:
    def other_seed(data):
        data["tracks"][0]["seed"] = 999  # fingerprints still describe seed 101

    stale = SyntheticDataset(_copy_of_frozen(tmp_path, other_seed))
    with pytest.raises(DatasetError, match=r"synth-01: generated \w+ does not match"):
        stale.load("synth-01")
    stale.load("synth-02")  # untouched tracks still load

    def drop_a_stem(data):
        data["tracks"][1]["active_stems"] = ["vocals", "drums"]

    with pytest.raises(DatasetError, match="does not match"):
        SyntheticDataset(_copy_of_frozen(tmp_path, drop_a_stem)).load("synth-02")

    def no_fingerprint(data):
        del data["tracks"][2]["fingerprint"]["drums"]

    with pytest.raises(DatasetError, match="generated drums does not match"):
        SyntheticDataset(_copy_of_frozen(tmp_path, no_fingerprint)).load("synth-03")


def test_a_list_for_another_protocol_or_dataset_is_refused(tmp_path) -> None:
    def older(data):
        data["protocol_version"] = 0

    with pytest.raises(DatasetError, match="frozen under protocol 0"):
        SyntheticDataset(_copy_of_frozen(tmp_path, older)).track_ids()

    def renamed(data):
        data["dataset"] = "musdb18hq_test"

    with pytest.raises(DatasetError, match="is for 'musdb18hq_test'"):
        SyntheticDataset(_copy_of_frozen(tmp_path, renamed)).track_ids()

    def repeated(data):
        data["tracks"].append(data["tracks"][0])

    with pytest.raises(DatasetError, match="no id twice"):
        SyntheticDataset(_copy_of_frozen(tmp_path, repeated)).track_ids()

    with pytest.raises(DatasetError, match="has not been frozen"):
        SyntheticDataset(tmp_path / "absent.json").track_ids()
    with pytest.raises(DatasetError, match=r"is not in synthetic\.json"):
        SyntheticDataset().load("synth-99")
    with pytest.raises(DatasetError, match="unknown dataset"):
        get_dataset("nope")


def test_fingerprint_ignores_the_last_bit_but_not_a_real_change() -> None:
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal((2, 44100))).astype(np.float32)
    nudged = np.nextafter(audio, np.float32(1.0))
    assert fingerprint(audio) == pytest.approx(fingerprint(nudged), abs=1e-7)
    assert fingerprint(audio) != pytest.approx(fingerprint(audio * 1.001), abs=1e-6)
    assert fingerprint(audio) != pytest.approx(fingerprint(np.roll(audio, 1000, axis=1)), abs=1e-6)
    assert fingerprint(np.zeros((2, 100), dtype=np.float32)) == [0.0, 0.0, 0.0]


def test_freeze_command_never_rewrites_an_existing_list(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(synthetic, "SPEC", synthetic.SPEC[:2])  # two short tracks: faster
    monkeypatch.setattr(synthetic, "DURATION_SECONDS", 2.0)
    path = tmp_path / "frozen" / "synthetic.json"
    assert synthetic.main(["--freeze", "--path", str(path)]) == 0
    first = path.read_bytes()
    assert json.loads(first)["tracks"][0]["id"] == "synth-01"
    assert synthetic.main(["--freeze", "--path", str(path)]) == 1
    assert "not rewritten" in capsys.readouterr().err
    assert path.read_bytes() == first
    assert synthetic.main(["--path", str(path)]) == 0
    assert "2 tracks match" in capsys.readouterr().out
    assert load_frozen(path, "synthetic").sha256 == hashlib.sha256(first).hexdigest()


def test_protocol_maps_outputs_to_inputs_and_targets() -> None:
    assert protocol.input_for(["vocals", "instrumental"]) == "mix"
    assert protocol.input_for(["lead_vocals", "backing_vocals"]) == "vocals"
    assert protocol.input_for(["dry"]) == "vocals_wet"
    assert protocol.target_for("other", other_contains_guitar_piano=True) == "other_4stem"
    assert protocol.target_for("other", other_contains_guitar_piano=False) == "other"
    assert protocol.target_for("drums", other_contains_guitar_piano=True) == "drums"
