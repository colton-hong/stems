"""The committed AudioSet class-to-stem table is complete, consistent and usable."""

from __future__ import annotations

import hashlib
import shutil

import pytest
import yaml

from stems_common.stems import PRODUCT_STEMS
from stems_worker.listening.audioset import (
    LISTENING_DIR,
    STEM_MAP_PATH,
    StemMapError,
    default_stem_map,
    load_labels,
    load_stem_map,
)


@pytest.fixture(scope="module")
def mapping():
    return load_stem_map()


def test_label_list_is_audiosets_527_classes(mapping) -> None:
    labels = mapping.labels
    assert len(labels.names) == len(labels.mids) == 527
    assert len(set(labels.mids)) == 527
    assert (labels.names[0], labels.mids[0]) == ("Speech", "/m/09x0r")
    assert labels.names[137] == "Music"
    assert labels.names[153] == "Piano"
    raw = (LISTENING_DIR / "audioset_class_labels.csv").read_bytes()
    assert labels.sha256 == hashlib.sha256(raw).hexdigest()


def test_every_mapped_class_exists_in_the_label_set(mapping) -> None:
    data = yaml.safe_load(STEM_MAP_PATH.read_text())
    assert set(data["stems"]) == set(PRODUCT_STEMS)
    for stem, names in data["stems"].items():
        for name in names:
            assert name in mapping.labels.names, f"{stem}: {name}"
        assert mapping.classes(stem) == names
    for name in data["unmapped"]:
        assert name in mapping.labels.names, name


def test_no_class_serves_two_stems(mapping) -> None:
    every = [i for indices in mapping.stems.values() for i in indices]
    assert len(every) == len(set(every))


def test_every_instrument_class_is_mapped_or_explicitly_left_out(mapping) -> None:
    names = mapping.labels.names
    first, last = names.index("Music"), names.index("Scratching (performance technique)")
    mapped = {names[i] for indices in mapping.stems.values() for i in indices}
    accounted = mapped | set(mapping.unmapped)
    missing = [n for n in names[first : last + 1] if n not in accounted]
    assert missing == []
    assert all(reason for reason in mapping.unmapped.values())


def test_the_reports_core_classes_are_where_it_says(mapping) -> None:
    assert {"Singing", "Speech"} <= set(mapping.classes("vocals"))
    assert {"Drum", "Drum kit"} <= set(mapping.classes("drums"))
    assert "Bass guitar" in mapping.classes("bass")
    assert {"Electric guitar", "Acoustic guitar"} <= set(mapping.classes("guitar"))
    assert "Piano" in mapping.classes("piano")
    assert {"Synthesizer", "Violin, fiddle", "Saxophone"} <= set(mapping.classes("other"))


def test_stem_scores_take_the_largest_probability_of_each_stems_classes(mapping) -> None:
    names = mapping.labels.names
    probabilities = [0.0] * 527
    probabilities[names.index("Singing")] = 0.7
    probabilities[names.index("Male singing")] = 0.9
    probabilities[names.index("Snare drum")] = 0.4
    probabilities[names.index("Music")] = 1.0  # unmapped: must not leak into any stem
    probabilities[names.index("Keyboard (musical)")] = 0.95
    scores = mapping.stem_scores(probabilities)
    assert scores == {
        "vocals": 0.9,
        "drums": 0.4,
        "bass": 0.0,
        "guitar": 0.0,
        "piano": 0.0,
        "other": 0.0,
    }
    with pytest.raises(StemMapError, match="expected 527 probabilities, got 10"):
        mapping.stem_scores([0.0] * 10)


def test_scores_by_class_id_do_not_depend_on_the_taggers_order(mapping) -> None:
    labels = mapping.labels
    by_mid = dict.fromkeys(labels.mids, 0.0)
    by_mid[labels.mids[labels.names.index("Bass guitar")]] = 0.8
    shuffled = dict(reversed(list(by_mid.items())))
    assert mapping.stem_scores_by_mid(shuffled)["bass"] == 0.8
    del shuffled["/m/09x0r"]
    with pytest.raises(StemMapError, match="1 AudioSet classes are missing"):
        mapping.stem_scores_by_mid(shuffled)


def _copy(tmp_path, change):
    shutil.copy(LISTENING_DIR / "audioset_class_labels.csv", tmp_path)
    data = yaml.safe_load(STEM_MAP_PATH.read_text())
    change(data)
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_inconsistent_mappings_are_refused(tmp_path) -> None:
    cases = [
        (lambda d: d["stems"]["piano"].append("Grand piano"), "'Grand piano' is not an AudioSet"),
        (lambda d: d["stems"]["other"].append("Piano"), "mapped to both piano and other"),
        (lambda d: d["stems"].pop("bass"), "stems must be exactly"),
        (lambda d: d["stems"].update(bass=[]), "bass has no classes"),
        (lambda d: d["unmapped"].update(Piano="x"), "both mapped to piano and listed"),
        (lambda d: d.update(combine="mean"), "unknown combine rule"),
        (lambda d: d.update(labels_sha256="0" * 64), "not the file the mapping was written"),
    ]
    for change, message in cases:
        with pytest.raises(StemMapError, match=message):
            load_stem_map(_copy(tmp_path, change))


def test_a_label_file_of_the_wrong_shape_is_refused(tmp_path) -> None:
    short = tmp_path / "labels.csv"
    short.write_text('index,mid,display_name\n0,/m/09x0r,"Speech"\n')
    with pytest.raises(StemMapError, match="1 classes, expected 527"):
        load_labels(short)
    short.write_text('index,mid,display_name\n1,/m/09x0r,"Speech"\n')
    with pytest.raises(StemMapError, match="indices are not"):
        load_labels(short)


def test_the_default_map_has_a_version_and_a_hash() -> None:
    mapping = default_stem_map()
    assert mapping.version == 1
    assert mapping.sha256 == hashlib.sha256(STEM_MAP_PATH.read_bytes()).hexdigest()
    assert default_stem_map() is mapping
