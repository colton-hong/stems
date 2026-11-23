"""The Clef-omni pre-registration: built from a dataset, written once, read only if committed."""

from __future__ import annotations

import hashlib
import json

import pytest

from stems_bench.clef import clips as c
from stems_bench.clef import prereg
from stems_bench.datasets.synthetic import SyntheticDataset
from stems_common.paths import REPO_ROOT
from stems_common.stems import PRODUCT_STEMS
from stems_worker.listening.audioset import default_stem_map

OPTIONS = {
    "window_seconds": 8.0,
    "part_b_targets": ("vocals", "drums", "bass", "guitar", "piano"),
    "reverb_clips": 1,
}


@pytest.fixture(scope="module")
def built():
    return prereg.build(SyntheticDataset(), created="2026-11-22", **OPTIONS)


def test_thresholds_are_the_reports():
    t = prereg.THRESHOLDS
    assert t["A1"]["min_accuracy"] == 0.85
    assert t["A2"]["min_accuracy"] == {
        "vocals": 0.80,
        "drums": 0.80,
        "bass": 0.80,
        "guitar": 0.80,
        "piano": 0.80,
        "other": 0.70,
    }
    assert t["A3"]["min_accuracy"] == 0.90
    assert (t["B1"]["min_auc_pooled"], t["B1"]["min_auc_hardest"]) == (0.85, 0.80)
    assert t["B1"]["hardest_sir_db"] == max(c.SIR_LEVELS_DB) == 30.0
    assert t["B2"]["min_spearman"] == 0.70
    assert t["B3"]["max_false_positive_rate"] == 0.10
    assert t["C"]["min_margin_points"] == 5.0
    assert set(t) == {"A1", "A2", "A3", "B1", "B2", "B3", "C"}
    assert c.SIR_LEVELS_DB == (30.0, 20.0, 12.0, 6.0)
    assert (c.PRESENT_RMS_DBFS, c.PRESENT_SECONDS, c.WINDOW_SECONDS) == (-40.0, 5.0, 30.0)
    assert c.REVERB_WET_DB == -12.0


def test_every_stem_has_a_question_and_the_legend_has_four_levels():
    assert set(prereg.PRESENCE_QUESTIONS) == set(PRODUCT_STEMS) == set(prereg.TARGET_NAMES)
    assert len(prereg.BLEED_LEGEND) == 4
    assert "{target}" in prereg.BLEED_QUESTION
    assert "{target}" in prereg.BLEED_SCORE_QUESTION


def test_the_built_file_is_plain_reproducible_data(built):
    again = prereg.build(SyntheticDataset(), created="2026-11-22", **OPTIONS)
    assert json.dumps(built, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert built["version"] == prereg.VERSION
    assert built["thresholds"] == prereg.THRESHOLDS
    assert built["excerpt"]["seconds"] == 8.0
    assert built["dataset"]["frozen_sha256"] == SyntheticDataset().frozen.sha256
    stem_map = default_stem_map()
    assert built["class_to_stem"]["sha256"] == stem_map.sha256
    assert built["class_to_stem"]["file"] == "models/listening/audioset_stem_map.yaml"
    assert built["bleed_levels"] == {
        "clean": 0,
        "+30 dB SIR": 1,
        "+20 dB SIR": 2,
        "+12 dB SIR": 3,
        "+6 dB SIR": 4,
    }
    assert built["clef_omni"]["lead_backing"]["options"] == list(c.LEAD_BACKING_ANSWERS)
    request = built["clef_omni"]["request"]
    assert request["clips_per_request"] == 1
    assert request["A"]["question_ids"]["piano"] == "present.piano"
    assert list(built)[-1] == "plan"


def test_counts_describe_the_plan(built):
    counts, plan = built["counts"], built["plan"]
    assert counts["tracks"] == len(plan["tracks"]) == 8
    assert counts["mix_decisions"] == 48
    for stem in PRODUCT_STEMS:
        present = sum(1 for t in plan["tracks"].values() if t["present"][stem])
        assert counts["present"][stem] == present
        assert counts["absent"][stem] == 8 - present
    assert counts["lead_backing_clips"] == len(plan["lead_backing"])
    assert sum(counts["lead_backing_answers"].values()) == counts["lead_backing_clips"]
    assert (counts["part_b_clean"], counts["part_b_bleed"], counts["reverb_clips"]) == (5, 20, 1)
    # The note about A3 quotes the number of excerpts without vocals.
    assert f"{counts['absent']['vocals']} of 8 excerpts" in " ".join(built["notes"])


def test_a_preregistration_is_written_once(built, tmp_path):
    path = tmp_path / "prereg" / "file.json"
    digest = prereg.write(built, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text(encoding="utf-8"))["counts"] == built["counts"]
    with pytest.raises(prereg.PreregError, match="written once"):
        prereg.write(built, path)


def test_an_uncommitted_or_missing_file_cannot_be_used(built, tmp_path):
    with pytest.raises(prereg.PreregError, match="does not exist"):
        prereg.load(tmp_path / "missing.json")
    # Inside the repository but not committed.
    path = REPO_ROOT / "runs" / "test-prereg-uncommitted.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(prereg.PreregError, match="not committed"):
            prereg.load(path)
    finally:
        path.unlink()


def test_the_committed_file_matches_the_code_and_the_frozen_inputs():
    if not prereg.PREREG_PATH.is_file():
        pytest.skip("the pre-registration has not been frozen yet")
    data = json.loads(prereg.PREREG_PATH.read_text(encoding="utf-8"))
    # None of these may drift after the file is committed.
    assert data["version"] == prereg.VERSION
    assert data["thresholds"] == prereg.THRESHOLDS
    assert data["decision_rule"] == prereg.DECISION_RULE
    assert data["clef_omni"]["presence_questions"]["per_stem"] == prereg.PRESENCE_QUESTIONS
    assert data["clef_omni"]["bleed_score"]["legend"] == list(prereg.BLEED_LEGEND)
    assert data["clef_omni"]["request"] == prereg.REQUEST
    assert data["amendments"] == prereg.AMENDMENTS
    assert all(a["before_any_request"] for a in data["amendments"])
    stem_map = default_stem_map()
    assert data["class_to_stem"]["sha256"] == stem_map.sha256
    assert data["class_to_stem"]["labels_sha256"] == stem_map.labels.sha256
    frozen = REPO_ROOT / "bench" / "frozen" / data["dataset"]["frozen_list"]
    assert data["dataset"]["frozen_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    ids = [t["id"] for t in json.loads(frozen.read_text(encoding="utf-8"))["tracks"]]
    assert list(data["plan"]["tracks"]) == ids
    assert len(ids) == 50
    assert data["excerpt"]["seconds"] == 30.0
    assert [i["target"] for i in data["plan"]["part_b"]] == list(c.PART_B_TARGETS)
    assert len({i["track_id"] for i in data["plan"]["part_b"]}) == 20
    assert len(data["plan"]["reverb"]) == 5
    for entry in data["plan"]["tracks"].values():
        assert entry["window"]["seconds"] <= 30.0
        assert set(entry["present"]) == set(PRODUCT_STEMS)
    for item in data["plan"]["part_b"]:
        present = data["plan"]["tracks"][item["track_id"]]["present"]
        assert present[item["target"]]
        assert present[item["contaminant"]]
