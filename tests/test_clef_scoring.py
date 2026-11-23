"""The Clef-omni test's statistics, each against a case worked by hand."""

from __future__ import annotations

import math

import pytest

from stems_bench.clef import scoring as s
from stems_bench.clef.prereg import THRESHOLDS
from stems_common.stems import PRODUCT_STEMS


def test_accuracy():
    assert s.accuracy([True, True, False, False], [True, False, False, True]) == 0.5
    assert s.accuracy([True], [True]) == 1.0
    assert s.accuracy([], []) is None
    with pytest.raises(ValueError, match="differ in length"):
        s.accuracy([True], [])


def test_auc_counts_pairs_and_half_for_ties():
    # Pairs (positive, negative): 0.35>0.1 yes, 0.35>0.4 no, 0.8>0.1 yes, 0.8>0.4 yes.
    assert s.auc([0.1, 0.4], [0.35, 0.8]) == 0.75
    assert s.auc([0.1, 0.2], [0.3, 0.4]) == 1.0
    assert s.auc([0.3, 0.4], [0.1, 0.2]) == 0.0
    assert s.auc([0.5], [0.5]) == 0.5
    # One tie among four pairs: (0.5, 0.5) half, (0.5, 0.2) yes, (0.9, 0.5) yes, (0.9, 0.2) yes.
    assert s.auc([0.5, 0.2], [0.5, 0.9]) == 3.5 / 4
    assert s.auc([], [0.5]) is None
    assert s.auc([0.5], []) is None


def test_ranks_share_the_mean_among_ties():
    assert s.ranks([10, 30, 20]) == [1.0, 3.0, 2.0]
    assert s.ranks([5, 6, 7, 8, 7]) == [1.0, 2.0, 3.5, 5.0, 3.5]
    assert s.ranks([2, 2, 2]) == [2.0, 2.0, 2.0]


def test_spearman_by_hand():
    assert s.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert s.spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)
    # Ranks 1..5 against 1, 2, 3.5, 5, 3.5: covariance 8, variances 10 and 9.5.
    assert s.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(8 / math.sqrt(95))
    # No ties, so the textbook formula applies: d = 0, 1, -1, 0; 1 - 6*2/(4*15) = 0.8.
    assert s.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert s.spearman([1, 2, 3], [4, 4, 4]) is None
    assert s.spearman([1], [1]) is None


def test_best_f1_threshold():
    truth = [True, True, False, False]
    # At 0.6: predicted T, F, F, F -> F1 2/3. At 0.4: T, T, T, F -> F1 0.8. At 0.3: all -> 2/3.
    best = s.best_f1_threshold(truth, [0.6, 0.4, 0.4, 0.3])
    assert best == {"threshold": 0.4, "f1": pytest.approx(0.8), "accuracy": 0.75}
    assert s.best_f1_threshold([False, False], [0.1, 0.9]) is None


def part_a(wrong: dict[str, int], tracks: int = 20, absent: int = 4) -> list[dict]:
    """``tracks`` decisions per stem, the last ``absent`` truly absent; ``wrong[stem]`` of
    the present ones are answered wrongly."""
    items = []
    for stem in PRODUCT_STEMS:
        for n in range(tracks):
            truth = n < tracks - absent
            right = n >= wrong.get(stem, 0)
            says_present = truth if right else not truth
            items.append(
                {"track_id": f"t{n}", "stem": stem, "truth": truth,
                 "probability": 0.9 if says_present else 0.1}
            )  # fmt: skip
    return items


def test_part_a_thresholds_at_and_just_under_the_line():
    # 120 decisions, 18 wrong: exactly 85 %. Per stem 3 wrong of 20: 85 %.
    result = s.score_part_a(part_a(dict.fromkeys(PRODUCT_STEMS, 3)), THRESHOLDS)
    assert result["A1"] == {
        "value": 0.85,
        "threshold": 0.85,
        "pass": True,
        "n": 120,
        "correct": 102,
    }
    assert result["A2"]["pass"] is True
    assert result["A2"]["per_stem"]["piano"]["value"] == 0.85
    # Vocals at 85 % is under A3's 90 %.
    assert result["A3"]["value"] == 0.85
    assert result["A3"]["pass"] is False
    assert (result["A3"]["n"], result["A3"]["absent"]) == (20, 4)
    assert result["A3"]["always_present_accuracy"] == 0.8


def test_part_a_per_stem_thresholds_differ_for_other():
    # 5 wrong of 20 is 75 %: enough for "other" (70 %), not for guitar (80 %).
    result = s.score_part_a(part_a({"other": 5, "guitar": 5}), THRESHOLDS)
    per_stem = result["A2"]["per_stem"]
    assert per_stem["other"]["value"] == 0.75
    assert per_stem["other"]["pass"] is True
    assert per_stem["guitar"]["pass"] is False
    assert result["A2"]["pass"] is False
    assert result["A1"]["value"] == pytest.approx(110 / 120)
    assert result["A3"]["pass"] is True


def test_a_probability_of_exactly_half_counts_as_present():
    items = [{"track_id": "t", "stem": stem, "truth": True, "probability": 0.5}
             for stem in PRODUCT_STEMS]  # fmt: skip
    assert s.score_part_a(items, THRESHOLDS)["A1"]["value"] == 1.0


def part_b(clean_p, bleed_p, scores=None) -> list[dict]:
    """``clean_p``: probabilities of the clean clips; ``bleed_p[level]``: of each level."""
    items = [{"clip_id": f"c{n}", "level": 0, "bleed_probability": p}
             for n, p in enumerate(clean_p)]  # fmt: skip
    for level, group in bleed_p.items():
        items += [{"clip_id": f"l{level}-{n}", "level": level, "bleed_probability": p}
                  for n, p in enumerate(group)]  # fmt: skip
    for item in items:
        item["bleed_score"] = None if scores is None else scores[item["level"]]
    return items


def test_part_b_by_hand():
    items = part_b(
        clean_p=[0.1, 0.2, 0.3, 0.6],
        bleed_p={1: [0.25, 0.7], 2: [0.8, 0.8], 3: [0.9, 0.9], 4: [0.95, 0.95]},
        scores={0: 0.2, 1: 0.9, 2: 1.5, 3: 2.2, 4: 2.9},
    )
    result = s.score_part_b(items, THRESHOLDS)
    # Hardest level against clean: 0.25 beats 0.1 and 0.2 (2 of 4); 0.7 beats all 4.
    assert result["B1"]["auc_hardest"] == 6 / 8
    # Pooled: those 6, plus the six louder clips beating every clean clip (24): 30 of 32.
    assert result["B1"]["auc_pooled"] == 30 / 32
    assert result["B1"]["pass"] is False  # 0.75 is under the 0.80 needed at +30 dB
    assert (result["B1"]["clean"], result["B1"]["bleed"], result["B1"]["hardest"]) == (4, 8, 2)
    # Scores rise with every level, so the ranks agree exactly up to ties within a level.
    assert result["B2"]["value"] == pytest.approx(1.0)
    assert result["B2"]["rated_by"] == "bleed_score"
    assert result["B2"]["pass"] is True
    # One clean clip of four has p >= 0.5.
    assert result["B3"]["value"] == 0.25
    assert (result["B3"]["flagged"], result["B3"]["pass"]) == (1, False)


def test_a_clean_clip_is_flagged_by_score_or_by_probability():
    items = part_b(clean_p=[0.1, 0.2, 0.5, 0.49], bleed_p={1: [0.9]})
    for item, score in zip(items, [0.2, 1.0, 0.3, 0.99, 3.0], strict=True):
        item["bleed_score"] = score
    result = s.score_part_b(items, THRESHOLDS)
    assert result["B3"]["flagged"] == 2  # score 1.0, and probability 0.5
    assert result["B3"]["value"] == 0.5


def test_without_a_rubric_score_the_probability_is_ranked_instead():
    items = part_b(clean_p=[0.1] * 10, bleed_p={1: [0.2], 2: [0.3], 3: [0.4], 4: [0.5]})
    result = s.score_part_b(items, THRESHOLDS)
    assert result["B2"]["rated_by"] == "bleed_probability"
    assert result["B2"]["value"] > 0.7
    assert result["B1"]["auc_pooled"] == 1.0
    assert result["B3"] == {
        "value": 0.0,
        "threshold": 0.10,
        "pass": True,
        "clean": 10,
        "flagged": 0,
    }
    assert result["B1"]["pass"] is True


def test_missing_groups_fail_rather_than_pass():
    result = s.score_part_b(part_b(clean_p=[], bleed_p={2: [0.9]}), THRESHOLDS)
    assert result["B1"]["auc_pooled"] is None
    assert result["B1"]["pass"] is False
    assert result["B3"]["pass"] is False
    assert result["B2"]["pass"] is False
    empty = s.score_part_a([], THRESHOLDS)
    assert empty["A1"]["pass"] is False
    assert empty["A2"]["pass"] is False


def good_b() -> list[dict]:
    return part_b(
        clean_p=[0.05] * 10,
        bleed_p={1: [0.6] * 2, 2: [0.7] * 2, 3: [0.8] * 2, 4: [0.9] * 2},
        scores={0: 0.1, 1: 1.2, 2: 1.8, 3: 2.4, 4: 2.9},
    )


def test_evaluate_and_the_decision_rule():
    passing = s.evaluate(part_a({}), good_b(), THRESHOLDS)
    assert passing["passes_a"] is True
    assert passing["passes_b"] is True
    # BEATs a little worse on A1 and equal on B1.
    beats = s.evaluate(part_a({"guitar": 8}), good_b(), THRESHOLDS)
    comparison = s.compare(passing, beats, THRESHOLDS)
    assert comparison["A1_margin_points"] == pytest.approx(8 / 120 * 100)
    assert comparison["B1_margin_points"] == 0.0
    assert comparison["pass"] is False  # the margin is needed on both
    assert s.verdict(passing, comparison)["clef_omni"] == "passes, but BEATs preferred"
    assert s.verdict(passing, comparison)["listening_layer"].startswith("BEATs")
    ahead = {"A1_margin_points": 6.0, "B1_margin_points": 5.0, "threshold": 5.0, "pass": True}
    assert s.verdict(passing, ahead) == {
        "clef_omni": "gating",
        "listening_layer": "Clef-omni",
        "passes_a": True,
        "passes_b": True,
        "beats_margin_met": True,
    }
    bad_b = s.evaluate(part_a({}), part_b([0.9] * 4, {1: [0.1], 2: [0.1]}), THRESHOLDS)
    assert s.verdict(bad_b, ahead)["clef_omni"] == "presence and routing only"
    bad_a = s.evaluate(part_a(dict.fromkeys(PRODUCT_STEMS, 10)), good_b(), THRESHOLDS)
    assert s.verdict(bad_a, ahead)["clef_omni"] == "rejected"


def test_the_margin_is_in_points_and_five_is_enough():
    clef = {"A1": {"value": 0.90}, "B1": {"auc_pooled": 0.91}}
    beats = {"A1": {"value": 0.85}, "B1": {"auc_pooled": 0.86}}
    result = s.compare(clef, beats, THRESHOLDS)
    assert result["A1_margin_points"] == pytest.approx(5.0)
    assert result["B1_margin_points"] == pytest.approx(5.0)
    assert result["pass"] is True
    beats["B1"]["auc_pooled"] = 0.87
    assert s.compare(clef, beats, THRESHOLDS)["pass"] is False
    beats["B1"]["auc_pooled"] = None
    assert s.compare(clef, beats, THRESHOLDS)["pass"] is False
