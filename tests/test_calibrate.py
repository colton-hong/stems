"""Choosing and checking the listening thresholds (the analysis; no audio model here)."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from stems_bench import calibrate as cal  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402

SIRS = (None, 30.0, 20.0, 12.0, 6.0)


def scores(**given: float) -> dict[str, float]:
    return {stem: given.get(stem, 0.01) for stem in PRODUCT_STEMS}


def stem_row(set_name: str, track: str, target: str, sir, own: float, other: float) -> dict:
    """A stem clip whose own instrument scores ``own`` and whose contaminant scores ``other``."""
    contaminant = "drums" if target != "drums" else "bass"
    return {
        "id": f"stem:{track}:{target}:{sir}",
        "set": set_name,
        "kind": "stem",
        "track_id": track,
        "target": target,
        "contaminant": contaminant,
        "sir_db": sir,
        "level": cal.LEVEL_OF_SIR[sir],
        "scores": scores(**{target: own, contaminant: other}),
    }


def mix_row(set_name: str, track: str, present: dict[str, float]) -> dict:
    return {
        "id": f"mix:{track}",
        "set": set_name,
        "kind": "mix",
        "track_id": track,
        "present": {stem: stem in present for stem in PRODUCT_STEMS},
        "scores": scores(**present),
    }


def separable(set_name: str, tracks: int = 12) -> list[dict]:
    """A tagger that works: own instrument scores high, bleed score rises with the level."""
    by_level = {None: 0.02, 30.0: 0.08, 20.0: 0.3, 12.0: 0.4, 6.0: 0.8}
    rows = []
    for n in range(tracks):
        track = f"{set_name}-{n:02d}"
        rows.append(mix_row(set_name, track, {"vocals": 0.7, "bass": 0.6}))
        for target in ("vocals", "bass"):
            rows += [stem_row(set_name, track, target, sir, 0.9, by_level[sir]) for sir in SIRS]
    return rows


def useless(set_name: str, tracks: int = 12) -> list[dict]:
    """A tagger whose bleed score has nothing to do with the bleed."""
    rng = np.random.default_rng(3)
    rows = []
    for n in range(tracks):
        track = f"{set_name}-{n:02d}"
        rows.append(mix_row(set_name, track, {"vocals": 0.7}))
        for target in ("vocals", "bass"):
            rows += [
                stem_row(set_name, track, target, sir, 0.9, float(rng.uniform(0.02, 0.6)))
                for sir in SIRS
            ]
    return rows


# -- the fixed rules ---------------------------------------------------------------------


def test_the_rules_are_what_the_report_says_they_are() -> None:
    assert cal.LEVEL_OF_SIR == {
        None: "none",
        30.0: "slight",
        20.0: "moderate",
        12.0: "moderate",
        6.0: "severe",
    }
    assert frozenset({"none", "slight"}) == cal.ACCEPTABLE
    assert cal.SETS["calibration"]["dataset"] == "moisesdb_tuning"
    assert cal.SETS["heldout"]["dataset"] == "moisesdb_subset"
    assert cal.SETS["calibration"]["seed"] != cal.SETS["heldout"]["seed"]
    assert set(cal.CRITERIA) == {
        "presence_decided_accuracy_min",
        "presence_coverage_min",
        "bleed_auc_min",
        "bleed_balanced_accuracy_min",
        "clean_flagged_max",
    }


def test_the_two_sets_share_no_track() -> None:
    from stems_bench.datasets import load_frozen
    from stems_common.paths import REPO_ROOT

    frozen = REPO_ROOT / "bench" / "frozen"
    tuning = load_frozen(frozen / "moisesdb_tuning.json", "moisesdb_tuning")
    subset = load_frozen(frozen / "moisesdb_subset.json", "moisesdb_subset")
    assert not {t["id"] for t in tuning.tracks} & {t["id"] for t in subset.tracks}
    assert len(tuning.tracks) == 30
    assert len(subset.tracks) == 50


# -- planning ----------------------------------------------------------------------------


def test_plan_takes_present_stems_and_a_present_contaminant_reproducibly() -> None:
    present = {
        "t1": {stem: stem in ("vocals", "drums", "bass", "guitar") for stem in PRODUCT_STEMS},
        "t2": {stem: stem in ("vocals", "piano") for stem in PRODUCT_STEMS},
        "t3": {stem: stem == "vocals" for stem in PRODUCT_STEMS},  # nothing to bleed in
        "t4": dict.fromkeys(PRODUCT_STEMS, False),
    }
    items = cal.plan_items(present, seed=5)
    assert items == cal.plan_items(present, seed=5)
    assert items != cal.plan_items(present, seed=6)
    by_track: dict[str, list] = {}
    for item in items:
        by_track.setdefault(item["track_id"], []).append(item)
        assert present[item["track_id"]][item["target"]]
        assert present[item["track_id"]][item["contaminant"]]
        assert item["target"] != item["contaminant"]
    assert sorted(by_track) == ["t1", "t2"]
    assert len(by_track["t1"]) == 3  # at most three of its four
    assert len({i["target"] for i in by_track["t1"]}) == 3
    assert sorted(i["target"] for i in by_track["t2"]) == ["piano", "vocals"]


# -- choosing ----------------------------------------------------------------------------


def test_presence_pairs_come_from_the_right_scores() -> None:
    clean = stem_row("calibration", "t", "vocals", None, 0.9, 0.02)
    bled = stem_row("calibration", "t", "vocals", 6.0, 0.8, 0.7)
    pairs = cal.presence_decisions([clean, bled], "stem")
    # Clean: its own instrument is there and the five others are not.
    assert pairs.count((0.9, True)) == 1
    assert sum(1 for _, truth in pairs if not truth) == 5
    # Bled: only the target counts; the contaminant is there, faintly, and is left out.
    assert pairs.count((0.8, True)) == 1
    assert len(pairs) == 7
    mix = mix_row("calibration", "t", {"vocals": 0.6})
    assert sorted(cal.presence_decisions([mix], "mix"))[-1] == (0.6, True)
    assert len(cal.presence_decisions([mix], "mix")) == 6


def test_bleed_score_is_the_strongest_other_source() -> None:
    row = stem_row("calibration", "t", "vocals", 12.0, 0.9, 0.4)
    row["scores"]["piano"] = 0.55
    assert cal.bleed_scores([row]) == [(0.55, "moderate")]


def test_presence_thresholds_are_the_most_generous_that_stay_reliable() -> None:
    pairs = [(0.9, True)] * 30 + [(0.5, True)] * 19 + [(0.5, False)] + [(0.05, False)] * 40
    pairs += [(0.2, False)] * 8 + [(0.2, True)] * 2
    # From 0.5 up: 49 of 50 right (98%). From 0.2 up: 51 of 60 (85%), not reliable enough.
    assert cal.choose_present_min(pairs) == 0.5
    # Up to 0.05: all 40 absent. Up to 0.2: 48 of 50 (96%), still reliable; up to 0.5: 49 of 70.
    assert cal.choose_absent_max(pairs) == 0.2
    outcome = cal.presence_outcome(pairs, 0.5, 0.2)
    assert outcome["decisions"] == 100
    assert outcome["coverage"] == 1.0
    assert outcome["decided_accuracy"] == pytest.approx(97 / 100)
    # With too few decisions on one side, no threshold is claimed.
    assert cal.choose_present_min([(0.9, True)] * 5 + [(0.1, False)] * 50) is None


def test_best_split_maximises_balanced_accuracy_and_respects_its_bounds() -> None:
    scores_ = [(0.1, False)] * 10 + [(0.3, False)] * 2 + [(0.3, True)] * 3 + [(0.6, True)] * 10
    assert cal.best_split(scores_) == 0.3
    ba, caught, passed = cal.balanced_accuracy(scores_, 0.3)
    assert (caught, passed) == (1.0, pytest.approx(10 / 12))
    assert ba == pytest.approx((1 + 10 / 12) / 2)
    assert cal.best_split(scores_, above=0.3) == 0.6
    assert cal.best_split(scores_, below=0.3) == 0.1
    assert cal.best_split([(0.5, True)] * 3) is None  # one class only: nothing to split


def test_a_tagger_that_works_is_calibrated_and_passes_the_bar() -> None:
    result = cal.analyse(separable("calibration") + separable("heldout"))
    chosen = result["chosen"]
    assert result["usable"]
    assert chosen["absent_max"] < chosen["present_min"]
    assert chosen["slight_min"] < chosen["moderate_min"] < chosen["severe_min"]
    assert (chosen["slight_min"], chosen["moderate_min"], chosen["severe_min"]) == (0.08, 0.3, 0.8)
    heldout = result["heldout"]
    assert heldout["bleed"]["auc"] == 1.0
    assert heldout["bleed"]["exact_level"] == 1.0
    assert heldout["bleed"]["clean_flagged"] == 0.0
    assert heldout["presence_on_stems"]["decided_accuracy"] == 1.0
    assert result["verdict"]["calibrated"] is True
    assert all(check["met"] for check in result["verdict"]["checks"].values())
    assert result["counts"]["heldout"] == {"mix_clips": 12, "stem_clips": 120, "tracks": 12}


def test_a_tagger_that_cannot_hear_bleed_fails_the_bar_whatever_is_chosen() -> None:
    result = cal.analyse(useless("calibration") + useless("heldout"))
    verdict = result["verdict"]
    assert verdict["calibrated"] is False
    if result["usable"]:
        assert not verdict["checks"]["bleed_auc_min"]["met"]
        assert 0.3 < result["heldout"]["bleed"]["auc"] < 0.7  # chance, give or take


def test_thresholds_chosen_on_one_set_are_only_ever_checked_on_the_other() -> None:
    good, bad = separable("calibration"), useless("heldout")
    result = cal.analyse(good + bad)
    # Chosen from the set that separates; judged on the one that does not.
    assert result["chosen"]["moderate_min"] == 0.3
    assert result["calibration"]["bleed"]["auc"] == 1.0
    assert result["verdict"]["calibrated"] is False
    with pytest.raises(cal.CalibrationError, match="both"):
        cal.analyse(good)


def test_the_verdict_needs_every_check() -> None:
    passing = {
        "presence_on_stems": {"decided_accuracy": 0.95, "coverage": 0.6},
        "bleed": {"auc": 0.85, "balanced_accuracy": 0.8, "clean_flagged": 0.05},
    }
    assert cal.verdict(passing)["calibrated"] is True
    for section, key, value in (
        ("presence_on_stems", "decided_accuracy", 0.89),
        ("presence_on_stems", "coverage", 0.49),
        ("bleed", "auc", 0.79),
        ("bleed", "balanced_accuracy", 0.74),
        ("bleed", "clean_flagged", 0.11),
        ("bleed", "auc", None),
    ):
        changed = {name: dict(part) for name, part in passing.items()}
        changed[section][key] = value
        assert cal.verdict(changed)["calibrated"] is False, (key, value)
    # Exactly at the bar passes.
    exact = {
        "presence_on_stems": {"decided_accuracy": 0.90, "coverage": 0.50},
        "bleed": {"auc": 0.80, "balanced_accuracy": 0.75, "clean_flagged": 0.10},
    }
    assert cal.verdict(exact)["calibrated"] is True


# -- the report --------------------------------------------------------------------------

RUN = {
    "run_id": "test-run",
    "finished": "2026-11-24T12:00:00+00:00",
    "tagger": "fake",
    "device": "cpu",
    "git": {"commit": "a" * 40, "dirty": False},
    "rules_commit": "b" * 40,
}
CURRENT = {
    "present_min": 0.5,
    "absent_max": 0.2,
    "slight_min": 0.2,
    "moderate_min": 0.5,
    "severe_min": 0.8,
}


def test_report_states_every_threshold_check_and_the_verdict() -> None:
    text = cal.render(cal.analyse(separable("calibration") + separable("heldout")), RUN, CURRENT)
    assert "`runs/calibration/test-run/scores.jsonl`" in text
    assert "at commit `bbbbbbbbbbbb`" in text
    assert "| `listening.bleed.moderate_min` | 0.5 | 0.3000 |" in text
    assert "| Bleed AUC, at least | 0.80 | 1.000 | pass |" in text
    assert "| Clean stems flagged, at most | 10.0% | 0.0% | pass |" in text
    assert "`listening.calibrated` is set to **true**" in text
    assert "| 0.3000 (chosen) |" in text

    failing = cal.render(cal.analyse(separable("calibration") + useless("heldout")), RUN, CURRENT)
    assert "stays **false**" in failing
    assert "| fail |" in failing
    assert "set to **true**" not in failing


def test_report_compares_with_the_thresholds_used_until_now() -> None:
    rows = separable("calibration") + separable("heldout")
    analysis = cal.analyse(rows, CURRENT)
    # The old guesses (moderate from 0.5) miss the bleed at 0.3 and 0.4 that the chosen
    # threshold catches.
    assert analysis["before_heldout"]["bleed"]["balanced_accuracy"] < 1.0
    assert analysis["heldout"]["bleed"]["balanced_accuracy"] == 1.0
    text = cal.render(analysis, RUN, CURRENT)
    assert "## Against the values used until now (held out)" in text
    assert "| Until now |" in text
    assert "| Chosen |" in text
    assert "Against the values used until now" not in cal.render(cal.analyse(rows), RUN, CURRENT)


def test_report_says_so_when_no_thresholds_can_be_chosen() -> None:
    few = separable("calibration", tracks=1)[:3] + separable("heldout", tracks=1)
    analysis = cal.analyse(few)
    assert analysis["usable"] is False
    assert analysis["verdict"]["calibrated"] is False
    text = cal.render(analysis, RUN, CURRENT)
    assert "No usable set of thresholds was found" in text
    assert "none found" in text


# -- presence in a full mix --------------------------------------------------------------


def mixes(set_name: str, *, piano_works: bool, count: int = 24) -> list[dict]:
    """Mixes that all have vocals; half have piano, the other half guitar."""
    rng = np.random.default_rng(11 if set_name == "calibration" else 12)
    rows = []
    for n in range(count):
        heard = {"vocals": 0.2}
        if n % 2 == 0:
            heard["piano"] = 0.05 if piano_works else float(rng.uniform(0.001, 0.02))
        if n % 2 == 1:
            heard["guitar"] = float(rng.uniform(0.001, 0.3))
        row = mix_row(set_name, f"{set_name}-{n:02d}", heard)
        # What a tagger gives an instrument that is not there.
        for stem in ("piano", "guitar"):
            if stem not in heard:
                row["scores"][stem] = 0.004 if stem == "piano" else float(rng.uniform(0.001, 0.3))
        if not piano_works and "piano" not in heard:
            row["scores"]["piano"] = float(rng.uniform(0.001, 0.02))
        rows.append(row)
    return rows


def test_the_mix_rules_are_what_the_report_says_they_are() -> None:
    assert cal.MIX_MIN_EACH == 10
    assert cal.MIX_CRITERIA == {
        "auc_min": 0.80,
        "decided_accuracy_min": 0.90,
        "coverage_min": 0.50,
    }
    assert cal.STEM_RULES_COMMIT.startswith("7d58a494a794")


def test_an_instrument_heard_in_a_mix_gets_its_own_thresholds_and_passes() -> None:
    mix = cal.analyse_mix(
        mixes("calibration", piano_works=True), mixes("heldout", piano_works=True)
    )
    piano = mix["piano"]
    assert piano["calibration"] == piano["heldout"] == {"with": 12, "without": 12}
    assert (piano["judged"], piano["passed"]) == (True, True)
    # Far below anything a stem threshold would be: a mix has its own scale.
    assert (piano["present_min"], piano["absent_max"]) == (0.05, 0.004)
    assert (piano["auc"], piano["coverage"], piano["decided_accuracy"]) == (1.0, 1.0, 1.0)
    # Guitar scores the same with and without: judged, and failed.
    assert (mix["guitar"]["judged"], mix["guitar"]["passed"]) == (True, False)
    # Every mix has vocals and none has drums: there is nothing to tell apart.
    for stem in ("vocals", "drums"):
        assert mix[stem]["judged"] is False
        assert mix[stem]["passed"] is False
        assert mix[stem]["auc"] is None


def test_mix_thresholds_are_chosen_on_one_set_and_checked_on_the_other() -> None:
    mix = cal.analyse_mix(
        mixes("calibration", piano_works=True), mixes("heldout", piano_works=False)
    )
    assert (mix["piano"]["present_min"], mix["piano"]["absent_max"]) == (0.05, 0.004)
    assert mix["piano"]["passed"] is False


def test_the_report_says_which_instruments_can_be_named_in_a_mix() -> None:
    def report(works: bool, commit: str | None) -> str:
        rows = (
            separable("calibration")
            + separable("heldout")
            + mixes("calibration", piano_works=works)
            + mixes("heldout", piano_works=works)
        )
        return cal.render(cal.analyse(rows), {**RUN, "mix_rules_commit": commit}, CURRENT)

    good = report(True, "c" * 40)
    assert "## Presence in a full mix, with thresholds of its own" in good
    assert "at commit `cccccccccccc`" in good
    # The twelve mixes of `separable` have vocals and bass and nothing else.
    assert "| piano | 24, 24 | 1.000 | 0.0500 | 0.0100 | 100.0% | 100.0% | pass |" in good
    assert "| drums | 36, 36 | - | - | - | - | - | cannot be judged |" in good
    assert "**Passed: bass, piano.**" in good
    assert "Cannot be judged: vocals, drums, other." in good
    bad = report(False, None)
    assert "in the commit that adds this section" in bad
    assert "**Passed: bass.**" in bad
    assert "| piano | 24, 24 |" in bad
    assert "| fail |" in bad
    nothing = cal.render(
        cal.analyse(separable("calibration", tracks=9) + separable("heldout", tracks=9)),
        RUN,
        CURRENT,
    )
    assert "**No instrument passed.**" in nothing


# -- collecting, with a stand-in tagger --------------------------------------------------


def test_collect_scores_each_clip_once_and_resumes(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    assert torch
    from stems_bench.datasets import get_dataset
    from stems_common.settings import load_defaults
    from stems_worker.listening.audioset import default_stem_map

    stem_map = default_stem_map()
    classes = len(stem_map.labels.names)

    class Flat:
        """Says every class is equally likely: enough to drive the bookkeeping."""

        name, window_seconds = "flat", 10.0

        def __init__(self) -> None:
            self.calls = 0

        def class_probabilities(self, audio, sample_rate):
            self.calls += 1
            windows = max(1, int(np.ceil(audio.shape[-1] / sample_rate / self.window_seconds)))
            return np.full((windows, classes), 0.25)

    dataset = get_dataset("synthetic")
    tagger = Flat()
    out = tmp_path / "scores.jsonl"
    settings = load_defaults().listening
    added = cal.collect("calibration", dataset, tagger, settings, stem_map, out, seed=1)
    rows = cal.read_rows(out)
    assert added == len(rows) == tagger.calls
    mixes = [row for row in rows if row["kind"] == "mix"]
    stems = [row for row in rows if row["kind"] == "stem"]
    assert len(mixes) == len(dataset.track_ids())
    assert len(stems) % 5 == 0  # clean plus four bled versions of each
    assert {row["level"] for row in stems} == set(cal.LEVELS)
    assert len({row["id"] for row in rows}) == len(rows)
    for row in stems:
        assert row["target"] != row["contaminant"]
        assert set(row["scores"]) == set(PRODUCT_STEMS)
    plan = json.loads((tmp_path / "calibration-plan.json").read_text())
    assert plan["dataset"] == "synthetic"
    assert len(plan["items"]) * 5 == len(stems)
    # Run again: nothing is scored twice.
    assert cal.collect("calibration", dataset, tagger, settings, stem_map, out, seed=1) == 0
    assert tagger.calls == len(rows)
    # Lose the last three rows (an interrupted run): only those are redone.
    out.write_text("".join(line + "\n" for line in out.read_text().splitlines()[:-3]))
    assert cal.collect("calibration", dataset, tagger, settings, stem_map, out, seed=1) == 3
    assert {row["id"] for row in cal.read_rows(out)} == {row["id"] for row in rows}
