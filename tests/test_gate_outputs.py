"""The listening layer on real separator output: the bookkeeping and the analysis.

No audio model runs here. The rows that stand in for scored outputs are made up to have a
known answer, and the collecting step is driven by a stand-in tagger over a made-up
benchmark folder.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from stems_bench import gate_outputs as go  # noqa: E402
from stems_bench.clef import clips as c  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402

LISTENING = {
    "presence": {"present_min": 0.6, "absent_max": 0.05},
    "bleed": {"moderate_min": 0.3},
    "silent_fraction_min": 0.98,
}


def row(
    set_name: str,
    track: str,
    stem: str,
    model: str,
    *,
    present: bool = True,
    own: float = 0.8,
    bleed: float = 0.02,
    level: float = -20.0,
    sdr: float | None = 8.0,
    sir: float | None = 18.0,
    silence: float = 0.02,
) -> dict:
    other = "drums" if stem != "drums" else "bass"
    scores = dict.fromkeys(PRODUCT_STEMS, 0.001)
    scores.update({stem: own, other: bleed})
    return {
        "id": f"{set_name}:{model}:{track}:{stem}",
        "set": set_name,
        "model_id": model,
        "track_id": track,
        "stem": stem,
        "reference_present": present,
        "scores": scores,
        "level_dbfs": level,
        "mix_level_dbfs": -14.0,
        "silence_fraction": silence,
        "sdr_db": sdr if present else None,
        "sir_db": sir if present else None,
        "scored_seconds": 60 if present else 0,
    }


def world(set_name: str, *, level_works: bool, tagger_works: bool, bleed_works: bool) -> list[dict]:
    """Twenty tracks: piano is missing from half, and guitar has a good and a poor model."""
    rng = np.random.default_rng(5 if set_name == "calibration" else 6)
    rows = []
    for n in range(20):
        track = f"{set_name}-{n:02d}"
        there = n % 2 == 0
        quiet = -60.0 if level_works else float(rng.uniform(-40, -15))
        unheard = 0.01 if tagger_works else float(rng.uniform(0.01, 0.9))
        rows.append(
            row(
                set_name,
                track,
                "piano",
                "six",
                present=there,
                own=0.8 if there else unheard,
                level=-22.0 if there else quiet,
            )
        )
        good_bleed, poor_bleed = (0.02, 0.4) if bleed_works else rng.uniform(0.02, 0.4, 2)
        rows.append(row(set_name, track, "guitar", "six", sdr=9.0, bleed=float(good_bleed)))
        rows.append(row(set_name, track, "guitar", "four", sdr=4.0, bleed=float(poor_bleed)))
    return rows


# -- the fixed rules ---------------------------------------------------------------------


def test_the_rules_are_what_the_report_says_they_are() -> None:
    assert go.SETS == {"calibration": "moisesdb_tuning", "heldout": "moisesdb_subset"}
    assert go.EMPTY_RULES == ("level", "tagger_config", "tagger_rechosen")
    assert go.TAGGER_ABSENT_MAX == 0.0207
    assert (go.EMPTY_RELIABILITY, go.MIN_SUPPORT) == (0.95, 10)
    assert (go.PAIR_GAP_DB, go.PAIR_MIN_SECONDS, go.PAIR_RULE) == (3.0, 10, "bleed")
    assert go.CRITERIA == {
        "empty_precision_min": 0.90,
        "empty_recall_min": 0.50,
        "pair_accuracy_min": 0.80,
        "pair_support_min": 30,
    }


# -- signals -----------------------------------------------------------------------------


def test_the_signals_are_read_as_a_job_reads_them() -> None:
    one = row("heldout", "t", "piano", "m", own=0.7, bleed=0.2, level=-30.0)
    assert go.own_score(one) == 0.7
    assert go.bleed_score(one) == 0.2  # the strongest other source, never its own
    assert go.relative_level(one) == -16.0


def test_benchmark_windows_inside_the_excerpt_are_used_and_gaps_dropped() -> None:
    values = [float(i) for i in range(10)]
    # 2.5 s to 7.5 s: windows 3, 4, 5 and 6 lie wholly inside.
    assert go.frames_in(values, c.Window(250, 500), 100, 1.0) == [3.0, 4.0, 5.0, 6.0]
    assert go.frames_in(values, c.Window(200, 300), 100, 1.0) == [2.0, 3.0, 4.0]
    gaps = [1.0, None, float("nan"), 4.0]
    assert go.frames_in(gaps, c.Window(0, 400), 100, 1.0) == [1.0, 4.0]
    assert go.frames_in([], c.Window(0, 400), 100, 1.0) == []


# -- question 1: empty stems -------------------------------------------------------------


def test_an_empty_rule_is_judged_by_what_it_calls_empty_and_what_it_misses() -> None:
    pairs = [(-60.0, False)] * 8 + [(-50.0, True)] * 2 + [(-20.0, True)] * 10 + [(-25.0, False)] * 2
    outcome = go.empty_outcome(pairs, -50.0)
    assert (outcome["called_empty"], outcome["right"], outcome["empty"]) == (10, 8, 10)
    assert outcome["precision"] == 0.8
    assert outcome["recall"] == 0.8
    assert 0.5 < outcome["auc"] < 1.0
    nothing = go.empty_outcome(pairs, None)
    assert nothing["called_empty"] is None
    assert nothing["auc"] == outcome["auc"]
    assert go.empty_verdict(nothing) == {"enough": False, "passed": False}


def test_the_gates_own_silence_rule_is_shown_beside_the_rules_on_trial() -> None:
    rows = [
        row("heldout", "a", "piano", "m", present=False, silence=1.0),
        row("heldout", "b", "piano", "m", present=False, silence=0.99),
        row("heldout", "c", "piano", "m", present=False, silence=0.5),  # leftovers, not silence
        row("heldout", "d", "piano", "m", present=False, silence=0.9),
        row("heldout", "e", "piano", "m", silence=0.985),  # a real part that plays once
        row("heldout", "f", "piano", "m", silence=0.1),
    ]
    assert go.silence_outcome(rows, 0.98) == {
        "threshold": 0.98,
        "called_empty": 3,
        "right": 2,
        "precision": 2 / 3,
        "recall": 0.5,
    }
    assert go.silence_outcome(rows[4:], 0.999)["precision"] is None


def test_the_threshold_is_the_most_generous_that_stays_reliable() -> None:
    pairs = [(-60.0, False)] * 12 + [(-40.0, False)] * 8 + [(-30.0, True)] * 5 + [(-10.0, True)] * 9
    # Up to -40 every call is right; taking -30 in would make 5 of 25 wrong.
    assert go.choose_empty_max(pairs) == -40.0
    assert go.choose_empty_max([(-60.0, False)] * 9) is None  # too few to choose on


def test_rules_that_work_pass_and_the_first_in_order_is_the_one_to_use() -> None:
    rows = [
        *world("calibration", level_works=True, tagger_works=True, bleed_works=True),
        *world("heldout", level_works=True, tagger_works=True, bleed_works=True),
    ]
    analysis = go.analyse(rows, LISTENING)
    rules = analysis["empty"]["rules"]
    assert all(rules[rule]["passed"] for rule in go.EMPTY_RULES)
    assert analysis["empty"]["use"] == "level"
    assert rules["level"]["heldout"]["threshold"] == -46.0  # -60 dBFS under a -14 dBFS mix
    assert rules["tagger_config"]["heldout"]["threshold"] == go.TAGGER_ABSENT_MAX
    assert rules["level"]["heldout"]["precision"] == rules["level"]["heldout"]["recall"] == 1.0
    assert analysis["counts"]["heldout"] == {
        "outputs": 60,
        "tracks": 20,
        "models": ["four", "six"],
        "empty": 10,
    }


def test_when_the_level_says_nothing_the_tagger_rule_is_the_one_to_use() -> None:
    rows = [
        *world("calibration", level_works=False, tagger_works=True, bleed_works=True),
        *world("heldout", level_works=False, tagger_works=True, bleed_works=True),
    ]
    analysis = go.analyse(rows, LISTENING)
    assert analysis["empty"]["rules"]["level"]["passed"] is False
    assert analysis["empty"]["use"] == "tagger_config"


def test_when_nothing_tells_empty_from_real_no_rule_is_chosen() -> None:
    rows = [
        *world("calibration", level_works=False, tagger_works=False, bleed_works=False),
        *world("heldout", level_works=False, tagger_works=False, bleed_works=False),
    ]
    analysis = go.analyse(rows, LISTENING)
    assert analysis["empty"]["use"] is None
    assert not any(analysis["empty"]["rules"][rule]["passed"] for rule in go.EMPTY_RULES)
    assert "no rule passed" in go.render(analysis, RUN)


def test_thresholds_come_from_the_calibration_outputs_only() -> None:
    calibration = world("calibration", level_works=True, tagger_works=True, bleed_works=True)
    heldout = world("heldout", level_works=True, tagger_works=True, bleed_works=True)
    before = go.analyse([*calibration, *heldout], LISTENING)
    for item in heldout:  # make the held-out empties loud: the threshold must not follow
        if not item["reference_present"]:
            item["level_dbfs"] = -20.0
    after = go.analyse([*calibration, *heldout], LISTENING)
    assert (
        after["empty"]["rules"]["level"]["heldout"]["threshold"]
        == before["empty"]["rules"]["level"]["heldout"]["threshold"]
    )
    assert after["empty"]["rules"]["level"]["passed"] is False
    assert after["empty"]["use"] == "tagger_config"


# -- question 2: the better of two -------------------------------------------------------


def test_pairs_need_two_models_a_present_reference_and_a_real_gap() -> None:
    rows = [
        row("heldout", "a", "guitar", "one", sdr=9.0),
        row("heldout", "a", "guitar", "two", sdr=5.0),
        row("heldout", "a", "guitar", "three", sdr=7.5),  # 1.5 dB from "one": no pair with it
        row("heldout", "a", "piano", "one", present=False),
        row("heldout", "a", "piano", "two", present=False),
        row("heldout", "b", "guitar", "one", sdr=9.0),
        {**row("heldout", "b", "guitar", "two", sdr=2.0), "scored_seconds": 4},  # too short
        row("heldout", "c", "bass", "one", sdr=9.0),  # nothing to compare with
    ]
    pairs = go.output_pairs(rows)
    assert [(b["model_id"], w["model_id"]) for b, w in pairs] == [("one", "two")]
    assert pairs[0][0]["sdr_db"] > pairs[0][1]["sdr_db"]


def test_a_rule_is_right_when_it_prefers_the_better_output_and_a_tie_is_half() -> None:
    better = row("heldout", "a", "guitar", "one", sdr=9.0, bleed=0.1)
    worse = row("heldout", "a", "guitar", "two", sdr=4.0, bleed=0.3)
    same = row("heldout", "b", "guitar", "two", sdr=4.0, bleed=0.1)
    assert go.pair_accuracy([(better, worse)], "bleed") == {"pairs": 1, "accuracy": 1.0}
    assert go.pair_accuracy([(worse, better)], "bleed")["accuracy"] == 0.0
    assert go.pair_accuracy([(better, same)], "bleed")["accuracy"] == 0.5
    assert go.pair_accuracy([], "bleed") == {"pairs": 0, "accuracy": None}


def test_a_bleed_score_that_follows_quality_passes_and_one_that_does_not_fails() -> None:
    def both(works: bool) -> list[dict]:
        return [
            *world("calibration", level_works=True, tagger_works=True, bleed_works=works),
            *world("heldout", level_works=True, tagger_works=True, bleed_works=works),
            # Twenty more held-out tracks, to have more than thirty pairs.
            *[
                {**item, "track_id": item["track_id"] + "x", "id": item["id"] + "x"}
                for item in world("heldout", level_works=True, tagger_works=True, bleed_works=works)
            ],
        ]

    good = go.analyse(both(True), LISTENING)["pairs"]
    assert (good["rules"]["bleed"], good["passed"]) == ({"pairs": 40, "accuracy": 1.0}, True)
    assert good["by_stem"] == {"guitar": {"pairs": 40, "accuracy": 1.0}}
    bad = go.analyse(both(False), LISTENING)["pairs"]
    assert bad["enough"] is True
    assert bad["passed"] is False
    assert 0.2 < bad["rules"]["bleed"]["accuracy"] < 0.8


def test_too_few_pairs_give_no_verdict() -> None:
    rows = [
        *world("calibration", level_works=True, tagger_works=True, bleed_works=True),
        *world("heldout", level_works=True, tagger_works=True, bleed_works=True),
    ]
    pairs = go.analyse(rows, LISTENING)["pairs"]
    assert (pairs["rules"]["bleed"]["pairs"], pairs["enough"], pairs["passed"]) == (
        20,
        False,
        False,
    )
    assert "not enough pairs to say" in go.render(go.analyse(rows, LISTENING), RUN)


def test_rank_correlation() -> None:
    assert go.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert go.spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)
    assert go.spearman([1, 2], [1, 2]) is None
    assert go.spearman([1, 1, 1], [1, 2, 3]) is None


# -- the empty-stem rule against whole songs ---------------------------------------------


def test_the_whole_song_rules_are_what_the_report_says_they_are() -> None:
    assert go.IN_SONG_SECONDS == 5.0
    assert go.WHOLE_RULES == ("minute", "whole")
    assert go.IN_FORCE_DB == -19.81
    assert go.RULES_COMMIT.startswith("5b63f0807f6e")


def songs(set_name: str, *, elsewhere: int, whole_works: bool = True) -> tuple[list, list]:
    """Twenty-four piano outputs and their whole-song levels.

    Eight songs have a piano throughout, eight have none, and ``elsewhere`` of the other
    eight have one that plays only outside the loudest minute (the rest of them have it
    throughout): in that minute its output is leftovers, over the whole song it is not.
    """
    rng = np.random.default_rng(8 if set_name == "calibration" else 9)
    rows, levels = [], []
    for n in range(24):
        kind = (
            "there" if n < 8 else "absent" if n < 16 else "late" if n - 16 < elsewhere else "there"
        )
        one = row(
            set_name,
            f"{set_name}-{n:02d}",
            "piano",
            "six",
            present=kind == "there",
            level=-22.0 if kind == "there" else -60.0,
        )
        over_song = {"there": -8.0, "late": -14.0, "absent": -46.0}[kind]
        if not whole_works:
            over_song = float(rng.uniform(-50.0, -5.0))
        rows.append(one)
        levels.append(
            {
                "id": one["id"],
                "set": set_name,
                "whole_level_dbfs": -14.0 + over_song,
                "whole_mix_level_dbfs": -14.0,
                "reference_sounding_seconds": 0.0 if kind == "absent" else 40.0,
            }
        )
    return rows, levels


def both_sets(**options) -> tuple[list, list]:
    first, second = songs("calibration", **options), songs("heldout", **options)
    return first[0] + second[0], first[1] + second[1]


def test_truth_is_the_whole_song_and_levels_are_against_the_mix() -> None:
    rows, levels = songs("heldout", elsewhere=1)
    joined = {**rows[16], **levels[16]}  # plays only outside the loudest minute
    assert joined["reference_present"] is False
    assert go.in_song(joined) is True
    assert go.in_song({**joined, "reference_sounding_seconds": 4.9}) is False
    assert go.whole_level(joined) == -14.0
    assert go.relative_level(joined) == -46.0


def test_too_few_calls_decide_nothing_in_the_rules_favour() -> None:
    # Eight empty songs per set is under the ten the bar asks for. A rule that is never
    # wrong still does not pass on that, and only a pass keeps it.
    whole = go.analyse_whole(*both_sets(elsewhere=0))
    minute = whole["rules"]["minute"]
    assert (minute["heldout"]["threshold"], minute["heldout"]["called_empty"]) == (-19.81, 8)
    assert minute["heldout"]["right"] == 8
    assert (minute["enough"], minute["passed"]) == (False, False)
    assert (whole["absent_in_minute"], whole["not_in_song"], whole["elsewhere"]) == (8, 8, 0)
    assert whole["decision"] == "off"


def many(**options) -> tuple[list, list]:
    """Three copies of each set under other track names, to have enough of every kind."""
    rows, levels = both_sets(**options)
    out_rows, out_levels = [], []
    for copy in "abc":
        for one, level in zip(rows, levels, strict=True):
            out_rows.append({**one, "id": one["id"] + copy, "track_id": one["track_id"] + copy})
            out_levels.append({**level, "id": level["id"] + copy})
    return out_rows, out_levels


def test_the_three_outcomes() -> None:
    kept = go.analyse_whole(*many(elsewhere=0))
    assert kept["decision"] == "keep"
    assert kept["rules"]["minute"]["passed"] is True

    # A quarter of the songs have a piano that the loudest minute does not show.
    replaced = go.analyse_whole(*many(elsewhere=8))
    minute, whole = replaced["rules"]["minute"], replaced["rules"]["whole"]
    assert (replaced["elsewhere"], replaced["elsewhere_called_empty"]) == (24, 24)
    assert minute["heldout"]["precision"] == 0.5  # half of what it calls empty has a piano
    assert minute["passed"] is False
    assert whole["heldout"]["threshold"] == -46.0
    assert (whole["heldout"]["precision"], whole["heldout"]["recall"]) == (1.0, 1.0)
    assert replaced["decision"] == "replace"

    off = go.analyse_whole(*many(elsewhere=8, whole_works=False))
    assert off["rules"]["whole"]["passed"] is False
    assert off["decision"] == "off"


def test_outputs_without_a_whole_song_level_are_counted_and_left_out() -> None:
    rows, levels = many(elsewhere=0)
    whole = go.analyse_whole(rows, levels[:-5])
    assert whole["without_levels"] == 5
    assert whole["outputs"] == {"calibration": 72, "heldout": 67}


def test_the_report_gains_the_whole_song_section_only_when_levels_exist() -> None:
    rows, levels = many(elsewhere=8)
    analysis = go.analyse(rows, LISTENING)
    assert "against the whole song" not in go.render(analysis, RUN)
    analysis["whole"] = go.analyse_whole(rows, levels)
    text = go.render(analysis, {**RUN, "whole_rules_commit": "a" * 40})
    assert "## The empty-stem rule against the whole song" in text
    assert "at commit `aaaaaaaaaaaa`" in text
    assert "Of the 72 held-out outputs, 48 have a reference that is absent" in text
    assert "24 have an instrument that is not in the song at all" in text
    assert "the rule in force calls 24 of them empty" in text
    assert "| -19.8 dB | " in text
    assert "| -46.0 dB | " in text
    assert "**The rule is replaced.**" in text
    assert text.index("against the whole song") < text.index("## Limits")
    json.dumps(analysis)


# -- retrying an empty stem --------------------------------------------------------------

CHAINS = {"guitar": ["six", "solo"], "piano": ["six", "four"], "bass": ["four", "six"]}


def attempt(track: str, stem: str, model: str, *, level: float, sounding: float, sdr) -> tuple:
    """One kept output and its whole-song level: ``level`` dB against the mix."""
    one = row("heldout", track, stem, model, present=sounding > 0)
    one["track_sdr_db"] = sdr
    return one, {
        "id": one["id"],
        "set": "heldout",
        "whole_level_dbfs": -14.0 + level,
        "whole_mix_level_dbfs": -14.0,
        "reference_sounding_seconds": sounding,
    }


def retry_world() -> tuple[list, list]:
    made = [
        # No guitar in the song; the default is silent; the fallback finds "something".
        attempt("a", "guitar", "six", level=-80.0, sounding=0.0, sdr=None),
        attempt("a", "guitar", "solo", level=-12.0, sounding=0.0, sdr=None),
        # No guitar; both models say so.
        attempt("b", "guitar", "six", level=-80.0, sounding=0.0, sdr=None),
        attempt("b", "guitar", "solo", level=-40.0, sounding=0.0, sdr=None),
        # A guitar the default missed and the fallback found.
        attempt("c", "guitar", "six", level=-30.0, sounding=90.0, sdr=-0.2),
        attempt("c", "guitar", "solo", level=-9.0, sounding=90.0, sdr=4.0),
        # A guitar the default missed; the fallback is louder and no better.
        attempt("d", "guitar", "six", level=-30.0, sounding=90.0, sdr=0.3),
        attempt("d", "guitar", "solo", level=-9.0, sounding=90.0, sdr=-1.0),
        # A guitar the default found: nothing is retried.
        attempt("e", "guitar", "six", level=-8.0, sounding=90.0, sdr=7.0),
        attempt("e", "guitar", "solo", level=-9.0, sounding=90.0, sdr=4.0),
        # Exactly on the threshold counts as empty; the fallback rescues it.
        attempt("f", "piano", "six", level=-21.68, sounding=30.0, sdr=None),
        attempt("f", "piano", "four", level=-10.0, sounding=30.0, sdr=1.5),
        # The fallback's output was not kept for this track: not a pair.
        attempt("g", "piano", "six", level=-80.0, sounding=0.0, sdr=None),
    ]
    return [m[0] for m in made], [m[1] for m in made]


def test_the_retry_rules_are_what_the_report_says_they_are() -> None:
    assert go.RETRY_EMPTY_DB == -21.68


def test_each_empty_first_attempt_is_a_rescue_a_wrong_delivery_or_neither() -> None:
    rows, levels = retry_world()
    retry = go.analyse_retry(rows, levels, CHAINS)
    assert retry["by_stem"]["guitar"] == {
        "default": "six",
        "fallback": "solo",
        "pairs": 5,
        "empty": 4,
        "stays_empty": 1,
        "rescue": 1,
        "wrong": 1,
        "no_better": 1,
    }
    assert retry["by_stem"]["piano"]["pairs"] == 1
    assert retry["by_stem"]["piano"]["rescue"] == 1
    assert "bass" not in retry["by_stem"]  # no outputs kept for its chain
    assert retry["total"] == {
        "pairs": 6, "empty": 5, "stays_empty": 1, "rescue": 2, "wrong": 1, "no_better": 1
    }  # fmt: skip
    assert [(c["track_id"], c["outcome"]) for c in retry["cases"]] == [
        ("a", "wrong"),
        ("c", "rescue"),
        ("d", "no_better"),
        ("f", "rescue"),
    ]
    assert retry["decision"] == "retry"  # two rescues against one wrong delivery


def test_retrying_stops_unless_rescues_outnumber_wrong_deliveries() -> None:
    rows, levels = retry_world()
    keep = [i for i, one in enumerate(rows) if one["track_id"] != "f"]
    tied = go.analyse_retry([rows[i] for i in keep], [levels[i] for i in keep], CHAINS)
    assert (tied["total"]["rescue"], tied["total"]["wrong"], tied["decision"]) == (1, 1, "stop")
    nothing = go.analyse_retry(rows[8:10], levels[8:10], CHAINS)
    assert (nothing["total"]["empty"], nothing["decision"]) == (0, "stop")
    # Calibration outputs are not counted: the question is asked of the held-out ones.
    moved = [{**one, "set": "calibration"} for one in rows]
    assert go.analyse_retry(moved, levels, CHAINS)["total"]["pairs"] == 0


def test_the_report_states_the_counts_and_what_the_gate_does() -> None:
    rows, levels = retry_world()
    analysis = go.analyse(rows, LISTENING)
    assert "tried again" not in go.render(analysis, RUN)
    analysis["retry"] = go.analyse_retry(rows, levels, CHAINS)
    text = go.render(analysis, {**RUN, "retry_rules_commit": "d" * 40})
    assert "## Should a stem found empty be tried again?" in text
    assert "at commit `dddddddddddd`" in text
    assert "| guitar | `six`, then `solo` | 5 | 4 | 1 | 1 | 1 | 1 |" in text
    assert "| all | | 6 | 5 | 1 | 2 | 1 | 1 |" in text
    assert "**The gate keeps retrying.** 2 rescues against 1 wrong deliveries." in text
    keep = [i for i, one in enumerate(rows) if one["track_id"] != "f"]
    analysis["retry"] = go.analyse_retry([rows[i] for i in keep], [levels[i] for i in keep], CHAINS)
    assert "**The gate stops retrying empty stems.** 1 rescues against 1" in go.render(
        analysis, RUN
    )
    json.dumps(analysis)


# -- the report --------------------------------------------------------------------------

RUN = {
    "run_id": "gate-outputs-test",
    "finished": "2026-11-24T12:00:00+00:00",
    "tagger": "scripted",
    "device": "cpu",
    "git": {"commit": "0123456789abcdef", "dirty": False},
    "rules_commit": "fedcba9876543210",
}


def test_the_report_states_each_rule_its_bar_and_the_verdicts() -> None:
    rows = [
        *world("calibration", level_works=True, tagger_works=True, bleed_works=False),
        *world("heldout", level_works=True, tagger_works=True, bleed_works=False),
        *[
            {**item, "track_id": item["track_id"] + "x", "id": item["id"] + "x"}
            for item in world("heldout", level_works=True, tagger_works=True, bleed_works=False)
        ],
    ]
    analysis = go.analyse(rows, LISTENING)
    text = go.render(analysis, RUN)
    assert "commit `0123456789ab`" in text
    assert "at commit `fedcba987654`" in text
    assert "| Calibration (`moisesdb_tuning`) | 20 | 2 | 60 | 10 |" in text
    assert "| Held out (`moisesdb_subset`) | 40 | 2 | 120 | 20 |" in text
    assert "-46.0 dB" in text
    assert "0.0207" in text
    assert "right at least 90.0%, found at least 50.0%" in text
    assert "at least 98.0% of its length" in text
    assert "**Empty stems: the rule to use is `level`**" in text
    assert "**The better of two: failed.**" in text
    assert "(80.0% needed)" in text
    by_stem = {entry["stem"]: entry for entry in analysis["by_stem"]}
    assert set(by_stem) == {"guitar", "piano"}
    assert (by_stem["piano"]["outputs"], by_stem["piano"]["empty"]) == (40, 20)
    assert by_stem["piano"]["scored"] == by_stem["piano"]["called_present"] == 20
    json.dumps(analysis)  # the analysis is saved next to the scores


# -- collecting, with a stand-in tagger and a made-up benchmark folder -------------------


def fake_bench(bench_dir, dataset, models: dict[str, list[str]], *, skip: tuple = ()) -> None:
    """Kept outputs and result files as a benchmark run leaves them."""
    for model, stems in models.items():
        for number, track_id in enumerate(dataset.track_ids()):
            if (model, track_id) in skip:
                continue
            track = dataset.load(track_id)
            key = f"{number:064x}"
            folder = bench_dir / "estimates" / dataset.name / model / key
            folder.mkdir(parents=True)
            seconds = int(track.inputs["mix"].shape[-1] / track.sample_rate)
            outputs = {}
            for stem in stems:
                target = "other_4stem" if (model, stem) == ("four", "other") else stem
                np.save(folder / f"{stem}.npy", 0.5 * track.references[target])
                outputs[stem] = {
                    "target": target,
                    "derived": False,
                    "bss": {
                        "sdr": 6.0,
                        "window_seconds": 1.0,
                        "frames": {
                            "sdr": [float(i) for i in range(seconds)],
                            "sir": [None, *[20.0] * (seconds - 1)],
                        },
                    },
                }
            result = {"kind": "model", "input": "mix", "track_id": track_id, "outputs": outputs}
            path = bench_dir / "results" / dataset.name / model / f"{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))


def test_collect_scores_each_kept_output_once_and_resumes(tmp_path) -> None:
    pytest.importorskip("torch")
    from stems_bench.datasets import get_dataset
    from stems_common.settings import load_defaults
    from stems_worker.listening.audioset import default_stem_map

    stem_map = default_stem_map()
    classes = len(stem_map.labels.names)

    class Flat:
        name, window_seconds = "flat", 10.0

        def __init__(self) -> None:
            self.calls = 0

        def class_probabilities(self, audio, sample_rate):
            self.calls += 1
            windows = max(1, int(np.ceil(audio.shape[-1] / sample_rate / self.window_seconds)))
            return np.full((windows, classes), 0.25)

    dataset = get_dataset("synthetic")
    tracks = dataset.track_ids()
    bench = tmp_path / "bench"
    fake_bench(
        bench,
        dataset,
        {"six": ["guitar", "piano"], "four": ["bass", "other"], "partial": ["vocals"]},
        skip=(("partial", tracks[-1]),),
    )
    # A job that is still running has outputs and no result yet: it is not there.
    (bench / "estimates" / dataset.name / "six" / ("f" * 64)).mkdir()
    listed = list(go.kept_outputs(bench, dataset.name))
    # The four-stem "other" is scored against a different target and is left out.
    assert {(item["model_id"], item["stem"]) for item in listed} == {
        ("six", "guitar"),
        ("six", "piano"),
        ("four", "bass"),
        ("partial", "vocals"),
    }
    assert go.complete_models(listed, tracks) == {"four": ["bass"], "six": ["guitar", "piano"]}

    tagger, out = Flat(), tmp_path / "run" / "scores.jsonl"
    settings = load_defaults().listening
    args = ("heldout", dataset, tagger, settings, stem_map, out)
    added = go.collect(*args, bench_dir=bench)
    rows = go.read_rows(out)
    assert added == len(rows) == tagger.calls == 3 * len(tracks)
    assert {row["model_id"] for row in rows} == {"four", "six"}  # "partial" misses a track
    assert len({row["id"] for row in rows}) == len(rows)
    one = rows[0]
    assert set(one["scores"]) == set(PRODUCT_STEMS)
    assert one["reference_present"] is True
    assert one["set"] == "heldout"
    # Half the reference's level, under a mix of six stems.
    assert one["level_dbfs"] < one["mix_level_dbfs"]
    # The whole 12-second track is the excerpt: the median of windows 0..11, and of the
    # eleven SIR windows that were scored.
    assert (one["sdr_db"], one["sir_db"], one["scored_seconds"]) == (5.5, 20.0, 12)
    assert one["track_sdr_db"] == 6.0
    plan = json.loads((tmp_path / "run" / "heldout-plan.json").read_text())
    assert plan["models"] == {"four": ["bass"], "six": ["guitar", "piano"]}
    assert plan["excerpts_from"] == "computed"

    # Run again: nothing is scored twice.
    assert go.collect(*args, bench_dir=bench) == 0
    # Lose the last two rows (an interrupted run): only those are redone.
    out.write_text("".join(line + "\n" for line in out.read_text().splitlines()[:-2]))
    assert go.collect(*args, bench_dir=bench) == 2
    assert {row["id"] for row in go.read_rows(out)} == {row["id"] for row in rows}
    # The benchmark finishes the missing track later: the models that count do not change.
    fake_bench(bench, dataset, {"late": ["drums"]})
    assert go.collect(*args, bench_dir=bench) == 0

    # Whole-song levels of the same outputs: no tagger, a row each, and resumable.
    calls = tagger.calls
    levels_path = tmp_path / "run" / "levels.jsonl"
    rows = go.read_rows(out)
    assert go.collect_levels("heldout", dataset, rows, levels_path, bench_dir=bench) == len(rows)
    assert go.collect_levels("calibration", dataset, rows, levels_path, bench_dir=bench) == 0
    levels = go.read_rows(levels_path)
    assert {level["id"] for level in levels} == {row["id"] for row in rows}
    assert tagger.calls == calls
    one_level = levels[0]
    assert one_level["track_seconds"] == 12.0
    assert one_level["whole_level_dbfs"] < one_level["whole_mix_level_dbfs"]
    assert 0.0 <= one_level["reference_sounding_seconds"] <= 12.0
    # The synthetic track is shorter than a minute, so the minute is the whole song.
    assert one_level["whole_level_dbfs"] == rows[0]["level_dbfs"]
    assert go.collect_levels("heldout", dataset, rows, levels_path, bench_dir=bench) == 0
    whole = go.analyse_whole(rows, levels)
    assert whole["outputs"] == {"calibration": 0, "heldout": len(rows)}
    # A stem of the synthetic set that sounds for under five seconds is not "in the song".
    brief = sum(1 for level in levels if level["reference_sounding_seconds"] < go.IN_SONG_SECONDS)
    assert whole["not_in_song"] == brief


def test_excerpts_of_an_earlier_run_are_reused_only_for_the_same_tracks(tmp_path) -> None:
    from stems_bench.datasets import get_dataset

    dataset = get_dataset("synthetic")
    tracks = dataset.track_ids()
    earlier = tmp_path / "earlier.json"
    earlier.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "windows": {t: [100, 2000] for t in tracks},
                "present": {t: dict.fromkeys(PRODUCT_STEMS, False) for t in tracks},
            }
        )
    )
    plan = go.make_plan("heldout", dataset, tmp_path / "bench", tmp_path / "a.json", reuse=earlier)
    assert plan["excerpts_from"] == str(earlier)
    assert plan["windows"][tracks[0]] == [100, 2000]
    earlier.write_text(
        json.dumps({"dataset": "synthetic", "windows": {"x": [0, 1]}, "present": {}})
    )
    plan = go.make_plan("heldout", dataset, tmp_path / "bench", tmp_path / "b.json", reuse=earlier)
    assert plan["excerpts_from"] == "computed"
    assert plan["windows"][tracks[0]][0] == 0


def test_an_output_of_the_wrong_length_stops_the_run(tmp_path) -> None:
    pytest.importorskip("torch")
    from stems_bench.datasets import get_dataset
    from stems_common.settings import load_defaults
    from stems_worker.listening.audioset import default_stem_map

    dataset = get_dataset("synthetic")
    bench = tmp_path / "bench"
    fake_bench(bench, dataset, {"six": ["piano"]})
    first = sorted((bench / "estimates" / dataset.name / "six").iterdir())[0]
    np.save(first / "piano.npy", np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(go.OutputsError, match="not an output for this track"):
        go.collect(
            "heldout",
            dataset,
            object(),
            load_defaults().listening,
            default_stem_map(),
            tmp_path / "scores.jsonl",
            bench_dir=bench,
        )


def test_report_refuses_a_run_that_has_not_finished(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(go.cal, "CALIBRATION_DIR", tmp_path)
    assert go.main(["report", "--run-id", "nothing"]) == 2
    assert "has not finished" in capsys.readouterr().err
