"""Calibrate the listening layer's thresholds on stems whose contents are known (P7.1).

    python -m stems_bench.calibrate collect            # score the clips (resumable)
    python -m stems_bench.calibrate report --out docs/gate-calibration.md

The listening layer turns tagger scores into three things the gate uses: whether a stem
contains its own instrument (``listening.presence``), and how much of another instrument
has bled into it (``listening.bleed``). Until now those thresholds were guesses and
``listening.calibrated`` was false, so the gate did not act on them.

**The data.** Ground-truth stems from MoisesDB, cut to the loudest minute of each track:

- *clean* stems: one instrument, nothing else;
- *bled* stems: the same stem with another stem of the same track mixed in at a known
  level (signal-to-interference ratio), which is what a poor separation sounds like;
- the *mix*, where which instruments are present is known from the stems.

Thresholds are chosen on the tuning tracks (``moisesdb_tuning``, which no benchmark test
uses) and then checked on tracks they were not chosen on (``moisesdb_subset``).

**Everything that decides the outcome is fixed below, before any score is seen:** which
bleed level each mixing ratio counts as, the rule that picks each threshold, and the bar
the held-out check must clear for ``listening.calibrated`` to be set to true. The commit
that added this file is recorded in the report.

Scores come from the product's own code path (``stems_worker.listening.features.hear``),
so a threshold chosen here means the same thing in a job. One difference remains and is
stated in the report: a job scores a whole stem, this scores a one-minute excerpt.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.clef import clips as c
from stems_bench.clef.scoring import auc
from stems_bench.run import git_state, write_json
from stems_common.paths import RUNS_DIR
from stems_common.stems import PRODUCT_STEMS

CALIBRATION_DIR = RUNS_DIR / "calibration"
DEFAULT_RUN = "gate-v1"

# -- fixed before any score was seen -------------------------------------------------------

SETS: dict[str, dict[str, Any]] = {
    # Thresholds are chosen on this list ...
    "calibration": {"dataset": "moisesdb_tuning", "seed": 7101},
    # ... and checked on this one.
    "heldout": {"dataset": "moisesdb_subset", "seed": 7102},
}
EXCERPT_SECONDS = 60.0
# Clean stems (and their bled versions) taken from each track, chosen by the seed.
TARGETS_PER_TRACK = 3
# What each mixing ratio counts as. A convention, not a measurement: 30 dB below the
# stem is faint, 20 and 12 dB are plainly audible, 6 dB is a second instrument.
LEVEL_OF_SIR: dict[float | None, str] = {
    None: "none",
    30.0: "slight",
    20.0: "moderate",
    12.0: "moderate",
    6.0: "severe",
}
LEVELS = ("none", "slight", "moderate", "severe")
# The gate delivers a stem with no or slight bleed and retries one with more.
ACCEPTABLE = frozenset({"none", "slight"})

# How each threshold is chosen, on the calibration set:
#   present_min   the lowest score at which "present" is right this often
#   absent_max    the highest score at which "absent" is right this often
#   moderate_min  the score that best separates acceptable from not (balanced accuracy)
#   slight_min    the same for clean against any bleed, kept below moderate_min
#   severe_min    the same for severe against the rest, kept above moderate_min
PRESENCE_RELIABILITY = 0.95
# Fewer decisions than this on one side of a threshold prove nothing.
MIN_SUPPORT = 10

# What the held-out check must show for listening.calibrated to become true. All of it.
CRITERIA: dict[str, float] = {
    # Of the presence answers the layer commits to (present or absent), this many are right.
    "presence_decided_accuracy_min": 0.90,
    # And it commits to an answer at least this often (the rest are "cannot tell").
    "presence_coverage_min": 0.50,
    # Bleed score ranks unacceptable stems above acceptable ones this well.
    "bleed_auc_min": 0.80,
    # At the chosen threshold, mean of the two hit rates.
    "bleed_balanced_accuracy_min": 0.75,
    # Clean stems wrongly called moderate or worse.
    "clean_flagged_max": 0.10,
}


# The commit at which everything above was fixed, before any clip was scored.
STEM_RULES_COMMIT = "7d58a494a794e18ae92bc136c86b29e46827c103"

# -- presence in a full mix (added in Phase 8) ---------------------------------------------
# The stem calibration showed that thresholds chosen on stems do not work on a mix. A mix
# gets thresholds of its own, one pair per instrument, chosen by the same two rules
# (choose_present_min, choose_absent_max) on the calibration mixes and checked on the
# held-out mixes. This block was written after the stem results were known and before
# any threshold for a mix was computed; the scores are those of the same run.
#
# An instrument is judged only if at least this many mixes have it and this many lack it,
# in the calibration set (to choose on) and in the held-out set (to check on).
MIX_MIN_EACH = 10
# What the held-out mixes must show before track_features may name the instrument.
MIX_CRITERIA: dict[str, float] = {
    # The score ranks a mix with the instrument above a mix without it this often.
    "auc_min": 0.80,
    # Of the answers it commits to (present or absent), this many are right.
    "decided_accuracy_min": 0.90,
    # And it commits to an answer at least this often.
    "coverage_min": 0.50,
}
# The commit that added the block above. Filled in by the commit after it.
MIX_RULES_COMMIT: str | None = "7f6c21e1a87d1b5704a06ddd98b254407bba9158"


class CalibrationError(ValueError):
    """The calibration cannot be run or reported as asked."""


# -- collecting scores ---------------------------------------------------------------------


def plan_items(
    present: Mapping[str, Mapping[str, bool]], seed: int, per_track: int = TARGETS_PER_TRACK
) -> list[dict[str, str]]:
    """Which clean stems to take from each track and what to bleed into each.

    For every track, up to ``per_track`` of its present stems are drawn with ``seed``;
    each gets one contaminant drawn from the track's other present stems. A track with
    fewer than two present stems gives nothing: there is nothing to bleed.
    """
    rng = np.random.default_rng(seed)
    items: list[dict[str, str]] = []
    for track_id in sorted(present):
        there = [stem for stem in PRODUCT_STEMS if present[track_id][stem]]
        if len(there) < 2:
            continue
        targets = sorted(rng.choice(there, size=min(per_track, len(there)), replace=False))
        for target in targets:
            others = [stem for stem in there if stem != target]
            items.append(
                {
                    "track_id": track_id,
                    "target": str(target),
                    "contaminant": str(rng.choice(others)),
                }
            )
    return items


def _row_id(kind: str, track_id: str, target: str | None, sir: float | None) -> str:
    level = "clean" if sir is None else f"sir{sir:02.0f}"
    return f"{kind}:{track_id}" if kind == "mix" else f"{kind}:{track_id}:{target}:{level}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect(
    set_name: str,
    dataset: Any,
    tagger: Any,
    settings: Any,
    stem_map: Any,
    out_path: Path,
    *,
    seed: int,
    progress: Any = None,
) -> int:
    """Score every clip of one set, appending a row per clip. Returns how many were new.

    A clip whose row is already in ``out_path`` is skipped, so an interrupted run is
    continued by running it again.
    """
    from stems_worker.listening.features import hear

    done = {row["id"] for row in read_rows(out_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0

    def score(row_id: str, audio: np.ndarray, rate: int, facts: dict[str, Any]) -> None:
        nonlocal added
        if row_id in done:
            return
        heard = hear(np.asarray(audio, dtype=np.float32), rate, tagger, settings, stem_map)
        row = {
            "id": row_id,
            "set": set_name,
            **facts,
            "windows": int(heard.probabilities.shape[0]),
            "audible_windows": int(heard.audible.sum()),
            "scores": {stem: round(value, 6) for stem, value in heard.stems(stem_map).items()},
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        done.add(row_id)
        added += 1

    track_ids = dataset.track_ids()
    windows: dict[str, c.Window] = {}
    present: dict[str, dict[str, bool]] = {}
    plan_path = out_path.with_name(f"{set_name}-plan.json")
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        # First pass: where each track's loudest minute is and what plays in it.
        for number, track_id in enumerate(track_ids, start=1):
            track = dataset.load(track_id)
            window = c.loudest_window(track.inputs["mix"], track.sample_rate, EXCERPT_SECONDS)
            windows[track_id] = window
            present[track_id] = c.presence(track.references, window, track.sample_rate)
            if progress:
                progress(f"{set_name}: planned {number} of {len(track_ids)} tracks")
        plan = {
            "set": set_name,
            "dataset": dataset.name,
            "seed": seed,
            "windows": {t: [w.start, w.length] for t, w in windows.items()},
            "present": present,
            "items": plan_items(present, seed),
        }
        write_json(plan_path, plan)
    by_track: dict[str, list[dict[str, str]]] = {}
    for item in plan["items"]:
        by_track.setdefault(item["track_id"], []).append(item)

    for number, track_id in enumerate(track_ids, start=1):
        start, length = plan["windows"][track_id]
        window = c.Window(start, length)
        wanted = [_row_id("mix", track_id, None, None)]
        for item in by_track.get(track_id, []):
            wanted += [_row_id("stem", track_id, item["target"], sir) for sir in LEVEL_OF_SIR]
        if all(row_id in done for row_id in wanted):
            continue
        track = dataset.load(track_id)
        rate = track.sample_rate
        score(
            _row_id("mix", track_id, None, None),
            window.cut(track.inputs["mix"]),
            rate,
            {"kind": "mix", "track_id": track_id, "present": plan["present"][track_id]},
        )
        for item in by_track.get(track_id, []):
            target = window.cut(track.references[item["target"]])
            contaminant = window.cut(track.references[item["contaminant"]])
            for sir, level in LEVEL_OF_SIR.items():
                audio = target if sir is None else c.mix_at_sir(target, contaminant, sir)[0]
                score(
                    _row_id("stem", track_id, item["target"], sir),
                    audio,
                    rate,
                    {
                        "kind": "stem",
                        "track_id": track_id,
                        "target": item["target"],
                        "contaminant": item["contaminant"],
                        "sir_db": sir,
                        "level": level,
                    },
                )
        if progress:
            progress(f"{set_name}: scored {number} of {len(track_ids)} tracks")
    return added


# -- analysis (pure) -----------------------------------------------------------------------


def presence_decisions(rows: Iterable[Mapping[str, Any]], context: str) -> list[tuple[float, bool]]:
    """(score, is the instrument really there) pairs.

    ``stem``: on a clean stem, its own instrument is there and the other five are not; on
    a bled stem the target is there (the contaminant is left out: it is there, faintly).
    ``mix``: every instrument of every mix, with the truth from the stems.
    """
    pairs: list[tuple[float, bool]] = []
    for row in rows:
        if row["kind"] != context:
            continue
        if context == "mix":
            pairs += [(row["scores"][s], bool(row["present"][s])) for s in PRODUCT_STEMS]
            continue
        pairs.append((row["scores"][row["target"]], True))
        if row["level"] == "none":
            pairs += [(row["scores"][s], False) for s in PRODUCT_STEMS if s != row["target"]]
    return pairs


def bleed_scores(rows: Iterable[Mapping[str, Any]]) -> list[tuple[float, str]]:
    """(bleed score, true level) per stem clip: the strongest other source, as in a job."""
    pairs = []
    for row in rows:
        if row["kind"] == "stem":
            others = [v for s, v in row["scores"].items() if s != row["target"]]
            pairs.append((max(others), row["level"]))
    return pairs


def _candidates(scores: Sequence[float]) -> list[float]:
    return sorted({round(float(s), 6) for s in scores})


def choose_present_min(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """The lowest score from which "present" is right ``PRESENCE_RELIABILITY`` of the time."""
    for threshold in _candidates([s for s, _ in pairs]):
        said = [truth for score, truth in pairs if score >= threshold]
        if len(said) >= MIN_SUPPORT and sum(said) / len(said) >= PRESENCE_RELIABILITY:
            return threshold
    return None


def choose_absent_max(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """The highest score up to which "absent" is right ``PRESENCE_RELIABILITY`` of the time."""
    for threshold in reversed(_candidates([s for s, _ in pairs])):
        said = [not truth for score, truth in pairs if score <= threshold]
        if len(said) >= MIN_SUPPORT and sum(said) / len(said) >= PRESENCE_RELIABILITY:
            return threshold
    return None


def presence_outcome(
    pairs: Sequence[tuple[float, bool]], present_min: float, absent_max: float
) -> dict[str, Any]:
    """How the three-way answer (present, absent, cannot tell) does on ``pairs``."""
    decided = [(s >= present_min, t) for s, t in pairs if s >= present_min or s <= absent_max]
    right = sum(1 for said, truth in decided if said == truth)
    return {
        "decisions": len(pairs),
        "decided": len(decided),
        "coverage": len(decided) / len(pairs) if pairs else None,
        "decided_accuracy": right / len(decided) if decided else None,
        "said_present": sum(1 for said, _ in decided if said),
        "said_absent": sum(1 for said, _ in decided if not said),
        "truly_present": sum(1 for _, truth in pairs if truth),
    }


def balanced_accuracy(
    scores: Sequence[tuple[float, bool]], threshold: float
) -> tuple[float, float, float] | None:
    """(balanced accuracy, hit rate on positives, hit rate on negatives) at ``threshold``."""
    positives = [s for s, bad in scores if bad]
    negatives = [s for s, bad in scores if not bad]
    if not positives or not negatives:
        return None
    caught = sum(1 for s in positives if s >= threshold) / len(positives)
    passed = sum(1 for s in negatives if s < threshold) / len(negatives)
    return (caught + passed) / 2, caught, passed


def best_split(
    scores: Sequence[tuple[float, bool]], *, above: float | None = None, below: float | None = None
) -> float | None:
    """The threshold with the best balanced accuracy, optionally kept inside (above, below).

    Of equally good thresholds the lowest wins, so that a tie flags more, not less.
    """
    best: tuple[float, float] | None = None
    for threshold in _candidates([s for s, _ in scores]):
        if (above is not None and threshold <= above) or (below is not None and threshold >= below):
            continue
        outcome = balanced_accuracy(scores, threshold)
        if outcome is not None and (best is None or outcome[0] > best[0] + 1e-12):
            best = (outcome[0], threshold)
    return None if best is None else best[1]


def choose_thresholds(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Every threshold, by the rules at the top of this file, from calibration rows."""
    stem_pairs = presence_decisions(rows, "stem")
    bleed = bleed_scores(rows)
    moderate = best_split([(s, level not in ACCEPTABLE) for s, level in bleed])
    slight = severe = None
    if moderate is not None:
        slight = best_split([(s, level != "none") for s, level in bleed], below=moderate)
        severe = best_split([(s, level == "severe") for s, level in bleed], above=moderate)
    return {
        "present_min": choose_present_min(stem_pairs),
        "absent_max": choose_absent_max(stem_pairs),
        "slight_min": slight,
        "moderate_min": moderate,
        "severe_min": severe,
    }


def usable(thresholds: Mapping[str, float | None]) -> bool:
    """Whether the chosen values form a valid configuration."""
    values = [
        thresholds[k]
        for k in ("absent_max", "present_min", "slight_min", "moderate_min", "severe_min")
    ]
    if any(v is None for v in values):
        return False
    absent, present, slight, moderate, severe = values  # type: ignore[misc]
    return absent < present and slight < moderate < severe  # type: ignore[operator]


def level_of(score: float, thresholds: Mapping[str, float]) -> str:
    if score >= thresholds["severe_min"]:
        return "severe"
    if score >= thresholds["moderate_min"]:
        return "moderate"
    return "slight" if score >= thresholds["slight_min"] else "none"


def evaluate(rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """How ``thresholds`` do on ``rows``: presence on stems and on mixes, and bleed."""
    bleed = bleed_scores(rows)
    binary = [(s, level not in ACCEPTABLE) for s, level in bleed]
    at = balanced_accuracy(binary, thresholds["moderate_min"])
    clean = [s for s, level in bleed if level == "none"]
    by_level = {
        level: {
            "clips": sum(1 for _, lv in bleed if lv == level),
            "median_score": float(np.median([s for s, lv in bleed if lv == level]))
            if any(lv == level for _, lv in bleed)
            else None,
            "called": {
                called: sum(
                    1 for s, lv in bleed if lv == level and level_of(s, thresholds) == called
                )
                for called in LEVELS
            },
        }
        for level in LEVELS
    }
    return {
        "presence_on_stems": presence_outcome(
            presence_decisions(rows, "stem"), thresholds["present_min"], thresholds["absent_max"]
        ),
        "presence_on_mixes": presence_outcome(
            presence_decisions(rows, "mix"), thresholds["present_min"], thresholds["absent_max"]
        ),
        "bleed": {
            "clips": len(bleed),
            "auc": auc([s for s, bad in binary if not bad], [s for s, bad in binary if bad]),
            "balanced_accuracy": None if at is None else at[0],
            "unacceptable_caught": None if at is None else at[1],
            "acceptable_passed": None if at is None else at[2],
            "clean_flagged": sum(1 for s in clean if s >= thresholds["moderate_min"]) / len(clean)
            if clean
            else None,
            "exact_level": sum(1 for s, lv in bleed if level_of(s, thresholds) == lv) / len(bleed)
            if bleed
            else None,
            "by_level": by_level,
        },
    }


def mix_pairs(rows: Iterable[Mapping[str, Any]], stem: str) -> list[tuple[float, bool]]:
    """(score for ``stem``, is it really playing) for every mix."""
    return [
        (float(row["scores"][stem]), bool(row["present"][stem]))
        for row in rows
        if row["kind"] == "mix"
    ]


def analyse_mix(
    calibration: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per instrument: thresholds for a full mix from one set, and how they do on the other."""
    out: dict[str, dict[str, Any]] = {}
    for stem in PRODUCT_STEMS:
        chosen_on, checked_on = mix_pairs(calibration, stem), mix_pairs(heldout, stem)
        counts = {
            name: {
                "with": sum(1 for _, there in pairs if there),
                "without": sum(1 for _, there in pairs if not there),
            }
            for name, pairs in (("calibration", chosen_on), ("heldout", checked_on))
        }
        entry: dict[str, Any] = {
            **counts,
            "judged": min(n for part in counts.values() for n in part.values()) >= MIX_MIN_EACH,
            "present_min": None,
            "absent_max": None,
            "auc": None,
            "coverage": None,
            "decided_accuracy": None,
            "passed": False,
        }
        if entry["judged"]:
            present_min, absent_max = choose_present_min(chosen_on), choose_absent_max(chosen_on)
            entry.update(
                present_min=present_min,
                absent_max=absent_max,
                auc=auc(
                    [s for s, there in checked_on if not there],
                    [s for s, there in checked_on if there],
                ),
            )
            if present_min is not None and absent_max is not None and absent_max < present_min:
                outcome = presence_outcome(checked_on, present_min, absent_max)
                entry.update(
                    coverage=outcome["coverage"], decided_accuracy=outcome["decided_accuracy"]
                )
                entry["passed"] = bool(
                    entry["auc"] >= MIX_CRITERIA["auc_min"]
                    and outcome["decided_accuracy"] is not None
                    and outcome["decided_accuracy"] >= MIX_CRITERIA["decided_accuracy_min"]
                    and outcome["coverage"] >= MIX_CRITERIA["coverage_min"]
                )
        out[stem] = entry
    return out


def verdict(heldout: Mapping[str, Any]) -> dict[str, Any]:
    """Each criterion with its measured value, and whether all of them hold."""
    stems, bleed = heldout["presence_on_stems"], heldout["bleed"]
    measured = {
        "presence_decided_accuracy_min": stems["decided_accuracy"],
        "presence_coverage_min": stems["coverage"],
        "bleed_auc_min": bleed["auc"],
        "bleed_balanced_accuracy_min": bleed["balanced_accuracy"],
        "clean_flagged_max": bleed["clean_flagged"],
    }
    checks = {}
    for name, needed in CRITERIA.items():
        value = measured[name]
        met = value is not None and (value <= needed if name.endswith("_max") else value >= needed)
        checks[name] = {"measured": value, "needed": needed, "met": bool(met)}
    return {"checks": checks, "calibrated": all(check["met"] for check in checks.values())}


def trade_off(
    scores: Sequence[tuple[float, bool]], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Hit rates at a few thresholds: the trade-off a single number hides."""
    table = []
    for threshold in thresholds:
        outcome = balanced_accuracy(scores, threshold)
        if outcome is not None:
            table.append(
                {
                    "threshold": threshold,
                    "caught": outcome[1],
                    "passed": outcome[2],
                    "balanced_accuracy": outcome[0],
                }
            )
    return table


def analyse(
    rows: Sequence[Mapping[str, Any]], before: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Choose on the calibration rows, check on the held-out rows.

    ``before`` is the set of thresholds in use until now; given, it is scored on the same
    held-out rows, so the report can show what the calibration changed.
    """
    calibration = [row for row in rows if row["set"] == "calibration"]
    heldout = [row for row in rows if row["set"] == "heldout"]
    if not calibration or not heldout:
        raise CalibrationError("both the calibration set and the held-out set must be scored")
    chosen = choose_thresholds(calibration)
    result: dict[str, Any] = {
        "chosen": chosen,
        "usable": usable(chosen),
        "counts": {
            name: {
                "mix_clips": sum(1 for r in part if r["kind"] == "mix"),
                "stem_clips": sum(1 for r in part if r["kind"] == "stem"),
                "tracks": len({r["track_id"] for r in part}),
            }
            for name, part in (("calibration", calibration), ("heldout", heldout))
        },
    }
    result["mix"] = analyse_mix(calibration, heldout)
    if not result["usable"]:
        result["verdict"] = {"checks": {}, "calibrated": False}
        return result
    thresholds = {k: float(v) for k, v in chosen.items()}  # type: ignore[arg-type]
    result["calibration"] = evaluate(calibration, thresholds)
    result["heldout"] = evaluate(heldout, thresholds)
    if before is not None:
        result["before"] = dict(before)
        result["before_heldout"] = evaluate(heldout, before)
    result["verdict"] = verdict(result["heldout"])
    grid = sorted({0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, thresholds["moderate_min"]})
    result["bleed_trade_off"] = {
        name: trade_off([(s, lv not in ACCEPTABLE) for s, lv in bleed_scores(part)], grid)
        for name, part in (("calibration", calibration), ("heldout", heldout))
    }
    return result


# -- the report ----------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(analysis: Mapping[str, Any], run: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """The report as Markdown. Every figure comes from ``analysis`` or ``run``."""
    chosen, counts = analysis["chosen"], analysis["counts"]
    lines = [
        "# Gate calibration: the listening thresholds",
        "",
        f"Generated from `runs/calibration/{run['run_id']}/scores.jsonl` "
        f"(scored {run['finished']}, tagger `{run['tagger']}` on {run['device']}, "
        f"commit `{str(run['git']['commit'])[:12]}`"
        f"{', with uncommitted changes' if run['git'].get('dirty') else ''}). "
        "Reproduce with `python -m stems_bench.calibrate collect` and then `report`.",
        "",
        "The rules that pick each threshold, the bleed level each mixing ratio counts as, "
        "and the bar for switching the gate on were written into "
        f"`bench/stems_bench/calibrate.py` at commit `{run['rules_commit'][:12]}`, "
        "before any of these clips was scored.",
        "",
        "## What was scored",
        "",
        "| Set | Tracks | Mix excerpts | Stem excerpts (clean and bled) |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("calibration", "Calibration (`moisesdb_tuning`)"),
        ("heldout", "Held out (`moisesdb_subset`)"),
    ):
        part = counts[name]
        lines.append(f"| {label} | {part['tracks']} | {part['mix_clips']} | {part['stem_clips']} |")
    lines += [
        "",
        f"Each excerpt is the loudest {EXCERPT_SECONDS:.0f} seconds of its track. From each "
        f"track up to {TARGETS_PER_TRACK} stems were taken clean and with another stem of the "
        "same track mixed in at 30, 20, 12 and 6 dB below it, counted as slight, moderate, "
        "moderate and severe bleed.",
        "",
        "## Chosen values",
        "",
        "| Setting | Was (a guess) | Chosen on the calibration set |",
        "|---|---:|---:|",
    ]
    names = {
        "present_min": "presence.present_min",
        "absent_max": "presence.absent_max",
        "slight_min": "bleed.slight_min",
        "moderate_min": "bleed.moderate_min",
        "severe_min": "bleed.severe_min",
    }
    for key, label in names.items():
        value = chosen[key]
        shown = "none found" if value is None else f"{value:.4f}"
        lines.append(f"| `listening.{label}` | {current[key]} | {shown} |")
    if not analysis["usable"]:
        lines += [
            "",
            "**No usable set of thresholds was found**: at least one rule could not be met "
            "on the calibration set, or the values do not rise in the order the settings "
            "need. Nothing was changed and `listening.calibrated` stays false.",
            "",
        ]
        return "\n".join(lines + _mix_section(analysis, run))

    def presence_rows(label: str, part: Mapping[str, Any]) -> str:
        return (
            f"| {label} | {part['decisions']} | {_pct(part['coverage'])} | "
            f"{_pct(part['decided_accuracy'])} | {part['said_present']} | {part['said_absent']} |"
        )

    lines += [
        "",
        "## Presence: is the instrument there?",
        "",
        'The layer answers present, absent, or cannot tell. "Committed" is how often it '
        'gives one of the first two; "right" is how often that answer is correct.',
        "",
        "| Where | Decisions | Committed | Right when committed | Said present | Said absent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for set_name, label in (("calibration", "calibration"), ("heldout", "held out")):
        part = analysis[set_name]
        lines.append(presence_rows(f"Stems, {label}", part["presence_on_stems"]))
        lines.append(presence_rows(f"Mixes, {label}", part["presence_on_mixes"]))
    lines += [
        "",
        "## Bleed: how much of another instrument is in the stem?",
        "",
        "| Set | Clips | AUC | Balanced accuracy | Unacceptable caught | Acceptable passed "
        "| Clean flagged | Exact level |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for set_name, label in (("calibration", "Calibration"), ("heldout", "Held out")):
        b = analysis[set_name]["bleed"]
        lines.append(
            f"| {label} | {b['clips']} | {_num(b['auc'])} | {_pct(b['balanced_accuracy'])} | "
            f"{_pct(b['unacceptable_caught'])} | {_pct(b['acceptable_passed'])} | "
            f"{_pct(b['clean_flagged'])} | {_pct(b['exact_level'])} |"
        )
    lines += [
        "",
        "Held out, by true level: the median bleed score, and what the chosen thresholds call it.",
        "",
        "| True level | Clips | Median score | Called none | slight | moderate | severe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        entry = analysis["heldout"]["bleed"]["by_level"][level]
        called = entry["called"]
        lines.append(
            f"| {level} | {entry['clips']} | {_num(entry['median_score'], 4)} | {called['none']} | "
            f"{called['slight']} | {called['moderate']} | {called['severe']} |"
        )
    lines += [
        "",
        "The trade-off on the held-out set: move the moderate threshold and this is what is "
        "caught and what is wrongly flagged.",
        "",
        "| Threshold | Unacceptable caught | Acceptable passed | Balanced accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for row in analysis["bleed_trade_off"]["heldout"]:
        mark = " (chosen)" if row["threshold"] == chosen["moderate_min"] else ""
        lines.append(
            f"| {row['threshold']:.4f}{mark} | {_pct(row['caught'])} | {_pct(row['passed'])} | "
            f"{_pct(row['balanced_accuracy'])} |"
        )
    if "before_heldout" in analysis:
        lines += [
            "",
            "## Against the values used until now (held out)",
            "",
            "| Thresholds | Stems: committed | right | Mixes: committed | right "
            "| Bleed balanced accuracy | Clean flagged |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, part in (
            ("Until now", analysis["before_heldout"]),
            ("Chosen", analysis["heldout"]),
        ):
            stems, mixes, b = part["presence_on_stems"], part["presence_on_mixes"], part["bleed"]
            lines.append(
                f"| {label} | {_pct(stems['coverage'])} | {_pct(stems['decided_accuracy'])} | "
                f"{_pct(mixes['coverage'])} | {_pct(mixes['decided_accuracy'])} | "
                f"{_pct(b['balanced_accuracy'])} | {_pct(b['clean_flagged'])} |"
            )
        lines += [
            "",
            "On mixes neither set of thresholds works: an instrument that is playing in a "
            "full mix scores far lower than the same instrument on its own stem, so "
            '"absent" is usually the wrong answer there. The thresholds were chosen on '
            "stems, which is what the gate judges.",
        ]
    verdict_ = analysis["verdict"]
    labels = {
        "presence_decided_accuracy_min": "Presence answers that are right, at least",
        "presence_coverage_min": "Presence answered at all, at least",
        "bleed_auc_min": "Bleed AUC, at least",
        "bleed_balanced_accuracy_min": "Bleed balanced accuracy, at least",
        "clean_flagged_max": "Clean stems flagged, at most",
    }
    lines += [
        "",
        "## The bar for switching the gate on (held out, stems)",
        "",
        "| Check | Needed | Measured | Result |",
        "|---|---:|---:|---|",
    ]
    for name, check in verdict_["checks"].items():
        as_ratio = name == "bleed_auc_min"
        needed = _num(check["needed"], 2) if as_ratio else _pct(check["needed"])
        measured = _num(check["measured"]) if as_ratio else _pct(check["measured"])
        lines.append(
            f"| {labels[name]} | {needed} | {measured} | {'pass' if check['met'] else 'fail'} |"
        )
    lines += ["", "## Verdict", ""]
    if verdict_["calibrated"]:
        lines.append(
            "Every check passed. `listening.calibrated` is set to **true**: the gate may act "
            "on what the tagger hears."
        )
    else:
        failed = [name for name, check in verdict_["checks"].items() if not check["met"]]
        lines.append(
            f"{len(failed)} of {len(verdict_['checks'])} checks failed. `listening.calibrated` "
            "stays **false**: the gate keeps delivering stems it cannot vouch for with a note, "
            "and does not retry on the tagger's word."
        )
    lines.append("")
    return "\n".join(lines + _mix_section(analysis, run))


def _mix_section(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> list[str]:
    """The part of the report about hearing an instrument in a full mix."""
    mix, counts = analysis["mix"], analysis["counts"]
    added = run.get("mix_rules_commit")
    when = f"at commit `{added[:12]}`" if added else "in the commit that adds this section"
    lines = [
        "## Presence in a full mix, with thresholds of its own",
        "",
        "The thresholds above were chosen on stems and do not work on a mix. This part "
        "gives a mix its own thresholds, one pair per instrument, chosen by the same two "
        f"rules on the {counts['calibration']['mix_clips']} calibration mixes and checked on "
        f"the {counts['heldout']['mix_clips']} held-out mixes. Its rules and its bar were added "
        f"to `bench/stems_bench/calibrate.py` {when}: after the results above were known, "
        "and before any threshold for a mix was computed. The scores are those of the same "
        "run; nothing was scored again.",
        "",
        f"An instrument is judged only if at least {MIX_MIN_EACH} mixes have it and at least "
        f"{MIX_MIN_EACH} lack it, in each set. AUC is the chance that a mix with the "
        "instrument scores above a mix without it.",
        "",
        "| Instrument | Mixes without it (calibration, held out) | AUC | `present_min` | "
        "`absent_max` | Answered | Right | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    def found(value: float | None) -> str:
        return "none found" if value is None else f"{value:.4f}"

    for stem, entry in mix.items():
        without = f"{entry['calibration']['without']}, {entry['heldout']['without']}"
        if not entry["judged"]:
            lines.append(f"| {stem} | {without} | - | - | - | - | - | cannot be judged |")
            continue
        lines.append(
            f"| {stem} | {without} | {_num(entry['auc'])} | "
            f"{found(entry['present_min'])} | {found(entry['absent_max'])}"
            f" | {_pct(entry['coverage'])} | {_pct(entry['decided_accuracy'])} | "
            f"{'pass' if entry['passed'] else 'fail'} |"
        )
    passed = [stem for stem, entry in mix.items() if entry["passed"]]
    unjudged = [stem for stem, entry in mix.items() if not entry["judged"]]
    lines += [
        "",
        f"Needed on the held-out mixes: AUC at least {MIX_CRITERIA['auc_min']:.2f}, right at "
        f"least {_pct(MIX_CRITERIA['decided_accuracy_min'])} of the times it answers, and an "
        f"answer at least {_pct(MIX_CRITERIA['coverage_min'])} of the time.",
        "",
    ]
    if unjudged:
        lines += [
            f"Cannot be judged: {', '.join(unjudged)}. Too few of these mixes are without "
            "them (or with them) to choose a threshold or to check one.",
            "",
        ]
    unchosen = [
        stem
        for stem, entry in mix.items()
        if entry["judged"] and (entry["present_min"] is None or entry["absent_max"] is None)
    ]
    if unchosen:
        lines += [
            f"No thresholds could be chosen for {', '.join(unchosen)}: on the calibration "
            f"mixes no cut-off gave at least {MIN_SUPPORT} answers of which "
            f'{_pct(PRESENCE_RELIABILITY)} were right, for "present" and for "absent" '
            "both. An AUC well above 0.5 there means the score does carry something about "
            "that instrument in a mix; these mixes were too few, or the scores too mixed, to "
            "turn it into a rule.",
            "",
        ]
    if passed:
        lines.append(
            f"**Passed: {', '.join(passed)}.** `track_features` names these in a mix, with "
            "the thresholds above (`listening.mix_presence`), and no other instrument."
        )
    else:
        lines.append(
            "**No instrument passed.** `listening.mix_presence` is empty: `track_features` "
            'names no instrument in a mix and leaves the vocal presence "unknown", '
            "instead of guessing."
        )
    lines.append("")
    return lines


# -- commands ------------------------------------------------------------------------------


def _run_dir(run_id: str) -> Path:
    return CALIBRATION_DIR / run_id


def cmd_collect(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset
    from stems_common.settings import load_defaults
    from stems_worker.listening.audioset import default_stem_map
    from stems_worker.listening.beats import BeatsTagger

    run_dir = _run_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = load_defaults().listening
    stem_map = default_stem_map()
    tagger = BeatsTagger.from_registry()
    tagger.load(args.device)
    status_path = run_dir / "status.json"
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(status_path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    try:
        for name, spec in SETS.items():
            if args.only and name != args.only:
                continue
            dataset = get_dataset(spec["dataset"])
            collect(
                name,
                dataset,
                tagger,
                settings,
                stem_map,
                run_dir / "scores.jsonl",
                seed=spec["seed"],
                progress=progress,
            )
    finally:
        tagger.unload()
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "finished": finished,
            "tagger": tagger.name,
            "device": args.device,
            "top_fraction": settings.presence.top_fraction,
            "sets": SETS,
            "git": git_state(),
        },
    )
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from stems_common.settings import load_defaults

    run_dir = _run_dir(args.run_id)
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        print(f"calibrate: {run_path} is missing; run `collect` first", file=sys.stderr)
        return 2
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["rules_commit"] = STEM_RULES_COMMIT
    run["mix_rules_commit"] = MIX_RULES_COMMIT
    # The thresholds in use when this run was first reported. Kept with the run, so that
    # applying the chosen values to the config does not change what "until now" means.
    before_path = run_dir / "before.json"
    if before_path.is_file():
        current = json.loads(before_path.read_text(encoding="utf-8"))
    else:
        listening = load_defaults().listening
        current = {
            "present_min": listening.presence.present_min,
            "absent_max": listening.presence.absent_max,
            "slight_min": listening.bleed.slight_min,
            "moderate_min": listening.bleed.moderate_min,
            "severe_min": listening.bleed.severe_min,
        }
        write_json(before_path, current)
    analysis = analyse(read_rows(run_dir / "scores.jsonl"), current)
    write_json(run_dir / "analysis.json", analysis)
    text = render(analysis, run, current)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    print(f"calibrated: {analysis['verdict']['calibrated']}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.calibrate",
        description="Calibrate the listening thresholds on stems whose contents are known.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collect_ = commands.add_parser("collect", help="score the clips with the tagger (resumable)")
    collect_.add_argument("--run-id", default=DEFAULT_RUN)
    collect_.add_argument("--device", default="cpu")
    collect_.add_argument("--only", choices=tuple(SETS))
    collect_.set_defaults(run=cmd_collect)
    report = commands.add_parser("report", help="choose thresholds and write the report")
    report.add_argument("--run-id", default=DEFAULT_RUN)
    report.add_argument("--out")
    report.set_defaults(run=cmd_report)
    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except CalibrationError as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
