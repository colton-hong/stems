"""Can the listening layer judge real separator output? Measured against the benchmark (P8).

    python -m stems_bench.gate_outputs collect            # score the kept outputs (resumable)
    python -m stems_bench.gate_outputs levels             # their levels over whole songs
    python -m stems_bench.gate_outputs report --out docs/gate-on-real-output.md

P7.1 (``stems_bench.calibrate``) measured the tagger on reference stems with another stem
mixed in by hand. This asks the same of what the gate really sees: the output of real
separators. The benchmark has already scored those outputs against the references, so
how good each one is does not have to be guessed.

**The data.** A benchmark run keeps the outputs that its blends are made of, under
``runs/bench/estimates/<dataset>/<model>/<job>/<stem>.npy``, and a result file per job with
BSS Eval scores for every one-second window. For every kept output that is one of the six
product stems, this takes the loudest minute of its track (the rule P7.1 used) and records:

- the tagger's scores for that minute, through the product's own code path;
- the level of the output and of the mix in that minute;
- the measured SDR and SIR of that minute (the median of the benchmark's windows in it);
- whether the reference stem is present in that minute at all.

**The two questions, and everything that decides their answers, are fixed below before any
output is scored.** The commit that last changed this file is recorded in the report.

1. *Empty stems.* A song with no piano still gives a "piano" stem: leftovers. Can that be
   told from a real one? Three rules are tried, in the order they would be preferred.
2. *The better of two.* The gate retries with another model when a stem looks bad. That
   only helps if it can tell which of two outputs of the same stem is better. The bleed
   score is what the gate would use, so that is the rule on trial.

Thresholds are chosen on outputs for the tuning tracks (``moisesdb_tuning``) and checked on
outputs for the frozen test subset (``moisesdb_subset``). Only models with a kept output
for every track of a list are used for it, so a run that is still going cannot change
which outputs count; the models found are written to the plan file on the first run and
reused after that.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench import calibrate as cal
from stems_bench.clef import clips as c
from stems_bench.clef.scoring import auc, ranks
from stems_bench.run import BENCH_DIR, git_state, write_json
from stems_common.levels import rms_dbfs, silence_fraction
from stems_common.stems import PRODUCT_STEMS

DEFAULT_RUN = "gate-outputs-v1"

# -- fixed before any output was scored ----------------------------------------------------

SETS: dict[str, str] = {
    # Thresholds are chosen on outputs for this list ...
    "calibration": "moisesdb_tuning",
    # ... and every figure that decides a verdict is from outputs for this one.
    "heldout": "moisesdb_subset",
}
EXCERPT_SECONDS = cal.EXCERPT_SECONDS

# Question 1: empty stems. An output is "empty" when its reference is not present in the
# excerpt (the presence rule of P7.1, applied to the reference). Each rule says "empty"
# when its signal is at or below a threshold.
#   level            output RMS minus mix RMS in the excerpt, in dB; threshold chosen on
#                    the calibration outputs
#   tagger_config    the tagger's score for the stem's own instrument; threshold is the
#                    configured listening.presence.absent_max, chosen in P7.1
#   tagger_rechosen  the same score; threshold chosen on the calibration outputs
# A threshold chosen here is the highest at which "empty" was right EMPTY_RELIABILITY of
# the time on the calibration outputs, with at least MIN_SUPPORT outputs called empty.
EMPTY_RULES = ("level", "tagger_config", "tagger_rechosen")
TAGGER_ABSENT_MAX = 0.0207
EMPTY_RELIABILITY = cal.PRESENCE_RELIABILITY
MIN_SUPPORT = cal.MIN_SUPPORT

# Question 2: the better of two. Two outputs of the same stem of the same track from
# different models, both with the reference present, whose measured SDR in the excerpt
# differs by at least PAIR_GAP_DB. The rule on trial prefers the lower bleed score (the
# tagger's strongest score for any other instrument). A tie counts as half right.
PAIR_GAP_DB = 3.0
# Each output needs this many scored one-second windows in the excerpt to be compared.
PAIR_MIN_SECONDS = 10
PAIR_RULE = "bleed"

# What the held-out outputs must show.
CRITERIA: dict[str, float] = {
    # Of the outputs a rule calls empty, this many are.
    "empty_precision_min": 0.90,
    # Of the empty outputs, the rule finds this many.
    "empty_recall_min": 0.50,
    # Of the pairs, the bleed score prefers the better output this often.
    "pair_accuracy_min": 0.80,
    # Fewer pairs than this prove nothing either way.
    "pair_support_min": 30,
}


# The commit at which everything above was fixed, before any output was scored.
RULES_COMMIT = "5b63f0807f6ecab7c12c710954cfe4b38afb4ef4"

# -- whole songs (added in Phase 9) --------------------------------------------------------
# Question 1 called an output empty when its reference is absent in the loudest minute,
# and the gate's note for such a stem says the instrument "may not be in the song". This
# checks the rule against the whole song. The block was written after the results above
# were known and before any whole-song level or whole-song truth was computed.
#
# Truth: the instrument is in the song if its reference sounds (is above the silence level
# of stems_common.levels) for at least this many seconds in total, anywhere in the track.
IN_SONG_SECONDS = 5.0
# Two rules, each "empty when the output's level against the mix is at or below a
# threshold":
#   minute  the rule in force: over the mix's loudest minute, at the configured threshold
#   whole   over the whole song, with a threshold chosen on the calibration outputs
#           against the whole-song truth, by the same choosing rule as before
WHOLE_RULES = ("minute", "whole")
IN_FORCE_DB = -19.81
# What is done with the result, on the held-out outputs, against the whole-song truth and
# by the same bar (CRITERIA, MIN_SUPPORT):
#   keep     the rule in force passes: it stays, whatever the other does
#   replace  it fails and "whole" passes: "whole" takes its place
#   off      both fail: the rule is switched off
# The commit that added this block. Filled in by the commit after it.
WHOLE_RULES_COMMIT: str | None = "fcd175726d06774dfa6f9e3c25b0f9e90aa6a93a"


# -- retrying an empty stem (added in Phase 9, after P9.2) ---------------------------------
# The gate tries a stem it finds empty again with the fallback model, and delivers the
# retry if that is not empty. A real job (docs/product-vs-benchmark.md) showed the bad
# side of this: a song with no guitar, a silent first attempt, and a fallback model whose
# leftovers were delivered as the guitar. This counts both sides on the kept outputs. The
# block was written after that job and before any of these counts was made.
#
# For every held-out track and stem where the kept outputs hold both the default model's
# output and the fallback model's, and the default model's is empty by the rule in force
# (level against the mix over the whole song at or below RETRY_EMPTY_DB):
#   stays empty   the fallback's output is empty too: the retry changes nothing
#   rescue        the instrument is in the song, the fallback's output is not empty, and
#                 its measured SDR over the track is above 0 dB and above the default's
#   wrong         the instrument is not in the song and the fallback's output is not empty:
#                 leftovers would be delivered as the instrument
#   no better     in the song, not empty, but the SDR is not above 0 dB and the default's
RETRY_EMPTY_DB = -21.68
# What is done with the result: the gate keeps retrying empty stems only if rescues
# outnumber wrong deliveries. Otherwise (fewer, or as many, or none of either) it stops:
# a stem found empty is delivered at once with its note.
RETRY_RULES_COMMIT: str | None = "a537158b22f8a127f394195f1cd2c72c0b2751aa"


class OutputsError(ValueError):
    """The measurement cannot be run or reported as asked."""


# -- finding the kept outputs --------------------------------------------------------------


def kept_outputs(bench_dir: Path, dataset: str) -> Iterator[dict[str, Any]]:
    """Every kept output of a product stem that has a finished result, in a fixed order.

    Left out: outputs whose job has no result file yet, outputs scored against something
    other than the stem they are named after (a four-stem model's "other"), and models
    that were fed anything but the mix (the vocal split, which gets the vocals stem).
    """
    root = bench_dir / "estimates" / dataset
    if not root.is_dir():
        return
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for job_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            result_path = bench_dir / "results" / dataset / model_dir.name / f"{job_dir.name}.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("kind") != "model" or result.get("input") != "mix":
                continue
            for path in sorted(job_dir.glob("*.npy")):
                scored = result.get("outputs", {}).get(path.stem)
                if scored is None or path.stem not in PRODUCT_STEMS:
                    continue
                if scored.get("target") != path.stem or scored.get("derived"):
                    continue
                yield {
                    "model_id": model_dir.name,
                    "track_id": result["track_id"],
                    "stem": path.stem,
                    "job_key": job_dir.name,
                    "path": path,
                    "bss": scored.get("bss") or {},
                }


def complete_models(
    outputs: Iterable[Mapping[str, Any]], track_ids: Sequence[str]
) -> dict[str, list[str]]:
    """Model -> its product stems, for models with a kept output for every track."""
    tracks: dict[str, set[str]] = {}
    stems: dict[str, set[str]] = {}
    for item in outputs:
        tracks.setdefault(item["model_id"], set()).add(item["track_id"])
        stems.setdefault(item["model_id"], set()).add(item["stem"])
    wanted = set(track_ids)
    return {
        model: [stem for stem in PRODUCT_STEMS if stem in stems[model]]
        for model in sorted(tracks)
        if wanted <= tracks[model]
    }


def frames_in(
    values: Sequence[float | None], window: c.Window, sample_rate: int, frame_seconds: float
) -> list[float]:
    """The benchmark's per-window scores that lie wholly inside ``window``, without gaps.

    Window ``i`` of the benchmark covers ``[i, i + 1) * frame_seconds`` of the track. A
    window the benchmark could not score (the reference was silent there) is left out.
    """
    first = math.ceil(window.start / sample_rate / frame_seconds - 1e-9)
    last = math.floor((window.start + window.length) / sample_rate / frame_seconds + 1e-9)
    inside = values[max(first, 0) : max(last, 0)]
    return [float(v) for v in inside if v is not None and math.isfinite(v)]


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def read_rows(path: Path) -> list[dict[str, Any]]:
    return cal.read_rows(path)


# -- collecting scores ---------------------------------------------------------------------


def make_plan(
    set_name: str,
    dataset: Any,
    bench_dir: Path,
    plan_path: Path,
    *,
    reuse: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Where each track's loudest minute is, what plays in it, and which models count.

    Written once and read back after that. ``reuse`` names a P7.1 plan for the same
    tracks: the excerpts and what is present in them are taken from it (the rule and the
    audio are the same, so the answer is too), which saves reading every stem again.
    """
    if plan_path.is_file():
        return json.loads(plan_path.read_text(encoding="utf-8"))
    track_ids = list(dataset.track_ids())
    windows: dict[str, list[int]] = {}
    present: dict[str, dict[str, bool]] = {}
    source = "computed"
    if reuse is not None and reuse.is_file():
        earlier = json.loads(reuse.read_text(encoding="utf-8"))
        if earlier.get("dataset") == dataset.name and set(earlier["windows"]) == set(track_ids):
            windows, present, source = earlier["windows"], earlier["present"], str(reuse)
    if not windows:
        for number, track_id in enumerate(track_ids, start=1):
            track = dataset.load(track_id)
            window = c.loudest_window(track.inputs["mix"], track.sample_rate, EXCERPT_SECONDS)
            windows[track_id] = [window.start, window.length]
            present[track_id] = c.presence(track.references, window, track.sample_rate)
            if progress:
                progress(f"{set_name}: planned {number} of {len(track_ids)} tracks")
    plan = {
        "set": set_name,
        "dataset": dataset.name,
        "windows": windows,
        "present": present,
        "excerpts_from": source,
        "models": complete_models(kept_outputs(bench_dir, dataset.name), track_ids),
    }
    write_json(plan_path, plan)
    return plan


def collect(
    set_name: str,
    dataset: Any,
    tagger: Any,
    settings: Any,
    stem_map: Any,
    out_path: Path,
    *,
    bench_dir: Path = BENCH_DIR,
    reuse: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Score every kept output of one set, a row per output. Returns how many were new.

    An output whose row is already in ``out_path`` is skipped, so an interrupted run is
    continued by running it again.
    """
    from stems_worker.listening.features import hear

    plan = make_plan(
        set_name,
        dataset,
        bench_dir,
        out_path.with_name(f"{set_name}-plan.json"),
        reuse=reuse,
        progress=progress,
    )
    done = {row["id"] for row in read_rows(out_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_track: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for item in kept_outputs(bench_dir, dataset.name):
        if item["stem"] not in plan["models"].get(item["model_id"], []):
            continue
        item["id"] = f"{set_name}:{item['model_id']}:{item['track_id']}:{item['stem']}"
        if item["id"] in seen:  # a second kept run of the same model and track
            continue
        seen.add(item["id"])
        by_track.setdefault(item["track_id"], []).append(item)

    added = 0
    track_ids = [t for t in dataset.track_ids() if t in by_track]
    for number, track_id in enumerate(track_ids, start=1):
        todo = [item for item in by_track[track_id] if item["id"] not in done]
        if not todo:
            continue
        track = dataset.load(track_id)
        rate = track.sample_rate
        mix = track.inputs["mix"]
        window = c.Window(*plan["windows"][track_id])
        mix_level = round(rms_dbfs(window.cut(mix)), 2)
        for item in todo:
            estimate = np.load(item["path"], mmap_mode="r")
            if estimate.shape[-1] != mix.shape[-1]:
                raise OutputsError(
                    f"{item['path']} has {estimate.shape[-1]} samples, the mix "
                    f"{mix.shape[-1]}: not an output for this track"
                )
            excerpt = np.array(window.cut(estimate), dtype=np.float32)
            heard = hear(excerpt, rate, tagger, settings, stem_map)
            frames = item["bss"].get("frames") or {}
            seconds = float(item["bss"].get("window_seconds") or 1.0)
            sdr = frames_in(frames.get("sdr") or [], window, rate, seconds)
            sir = frames_in(frames.get("sir") or [], window, rate, seconds)
            row = {
                "id": item["id"],
                "set": set_name,
                "model_id": item["model_id"],
                "track_id": track_id,
                "stem": item["stem"],
                "job_key": item["job_key"],
                "reference_present": bool(plan["present"][track_id][item["stem"]]),
                "windows": int(heard.probabilities.shape[0]),
                "audible_windows": int(heard.audible.sum()),
                "scores": {s: round(v, 6) for s, v in heard.stems(stem_map).items()},
                "level_dbfs": round(rms_dbfs(excerpt), 2),
                "mix_level_dbfs": mix_level,
                "silence_fraction": round(silence_fraction(excerpt, rate), 4),
                "sdr_db": _median(sdr),
                "sir_db": _median(sir),
                "scored_seconds": len(sdr),
                "track_sdr_db": item["bss"].get("sdr"),
            }
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            done.add(row["id"])
            added += 1
        if progress:
            progress(f"{set_name}: scored {number} of {len(track_ids)} tracks")
    return added


def collect_levels(
    set_name: str,
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    bench_dir: Path = BENCH_DIR,
    progress: Callable[[str], None] | None = None,
) -> int:
    """For every scored output of one set, its level over the whole song. No tagger.

    A row per output: the output's level and the mix's over the whole track, and for how
    many seconds the reference stem sounds anywhere in it. Outputs already in
    ``out_path`` are skipped. Returns how many were new.
    """
    done = {row["id"] for row in read_rows(out_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_track: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["set"] == set_name and row["id"] not in done:
            by_track.setdefault(row["track_id"], []).append(row)
    added = 0
    track_ids = [t for t in dataset.track_ids() if t in by_track]
    for number, track_id in enumerate(track_ids, start=1):
        track = dataset.load(track_id)
        rate = track.sample_rate
        mix = track.inputs["mix"]
        seconds = mix.shape[-1] / rate
        mix_level = round(rms_dbfs(mix), 2)
        sounding: dict[str, float] = {}
        for row in by_track[track_id]:
            stem = row["stem"]
            if stem not in sounding:
                silent = silence_fraction(track.references[stem], rate)
                sounding[stem] = round((1.0 - silent) * seconds, 2)
            path = (
                bench_dir
                / "estimates"
                / dataset.name
                / row["model_id"]
                / row["job_key"]
                / f"{stem}.npy"
            )
            estimate = np.load(path, mmap_mode="r")
            level = {
                "id": row["id"],
                "set": set_name,
                "track_seconds": round(seconds, 2),
                "whole_level_dbfs": round(rms_dbfs(estimate), 2),
                "whole_mix_level_dbfs": mix_level,
                "reference_sounding_seconds": sounding[stem],
            }
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(level) + "\n")
            added += 1
        if progress:
            progress(f"{set_name}: levels of {number} of {len(track_ids)} tracks")
    return added


# -- analysis (pure) -----------------------------------------------------------------------


def own_score(row: Mapping[str, Any]) -> float:
    return float(row["scores"][row["stem"]])


def bleed_score(row: Mapping[str, Any]) -> float:
    """The strongest other source the tagger hears in the output, as in a job."""
    return max(float(v) for s, v in row["scores"].items() if s != row["stem"])


def relative_level(row: Mapping[str, Any]) -> float:
    return float(row["level_dbfs"]) - float(row["mix_level_dbfs"])


SIGNAL_OF_RULE: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "level": relative_level,
    "tagger_config": own_score,
    "tagger_rechosen": own_score,
}


def empty_pairs(rows: Iterable[Mapping[str, Any]], rule: str) -> list[tuple[float, bool]]:
    """(signal, is the reference really there) per output."""
    signal = SIGNAL_OF_RULE[rule]
    return [(round(signal(row), 6), bool(row["reference_present"])) for row in rows]


def choose_empty_max(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """The highest signal up to which "empty" was right ``EMPTY_RELIABILITY`` of the time."""
    return cal.choose_absent_max(pairs)


def empty_thresholds(calibration: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    return {
        "level": choose_empty_max(empty_pairs(calibration, "level")),
        "tagger_config": TAGGER_ABSENT_MAX,
        "tagger_rechosen": choose_empty_max(empty_pairs(calibration, "tagger_rechosen")),
    }


def empty_outcome(pairs: Sequence[tuple[float, bool]], threshold: float | None) -> dict[str, Any]:
    """How "empty when the signal is at or below ``threshold``" does on ``pairs``."""
    empties = [s for s, there in pairs if not there]
    real = [s for s, there in pairs if there]
    out: dict[str, Any] = {
        "outputs": len(pairs),
        "empty": len(empties),
        "threshold": threshold,
        # The chance that a real stem's signal is above an empty one's.
        "auc": auc(empties, real),
        "called_empty": None,
        "right": None,
        "precision": None,
        "recall": None,
    }
    if threshold is None:
        return out
    called = [there for s, there in pairs if s <= threshold]
    right = sum(1 for there in called if not there)
    out.update(
        called_empty=len(called),
        right=right,
        precision=right / len(called) if called else None,
        recall=right / len(empties) if empties else None,
    )
    return out


def silence_outcome(rows: Sequence[Mapping[str, Any]], minimum: float) -> dict[str, Any]:
    """What the gate's existing "mostly silent" rule makes of the same outputs.

    Not one of the rules on trial: it already acts in the gate, and is here to show how
    many of the empty outputs it catches by itself.
    """
    called = [row for row in rows if float(row["silence_fraction"]) >= minimum]
    right = sum(1 for row in called if not row["reference_present"])
    empties = sum(1 for row in rows if not row["reference_present"])
    return {
        "threshold": minimum,
        "called_empty": len(called),
        "right": right,
        "precision": right / len(called) if called else None,
        "recall": right / empties if empties else None,
    }


def empty_verdict(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Does one rule clear the bar on the held-out outputs?"""
    enough = (
        outcome["called_empty"] is not None
        and outcome["called_empty"] >= MIN_SUPPORT
        and outcome["empty"] >= MIN_SUPPORT
    )
    passed = (
        enough
        and outcome["precision"] >= CRITERIA["empty_precision_min"]
        and outcome["recall"] >= CRITERIA["empty_recall_min"]
    )
    return {"enough": bool(enough), "passed": bool(passed)}


def comparable(row: Mapping[str, Any]) -> bool:
    return (
        bool(row["reference_present"])
        and row.get("sdr_db") is not None
        and int(row.get("scored_seconds") or 0) >= PAIR_MIN_SECONDS
    )


def output_pairs(
    rows: Iterable[Mapping[str, Any]], gap_db: float = PAIR_GAP_DB
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """(better, worse) outputs of one stem of one track whose SDR differs by ``gap_db``."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if comparable(row):
            groups.setdefault((row["track_id"], row["stem"]), []).append(row)
    pairs = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["model_id"])
        for i, first in enumerate(group):
            for second in group[i + 1 :]:
                if abs(first["sdr_db"] - second["sdr_db"]) >= gap_db:
                    better, worse = (
                        (first, second) if first["sdr_db"] > second["sdr_db"] else (second, first)
                    )
                    pairs.append((better, worse))
    return pairs


# How much each rule prefers the first output over the second: positive means "first".
PAIR_RULES: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], float]] = {
    "bleed": lambda a, b: bleed_score(b) - bleed_score(a),
    "own": lambda a, b: own_score(a) - own_score(b),
    "own_minus_bleed": lambda a, b: (
        (own_score(a) - bleed_score(a)) - (own_score(b) - bleed_score(b))
    ),
    "level": lambda a, b: relative_level(a) - relative_level(b),
}


def pair_accuracy(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], rule: str
) -> dict[str, Any]:
    """How often ``rule`` prefers the better output. A tie counts as half."""
    prefer = PAIR_RULES[rule]
    points = 0.0
    for better, worse in pairs:
        margin = prefer(better, worse)
        points += 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
    return {"pairs": len(pairs), "accuracy": points / len(pairs) if pairs else None}


def pair_verdict(outcome: Mapping[str, Any]) -> dict[str, Any]:
    enough = outcome["pairs"] >= CRITERIA["pair_support_min"]
    passed = enough and outcome["accuracy"] >= CRITERIA["pair_accuracy_min"]
    return {"enough": bool(enough), "passed": bool(passed)}


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation; None with fewer than three points or no spread."""
    if len(xs) < 3:
        return None
    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / math.sqrt(sxx * syy)


def by_stem(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict]:
    """Per stem, what the measured scores are and what the configured thresholds say."""
    present_min = config["presence"]["present_min"]
    absent_max = config["presence"]["absent_max"]
    moderate_min = config["bleed"]["moderate_min"]
    table = []
    for stem in PRODUCT_STEMS:
        mine = [row for row in rows if row["stem"] == stem]
        real = [row for row in mine if comparable(row)]
        if not mine:
            continue
        with_sir = [row for row in real if row.get("sir_db") is not None]
        table.append(
            {
                "stem": stem,
                "outputs": len(mine),
                "empty": sum(1 for row in mine if not row["reference_present"]),
                "scored": len(real),
                "median_sdr_db": _median([row["sdr_db"] for row in real]),
                "median_sir_db": _median([row["sir_db"] for row in with_sir]),
                "median_own": _median([own_score(row) for row in real]),
                "median_bleed": _median([bleed_score(row) for row in real]),
                "called_present": sum(1 for row in real if own_score(row) >= present_min),
                "called_absent": sum(1 for row in real if own_score(row) <= absent_max),
                "called_bled": sum(1 for row in real if bleed_score(row) >= moderate_min),
                "bleed_vs_sir": spearman(
                    [bleed_score(row) for row in with_sir], [row["sir_db"] for row in with_sir]
                ),
                "bleed_vs_sdr": spearman(
                    [bleed_score(row) for row in real], [row["sdr_db"] for row in real]
                ),
            }
        )
    return table


def in_song(row: Mapping[str, Any]) -> bool:
    return float(row["reference_sounding_seconds"]) >= IN_SONG_SECONDS


def whole_level(row: Mapping[str, Any]) -> float:
    return float(row["whole_level_dbfs"]) - float(row["whole_mix_level_dbfs"])


WHOLE_SIGNALS: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "minute": relative_level,
    "whole": whole_level,
}


def analyse_whole(
    rows: Sequence[Mapping[str, Any]], levels: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The empty-stem rule against whole-song truth. ``levels`` are ``collect_levels`` rows."""
    by_id = {level["id"]: level for level in levels}
    joined = [{**row, **by_id[row["id"]]} for row in rows if row["id"] in by_id]
    calibration = [row for row in joined if row["set"] == "calibration"]
    heldout = [row for row in joined if row["set"] == "heldout"]

    def pairs(part: Sequence[Mapping[str, Any]], rule: str) -> list[tuple[float, bool]]:
        return [(round(WHOLE_SIGNALS[rule](row), 6), in_song(row)) for row in part]

    thresholds = {
        "minute": IN_FORCE_DB,
        "whole": choose_empty_max(pairs(calibration, "whole")),
    }
    rules: dict[str, Any] = {}
    for rule in WHOLE_RULES:
        outcome = empty_outcome(pairs(heldout, rule), thresholds[rule])
        rules[rule] = {
            "calibration": empty_outcome(pairs(calibration, rule), thresholds[rule]),
            "heldout": outcome,
            **empty_verdict(outcome),
        }
    decision = (
        "keep" if rules["minute"]["passed"] else "replace" if rules["whole"]["passed"] else "off"
    )
    # Outputs the first measurement counted as empty although the instrument is in the song.
    elsewhere = [row for row in heldout if not row["reference_present"] and in_song(row)]
    return {
        "outputs": {"calibration": len(calibration), "heldout": len(heldout)},
        "without_levels": len(rows) - len(joined),
        "not_in_song": sum(1 for row in heldout if not in_song(row)),
        "absent_in_minute": sum(1 for row in heldout if not row["reference_present"]),
        "elsewhere": len(elsewhere),
        "elsewhere_called_empty": sum(1 for row in elsewhere if relative_level(row) <= IN_FORCE_DB),
        "in_minute_not_in_song": sum(
            1 for row in heldout if row["reference_present"] and not in_song(row)
        ),
        "rules": rules,
        "decision": decision,
    }


def analyse_retry(
    rows: Sequence[Mapping[str, Any]],
    levels: Sequence[Mapping[str, Any]],
    chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """What retrying an empty stem with the fallback model would do, on the held-out outputs.

    ``chains`` maps a stem to (default model, fallback model), as the product is configured.
    """
    by_id = {level["id"]: level for level in levels}
    joined = {
        (row["track_id"], row["stem"], row["model_id"]): {**row, **by_id[row["id"]]}
        for row in rows
        if row["set"] == "heldout" and row["id"] in by_id
    }

    def empty(row: Mapping[str, Any]) -> bool:
        return whole_level(row) <= RETRY_EMPTY_DB

    def sdr(row: Mapping[str, Any]) -> float | None:
        value = row.get("track_sdr_db")
        return None if value is None else float(value)

    by_stem: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for stem, (default, fallback) in chains.items():
        counts = dict.fromkeys(("pairs", "empty", "stays_empty", "rescue", "wrong", "no_better"), 0)
        for track_id in sorted({key[0] for key in joined if key[1] == stem}):
            first = joined.get((track_id, stem, default))
            second = joined.get((track_id, stem, fallback))
            if first is None or second is None:
                continue
            counts["pairs"] += 1
            if not empty(first):
                continue
            counts["empty"] += 1
            if empty(second):
                outcome = "stays_empty"
            elif not in_song(first):
                outcome = "wrong"
            else:
                a, b = sdr(first), sdr(second)
                better = b is not None and b > 0 and (a is None or b > a)
                outcome = "rescue" if better else "no_better"
            counts[outcome] += 1
            if outcome != "stays_empty":
                cases.append(
                    {
                        "track_id": track_id,
                        "stem": stem,
                        "outcome": outcome,
                        "default_sdr_db": sdr(first),
                        "fallback_sdr_db": sdr(second),
                        "fallback_level_db": round(whole_level(second), 1),
                    }
                )
        if counts["pairs"]:
            by_stem[stem] = {"default": default, "fallback": fallback, **counts}
    total = {
        key: sum(entry[key] for entry in by_stem.values())
        for key in ("pairs", "empty", "stays_empty", "rescue", "wrong", "no_better")
    }
    return {
        "by_stem": by_stem,
        "total": total,
        "cases": cases,
        "decision": "retry" if total["rescue"] > total["wrong"] else "stop",
    }


def analyse(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the report states, from the scored rows and the configured thresholds.

    config holds the thresholds in force when the outputs were scored:
    presence and bleed of listening, and the gate's silent_fraction_min.
    """
    calibration = [row for row in rows if row["set"] == "calibration"]
    heldout = [row for row in rows if row["set"] == "heldout"]
    thresholds = empty_thresholds(calibration)
    empties: dict[str, Any] = {}
    for rule in EMPTY_RULES:
        outcome = empty_outcome(empty_pairs(heldout, rule), thresholds[rule])
        empties[rule] = {
            "calibration": empty_outcome(empty_pairs(calibration, rule), thresholds[rule]),
            "heldout": outcome,
            **empty_verdict(outcome),
        }
    use = next((rule for rule in EMPTY_RULES if empties[rule]["passed"]), None)
    pairs = output_pairs(heldout)
    by_rule = {rule: pair_accuracy(pairs, rule) for rule in PAIR_RULES}
    pairs_by_stem = {
        stem: pair_accuracy([p for p in pairs if p[0]["stem"] == stem], PAIR_RULE)
        for stem in PRODUCT_STEMS
        if any(p[0]["stem"] == stem for p in pairs)
    }
    return {
        "counts": {
            name: {
                "outputs": len(part),
                "tracks": len({row["track_id"] for row in part}),
                "models": sorted({row["model_id"] for row in part}),
                "empty": sum(1 for row in part if not row["reference_present"]),
            }
            for name, part in (("calibration", calibration), ("heldout", heldout))
        },
        "empty": {
            "rules": empties,
            "use": use,
            "silence_rule": silence_outcome(heldout, config["silent_fraction_min"]),
        },
        "pairs": {
            "gap_db": PAIR_GAP_DB,
            "rules": by_rule,
            "by_stem": pairs_by_stem,
            **pair_verdict(by_rule[PAIR_RULE]),
        },
        "by_stem": by_stem(heldout, config),
        "criteria": CRITERIA,
    }


# -- report --------------------------------------------------------------------------------

_pct, _num = cal._pct, cal._num

RULE_NAMES = {
    "level": "Level: the output is this far below the mix, or further",
    "tagger_config": "Tagger: own-instrument score at most the configured `absent_max`",
    "tagger_rechosen": "Tagger: own-instrument score, threshold chosen on these outputs",
}
PAIR_NAMES = {
    "bleed": "Lower bleed score (the rule on trial)",
    "own": "Higher own-instrument score",
    "own_minus_bleed": "Higher own-instrument score minus bleed score",
    "level": "Louder output",
}


def _count(value: int | None) -> str:
    return "-" if value is None else str(value)


def _threshold(rule: str, value: float | None) -> str:
    if value is None:
        return "none found"
    return f"{value:.1f} dB" if rule == "level" else f"{value:.4f}"


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    counts, empty, pairs = analysis["counts"], analysis["empty"], analysis["pairs"]
    crit = analysis["criteria"]
    git = run.get("git") or {}
    commit = str(git.get("commit") or "unknown")[:12] + (
        ", with uncommitted changes" if git.get("dirty") else ""
    )
    lines = [
        "# The listening layer on real separator output",
        "",
        f"Generated from `runs/calibration/{run['run_id']}/scores.jsonl` (scored "
        f"{run['finished']}, tagger `{run['tagger']}` on {run['device']}, commit `{commit}`). "
        "Reproduce with `python -m stems_bench.gate_outputs collect` and then `report`; "
        "that needs the benchmark's kept outputs under `runs/bench/estimates/`, which are "
        "not in the repository.",
        "",
        "The two questions, the rules tried, how each threshold is chosen and the bar each "
        "must clear were written into `bench/stems_bench/gate_outputs.py` at commit "
        f"`{str(run.get('rules_commit') or 'unknown')[:12]}`, before any of these outputs "
        "was scored.",
        "",
        "P7.1 (`docs/gate-calibration.md`) tested the tagger on reference stems with "
        "another stem mixed in by hand. This tests it on what the gate really sees: "
        "outputs of real separators, with their quality as the benchmark measured it.",
        "",
        "## What was scored",
        "",
        "| Set | Tracks | Models | Outputs | Of them empty |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (("calibration", "Calibration"), ("heldout", "Held out")):
        part = counts[name]
        lines.append(
            f"| {label} (`{SETS[name]}`) | {part['tracks']} | {len(part['models'])} | "
            f"{part['outputs']} | {part['empty']} |"
        )
    lines += [
        "",
        f"Each output is cut to the loudest {EXCERPT_SECONDS:.0f} seconds of its track. "
        '"Empty" means the reference stem is not present in that minute: the song has no '
        "such instrument there, and the output is leftovers. Only outputs the benchmark "
        "kept are here (the members of its blends), so weak models are under-represented.",
        "",
        "Models, held out: " + ", ".join(f"`{m}`" for m in counts["heldout"]["models"]) + ".",
        "",
        "## Question 1: can an empty stem be told from a real one?",
        "",
        "Each rule calls an output empty when its signal is at or below the threshold. "
        '"Right" is how many of the outputs it calls empty are; "found" is how many of '
        "the empty outputs it calls empty. AUC is the chance that a real stem's signal is "
        "above an empty one's.",
        "",
        "| Rule | Threshold | AUC | Called empty | Right | Found | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for rule in EMPTY_RULES:
        entry = empty["rules"][rule]
        held = entry["heldout"]
        result = "pass" if entry["passed"] else "fail" if entry["enough"] else "too few to say"
        lines.append(
            f"| {RULE_NAMES[rule]} | {_threshold(rule, held['threshold'])} | "
            f"{_num(held['auc'])} | {_count(held['called_empty'])} "
            f"| {_pct(held['precision'])} | {_pct(held['recall'])} | {result} |"
        )
    lines += [
        "",
        f"Needed on the held-out outputs: right at least "
        f"{_pct(crit['empty_precision_min'])}, found at least {_pct(crit['empty_recall_min'])}, "
        f"with at least {MIN_SUPPORT} outputs called empty and {MIN_SUPPORT} empty ones. "
        f"There were {counts['heldout']['empty']} empty outputs among "
        f"{counts['heldout']['outputs']}.",
        "",
        "For comparison, the rule the gate already acts on (an output that is silent for "
        f"at least {_pct(empty['silence_rule']['threshold'])} of its length is 'mostly "
        f"silent') calls {empty['silence_rule']['called_empty']} of these outputs empty; "
        f"{empty['silence_rule']['right']} of those are, which is "
        f"{_pct(empty['silence_rule']['recall'])} of the empty ones.",
        "",
        "The same rules on the outputs their thresholds were chosen on:",
        "",
        "| Rule | AUC | Called empty | Right | Found |",
        "|---|---:|---:|---:|---:|",
    ]
    for rule in EMPTY_RULES:
        cal_part = empty["rules"][rule]["calibration"]
        lines.append(
            f"| {RULE_NAMES[rule]} | {_num(cal_part['auc'])} | "
            f"{_count(cal_part['called_empty'])} | "
            f"{_pct(cal_part['precision'])} | {_pct(cal_part['recall'])} |"
        )
    lines += [
        "",
        "## Question 2: of two outputs of the same stem, which is better?",
        "",
        f"Pairs of outputs of one stem of one track from two models, whose measured SDR in "
        f"the excerpt differs by at least {pairs['gap_db']:g} dB. A rule is right when it "
        "prefers the one with the higher SDR. Guessing gives 50%.",
        "",
        "| Rule | Pairs | Prefers the better one |",
        "|---|---:|---:|",
    ]
    for rule, outcome in pairs["rules"].items():
        lines.append(f"| {PAIR_NAMES[rule]} | {outcome['pairs']} | {_pct(outcome['accuracy'])} |")
    lines += [
        "",
        f"Needed: at least {_pct(crit['pair_accuracy_min'])} on at least "
        f"{crit['pair_support_min']:g} pairs, for the bleed score. The other rules are shown "
        "for comparison and decide nothing.",
        "",
    ]
    if pairs["by_stem"]:
        lines += [
            "The bleed score by stem:",
            "",
            "| Stem | Pairs | Prefers the better one |",
            "|---|---:|---:|",
        ]
        for stem, outcome in pairs["by_stem"].items():
            lines.append(f"| {stem} | {outcome['pairs']} | {_pct(outcome['accuracy'])} |")
        lines.append("")
    lines += [
        "## What the configured thresholds say about real outputs (held out)",
        "",
        'Outputs whose reference is present, by stem. "Called present" and "called '
        'absent" use `listening.presence`; "called bled" is a bleed score at or above '
        "`listening.bleed.moderate_min`. The last two columns are rank correlations of the "
        "bleed score with the measured SIR and SDR: a bleed score that worked would be "
        "strongly negative in both.",
        "",
        "| Stem | Outputs | Median SDR | Median SIR | Median own score | Called present | "
        "Called absent | Median bleed score | Called bled | Bleed vs SIR | Bleed vs SDR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["by_stem"]:
        lines.append(
            f"| {row['stem']} | {row['scored']} | {_num(row['median_sdr_db'], 2)} | "
            f"{_num(row['median_sir_db'], 2)} | {_num(row['median_own'])} | "
            f"{row['called_present']} | {row['called_absent']} | {_num(row['median_bleed'])} | "
            f"{row['called_bled']} | {_num(row['bleed_vs_sir'], 2)} | "
            f"{_num(row['bleed_vs_sdr'], 2)} |"
        )
    lines += ["", "## Verdict", ""]
    if empty["use"] is None:
        lines.append(
            "**Empty stems: no rule passed.** Nothing here tells leftovers from a real stem "
            'well enough to tell a user "this song has no such instrument".'
        )
    else:
        lines.append(
            f"**Empty stems: the rule to use is `{empty['use']}`** "
            f"({RULE_NAMES[empty['use']].lower()}; threshold "
            f"{_threshold(empty['use'], empty['rules'][empty['use']]['heldout']['threshold'])}). "
            "It is the first of the three, in the order fixed beforehand, that cleared the bar."
        )
    lines.append("")
    trial = pairs["rules"][PAIR_RULE]
    if not pairs["enough"]:
        lines.append(
            f"**The better of two: not enough pairs to say** ({trial['pairs']}, "
            f"{crit['pair_support_min']:g} needed)."
        )
    elif pairs["passed"]:
        lines.append(
            f"**The better of two: passed.** The bleed score preferred the better output in "
            f"{_pct(trial['accuracy'])} of {trial['pairs']} pairs."
        )
    else:
        lines.append(
            f"**The better of two: failed.** The bleed score preferred the better output in "
            f"{_pct(trial['accuracy'])} of {trial['pairs']} pairs "
            f"({_pct(crit['pair_accuracy_min'])} needed). A retry chosen on the tagger's word "
            "would not reliably pick the better stem, so the gate keeps not acting on it."
        )
    if analysis.get("whole"):
        lines += ["", *_whole_section(analysis["whole"], crit, run)]
    if analysis.get("retry"):
        lines += ["", *_retry_section(analysis["retry"], run)]
    lines += [
        "",
        "## Limits",
        "",
        "- One minute per track, not the whole stem; a job scores the whole stem.",
        "- Kept outputs only: the models the blends are made of. A model bad enough to be "
        "left out of every blend is not here.",
        "- One dataset (MoisesDB), one tagger, outputs made on this Mac's GPU.",
        '- "Empty" is decided from the reference in that minute. An instrument that plays '
        "elsewhere in the song counts as absent here.",
        "",
    ]
    return "\n".join(lines)


WHOLE_NAMES = {
    "minute": "In force: level against the mix over the mix's loudest minute",
    "whole": "Level against the mix over the whole song",
}
DECISIONS = {
    "keep": "**The rule in force stays.** It clears the bar against the whole song too.",
    "replace": (
        "**The rule is replaced.** Measured over the loudest minute it fails against the "
        "whole song; measured over the whole song it passes, and that is what the gate "
        "now uses."
    ),
    "off": (
        "**The rule is switched off.** Neither way of measuring the level clears the bar "
        "against the whole song."
    ),
}


def _whole_section(
    whole: Mapping[str, Any], crit: Mapping[str, float], run: Mapping[str, Any]
) -> list[str]:
    """The part of the report that checks the empty-stem rule against whole songs."""
    added = run.get("whole_rules_commit")
    when = f"at commit `{added[:12]}`" if added else "in the commit that adds this section"
    lines = [
        "## The empty-stem rule against the whole song",
        "",
        "Question 1 called an output empty when its reference is absent in the loudest "
        "minute, and the gate's note for such a stem says the instrument may not be in the "
        "song. Here the truth is the whole song: the instrument is in it if its reference "
        f"sounds for at least {IN_SONG_SECONDS:g} seconds in total, anywhere. The rules "
        f"below and what is done with the result were added to this file's code {when}, "
        "after the results above were known and before any whole-song figure was computed. "
        "No tagger is involved; the levels were read from the same kept outputs.",
        "",
        f"Of the {whole['outputs']['heldout']} held-out outputs, {whole['absent_in_minute']} "
        f"have a reference that is absent in the loudest minute, and {whole['not_in_song']} "
        f"have an instrument that is not in the song at all. The difference, "
        f"{whole['elsewhere']} outputs, is instruments that play only outside the loudest "
        f"minute; the rule in force calls {whole['elsewhere_called_empty']} of them empty.",
        "",
        "| Rule | Threshold | AUC | Called empty | Right | Found | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for rule in WHOLE_RULES:
        entry = whole["rules"][rule]
        held = entry["heldout"]
        result = "pass" if entry["passed"] else "fail" if entry["enough"] else "too few to say"
        lines.append(
            f"| {WHOLE_NAMES[rule]} | {_threshold('level', held['threshold'])} | "
            f"{_num(held['auc'])} | {_count(held['called_empty'])} | {_pct(held['precision'])} | "
            f"{_pct(held['recall'])} | {result} |"
        )
    lines += [
        "",
        f'"Right" and "found" are against the whole-song truth. Needed, as before: right '
        f"at least {_pct(crit['empty_precision_min'])}, found at least "
        f"{_pct(crit['empty_recall_min'])}.",
        "",
        DECISIONS[whole["decision"]],
    ]
    if whole["without_levels"]:
        lines += ["", f"{whole['without_levels']} scored outputs have no whole-song level yet."]
    return lines


def _retry_section(retry: Mapping[str, Any], run: Mapping[str, Any]) -> list[str]:
    """The part of the report on whether a stem found empty should be tried again."""
    added = run.get("retry_rules_commit")
    when = f"at commit `{added[:12]}`" if added else "in the commit that adds this section"
    total = retry["total"]
    lines = [
        "## Should a stem found empty be tried again?",
        "",
        "The gate tries an empty stem again with the fallback model and delivers the retry "
        "if that is not empty. A real job (`docs/product-vs-benchmark.md`) showed how that "
        "goes wrong: a song with no guitar, a silent first attempt, and a fallback model "
        "whose leftovers were delivered as the guitar. This counts both sides on the "
        "held-out outputs, for every track and stem where both the default model's output "
        f"and the fallback's were kept. The counting rules and what follows from them were "
        f"added to this file's code {when}, after that job and before these counts were made.",
        "",
        '"Empty" is the rule in force: at least '
        f"{abs(RETRY_EMPTY_DB):g} dB under the mix over the whole song. A rescue is an "
        "instrument that is in the song, where the fallback's output is not empty and its "
        "measured SDR is above 0 dB and above the default model's. A wrong delivery is an "
        "instrument that is not in the song, where the fallback's output is not empty.",
        "",
        "| Stem | Default, then fallback | Tracks | Default's output empty | Fallback's too | "
        "Rescued | Wrongly delivered | No better |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stem, entry in retry["by_stem"].items():
        lines.append(
            f"| {stem} | `{entry['default']}`, then `{entry['fallback']}` | {entry['pairs']} | "
            f"{entry['empty']} | {entry['stays_empty']} | {entry['rescue']} | {entry['wrong']} | "
            f"{entry['no_better']} |"
        )
    lines += [
        f"| all | | {total['pairs']} | {total['empty']} | {total['stays_empty']} | "
        f"{total['rescue']} | {total['wrong']} | {total['no_better']} |",
        "",
    ]
    if retry["decision"] == "retry":
        lines.append(
            f"**The gate keeps retrying.** {total['rescue']} rescues against "
            f"{total['wrong']} wrong deliveries."
        )
    else:
        lines.append(
            f"**The gate stops retrying empty stems.** {total['rescue']} rescues against "
            f"{total['wrong']} wrong deliveries: a stem found empty is now delivered at once "
            "with its note, and no other model is run for it."
        )
    lines += [
        "",
        "The stems here are the ones whose default and fallback models the benchmark kept "
        "outputs for; the six-stem `other` is not among them. The gate also calls a stem "
        "empty when it is silent for 98% of its length, which this count does not look at "
        "separately.",
    ]
    return lines


# -- command line --------------------------------------------------------------------------


def _run_dir(run_id: str) -> Path:
    return cal.CALIBRATION_DIR / run_id


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
        for name, dataset_name in SETS.items():
            if args.only and name != args.only:
                continue
            collect(
                name,
                get_dataset(dataset_name),
                tagger,
                settings,
                stem_map,
                run_dir / "scores.jsonl",
                reuse=_run_dir(args.reuse_plans) / f"{name}-plan.json",
                progress=progress,
            )
    except BaseException as error:
        write_json(
            status_path,
            {"state": "failed", "started": started, "error": type(error).__name__},
        )
        raise
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


def cmd_levels(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset

    run_dir = _run_dir(args.run_id)
    rows = read_rows(run_dir / "scores.jsonl")
    if not rows:
        raise OutputsError(f"{run_dir / 'scores.jsonl'} is missing or empty: run `collect` first")
    status_path = run_dir / "levels-status.json"
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(status_path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    try:
        for name, dataset_name in SETS.items():
            collect_levels(
                name, get_dataset(dataset_name), rows, run_dir / "levels.jsonl", progress=progress
            )
    except BaseException as error:
        write_json(
            status_path, {"state": "failed", "started": started, "error": type(error).__name__}
        )
        raise
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from stems_common.settings import load_defaults

    run_dir = _run_dir(args.run_id)
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise OutputsError(f"{run_path} is missing: the scoring run has not finished")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    # The rules as they stood when the outputs were scored: fixed at the first report.
    rules_path = run_dir / "rules.json"
    if rules_path.is_file():
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
    else:
        defaults = load_defaults()
        listening = defaults.listening.model_dump()
        rules = {
            "rules_commit": RULES_COMMIT,
            "config": {
                "presence": listening["presence"],
                "bleed": listening["bleed"],
                "silent_fraction_min": defaults.gate.rules.silent_fraction_min,
            },
        }
        write_json(rules_path, rules)
    rows = read_rows(run_dir / "scores.jsonl")
    analysis = analyse(rows, rules["config"])
    levels = read_rows(run_dir / "levels.jsonl")
    if levels:
        analysis["whole"] = analyse_whole(rows, levels)
        # The chains as the product was configured when they were first counted.
        chains_path = run_dir / "retry-chains.json"
        if chains_path.is_file():
            chains = json.loads(chains_path.read_text(encoding="utf-8"))
        else:
            from stems_models.registry import load_registry
            from stems_worker.decision.plan import config_fallback

            defaults, registry = load_defaults(), load_registry()
            chains = {
                stem: [
                    defaults.ensembles.slots[stem].single_best,
                    config_fallback(stem, defaults, registry),
                ]
                for stem in PRODUCT_STEMS
            }
            write_json(chains_path, chains)
        analysis["retry"] = analyse_retry(rows, levels, chains)
    write_json(run_dir / "analysis.json", analysis)
    text = render(
        analysis,
        {
            **run,
            "rules_commit": rules["rules_commit"],
            "whole_rules_commit": WHOLE_RULES_COMMIT,
            "retry_rules_commit": RETRY_RULES_COMMIT,
        },
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stems_bench.gate_outputs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="score the kept benchmark outputs")
    collect_parser.add_argument("--run-id", default=DEFAULT_RUN)
    collect_parser.add_argument("--device", default="cpu")
    collect_parser.add_argument("--only", choices=sorted(SETS))
    collect_parser.add_argument(
        "--reuse-plans",
        default=cal.DEFAULT_RUN,
        help="the P7.1 run whose excerpts are reused when it covers the same tracks",
    )
    collect_parser.set_defaults(run=cmd_collect)
    levels_parser = sub.add_parser(
        "levels", help="measure each scored output's level over the whole song (no tagger)"
    )
    levels_parser.add_argument("--run-id", default=DEFAULT_RUN)
    levels_parser.set_defaults(run=cmd_levels)
    report_parser = sub.add_parser("report", help="write the report from a finished run")
    report_parser.add_argument("--run-id", default=DEFAULT_RUN)
    report_parser.add_argument("--out")
    report_parser.set_defaults(run=cmd_report)
    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except OutputsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
