"""Can the product tell a dropout from a rest? (Phase 15)

    python -m stems_bench.gaps collect     # read the benchmark's kept outputs (resumable)
    python -m stems_bench.gaps report --out docs/dropout-or-rest.md

Phases 12 and 14 found, with the true stems beside them, that a stem can be silent for a
long stretch while its instrument plays (11 of 100 stems on twenty songs), that the
instrument is then in a stem of another name, and that the product says nothing about it.
The product has no true stem. It has the stem it made, the song, and the second model its
plan already names for every stem (the fallback). The question here: from those alone,
can it say "this stem may be missing something from 1:05 to 1:40" and be right?

**The data.** The benchmark's kept outputs (``runs/bench/estimates``); no model is run. A
stem is counted when both the default model's output and the fallback model's are kept for
every track of both lists: the tuning tracks (``moisesdb_tuning``) and the test tracks
(``moisesdb_subset``). A stem for which they are not is left out and named in the report.
These are the benchmark's outputs, not a job's.

**Definitions**, fixed here before any kept output is read for this:

- A *gap* is an unbroken stretch of tenths of a second in which the default model's stem
  is silent (under the silence level of ``stems_common.levels``), and inside which the
  song sounds for ``LONG_SECONDS`` or more. A stem the gate already notes over the whole
  song (silent, or very quiet against the song, by ``gate.rules``) has no gaps counted:
  it carries a note as it is.
- A gap is a *dropout* when the true stem sounds for ``LONG_SECONDS`` or more inside it
  (the long dropout of ``docs/dropouts.md``). Otherwise it is a *rest*.
- Three rules are tried. Each one flags gaps:
    ``silence``      every gap
    ``second``       gaps inside which the fallback model's stem sounds for
                     ``LONG_SECONDS`` or more
    ``loud_second``  gaps over which the fallback model's stem is at least T dB against
                     the song (its level minus the song's, both over the gap). T is chosen
                     on the tuning tracks: the lowest level at which the rule flags
                     ``MIN_FLAGGED`` gaps or more and at least ``BAR`` of them are
                     dropouts. If no level does that, the rule flags nothing.
- A rule is *reliable* on a list when it flags ``MIN_FLAGGED`` gaps or more there and at
  least ``BAR`` of the gaps it flags are dropouts. ``BAR`` and ``MIN_FLAGGED`` are the
  bar the empty-stem note was held to (``docs/gate-on-real-output.md``).
- **What is done with the result.** A rule that is reliable on the tuning tracks is tried
  on the test tracks, once. If more than one is, the one that finds the most dropouts on
  the tuning tracks is tried (the earlier in the list above, if they find as many). If it
  is reliable on the test tracks too, it *stands*, and a later phase may build a note on
  it. If no rule is reliable on the tuning tracks, or the one tried is not reliable on
  the test tracks, the answer is that nothing here shows the product can tell, and the
  report says for each rule whether it was wrong too often or flagged too few gaps to go
  by. Nothing in the product changes in this phase either way. The other rules' counts
  on the test tracks are shown for information and decide nothing.
- For information only: over each dropout, the share of the true stem that the fallback
  model's stem holds (the share of ``stems_bench.lost``), so that it can be seen whether
  a second model would have anything to put in the gap.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.calibrate import MIN_SUPPORT
from stems_bench.gate_outputs import CRITERIA, IN_SONG_SECONDS, kept_outputs, read_rows
from stems_bench.lost import shares
from stems_bench.run import BENCH_DIR, git_state, write_json
from stems_common.levels import (
    SILENCE_DBFS,
    SILENCE_WINDOW_SECONDS,
    level_below_mix_db,
    silence_fraction,
    sounding_windows,
)
from stems_common.paths import RUNS_DIR
from stems_common.stems import PRODUCT_STEMS

GAPS_DIR = RUNS_DIR / "gaps"
# The two lists, by the names the report uses.
SETS = {"tuning": "moisesdb_tuning", "test": "moisesdb_subset"}
LONG_SECONDS = IN_SONG_SECONDS
RULES = ("silence", "second", "loud_second")
BAR = CRITERIA["empty_precision_min"]
MIN_FLAGGED = MIN_SUPPORT
# The commit that holds everything above, made before any kept output was read for this.
# Three things in the report came after the run, decide nothing and change no figure: the
# nearest any level of the third rule comes to the bar, how many songs the dropouts are
# in (with what that leaves of the bar), and the noted stems that hold a dropout.
RULES_COMMIT: str | None = "131fbfa561cc73950c84e1d93c25eb27b6ef8f44"


class GapsError(ValueError):
    """The count cannot be made or reported as asked."""


# -- measuring one stem --------------------------------------------------------------------


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Every unbroken run of True in ``mask``, as (first index, one past the last)."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(first), int(last)) for first, last in zip(edges[::2], edges[1::2], strict=True)]


def is_noted(
    made: np.ndarray,
    mix: np.ndarray,
    sample_rate: int,
    *,
    silent_fraction_min: float,
    empty_below_mix_db: float | None,
) -> bool:
    """Does the gate note this stem over the whole song: silent, or very quiet against it?"""
    if silence_fraction(made, sample_rate) >= silent_fraction_min:
        return True
    return empty_below_mix_db is not None and level_below_mix_db(made, mix) <= empty_below_mix_db


def measure_stem(
    mix: np.ndarray, true: np.ndarray, made: np.ndarray, second: np.ndarray, sample_rate: int
) -> list[dict[str, Any]]:
    """The gaps of one made stem, and what the song, the true stem and the second model's
    stem do inside each. Seconds are to a tenth; all four must have the same length.
    """
    samples = mix.shape[-1]
    for name, audio in (("true stem", true), ("default model's", made), ("fallback's", second)):
        if audio.shape[-1] != samples:
            raise GapsError(
                f"the {name} audio has {audio.shape[-1]} samples and the song has {samples}"
            )
    width = max(1, round(SILENCE_WINDOW_SECONDS * sample_rate))
    song = sounding_windows(mix, sample_rate)
    plays = sounding_windows(true, sample_rate)
    speaks = sounding_windows(second, sample_rate)
    gaps: list[dict[str, Any]] = []
    for first, last in runs_of(~sounding_windows(made, sample_rate)):
        inside = float(song[first:last].sum()) * SILENCE_WINDOW_SECONDS
        if inside < LONG_SECONDS:
            continue
        low, high = first * width, min(last * width, samples)
        there = np.asarray(true[..., low:high])
        other = np.asarray(second[..., low:high])
        gap: dict[str, Any] = {
            "start": round(first * SILENCE_WINDOW_SECONDS, 1),
            "seconds": round((last - first) * SILENCE_WINDOW_SECONDS, 1),
            "song": round(inside, 1),
            "true": round(float(plays[first:last].sum()) * SILENCE_WINDOW_SECONDS, 1),
            "second": round(float(speaks[first:last].sum()) * SILENCE_WINDOW_SECONDS, 1),
            "second_below_song_db": round(level_below_mix_db(other, mix[..., low:high]), 2),
            "true_below_song_db": round(level_below_mix_db(there, mix[..., low:high]), 2),
        }
        held = shares(there, {"second": other}, np.ones(high - low, dtype=bool))
        if held is not None:
            gap["second_share"] = round(held["second"], 4)
        gaps.append(gap)
    return gaps


# -- the rules (pure) ----------------------------------------------------------------------


def is_dropout(gap: Mapping[str, Any]) -> bool:
    return gap["true"] >= LONG_SECONDS


def flags(rule: str, gap: Mapping[str, Any], threshold: float | None = None) -> bool:
    """Does ``rule`` flag this gap? ``threshold`` is T of ``loud_second``."""
    if rule == "silence":
        return True
    if rule == "second":
        return gap["second"] >= LONG_SECONDS
    if rule == "loud_second":
        return threshold is not None and gap["second_below_song_db"] >= threshold
    raise GapsError(f"unknown rule: {rule}")


def choose_threshold(gaps: Sequence[Mapping[str, Any]]) -> float | None:
    """T of ``loud_second``: the lowest level that is reliable on these gaps, or None."""
    for level in sorted({gap["second_below_song_db"] for gap in gaps}):
        said = [is_dropout(gap) for gap in gaps if gap["second_below_song_db"] >= level]
        if len(said) < MIN_FLAGGED:
            return None  # a higher level flags fewer still
        if sum(said) / len(said) >= BAR:
            return level
    return None


def nearest_level(gaps: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Of the levels that flag ``MIN_FLAGGED`` gaps or more, the one whose flagged gaps are
    most often dropouts (the lowest such level, if several). None if no level flags enough.
    """
    best: dict[str, Any] | None = None
    for level in sorted({gap["second_below_song_db"] for gap in gaps}):
        said = [is_dropout(gap) for gap in gaps if gap["second_below_song_db"] >= level]
        if len(said) < MIN_FLAGGED:
            break
        share = sum(said) / len(said)
        if best is None or share > best["share"]:
            best = {"level": level, "flagged": len(said), "right": sum(said), "share": share}
    return best


def outcome(
    gaps: Sequence[Mapping[str, Any]], rule: str, threshold: float | None = None
) -> dict[str, Any]:
    """How ``rule`` does on ``gaps``: what it flags, and how many of those are dropouts."""
    said = [is_dropout(gap) for gap in gaps if flags(rule, gap, threshold)]
    dropouts = sum(1 for gap in gaps if is_dropout(gap))
    right = sum(said)
    precision = right / len(said) if said else None
    return {
        "gaps": len(gaps),
        "dropouts": dropouts,
        "flagged": len(said),
        "right": right,
        "precision": precision,
        "recall": right / dropouts if dropouts else None,
        "reliable": len(said) >= MIN_FLAGGED and precision is not None and precision >= BAR,
    }


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def analyse(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the report shows, from the collected rows."""
    gaps: dict[str, list[dict[str, Any]]] = {}
    stems: dict[str, dict[str, dict[str, Any]]] = {}
    for name in SETS:
        mine = [row for row in rows if row["set"] == name]
        gaps[name] = [
            {**gap, "stem": row["stem"], "track_id": row["track_id"]}
            for row in mine
            if not row["noted"]
            for gap in row["gaps"]
        ]
        stems[name] = {}
        for stem in PRODUCT_STEMS:
            of_stem = [row for row in mine if row["stem"] == stem]
            if not of_stem:
                continue
            counted = [row for row in of_stem if not row["noted"]]
            theirs = [gap for gap in gaps[name] if gap["stem"] == stem]
            stems[name][stem] = {
                "stems": len(of_stem),
                "noted": len(of_stem) - len(counted),
                "with_a_gap": sum(1 for row in counted if row["gaps"]),
                "with_a_dropout": sum(
                    1 for row in counted if any(is_dropout(gap) for gap in row["gaps"])
                ),
                "gaps": len(theirs),
                "dropouts": sum(1 for gap in theirs if is_dropout(gap)),
            }
    threshold = choose_threshold(gaps["tuning"])
    outcomes = {
        name: {rule: outcome(gaps[name], rule, threshold) for rule in RULES} for name in SETS
    }
    reliable = [rule for rule in RULES if outcomes["tuning"][rule]["reliable"]]
    # max() keeps the first of equals, which is the earlier rule in the list.
    tried = max(reliable, key=lambda rule: outcomes["tuning"][rule]["right"]) if reliable else None
    if tried is None:
        verdict = "none"
    else:
        verdict = "stands" if outcomes["test"][tried]["reliable"] else "fails"
    second: dict[str, dict[str, Any]] = {}
    for name in SETS:
        dropped = [gap for gap in gaps[name] if is_dropout(gap)]
        rests = [gap for gap in gaps[name] if not is_dropout(gap)]
        second[name] = {
            "dropouts": len(dropped),
            "speaks_in_dropouts": sum(1 for gap in dropped if flags("second", gap)),
            "share": _median([gap["second_share"] for gap in dropped if "second_share" in gap]),
            "level_in_dropouts": _median([gap["second_below_song_db"] for gap in dropped]),
            "rests": len(rests),
            "speaks_in_rests": sum(1 for gap in rests if flags("second", gap)),
            "level_in_rests": _median([gap["second_below_song_db"] for gap in rests]),
        }
    return {
        "tracks": {
            name: len({row["track_id"] for row in rows if row["set"] == name}) for name in SETS
        },
        "stems": stems,
        "threshold": threshold,
        "nearest_level": nearest_level(gaps["tuning"]),
        "songs_with_a_dropout": {
            name: len({gap["track_id"] for gap in gaps[name] if is_dropout(gap)}) for name in SETS
        },
        "noted_with_a_dropout": {
            name: sum(
                1
                for row in rows
                if row["set"] == name
                and row["noted"]
                and any(is_dropout(gap) for gap in row["gaps"])
            )
            for name in SETS
        },
        "outcomes": outcomes,
        "reliable_on_tuning": reliable,
        "tried": tried,
        "verdict": verdict,
        "second": second,
    }


# -- the report ----------------------------------------------------------------------------

_RULE_TEXT = {
    "silence": "every gap",
    "second": "gaps in which the second model's stem sounds",
    "loud_second": "gaps over which the second model's stem is loud against the song",
}
_VERDICT = {
    "none": (
        "**No rule is reliable on the tuning tracks, so none was tried on the test tracks: "
        "nothing here shows that the product can tell a dropout from a rest.**"
    ),
    "fails": (
        "**The rule that was reliable on the tuning tracks is not reliable on the test "
        "tracks: nothing here shows that the product can tell a dropout from a rest.**"
    ),
    "stands": (
        "**The rule stands: it was reliable on the tuning tracks and, tried once, on the "
        "test tracks too.** A later phase may build a note on it."
    ),
}


def why(result: Mapping[str, Any]) -> str:
    """Whether a rule is reliable on a list and, if it is not, which half of the bar it missed."""
    if result["reliable"]:
        return "yes"
    return "no: too few flagged" if result["flagged"] < MIN_FLAGGED else "no: wrong too often"


def _percent(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{round(100 * value, digits) + 0.0:.{digits}f}%"


def _db(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} dB"


def _few(analysis: Mapping[str, Any]) -> list[str]:
    """How few dropouts there are to go by, and what the gate's note already covers."""
    songs = analysis.get("songs_with_a_dropout")
    if songs is None:
        return []
    found = {name: analysis["outcomes"][name]["silence"]["dropouts"] for name in SETS}
    lines = [
        f"- There are few dropouts to go by, and they come in clusters: {found['tuning']} on "
        f"the tuning tracks, in {songs['tuning']} songs, and {found['test']} on the test "
        f"tracks, in {songs['test']}. Gaps in one song are not independent of one another."
    ]
    # The most gaps a rule can flag and still have BAR of them be dropouts.
    most = int(found["tuning"] / BAR + 1e-9)
    if most < MIN_FLAGGED:
        lines.append(
            f"- With {found['tuning']} dropouts on the tuning tracks the bar could not be "
            f"met there by any rule: {MIN_FLAGGED} flagged gaps cannot be {_percent(BAR)} "
            "dropouts. That is a fault of the bar, which was set before the count."
        )
    else:
        span = f"exactly {most}" if most == MIN_FLAGGED else f"from {MIN_FLAGGED} to {most}"
        lines.append(
            f"- With {found['tuning']} dropouts on the tuning tracks, a rule can clear the "
            f"bar there only by flagging {span} gaps. The bar was set before the count; the "
            "rules above are nowhere near it."
        )
    noted = analysis["noted_with_a_dropout"]
    lines.append(
        "- A stem that carries the gate's note is not counted. Of the stems left out that "
        f"way, {noted['tuning']} on the tuning tracks and {noted['test']} on the test tracks "
        "hold a stretch that would have been a dropout."
    )
    return lines


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    models = run.get("models", {})
    left_out = run.get("left_out", [])
    covered = ", ".join(f"{stem} (`{pair[0]}`, then `{pair[1]}`)" for stem, pair in models.items())
    git = run.get("git") or {}
    at = ""
    if RULES_COMMIT is not None:
        at = f" and committed (`{RULES_COMMIT[:7]}`)"
    lines = [
        "# Can the product tell a dropout from a rest?",
        "",
        f"Written by `python -m stems_bench.gaps report` from the run `{run.get('run_id', '?')}`"
        + (f", made at commit `{git['commit'][:7]}`" if git.get("commit") else "")
        + (", with uncommitted changes" if git.get("dirty") else "")
        + ". No model was run: these are the benchmark's kept outputs for the "
        f"{analysis['tracks']['tuning']} tuning tracks and the {analysis['tracks']['test']} "
        "test tracks of MoisesDB, read beside the true stems. The definitions, the three "
        f"rules and what would be done with the result were written into "
        f"`bench/stems_bench/gaps.py`{at} before any kept output was read for this.",
        "",
        "A stem can be silent for a long stretch while its instrument plays "
        "(`docs/dropouts.md`, `docs/what-is-lost.md`), and the product says nothing. It "
        "has no true stem to know it by. It has the stem it made, the song, and the second "
        "model its plan names for the stem. Can it tell from those?",
        "",
        f"Counted: {covered or 'no stem'} (the default model, then the second)."
        + (
            f" Not counted, because the benchmark did not keep both models' outputs for "
            f"every track: {', '.join(left_out)}."
            if left_out
            else ""
        ),
        "",
        "## The answer",
        "",
        _VERDICT[analysis["verdict"]],
        "",
        f"A **gap** is an unbroken stretch in which the default model's stem is silent "
        f"(under {SILENCE_DBFS:g} dBFS) while the song sounds for {LONG_SECONDS:g} seconds "
        "or more. It is a **dropout** when the true stem sounds for that long inside it, "
        "and a **rest** when it does not. A stem that already carries the gate's note for "
        "the whole song has no gaps counted.",
        "",
        "## The gaps",
        "",
        "| List | Stem | Stems | Noted already | With a gap | With a dropout | Gaps | "
        "Dropouts | Rests |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SETS:
        for stem, entry in analysis["stems"][name].items():
            lines.append(
                f"| {name} | {stem} | {entry['stems']} | {entry['noted']} | "
                f"{entry['with_a_gap']} | {entry['with_a_dropout']} | {entry['gaps']} | "
                f"{entry['dropouts']} | {entry['gaps'] - entry['dropouts']} |"
            )
    threshold = analysis["threshold"]
    nearest = analysis.get("nearest_level")
    lines += [
        "",
        "## The three rules",
        "",
        f"A rule is reliable on a list when it flags {MIN_FLAGGED} gaps or more and at least "
        f"{_percent(BAR)} of them are dropouts. For the third rule the level was looked for "
        "on the tuning tracks: "
        + (
            f"{threshold:.2f} dB against the song."
            if threshold is not None
            else "no level is reliable there, so the rule flags nothing."
        )
        + (
            f" The nearest any level comes: at {nearest['level']:.2f} dB it flags "
            f"{nearest['flagged']} gaps, of which {nearest['right']} are dropouts "
            f"({_percent(nearest['share'])})."
            if threshold is None and nearest is not None
            else ""
        ),
        "",
        "| Rule | Flags | List | Flagged | Dropouts among them | Share | "
        "Of all dropouts | Reliable |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for rule in RULES:
        for name in SETS:
            result = analysis["outcomes"][name][rule]
            no_level = rule == "loud_second" and threshold is None
            lines.append(
                f"| `{rule}` | {_RULE_TEXT[rule]} | {name} | {result['flagged']} | "
                f"{result['right']} | {_percent(result['precision'])} | "
                f"{result['right']} of {result['dropouts']} | "
                f"{'no: no level found' if no_level else why(result)} |"
            )
    tried = analysis["tried"]
    lines += [
        "",
        (
            f"Tried on the test tracks: `{tried}`. The other rules' rows for the test tracks "
            "are for information and decide nothing."
            if tried
            else "No rule was tried on the test tracks; their rows are for information and "
            "decide nothing."
        ),
        "",
        "## What the second model has in a gap",
        "",
        "For information. Its stem sounds inside a gap when it does so for "
        f"{LONG_SECONDS:g} seconds or more; its level is against the song, over the gap; the "
        "share is how much of the true stem it holds over a dropout "
        "(`<true, second> / <true, true>`, as in `docs/what-is-lost.md`).",
        "",
        "| List | Dropouts | It sounds in | Its level there (median) | Share of the true "
        "stem (median) | Rests | It sounds in | Its level there (median) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SETS:
        entry = analysis["second"][name]
        lines.append(
            f"| {name} | {entry['dropouts']} | {entry['speaks_in_dropouts']} | "
            f"{_db(entry['level_in_dropouts'])} | {_percent(entry['share'])} | "
            f"{entry['rests']} | {entry['speaks_in_rests']} | {_db(entry['level_in_rests'])} |"
        )
    lines += [
        "",
        "## What this is and is not",
        "",
        f"- {len(models)} of the six stems, one dataset, one default model and one second "
        "model for each stem. It says nothing about the stems that are not counted.",
        "- The benchmark's outputs, not a job's. `docs/product-vs-benchmark.md` compares the two.",
        "- A rest may still hold some of the instrument: up to "
        f"{LONG_SECONDS:g} seconds of it, by the definition.",
        *_few(analysis),
        "- Nothing in the product was changed.",
        "",
    ]
    return "\n".join(lines)


# -- collecting ----------------------------------------------------------------------------


def index(bench_dir: Path, dataset: str) -> dict[tuple[str, str, str], Path]:
    """(model, track, stem) -> the kept output, the first one if a job was kept twice."""
    found: dict[tuple[str, str, str], Path] = {}
    for item in kept_outputs(bench_dir, dataset):
        found.setdefault((item["model_id"], item["track_id"], item["stem"]), item["path"])
    return found


def covered_stems(
    indexes: Mapping[str, Mapping[tuple[str, str, str], Path]],
    track_ids: Mapping[str, Sequence[str]],
    models: Mapping[str, tuple[str, str]],
) -> list[str]:
    """The stems whose two models both have a kept output for every track of every list."""
    return [
        stem
        for stem, pair in models.items()
        if all(
            (model, track_id, stem) in indexes[name]
            for name in indexes
            for track_id in track_ids[name]
            for model in pair
        )
    ]


def collect(
    set_name: str,
    dataset: Any,
    models: Mapping[str, tuple[str, str]],
    out_path: Path,
    *,
    silent_fraction_min: float,
    empty_below_mix_db: float | None,
    bench_dir: Path = BENCH_DIR,
    progress: Callable[[str], None] | None = None,
) -> int:
    """A row for every stem in ``models`` of every track of one list. Returns how many
    were new; rows already in ``out_path`` are not made again.
    """
    done = {row["id"] for row in read_rows(out_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = index(bench_dir, dataset.name)
    track_ids = dataset.track_ids()
    added = 0
    for number, track_id in enumerate(track_ids, start=1):
        wanted = [stem for stem in models if f"{set_name}/{track_id}/{stem}" not in done]
        if wanted:
            track = dataset.load(track_id)
            rate, mix = track.sample_rate, track.inputs["mix"]
            for stem in wanted:
                default, fallback = models[stem]
                made = np.load(paths[(default, track_id, stem)], mmap_mode="r")
                second = np.load(paths[(fallback, track_id, stem)], mmap_mode="r")
                true = track.references[stem]
                row = {
                    "id": f"{set_name}/{track_id}/{stem}",
                    "set": set_name,
                    "track_id": track_id,
                    "stem": stem,
                    "seconds": round(mix.shape[-1] / rate, 1),
                    "noted": is_noted(
                        made,
                        mix,
                        rate,
                        silent_fraction_min=silent_fraction_min,
                        empty_below_mix_db=empty_below_mix_db,
                    ),
                    "gaps": measure_stem(mix, true, made, second, rate),
                }
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                added += 1
        if progress:
            progress(f"{set_name}: {number} of {len(track_ids)} tracks")
    return added


def plan_models() -> dict[str, tuple[str, str]]:
    """Stem -> (the default model, the fallback the plan names for it), from the config."""
    from stems_common.settings import load_defaults
    from stems_models.registry import load_registry
    from stems_worker.decision.plan import NONE, config_fallback

    defaults, registry = load_defaults(), load_registry()
    pairs = {
        stem: (
            defaults.ensembles.slots[stem].single_best,
            config_fallback(stem, defaults, registry),
        )
        for stem in PRODUCT_STEMS
    }
    return {stem: pair for stem, pair in pairs.items() if NONE not in pair}


def cmd_collect(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset
    from stems_common.settings import load_defaults

    run_dir = GAPS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rules = load_defaults().gate.rules
    datasets = {name: get_dataset(dataset_name) for name, dataset_name in SETS.items()}
    pairs = plan_models()
    indexes = {name: index(BENCH_DIR, dataset.name) for name, dataset in datasets.items()}
    track_ids = {name: dataset.track_ids() for name, dataset in datasets.items()}
    stems = covered_stems(indexes, track_ids, pairs)
    if not stems:
        raise GapsError(
            "no stem has both models' outputs kept for every track: run the benchmark first"
        )
    models = {stem: pairs[stem] for stem in stems}
    status_path = run_dir / "status.json"
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(status_path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    try:
        for name, dataset in datasets.items():
            collect(
                name,
                dataset,
                models,
                run_dir / "rows.jsonl",
                silent_fraction_min=rules.silent_fraction_min,
                empty_below_mix_db=rules.empty_below_mix_db,
                progress=progress,
            )
    except BaseException as error:
        write_json(
            status_path, {"state": "failed", "started": started, "error": type(error).__name__}
        )
        raise
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "finished": finished,
            "sets": SETS,
            "models": {stem: list(pair) for stem, pair in models.items()},
            "left_out": [stem for stem in PRODUCT_STEMS if stem not in models],
            "silent_fraction_min": rules.silent_fraction_min,
            "empty_below_mix_db": rules.empty_below_mix_db,
            "git": git_state(),
        },
    )
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = GAPS_DIR / args.run_id
    status_path = run_dir / "status.json"
    state = (
        json.loads(status_path.read_text(encoding="utf-8")).get("state")
        if status_path.is_file()
        else None
    )
    if state != "done":
        print(
            f"error: the run {args.run_id} has not finished (state: {state}): run "
            "`python -m stems_bench.gaps collect` first",
            file=sys.stderr,
        )
        return 2
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    text = render(analyse(read_rows(run_dir / "rows.jsonl")), run)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.gaps",
        description="Can the product tell a dropout from a rest? Counted on kept outputs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="read the kept outputs (resumable)")
    collect_parser.add_argument("--run-id", default="gaps-v1")
    collect_parser.set_defaults(func=cmd_collect)
    report_parser = sub.add_parser("report", help="write the report from a finished run")
    report_parser.add_argument("--run-id", default="gaps-v1")
    report_parser.add_argument("--out", help="write the report here (default: print it)")
    report_parser.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GapsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
