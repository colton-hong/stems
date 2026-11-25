"""The two ways of making ``other``, joined (Phase 20).

    python -m stems_bench.other_joined collect   # the gaps of the default's own (resumable)
    python -m stems_bench.other_joined join      # choose the level, join, score (resumable)
    python -m stems_bench.other_joined report --out docs/other-joined.md

The default model's own ``other`` goes silent for a sixth of the time the instrument
plays; ``other`` made by subtraction (a four-stem model's ``other`` less the guitar and
the piano) hardly ever does, scores lower, and is not empty when a song has no ``other``
(``docs/other-by-subtraction.md``). They fail in opposite directions, so the obvious
thing to try is the first wherever it sounds and the second where the first is silent,
but only where the second is loud, because where a song has no ``other`` the second is
leftovers.

**The rule**, fixed here before the default model's kept ``other`` is read for this:

- *Own* is the default model's ``other``. *Subtracted* is the four-stem ``other`` of the
  model the plan names as ``other``'s fallback, less the default model's guitar and
  piano, as ``stems_worker.compose`` subtracts them.
- A *gap* is as in ``stems_bench.gaps``: an unbroken stretch of tenths of a second in
  which own is silent and inside which the song sounds for ``LONG_SECONDS`` or more. It
  is a *dropout* when the true ``other`` sounds for ``LONG_SECONDS`` or more inside it
  and a *rest* otherwise. Gaps are counted in every song, also where own carries the
  gate's note for the whole song: a song with no ``other`` is one long rest, and it is
  the case the join must not get wrong.
- The *joined* stem is own, except in every gap over which the subtracted stem is at
  least T dB against the song (its level less the song's, both over the gap): there it
  is the subtracted stem. The switch is made on the tenth of a second, with no fade; a
  product would fade, and the count does not need to.
- T is chosen on the tuning tracks by the rule of ``stems_bench.gaps``: the lowest level
  at which ``MIN_FLAGGED`` gaps or more are filled and at least ``BAR`` of those filled
  are dropouts. If no level does that, nothing is joined.
- **What is done with the result.** With T from the tuning tracks the join is tried on
  the test tracks, once. It *stands* if it is reliable there too: ``MIN_FLAGGED`` gaps or
  more filled, at least ``BAR`` of them dropouts. Then it is put to the owner under
  BLOCKED.md D15 as a further way of making ``other``, and a later phase may build it.
  If no T is found, or the join is not reliable on the test tracks, the report says so.
  Nothing in the product changes in this phase either way.
- Shown beside the verdict, for own, subtracted and joined, on each list: where the song
  has an ``other``, the dropped time and the stems with a long dropout, and the SDR of
  the benchmark's protocol with the number of one-second windows it could score (it
  cannot score a window in which the estimate is silent, which is how a stem that drops
  out keeps its score); where the song has none, the stem's level against the song and
  whether the product's rule would note it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench import gaps as gap_rules
from stems_bench.dropouts import LONG_SECONDS, measure
from stems_bench.dropouts_by_model import index, tally
from stems_bench.gate_outputs import IN_SONG_SECONDS, read_rows
from stems_bench.other_made import TAKEN_OUT, absent, four_stem_others
from stems_bench.run import BENCH_DIR, git_state, write_json
from stems_common.levels import SILENCE_WINDOW_SECONDS, level_below_mix_db, silence_fraction
from stems_common.paths import RUNS_DIR

OUT_DIR = RUNS_DIR / "other-joined"
SETS = {"tuning": "moisesdb_tuning", "test": "moisesdb_subset"}
VERSIONS = ("own", "subtracted", "joined")
BAR = gap_rules.BAR
MIN_FLAGGED = gap_rules.MIN_FLAGGED
RULE = "loud_second"  # the rule of stems_bench.gaps that the join is: fill where it is loud
# The commit that holds everything above, made before the default's kept `other` was read
# for this.
RULES_COMMIT: str | None = "5c539a1d25f15fbbb1e409e83cb83a10e01683bc"


class JoinError(ValueError):
    """The join cannot be made or reported as asked."""


# -- one song ------------------------------------------------------------------------------


def note(made: np.ndarray, mix: np.ndarray, rate: int, rules: Mapping[str, Any]) -> dict[str, Any]:
    """A stem's level against the song, and whether the product's rule would note it."""
    below = level_below_mix_db(made, mix)
    limit = rules["empty_below_mix_db"]
    noted = silence_fraction(made, rate) >= rules["silent_fraction_min"] or (
        limit is not None and below <= limit
    )
    return {"below_song_db": round(below, 2), "noted": bool(noted)}


def join(
    own: np.ndarray, subtracted: np.ndarray, found: Sequence[Mapping[str, Any]], rate: int
) -> np.ndarray:
    """Own, with the subtracted stem in each of the gaps given."""
    width = max(1, round(SILENCE_WINDOW_SECONDS * rate))
    joined = np.array(own, dtype=np.float32)
    for gap in found:
        low = round(gap["start"] / SILENCE_WINDOW_SECONDS) * width
        high = min(low + round(gap["seconds"] / SILENCE_WINDOW_SECONDS) * width, own.shape[-1])
        joined[..., low:high] = subtracted[..., low:high]
    return joined


def load_song(
    dataset: Any,
    track_id: str,
    kept: Mapping[tuple[str, str, str], Mapping[str, Any]],
    others: Mapping[tuple[str, str], Path],
    default_model: str,
    sub_model: str,
) -> dict[str, Any]:
    """The song, its true ``other``, the default's own and the subtracted one."""
    track = dataset.load(track_id)
    own = np.asarray(np.load(kept[(default_model, track_id, "other")]["path"]), dtype=np.float32)
    subtracted = np.load(others[(sub_model, track_id)]).astype(np.float32)
    for stem in TAKEN_OUT:
        subtracted = subtracted - np.load(kept[(default_model, track_id, stem)]["path"])
    return {
        "rate": track.sample_rate,
        "mix": track.inputs["mix"],
        "true": track.references["other"],
        "own": own,
        "subtracted": subtracted,
    }


def ready(
    dataset: Any,
    kept: Mapping[tuple[str, str, str], Any],
    others: Mapping[tuple[str, str], Any],
    default_model: str,
    sub_model: str,
) -> list[str]:
    """What is missing among the kept outputs for this list; empty when nothing is."""
    missing = []
    for track_id in dataset.track_ids():
        for stem in ("other", *TAKEN_OUT):
            if (default_model, track_id, stem) not in kept:
                missing.append(f"{default_model}'s {stem} of {track_id}")
        if (sub_model, track_id) not in others:
            missing.append(f"{sub_model}'s four-stem other of {track_id}")
    return missing


# -- collecting ----------------------------------------------------------------------------


def collect(
    set_name: str,
    dataset: Any,
    default_model: str,
    sub_model: str,
    out_path: Path,
    *,
    rules: Mapping[str, Any],
    bench_dir: Path = BENCH_DIR,
    progress: Callable[[str], None] | None = None,
) -> int:
    """A row a track: own and subtracted counted as they are, and the gaps of own with the
    subtracted stem's level over each. Returns how many rows were new.
    """
    done = {row["id"] for row in read_rows(out_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, others = index(bench_dir, dataset.name), four_stem_others(bench_dir, dataset.name)
    missing = ready(dataset, kept, others, default_model, sub_model)
    if missing:
        raise JoinError(f"not kept: {missing[0]} (and {len(missing) - 1} more)")
    track_ids = dataset.track_ids()
    added = 0
    for number, track_id in enumerate(track_ids, start=1):
        row_id = f"{set_name}/{track_id}"
        if row_id not in done:
            song = load_song(dataset, track_id, kept, others, default_model, sub_model)
            rate, mix, true = song["rate"], song["mix"], song["true"]
            row: dict[str, Any] = {"id": row_id, "set": set_name, "track_id": track_id}
            for version in ("own", "subtracted"):
                counted = measure(true, song[version], rate)
                row["sounding"] = counted["sounding"]
                row[version] = {
                    "dropped": counted["dropped"],
                    "longest": counted["longest"],
                    **note(song[version], mix, rate, rules),
                }
            row["gaps"] = gap_rules.measure_stem(mix, true, song["own"], song["subtracted"], rate)
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            added += 1
        if progress:
            progress(f"{set_name}: gaps of {number} of {len(track_ids)} tracks")
    return added


def gaps_of(rows: Sequence[Mapping[str, Any]], set_name: str) -> list[dict[str, Any]]:
    return [
        {**gap, "track_id": row["track_id"]}
        for row in rows
        if row["set"] == set_name
        for gap in row["gaps"]
    ]


def choose_level(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """T: chosen on the tuning tracks' gaps, by the rule of ``stems_bench.gaps``."""
    return gap_rules.choose_threshold(gaps_of(rows, "tuning"))


def filled(row: Mapping[str, Any], level: float | None) -> list[dict[str, Any]]:
    """The gaps of one song that the join fills at this level."""
    return [gap for gap in row["gaps"] if gap_rules.flags(RULE, gap, level)]


def join_songs(
    set_name: str,
    dataset: Any,
    default_model: str,
    sub_model: str,
    rows: Sequence[Mapping[str, Any]],
    level: float,
    out_path: Path,
    *,
    rules: Mapping[str, Any],
    bench_dir: Path = BENCH_DIR,
    progress: Callable[[str], None] | None = None,
    score: Callable[..., Any] | None = None,
) -> int:
    """Join every song of one list at ``level`` and count the result; score all three
    versions by the benchmark's protocol where the song has an ``other``. A row a track;
    rows made at another level are not reused. Returns how many rows were new.
    """
    if score is None:
        from stems_bench.metrics import bss_eval_stem as score

    done = {row["id"] for row in read_rows(out_path) if row.get("level") == level}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, others = index(bench_dir, dataset.name), four_stem_others(bench_dir, dataset.name)
    mine = [row for row in rows if row["set"] == set_name]
    added = 0
    for number, row in enumerate(mine, start=1):
        if row["id"] not in done:
            song = load_song(dataset, row["track_id"], kept, others, default_model, sub_model)
            rate, mix, true = song["rate"], song["mix"], song["true"]
            fill = filled(row, level)
            song["joined"] = join(song["own"], song["subtracted"], fill, rate)
            counted = measure(true, song["joined"], rate)
            result: dict[str, Any] = {
                "id": row["id"],
                "set": set_name,
                "track_id": row["track_id"],
                "level": level,
                "filled": len(fill),
                "joined": {
                    "dropped": counted["dropped"],
                    "longest": counted["longest"],
                    **note(song["joined"], mix, rate, rules),
                },
                "scores": {},
            }
            if row["sounding"] >= IN_SONG_SECONDS:
                for version in VERSIONS:
                    scored = score(true, song[version], mix, rate)
                    result["scores"][version] = {
                        "sdr": None if not np.isfinite(scored.sdr) else round(float(scored.sdr), 3),
                        "scored_windows": int(scored.scored_windows),
                        "windows": int(scored.windows),
                    }
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            added += 1
        if progress:
            progress(f"{set_name}: joined {number} of {len(mine)} tracks")
    return added


# -- analysis (pure) -----------------------------------------------------------------------


def _version_rows(
    rows: Sequence[Mapping[str, Any]], joined: Mapping[str, Mapping[str, Any]], set_name: str
) -> dict[str, dict[str, Any]]:
    """Version -> its tally where the song has an ``other``, its level where it has none."""
    out: dict[str, dict[str, Any]] = {}
    for version in VERSIONS:
        present, missing, scored, windows = [], [], 0, 0
        for row in rows:
            if row["set"] != set_name:
                continue
            extra = joined.get(row["id"])
            counted = row.get(version) if version != "joined" else (extra or {}).get("joined")
            if counted is None:
                continue
            if row["sounding"] >= IN_SONG_SECONDS:
                score = ((extra or {}).get("scores") or {}).get(version) or {}
                present.append({"sounding": row["sounding"], **counted, "sdr": score.get("sdr")})
                scored += score.get("scored_windows", 0)
                windows += score.get("windows", 0)
            else:
                missing.append(counted)
        out[version] = {
            "present": tally(present),
            "absent": absent(missing),
            "scored_windows": scored,
            "windows": windows,
        }
    return out


def analyse(
    rows: Sequence[Mapping[str, Any]], joined_rows: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Everything the report shows, from the gaps and, if a level was found, the joins."""
    level = choose_level(rows)
    joined = {row["id"]: row for row in joined_rows if row.get("level") == level}
    outcomes = {name: gap_rules.outcome(gaps_of(rows, name), RULE, level) for name in SETS}
    held = "stands" if outcomes["test"]["reliable"] else "fails"
    verdict = "none" if level is None else held
    return {
        "tracks": {name: sum(1 for row in rows if row["set"] == name) for name in SETS},
        "level": level,
        "nearest": gap_rules.nearest_level(gaps_of(rows, "tuning")),
        "outcomes": outcomes,
        "verdict": verdict,
        "joined_tracks": {
            name: sum(1 for row in rows if row["set"] == name and row["id"] in joined)
            for name in SETS
        },
        "versions": {name: _version_rows(rows, joined, name) for name in SETS},
    }


# -- the report ----------------------------------------------------------------------------

_VERDICT = {
    "none": (
        "**No level is reliable on the tuning tracks, so nothing was joined: the subtracted "
        "stem's loudness does not tell a dropout of the default's own `other` from a rest.**"
    ),
    "fails": (
        "**The level that was reliable on the tuning tracks is not reliable on the test "
        "tracks: the join does not stand.**"
    ),
    "stands": (
        "**The join stands: at the level chosen on the tuning tracks it is reliable on the "
        "test tracks too.** It goes to the owner under BLOCKED.md D15; nothing is built yet."
    ),
}
_NAMES = {
    "own": "the default's own",
    "subtracted": "made by subtraction",
    "joined": "the two joined",
}


def _percent(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{round(100 * value, digits) + 0.0:.{digits}f}%"


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    git = run.get("git") or {}
    level = analysis["level"]
    lines = [
        "# The two ways of making `other`, joined",
        "",
        "Written by `python -m stems_bench.other_joined report` from the run "
        f"`{run.get('run_id', '?')}`"
        + (f", made at commit `{git['commit'][:7]}`" if git.get("commit") else "")
        + (", with uncommitted changes" if git.get("dirty") else "")
        + ". The stems are the benchmark's kept outputs, read beside the true stems. The "
        "rule for joining, its bar and what each outcome leads to were written into "
        "`bench/stems_bench/other_joined.py`"
        + (f" and committed (`{RULES_COMMIT[:7]}`)" if RULES_COMMIT else "")
        + " before the default model's kept `other` was read for this.",
        "",
        f"**Own** is `{run.get('default_model', '?')}`'s `other`. **Subtracted** is "
        f"`{run.get('sub_model', '?')}`'s four-stem `other` less the default's guitar and "
        "piano. A **gap** is an unbroken stretch in which own is silent while the song "
        f"sounds for {LONG_SECONDS:g} seconds or more; it is a **dropout** when the true "
        "`other` sounds for that long inside it, and a **rest** otherwise. The **joined** "
        "stem is own, with the subtracted stem put into every gap over which the "
        "subtracted stem is at least T dB against the song.",
        "",
        "## The answer",
        "",
        _VERDICT[analysis["verdict"]],
        "",
        f"A level is reliable on a list when it fills {MIN_FLAGGED} gaps or more and at least "
        f"{_percent(BAR, 0)} of the gaps it fills are dropouts. On the tuning tracks: "
        + (
            f"T = {level:.2f} dB."
            if level is not None
            else "no level is."
            + (
                f" The nearest: at {analysis['nearest']['level']:.2f} dB, "
                f"{analysis['nearest']['flagged']} gaps filled, "
                f"{analysis['nearest']['right']} of them dropouts "
                f"({_percent(analysis['nearest']['share'], 0)})."
                if analysis.get("nearest")
                else ""
            )
        ),
        "",
        "| List | Tracks | Gaps | Dropouts | Filled | Dropouts among them | Share | "
        "Of all dropouts | Reliable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in SETS:
        result = analysis["outcomes"][name]
        lines.append(
            f"| {name} | {analysis['tracks'][name]} | {result['gaps']} | {result['dropouts']} | "
            f"{result['flagged']} | {result['right']} | {_percent(result['precision'], 0)} | "
            f"{result['right']} of {result['dropouts']} | "
            f"{'no: no level found' if level is None else gap_rules.why(result)} |"
        )
    lines += [
        "",
        "## What each way of making `other` does",
        "",
        "Where the song has an `other`: the time its stem is silent while the true one "
        "sounds, and the SDR of the benchmark's protocol. That protocol cannot score a "
        "one-second window in which the estimate is silent, so the column after it says how "
        "many windows the SDR stands on. Where the song has none: how far the stem is under "
        "the song, and whether the product's rule would note it as silent or very quiet.",
        "",
        "| List | How `other` is made | Songs with an `other` | Dropped | With a long dropout | "
        "SDR (dB, median) | Windows scored | Songs with none | There: dB under the song "
        "(median) | There: noted as empty |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SETS:
        for version in VERSIONS:
            entry = analysis["versions"][name][version]
            present, gone = entry["present"], entry["absent"]
            if version == "joined" and not analysis["joined_tracks"][name]:
                continue
            under = gone["below_song_db"]
            windows = (
                f"{entry['scored_windows']} of {entry['windows']}" if entry["windows"] else "-"
            )
            lines.append(
                f"| {name} | {_NAMES[version]} | {present['stems']} | "
                f"{_percent(present['share'])} | {present['long']} | {_number(present['sdr'])} | "
                f"{windows} | {gone['songs']} | {'-' if under is None else f'{-under:.1f}'} | "
                f"{gone['noted']} of {gone['songs']} |"
            )
    if level is None:
        lines += [
            "",
            "No stem was scored in this run: scoring is part of joining, and nothing was "
            "joined. `docs/other-by-subtraction.md` has the SDR of the two ways.",
        ]
    lines += [
        "",
        "## What this is and is not",
        "",
        "- One dataset, one default model, one four-stem model. The level is chosen on "
        "the tuning tracks and tried on the test tracks once.",
        "- The join switches on the tenth of a second with no fade, which a product would "
        "not do; what is counted here does not depend on it.",
        "- A higher SDR for a stem that drops out is partly the protocol's doing: the "
        "windows it is silent in are not scored. The windows column is there for that.",
        "- Nothing in the product was changed. The choice this bears on is BLOCKED.md D15.",
        "",
    ]
    return "\n".join(lines)


# -- commands ------------------------------------------------------------------------------


def plan() -> tuple[str, str, dict[str, Any]]:
    """From the config: the default model for ``other``, the four-stem model the plan names
    as its fallback, and the gate's two levels.
    """
    from stems_common.settings import load_defaults
    from stems_models.registry import load_registry
    from stems_worker.decision.plan import config_fallback

    defaults = load_defaults()
    rules = defaults.gate.rules
    return (
        defaults.ensembles.slots["other"].single_best,
        config_fallback("other", defaults, load_registry()),
        {
            "silent_fraction_min": rules.silent_fraction_min,
            "empty_below_mix_db": rules.empty_below_mix_db,
        },
    )


def _status(run_dir: Path, name: str) -> tuple[Path, str, Callable[[str], None]]:
    path = run_dir / name
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    return path, started, progress


def cmd_collect(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset

    run_dir = OUT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    default_model, sub_model, rules = plan()
    status_path, started, progress = _status(run_dir, "status.json")
    try:
        for name, dataset_name in SETS.items():
            collect(
                name,
                get_dataset(dataset_name),
                default_model,
                sub_model,
                run_dir / "rows.jsonl",
                rules=rules,
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
            "default_model": default_model,
            "sub_model": sub_model,
            **rules,
            "git": git_state(),
        },
    )
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    return 0


def _finished(run_dir: Path, name: str) -> bool:
    path = run_dir / name
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("state") == "done"


def cmd_join(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset

    run_dir = OUT_DIR / args.run_id
    if not _finished(run_dir, "status.json"):
        raise JoinError(f"the run {args.run_id} has no finished gaps: run `collect` first")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = read_rows(run_dir / "rows.jsonl")
    level = choose_level(rows)
    status_path, started, progress = _status(run_dir, "join-status.json")
    if level is None:
        print("no level is reliable on the tuning tracks: nothing is joined")
    else:
        rules = {key: run[key] for key in ("silent_fraction_min", "empty_below_mix_db")}
        try:
            for name, dataset_name in SETS.items():
                join_songs(
                    name,
                    get_dataset(dataset_name),
                    run["default_model"],
                    run["sub_model"],
                    rows,
                    level,
                    run_dir / "joined.jsonl",
                    rules=rules,
                    progress=progress,
                )
        except BaseException as error:
            write_json(
                status_path, {"state": "failed", "started": started, "error": type(error).__name__}
            )
            raise
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(
        status_path, {"state": "done", "started": started, "finished": finished, "level": level}
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = OUT_DIR / args.run_id
    if not (_finished(run_dir, "status.json") and _finished(run_dir, "join-status.json")):
        print(
            f"error: the run {args.run_id} has not finished: run `collect` and then `join` "
            "of `python -m stems_bench.other_joined` first",
            file=sys.stderr,
        )
        return 2
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    text = render(
        analyse(read_rows(run_dir / "rows.jsonl"), read_rows(run_dir / "joined.jsonl")), run
    )
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.other_joined",
        description="The default's own `other` with the subtracted one in its loud gaps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, text in (
        ("collect", cmd_collect, "count the gaps of the default's own `other` (resumable)"),
        ("join", cmd_join, "choose the level on the tuning tracks, join and score (resumable)"),
        ("report", cmd_report, "write the report from a finished run"),
    ):
        command = sub.add_parser(name, help=text)
        command.add_argument("--run-id", default="other-joined-v1")
        command.set_defaults(func=func)
        if name == "report":
            command.add_argument("--out", help="write the report here (default: print it)")
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except JoinError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
