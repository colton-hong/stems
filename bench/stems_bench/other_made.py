"""``other``, made the other way (Phase 19).

    python -m stems_bench.other_made collect   # compose from the kept outputs (resumable)
    python -m stems_bench.other_made report --out docs/other-by-subtraction.md

The product can make the stem ``other`` from models in two ways (``stems_worker.compose``).
It can take a six-stem model's own ``other``, which is what it does now. Or it can take a
four-stem model's ``other``, which still holds the guitar and the piano, and subtract the
guitar and piano stems from it, which is what it did before the defaults were chosen on
SDR in Phase 10. ``docs/what-is-lost.md`` found the first silent for 16.1% of the time the
true ``other`` sounds, on twenty songs; ``docs/dropouts.md`` found the second silent for
0.5%, on six. ``docs/dropouts-by-model.md`` could not count ``other`` at all, because no
six-stem ``other`` was kept. What subtraction needs was kept: the four-stem ``other`` of
some models, and the default model's guitar and piano.

**What is counted**, fixed here before any kept output is read for this:

- *The lists*: the MoisesDB test tracks and tuning tracks, which have a true six-stem
  ``other``.
- *The ways*: for every model whose four-stem ``other`` was kept for every track of a
  list, that ``other`` minus the default model's kept guitar and minus its kept piano,
  sample by sample, as ``stems_worker.compose`` subtracts them.
- *Where the song has an* ``other`` (its true stem sounds for ``IN_SONG_SECONDS`` or
  more): the true stem's sounding time, the time the way's stem is silent meanwhile
  (*dropped*, as ``stems_bench.dropouts.measure``) and the stems with a long dropout.
  Beside it, where the benchmark scored that very way (its ``other_composed`` blend of
  one model), the median of its SDR over the same tracks.
- *Where the song has none* (the true stem sounds for less than that): how far the way's
  ``other`` is below the song, over the whole song, as a median; and on how many of
  those songs the product's own rule would call it silent or very quiet
  (``gate.rules``). A stem made by subtraction holds what the three models disagree
  about, so this is where it is expected to cost.
- *The default's own* ``other`` was not kept. Its figures are those of the product's own
  run on the first twenty test tracks (``runs/product/lost-v1``): dropped time, the SDR
  that run measured, and on songs with no ``other`` whether the stem carried the note.
  The ways are counted again on those twenty tracks to stand beside it.
- **What is done with the result.** Nothing is decided here: ``other`` stays the default
  model's own, as PLAN.md decides (BLOCKED.md D15). The figures go to the owner there.
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

from stems_bench.dropouts import LONG_SECONDS, measure
from stems_bench.dropouts_by_model import index, tally
from stems_bench.gate_outputs import IN_SONG_SECONDS, read_rows
from stems_bench.run import BENCH_DIR, git_state, write_json
from stems_common.levels import level_below_mix_db, silence_fraction
from stems_common.paths import RUNS_DIR

OUT_DIR = RUNS_DIR / "other-made"
LISTS = {"MoisesDB test": "moisesdb_subset", "MoisesDB tuning": "moisesdb_tuning"}
PRODUCT_LIST = "MoisesDB test"
# What a four-stem model's "other" is scored against in the benchmark's results.
FOUR_STEM = "other_4stem"
TAKEN_OUT = ("guitar", "piano")
# The benchmark's own name for the subtraction it scored, among its blends' results.
SCORED_SLOT = "other_composed.single_best"
# The commit that holds everything above, made before any kept output was read for this.
# Filled in by the commit after it.
RULES_COMMIT: str | None = None


class OtherError(ValueError):
    """The count cannot be made or reported as asked."""


def way_name(model: str) -> str:
    return f"{model} minus guitar and piano"


# -- collecting ----------------------------------------------------------------------------


def four_stem_others(bench_dir: Path, dataset: str) -> dict[tuple[str, str], Path]:
    """(model, track) -> a kept four-stem ``other``, the first if a job was kept twice."""
    found: dict[tuple[str, str], Path] = {}
    root = bench_dir / "estimates" / dataset
    if not root.is_dir():
        return found
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for job_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            path = job_dir / "other.npy"
            result_path = bench_dir / "results" / dataset / model_dir.name / f"{job_dir.name}.json"
            if not path.is_file() or not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("kind") != "model" or result.get("input") != "mix":
                continue
            scored = result.get("outputs", {}).get("other") or {}
            if scored.get("target") == FOUR_STEM:
                found.setdefault((model_dir.name, result["track_id"]), path)
    return found


def scored_subtraction(
    bench_dir: Path, dataset: str, default_model: str
) -> dict[tuple[str, str], float]:
    """(model, track) -> the SDR the benchmark measured for that model's ``other`` minus
    the default model's guitar and piano, where it scored exactly that.
    """
    found: dict[tuple[str, str], float] = {}
    folder = bench_dir / "results" / dataset / "_ensembles" / SCORED_SLOT
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        members = result.get("members") or []
        minus = {m.get("slot"): m.get("members") for m in result.get("minus") or []}
        if len(members) != 1 or minus != {stem: [default_model] for stem in TAKEN_OUT}:
            continue
        scored = result.get("outputs", {}).get("other") or {}
        sdr = (scored.get("bss") or {}).get("sdr")
        if scored.get("target") == "other" and sdr is not None:
            found.setdefault((members[0], result["track_id"]), float(sdr))
    return found


def collect(
    list_name: str,
    dataset: Any,
    default_model: str,
    out_path: Path,
    *,
    silent_fraction_min: float,
    empty_below_mix_db: float | None,
    bench_dir: Path = BENCH_DIR,
    progress: Callable[[str], None] | None = None,
) -> int:
    """A row a track: the true ``other``'s sounding time, and for each way its dropped
    time, its level against the song and whether the gate would note it. Returns how many
    rows were new.
    """
    done = {row["id"] for row in read_rows(out_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    others = four_stem_others(bench_dir, dataset.name)
    kept = index(bench_dir, dataset.name)
    scores = scored_subtraction(bench_dir, dataset.name, default_model)
    track_ids = dataset.track_ids()
    models = [
        model
        for model in sorted({model for model, _ in others})
        if all((model, track_id) in others for track_id in track_ids)
    ]
    taken = all((default_model, t, stem) in kept for t in track_ids for stem in TAKEN_OUT)
    if not taken:
        models = []  # nothing can be subtracted without the default's guitar and piano
    added = 0
    for number, track_id in enumerate(track_ids, start=1):
        row_id = f"{list_name}/{track_id}"
        if row_id not in done:
            track = dataset.load(track_id)
            rate, mix, true = track.sample_rate, track.inputs["mix"], track.references["other"]
            ways: dict[str, Any] = {}
            sounding = round((1.0 - silence_fraction(true, rate)) * true.shape[-1] / rate, 1)
            if models:
                out = [np.load(kept[(default_model, track_id, s)]["path"]) for s in TAKEN_OUT]
            for model in models:
                made = np.load(others[(model, track_id)]).astype(np.float32)
                for stem in out:
                    made = made - stem
                if made.shape[-1] != true.shape[-1]:
                    raise OtherError(
                        f"{way_name(model)} of {track_id} has {made.shape[-1]} samples and the "
                        f"true stem has {true.shape[-1]}"
                    )
                counted = measure(true, made, rate)
                sounding = counted["sounding"]
                below = level_below_mix_db(made, mix)
                noted = silence_fraction(made, rate) >= silent_fraction_min or (
                    empty_below_mix_db is not None and below <= empty_below_mix_db
                )
                sdr = scores.get((model, track_id))
                ways[way_name(model)] = {
                    "dropped": counted["dropped"],
                    "longest": counted["longest"],
                    "below_song_db": round(below, 2),
                    "noted": bool(noted),
                    "sdr": None if sdr is None else round(sdr, 3),
                }
            row = {
                "id": row_id,
                "list": list_name,
                "track_id": track_id,
                "sounding": sounding,
                "ways": ways,
            }
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            added += 1
        if progress:
            progress(f"{list_name}: {number} of {len(track_ids)} tracks")
    return added


# -- analysis (pure) -----------------------------------------------------------------------


def absent(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One way over the songs that have no ``other``: each entry may carry the stem's
    level against the song (``below_song_db``) and says whether it was ``noted``.
    """
    levels = [e["below_song_db"] for e in entries if e.get("below_song_db") is not None]
    return {
        "songs": len(entries),
        "below_song_db": statistics.median(levels) if levels else None,
        "noted": sum(1 for e in entries if e["noted"]),
    }


def by_way(
    rows: Sequence[Mapping[str, Any]], list_name: str, *, only: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Way -> its tally where the song has an ``other``, and its level where it has none."""
    there: dict[str, list[dict[str, Any]]] = {}
    not_there: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["list"] != list_name or (only is not None and row["track_id"] not in only):
            continue
        present = row["sounding"] >= IN_SONG_SECONDS
        for way, counted in row["ways"].items():
            if present:
                there.setdefault(way, []).append({"sounding": row["sounding"], **counted})
            else:
                not_there.setdefault(way, []).append(counted)
    return {
        way: {"present": tally(there.get(way, [])), "absent": absent(not_there.get(way, []))}
        for way in sorted({*there, *not_there})
    }


def product_other(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, set[str]]:
    """The default model's own ``other`` from the product's run made with ``--lost``: its
    tally where the song has an ``other``, how often it was noted where it has none, and
    the ids of the run's tracks. None when the run has nothing to show.
    """
    present, missing = [], []
    tracks: set[str] = set()
    for record in records:
        lost, stems = record.get("lost"), record.get("stems") or {}
        if not lost or "other" not in stems:
            continue
        tracks.add(record["track_id"])
        entry = lost["stems"]["other"]
        if entry["sounding"] >= IN_SONG_SECONDS:
            present.append(
                {
                    "sounding": entry["sounding"],
                    "dropped": entry["dropped"],
                    "longest": entry["longest"],
                    "sdr": stems["other"].get("sdr_db"),
                }
            )
        else:
            missing.append({"noted": bool(stems["other"].get("flagged_empty"))})
    if not tracks:
        return None, tracks
    return {"present": tally(present), "absent": absent(missing)}, tracks


def analyse(
    rows: Sequence[Mapping[str, Any]], product: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Everything the report shows."""
    own, tracks = product_other(product)
    return {
        "tracks": {name: len({r["track_id"] for r in rows if r["list"] == name}) for name in LISTS},
        "lists": {name: by_way(rows, name) for name in LISTS},
        "own": own,
        "own_tracks": len(tracks),
        "beside_own": by_way(rows, PRODUCT_LIST, only=tracks) if own else {},
    }


# -- the report ----------------------------------------------------------------------------


def _percent(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{round(100 * value, digits) + 0.0:.{digits}f}%"


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


_HEAD = [
    "| How `other` is made | Songs with an `other` | True stem sounds (min) | Dropped (min) "
    "| Share | With a long dropout | SDR (dB, median) | Songs with none | There: dB under the "
    "song (median) | There: noted as empty |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]


def _row(name: str, entry: Mapping[str, Any]) -> str:
    present, gone = entry["present"], entry["absent"]
    level = gone["below_song_db"]
    return (
        f"| {name} | {present['stems']} | {present['sounding_minutes']:.1f} | "
        f"{present['dropped_minutes']:.1f} | {_percent(present['share'])} | {present['long']} | "
        f"{_number(present['sdr'])} | {gone['songs']} | "
        f"{'-' if level is None else f'{-level:.1f}'} | {gone['noted']} of {gone['songs']} |"
    )


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    git = run.get("git") or {}
    default = run.get("default_model", "?")
    lines = [
        "# `other`, made the other way",
        "",
        "Written by `python -m stems_bench.other_made report` from the run "
        f"`{run.get('run_id', '?')}`"
        + (f", made at commit `{git['commit'][:7]}`" if git.get("commit") else "")
        + (", with uncommitted changes" if git.get("dirty") else "")
        + ". No model was run: the stems here are put together from the benchmark's kept "
        "outputs and read beside the true stems. What is counted and what is done with it "
        "were written into `bench/stems_bench/other_made.py`"
        + (f" and committed (`{RULES_COMMIT[:7]}`)" if RULES_COMMIT else "")
        + " before any kept output was read for this.",
        "",
        "The product can make `other` in two ways from models. It can take a six-stem "
        f"model's own (`{default}`, what it does now). Or it can take a four-stem model's "
        "`other`, which still holds the guitar and the piano, and subtract the guitar and "
        "piano stems from it (what it did before Phase 10). The first is silent for a "
        "seventh of the time the true `other` sounds (`docs/what-is-lost.md`). This counts "
        "the second, for every four-stem `other` the benchmark kept, with the default "
        "model's guitar and piano taken out.",
        "",
        f"A stem is **dropped** for a tenth of a second when the true stem sounds in it and "
        f"the made one does not; a dropout is **long** from {LONG_SECONDS:g} seconds. A song "
        f"**has an `other`** when its true stem sounds for {IN_SONG_SECONDS:g} seconds or "
        "more. Where it has none, the table gives how far the made stem is below the song "
        "and how often the product's own rule would note it as silent or very quiet: there, "
        "anything in the stem is not `other`.",
    ]
    own = analysis["own"]
    if own:
        lines += [
            "",
            f"## Beside the default's own, on the product's {analysis['own_tracks']} tracks",
            "",
            "The default's own `other` was not kept by the benchmark. Its row is from the "
            "product's own run (`docs/what-is-lost.md`), with the SDR that run measured; the "
            "run did not record the stem's level, so that cell is empty.",
            "",
            *_HEAD,
            _row(f"`{default}`'s own (the default)", own),
            *[_row(f"`{way}`", entry) for way, entry in analysis["beside_own"].items()],
        ]
    else:
        lines += [
            "",
            "The product's own run was not found, so the default's own `other` is not here.",
        ]
    for name in LISTS:
        ways = analysis["lists"][name]
        lines += ["", f"## {name}: {analysis['tracks'][name]} tracks", ""]
        if not ways:
            lines.append("Nothing could be put together for this list from what was kept.")
            continue
        lines += [*_HEAD, *[_row(f"`{way}`", entry) for way, entry in ways.items()]]
    lines += [
        "",
        "## What this is and is not",
        "",
        "- A count, and it decides nothing: `other` stays the default model's own "
        "(BLOCKED.md D15).",
        "- The guitar and piano taken out are the default model's, as the product would take "
        "them out today. Before Phase 10 they were the same model's.",
        "- A stem that is never silent has not thereby got the instrument: subtraction leaves "
        "in `other` whatever the models disagree about. The columns for songs with no "
        "`other`, and SDR, are there for that.",
        "- SDR is this benchmark's and this product run's, a median over the songs counted "
        "in the row. It is shown only where one of them measured exactly that stem.",
        "- One dataset. Nothing in the product was changed.",
        "",
    ]
    return "\n".join(lines)


# -- commands ------------------------------------------------------------------------------


def cmd_collect(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset
    from stems_common.settings import load_defaults

    run_dir = OUT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    defaults = load_defaults()
    default_model = defaults.ensembles.slots["other"].single_best
    rules = defaults.gate.rules
    status_path = run_dir / "status.json"
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(status_path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    try:
        for name, dataset_name in LISTS.items():
            collect(
                name,
                get_dataset(dataset_name),
                default_model,
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
            "lists": LISTS,
            "default_model": default_model,
            "silent_fraction_min": rules.silent_fraction_min,
            "empty_below_mix_db": rules.empty_below_mix_db,
            "git": git_state(),
        },
    )
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from stems_bench.product_check import PRODUCT_DIR, read_results

    run_dir = OUT_DIR / args.run_id
    status_path = run_dir / "status.json"
    state = (
        json.loads(status_path.read_text(encoding="utf-8")).get("state")
        if status_path.is_file()
        else None
    )
    if state != "done":
        print(
            f"error: the run {args.run_id} has not finished (state: {state}): run "
            "`python -m stems_bench.other_made collect` first",
            file=sys.stderr,
        )
        return 2
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    product_dir = PRODUCT_DIR / args.product_run
    product = read_results(product_dir) if product_dir.is_dir() else []
    text = render(analyse(read_rows(run_dir / "rows.jsonl"), product), run)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.other_made",
        description="How `other` does when it is made by subtraction, from kept outputs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="compose from the kept outputs (resumable)")
    collect_parser.add_argument("--run-id", default="other-made-v1")
    collect_parser.set_defaults(func=cmd_collect)
    report_parser = sub.add_parser("report", help="write the report from a finished run")
    report_parser.add_argument("--run-id", default="other-made-v1")
    report_parser.add_argument("--product-run", default="lost-v1")
    report_parser.add_argument("--out", help="write the report here (default: print it)")
    report_parser.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OtherError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
