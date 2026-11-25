"""Dropouts by model (Phase 17).

    python -m stems_bench.dropouts_by_model collect   # read the kept outputs (resumable)
    python -m stems_bench.dropouts_by_model report --out docs/dropouts-by-model.md

The default model of every stem was chosen by SDR (``docs/benchmarks/recommendation.md``).
SDR gives a stem one number for a song, and that number hardly moves when the stem falls
silent for a while as its instrument plays. ``docs/dropouts.md`` and
``docs/what-is-lost.md`` counted that for the product's stems and found the default
model's ``other`` and piano silent for about a seventh of the time they play. Is that the
price of that model, or of any model? The benchmark kept the outputs its blends are made
of, so the same count can be made for other models with no model run.

**What is counted**, fixed here before any kept output is read for this:

- *The lists*: the MoisesDB test tracks, the MoisesDB tuning tracks and the MUSDB18-HQ
  test tracks (``LISTS``).
- *The choices* for a stem on a list: every model with a kept output of that stem for
  every track of the list; and the stem's configured two-model blend
  (``ensembles.slots``), made from its members' kept outputs as the product would make
  it (the configured method and weights), when both members are kept for every track.
- *The tracks* for a stem: those where the instrument is in the song, which is where its
  true stem sounds for ``IN_SONG_SECONDS`` or more.
- For each choice, over those tracks: how long the true stem sounds, how much of that
  time the choice's stem is silent (*dropped*, as ``stems_bench.dropouts.measure``), and
  how many stems have a long dropout (``LONG_SECONDS`` or more of the true stem inside
  one stretch of silence). Beside it, for a single model, the median of the SDR the
  benchmark measured for those same outputs, read from its results.
- *The default where it was not kept.* The benchmark did not keep every stem of the
  default model. Where it did not, the default's figures are those of the product's own
  run on the first twenty test tracks (``runs/product/lost-v1``,
  ``docs/what-is-lost.md``), and the other choices are counted again on those twenty
  tracks to stand beside it.
- **What is done with the result.** Nothing is decided here: the default of every stem
  stays what the rule of Phase 10 chose, on SDR. For each stem the report names the
  choice that drops the smallest share on the MoisesDB tuning tracks, the list choices
  are made on in this project, and shows it beside the default on the two test lists.
  Where that choice is not the default, it is put to the owner with both numbers.
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
from stems_bench.gate_outputs import IN_SONG_SECONDS, kept_outputs, read_rows
from stems_bench.run import BENCH_DIR, git_state, write_json
from stems_common.paths import RUNS_DIR
from stems_common.stems import PRODUCT_STEMS

OUT_DIR = RUNS_DIR / "dropouts-by-model"
# The lists, by the names the report uses. Choices are compared on the tuning list.
LISTS = {
    "MoisesDB test": "moisesdb_subset",
    "MoisesDB tuning": "moisesdb_tuning",
    "MUSDB18-HQ test": "musdb18hq_test",
}
TUNING = "MoisesDB tuning"
TESTS = ("MoisesDB test", "MUSDB18-HQ test")
# The list the product's own run was made on, for a default that was not kept.
PRODUCT_LIST = "MoisesDB test"
# The commit that holds everything above, made before any kept output was read for this.
# The report's table "The default against the rest" came after the run: it sets figures
# from the other tables side by side and adds none.
RULES_COMMIT: str | None = "af0a95049e6e3f89fc9a228ad181915a86e61eb3"


class ByModelError(ValueError):
    """The count cannot be made or reported as asked."""


def blend_name(members: Sequence[str]) -> str:
    return "blend of " + " and ".join(members)


# -- collecting ----------------------------------------------------------------------------


def index(bench_dir: Path, dataset: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(model, track, stem) -> the kept output's path and its SDR, the first if kept twice."""
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in kept_outputs(bench_dir, dataset):
        key = (item["model_id"], item["track_id"], item["stem"])
        found.setdefault(key, {"path": item["path"], "sdr": item["bss"].get("sdr")})
    return found


def choices_for(
    found: Mapping[tuple[str, str, str], Any],
    track_ids: Sequence[str],
    blends: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Stem -> choice -> the models it is made of, for the choices kept for every track.

    ``blends`` maps a stem to its configured blend: ``members`` (model ids) and how they
    are combined.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for stem in PRODUCT_STEMS:
        models = sorted({model for model, _, kept_stem in found if kept_stem == stem})
        whole = [m for m in models if all((m, t, stem) in found for t in track_ids)]
        choices = {model: [model] for model in whole}
        members = list((blends.get(stem) or {}).get("members") or [])
        if len(members) > 1 and all(member in whole for member in members):
            choices[blend_name(members)] = members
        if choices:
            out[stem] = choices
    return out


def collect(
    list_name: str,
    dataset: Any,
    blends: Mapping[str, Mapping[str, Any]],
    out_path: Path,
    *,
    bench_dir: Path = BENCH_DIR,
    progress: Callable[[str], None] | None = None,
) -> int:
    """A row for every track of one list: each stem's true sounding time, and each choice's
    dropped time and longest dropout. Returns how many rows were new.
    """
    from stems_worker.ensemble import Blender

    done = {row["id"] for row in read_rows(out_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    found = index(bench_dir, dataset.name)
    track_ids = dataset.track_ids()
    choices = choices_for(found, track_ids, blends)
    added = 0
    for number, track_id in enumerate(track_ids, start=1):
        row_id = f"{list_name}/{track_id}"
        if row_id not in done:
            track = dataset.load(track_id)
            rate = track.sample_rate
            stems: dict[str, Any] = {}
            for stem, mine in choices.items():
                true = track.references[stem]
                counted: dict[str, Any] = {}
                sounding = 0.0
                for choice, members in mine.items():
                    kept = [found[(member, track_id, stem)] for member in members]
                    audio = [np.asarray(np.load(k["path"], mmap_mode="r")) for k in kept]
                    if len(audio) == 1:
                        made, sdr = audio[0], kept[0]["sdr"]
                    else:
                        how = blends[stem]
                        made = Blender(audio).blend(how.get("weights"), how["method"])
                        sdr = None  # the benchmark's score for a blend is not read here
                    if made.shape[-1] != true.shape[-1]:
                        raise ByModelError(
                            f"{choice} for {stem} of {track_id} has {made.shape[-1]} samples "
                            f"and the true stem has {true.shape[-1]}"
                        )
                    result = measure(true, made, rate)
                    sounding = result["sounding"]
                    counted[choice] = {
                        "dropped": result["dropped"],
                        "longest": result["longest"],
                        "sdr": None if sdr is None else round(float(sdr), 3),
                    }
                stems[stem] = {"sounding": sounding, "choices": counted}
            row = {"id": row_id, "list": list_name, "track_id": track_id, "stems": stems}
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            added += 1
        if progress:
            progress(f"{list_name}: {number} of {len(track_ids)} tracks")
    return added


# -- analysis (pure) -----------------------------------------------------------------------


def tally(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One choice over the tracks counted for it: each entry has the true stem's
    ``sounding`` seconds and the choice's ``dropped``, ``longest`` and, perhaps, ``sdr``.
    """
    sounding = sum(e["sounding"] for e in entries)
    dropped = sum(e["dropped"] for e in entries)
    scores = [e["sdr"] for e in entries if e.get("sdr") is not None]
    return {
        "stems": len(entries),
        "sounding_minutes": sounding / 60.0,
        "dropped_minutes": dropped / 60.0,
        "share": dropped / sounding if sounding else None,
        "long": sum(1 for e in entries if e["longest"] >= LONG_SECONDS),
        "sdr": statistics.median(scores) if scores else None,
    }


def by_choice(
    rows: Sequence[Mapping[str, Any]], list_name: str, *, only: set[str] | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """Stem -> choice -> its tally on one list, over the tracks whose song has the
    instrument (and, with ``only``, over those tracks alone).
    """
    gathered: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if row["list"] != list_name or (only is not None and row["track_id"] not in only):
            continue
        for stem, entry in row["stems"].items():
            if entry["sounding"] < IN_SONG_SECONDS:
                continue
            for choice, counted in entry["choices"].items():
                gathered.setdefault(stem, {}).setdefault(choice, []).append(
                    {"sounding": entry["sounding"], **counted}
                )
    return {
        stem: {choice: tally(entries) for choice, entries in sorted(choices.items())}
        for stem in PRODUCT_STEMS
        if (choices := gathered.get(stem))
    }


def least(choices: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The choice that drops the smallest share; the first by name among equals."""
    shares = {name: entry["share"] for name, entry in choices.items() if entry["share"] is not None}
    return min(sorted(shares), key=shares.__getitem__) if shares else None


def product_entries(
    records: Sequence[Mapping[str, Any]], stem: str
) -> tuple[list[dict[str, Any]], set[str]]:
    """The product's own figures for one stem, from a run made with ``--lost``: an entry a
    track where the instrument is in the song, and the ids of the tracks in the run.
    """
    entries = []
    tracks: set[str] = set()
    for record in records:
        lost = record.get("lost")
        if not lost:
            continue
        tracks.add(record["track_id"])
        entry = lost["stems"][stem]
        if entry["sounding"] >= IN_SONG_SECONDS:
            entries.append(
                {
                    "sounding": entry["sounding"],
                    "dropped": entry["dropped"],
                    "longest": entry["longest"],
                }
            )
    return entries, tracks


def analyse(
    rows: Sequence[Mapping[str, Any]],
    defaults: Mapping[str, str],
    product: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Everything the report shows. ``defaults`` maps a stem to its default model and
    ``product`` holds the records of the product's own run, for a default not kept.
    """
    lists = {name: by_choice(rows, name) for name in LISTS}
    stems = [stem for stem in PRODUCT_STEMS if any(stem in lists[name] for name in LISTS)]
    chosen = {stem: least(lists[TUNING].get(stem, {})) for stem in stems}
    beside: dict[str, Any] = {}
    for stem in stems:
        default = defaults.get(stem)
        if default is None or default in lists[PRODUCT_LIST].get(stem, {}):
            continue
        entries, tracks = product_entries(product, stem)
        if not entries:
            beside[stem] = None  # the default was not kept, and the product's run is not there
            continue
        others = by_choice(rows, PRODUCT_LIST, only=tracks).get(stem, {})
        beside[stem] = {"default": tally(entries), "others": others, "tracks": len(tracks)}
    # The default beside the choice, other than itself, that drops least: on the product's
    # list where its output was kept, on the product's own tracks where it was not.
    against: dict[str, Any] = {}
    for stem in stems:
        default = defaults.get(stem)
        kept = lists[PRODUCT_LIST].get(stem, {})
        if default in kept:
            where, mine = PRODUCT_LIST, kept[default]
            others = {name: entry for name, entry in kept.items() if name != default}
        elif beside.get(stem):
            where = f"the product's {beside[stem]['tracks']} tracks"
            mine, others = beside[stem]["default"], beside[stem]["others"]
        else:
            continue
        other = least(others)
        against[stem] = {
            "where": where,
            "default": mine,
            "other": other,
            "other_entry": others.get(other) if other else None,
        }
    return {
        "against": against,
        "tracks": {name: len({r["track_id"] for r in rows if r["list"] == name}) for name in LISTS},
        "lists": lists,
        "stems": stems,
        "defaults": {stem: defaults.get(stem) for stem in stems},
        "least_on_tuning": chosen,
        "default_from_product": beside,
    }


# -- the report ----------------------------------------------------------------------------


def _percent(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{round(100 * value, digits) + 0.0:.{digits}f}%"


def _sdr(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _row(first: str, name: str, entry: Mapping[str, Any]) -> str:
    return (
        f"| {first} | {name} | {entry['stems']} | {entry['sounding_minutes']:.1f} | "
        f"{entry['dropped_minutes']:.1f} | {_percent(entry['share'])} | {entry['long']} | "
        f"{_sdr(entry['sdr'])} |"
    )


_HEAD = [
    "| Stem | Choice | Songs with it | True stem sounds (min) | Dropped (min) | Share | "
    "With a long dropout | SDR (dB, median) |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    git = run.get("git") or {}
    defaults = analysis["defaults"]

    def named(stem: str, choice: str) -> str:
        return f"`{choice}` (the default)" if choice == defaults.get(stem) else f"`{choice}`"

    lines = [
        "# Dropouts by model",
        "",
        "Written by `python -m stems_bench.dropouts_by_model report` from the run "
        f"`{run.get('run_id', '?')}`"
        + (f", made at commit `{git['commit'][:7]}`" if git.get("commit") else "")
        + (", with uncommitted changes" if git.get("dirty") else "")
        + ". No model was run: these are the benchmark's kept outputs, read beside the true "
        "stems. What is counted and what is done with it were written into "
        "`bench/stems_bench/dropouts_by_model.py`"
        + (f" and committed (`{RULES_COMMIT[:7]}`)" if RULES_COMMIT else "")
        + " before any kept output was read for this.",
        "",
        "The default model of every stem was chosen by SDR. SDR hardly moves when a stem "
        "falls silent for a while as its instrument plays, and the default model does that "
        "(`docs/dropouts.md`, `docs/what-is-lost.md`). This puts the same count beside SDR "
        "for every model whose output the benchmark kept, and for each stem's configured "
        "two-model blend. A stem is **dropped** for a tenth of a second when the true stem "
        f"sounds in it and the choice's stem does not; a dropout is **long** from "
        f"{LONG_SECONDS:g} seconds. Only songs that have the instrument are counted.",
        "",
        "## The choice that drops least, beside the default",
        "",
        "Named on the MoisesDB tuning tracks, where choices are made in this project, and "
        "shown on the two test lists. Nothing is decided by it.",
        "",
        "| Stem | Drops least on the tuning tracks | List | Its share dropped | Its SDR | "
        "The default's share dropped | The default's SDR |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for stem in analysis["stems"]:
        choice = analysis["least_on_tuning"].get(stem)
        if choice is None:
            lines.append(f"| {stem} | none was kept on the tuning tracks | | | | | |")
            continue
        for name in (TUNING, *TESTS):
            there = analysis["lists"][name].get(stem, {})
            mine, default = there.get(choice), there.get(defaults.get(stem) or "")
            lines.append(
                f"| {stem} | {named(stem, choice)} | {name} | "
                f"{_percent(mine['share']) if mine else '-'} | "
                f"{_sdr(mine['sdr']) if mine else '-'} | "
                f"{_percent(default['share']) if default else 'not kept'} | "
                f"{_sdr(default['sdr']) if default else '-'} |"
            )
    against = analysis.get("against") or {}
    if against:
        lines += [
            "",
            "## The default against the rest",
            "",
            "The default beside whichever other choice drops least on the same songs. This "
            "table was added after the run; every figure in it is in a table below.",
            "",
            "| Stem | Songs | The default drops | With a long dropout | The least another drops "
            "| With a long dropout | That choice | Its SDR | The default's SDR |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for stem, entry in against.items():
            mine, other = entry["default"], entry["other_entry"]
            lines.append(
                f"| {stem} | {mine['stems']} on {entry['where']} | {_percent(mine['share'])} | "
                f"{mine['long']} | {_percent(other['share']) if other else '-'} | "
                f"{other['long'] if other else '-'} | "
                f"{named(stem, entry['other']) if other else 'none was kept'} | "
                f"{_sdr(other['sdr']) if other else '-'} | {_sdr(mine['sdr'])} |"
            )
    for name in LISTS:
        there = analysis["lists"][name]
        lines += ["", f"## {name}: {analysis['tracks'][name]} tracks", ""]
        if not there:
            lines.append("No stem has a kept output for every track of this list.")
            continue
        lines += _HEAD
        for stem, choices in there.items():
            for choice, entry in choices.items():
                lines.append(_row(stem, named(stem, choice), entry))
    beside = analysis["default_from_product"]
    if beside:
        lines += [
            "",
            "## The default where the benchmark did not keep it",
            "",
            "For these stems the default model's output was not kept. Its row is from the "
            "product's own run on the first twenty test tracks (`docs/what-is-lost.md`), "
            "and the other choices are counted on the same tracks. The product's run does "
            "not carry SDR.",
            "",
        ]
        if any(entry is not None for entry in beside.values()):
            lines += _HEAD
        for stem, entry in beside.items():
            if entry is None:
                lines.append(f"The product's run was not found, so nothing stands for {stem}.")
                continue
            lines.append(_row(stem, named(stem, defaults[stem]), entry["default"]))
            for choice, other in entry["others"].items():
                lines.append(_row(stem, named(stem, choice), other))
    left_out = [stem for stem in PRODUCT_STEMS if stem not in analysis["stems"]]
    lines += [
        "",
        "## What this is and is not",
        "",
        "- A count, and it decides nothing: the defaults are as Phase 10's rule left them.",
        "- Only the outputs the benchmark kept, which are the members of its blends: a model "
        "that is in no blend for a stem is not here."
        + (f" No output was kept for: {', '.join(left_out)}." if left_out else ""),
        "- Dropped time says a stem is silent, not what is in it when it sounds: a model can "
        "drop nothing because it lets everything through. SDR is beside it for that reason.",
        "- A blend is silent only where all of its members are. Its SDR is not shown: the "
        "benchmark's scores for blends are in its own reports.",
        "- SDR is this benchmark's, a median over the songs counted here. It is not "
        "comparable with figures published elsewhere.",
        "- Nothing in the product was changed.",
        "",
    ]
    return "\n".join(lines)


# -- commands ------------------------------------------------------------------------------


def plan() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """From the config: each stem's default model, and its two-model blend with how it is
    combined.
    """
    from stems_common.settings import load_defaults

    ensembles = load_defaults().ensembles
    defaults: dict[str, str] = {}
    blends: dict[str, dict[str, Any]] = {}
    for stem in PRODUCT_STEMS:
        slot = ensembles.slots[stem]
        defaults[stem] = slot.single_best
        blends[stem] = {
            "members": list(slot.two_model_blend),
            "weights": getattr(slot, "two_model_weights", None),
            "method": getattr(slot, "blend_method", None) or ensembles.blend_method,
        }
    return defaults, blends


def cmd_collect(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset

    run_dir = OUT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    defaults, blends = plan()
    status_path = run_dir / "status.json"
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(status_path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    try:
        for name, dataset_name in LISTS.items():
            collect(
                name, get_dataset(dataset_name), blends, run_dir / "rows.jsonl", progress=progress
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
            "defaults": defaults,
            "blends": blends,
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
            "`python -m stems_bench.dropouts_by_model collect` first",
            file=sys.stderr,
        )
        return 2
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    product_dir = PRODUCT_DIR / args.product_run
    product = read_results(product_dir) if product_dir.is_dir() else []
    text = render(analyse(read_rows(run_dir / "rows.jsonl"), run["defaults"], product), run)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.dropouts_by_model",
        description="How much each kept model's stem is silent while the instrument plays.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="read the kept outputs (resumable)")
    collect_parser.add_argument("--run-id", default="by-model-v1")
    collect_parser.set_defaults(func=cmd_collect)
    report_parser = sub.add_parser("report", help="write the report from a finished run")
    report_parser.add_argument("--run-id", default="by-model-v1")
    report_parser.add_argument("--product-run", default="lost-v1")
    report_parser.add_argument("--out", help="write the report here (default: print it)")
    report_parser.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ByModelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
