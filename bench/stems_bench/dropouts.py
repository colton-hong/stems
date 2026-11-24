"""Where the instrument plays and the product delivers silence (P12.5).

    python -m stems_bench.dropouts --out docs/dropouts.md      # needs a listening run's audio

SDR is one number for a whole stem, and the gate looks at a stem's level over the whole
song. Neither says whether a stem that is fine for one minute is simply missing for the
next two. This counts that, where there is a true stem to count against: the songs of a
listening run (``python -m stems_bench.listen make``), whose audio is kept on disk.

**The measure**, fixed here before it was run over the songs (one case had been seen: the
bass of the sixth song, while the listening kit's levels were being checked):

- Audio is cut into the short windows that ``stems_common.levels`` already uses for
  silence (a tenth of a second), and a window *sounds* if its level reaches the project's
  silence level (-60 dBFS). Nothing new is tuned.
- A window is **dropped** when the true stem sounds in it and the product's stem does not.
- A **dropout** is one unbroken stretch of the product's silence; its length is the time
  the true stem sounds inside it. Gaps between the true stem's notes do not split it.
- A dropout is **long** at ``LONG_SECONDS``: the same five seconds that make an instrument
  count as being in a song (``gate_outputs.IN_SONG_SECONDS``).

Nothing is decided by this. It is a count, written to a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.gate_outputs import IN_SONG_SECONDS
from stems_bench.listen import DEFAULT_RUN, LISTEN_DIR, SYSTEM_NAMES
from stems_bench.run import write_json
from stems_common.levels import (
    SILENCE_DBFS,
    SILENCE_WINDOW_SECONDS,
    level_below_mix_db,
    sounding_windows,
)
from stems_common.stems import PRODUCT_STEMS

LONG_SECONDS = IN_SONG_SECONDS
VERSIONS = ("new", "old")
SAMPLE_RATE = 44100


class DropoutError(ValueError):
    """The count cannot be made as asked."""


def measure(true: np.ndarray, made: np.ndarray, sample_rate: int) -> dict[str, float]:
    """How much of the true stem's sounding time the made stem is silent for.

    Seconds, to a tenth: ``sounding`` (the true stem), ``dropped`` (of those, with the made
    stem silent) and ``longest`` (the most the true stem sounds inside one unbroken stretch
    of the made stem's silence). If the two differ in length the shorter decides.
    """
    samples = min(true.shape[-1], made.shape[-1])
    sounds = sounding_windows(true[..., :samples], sample_rate)
    silent = ~sounding_windows(made[..., :samples], sample_rate)
    dropped = sounds & silent
    longest = 0
    if dropped.any():
        # Each window that is not silent starts a new stretch; count the drops per stretch.
        stretch = np.cumsum(~silent)
        longest = int(np.bincount(stretch[dropped]).max())
    return {
        "sounding": round(float(sounds.sum()) * SILENCE_WINDOW_SECONDS, 1),
        "dropped": round(float(dropped.sum()) * SILENCE_WINDOW_SECONDS, 1),
        "longest": round(longest * SILENCE_WINDOW_SECONDS, 1),
    }


def collect(run_dir: Path, *, empty_below_mix_db: float | None) -> list[dict[str, Any]]:
    """One record for each stem of each song of a listening run, for each version."""
    from stems_worker.audio import load_audio

    stems_dir = run_dir / "stems"
    key_path = run_dir / "key.json"
    if not key_path.is_file() or not (stems_dir / "done").is_dir():
        raise DropoutError(f"{run_dir} has no finished listening run: run `listen make` first")
    codes = {
        entry["id"]: code
        for code, entry in json.loads(key_path.read_text(encoding="utf-8"))["tracks"].items()
    }
    records: list[dict[str, Any]] = []
    for track_id, code in sorted(codes.items(), key=lambda pair: pair[1]):
        made = json.loads((stems_dir / "done" / f"{track_id}.json").read_text(encoding="utf-8"))
        mix = load_audio(stems_dir / "mix" / f"{track_id}.wav", SAMPLE_RATE).samples
        for stem in made["stems"]:
            true = load_audio(stems_dir / "reference" / track_id / f"{stem}.flac", SAMPLE_RATE)
            for version in VERSIONS:
                audio = load_audio(stems_dir / version / track_id / f"{stem}.flac", SAMPLE_RATE)
                below = level_below_mix_db(audio.samples, mix)
                records.append(
                    {
                        "track": code,
                        "stem": stem,
                        "version": version,
                        **measure(true.samples, audio.samples, SAMPLE_RATE),
                        "below_song_db": round(below, 2),
                        "noted": empty_below_mix_db is not None and below < empty_below_mix_db,
                    }
                )
    return records


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals for a set of records."""
    sounding = sum(r["sounding"] for r in records)
    dropped = sum(r["dropped"] for r in records)
    long = [r for r in records if r["longest"] >= LONG_SECONDS]
    return {
        "stems": len(records),
        "sounding_minutes": sounding / 60.0,
        "dropped_minutes": dropped / 60.0,
        "share": dropped / sounding if sounding else None,
        "long": len(long),
        "long_noted": sum(1 for r in long if r["noted"]),
    }


def _row(label: str, total: Mapping[str, Any]) -> str:
    share = "-" if total["share"] is None else f"{100 * total['share']:.1f}%"
    return (
        f"| {label} | {total['stems']} | {total['sounding_minutes']:.1f} | "
        f"{total['dropped_minutes']:.1f} | {share} | {total['long']} |"
    )


def render(records: Sequence[Mapping[str, Any]], run: Mapping[str, Any]) -> str:
    old_models: Mapping[str, str] = run.get("old_models") or {}
    new_models: Mapping[str, str] = run.get("new_models") or {}
    same = [
        stem
        for stem in PRODUCT_STEMS
        if stem in old_models and old_models[stem] == new_models.get(stem)
    ]
    head = "| | Stems | True stem sounds (min) | Dropped (min) | Share | With a long dropout |"
    rule = "|---|---:|---:|---:|---:|---:|"
    lines = [
        "# Where the instrument plays and the product delivers silence",
        "",
        f"Written by `python -m stems_bench.dropouts` from the audio of the listening run "
        f"`{run['run_id']}`: {run.get('tracks', '?')} songs of the frozen MoisesDB test list, "
        "each put through the product's own job code with the models it uses now and with "
        "the ones it used before the recommendation of Phase 10, and the true stems beside "
        "them. SDR gives a stem one number and the gate looks at its level over the whole "
        "song; this asks instead whether a stem is simply missing for part of the song.",
        "",
        f"A tenth of a second is **dropped** when the true stem sounds in it (reaches "
        f"{SILENCE_DBFS:g} dBFS, the project's silence level) and the product's stem does "
        "not. A **dropout** is one unbroken stretch of the product's silence, as long as the "
        f"true stem sounds inside it; it is **long** from {LONG_SECONDS:g} seconds, the time "
        "that makes an instrument count as being in a song. These definitions were written "
        "into `bench/stems_bench/dropouts.py` before the count was run; one case had been "
        "seen by then (the bass of the sixth song).",
        "",
        head,
        rule,
    ]
    for version in VERSIONS:
        mine = [r for r in records if r["version"] == version]
        lines.append(_row(SYSTEM_NAMES[version], summarise(mine)))
    if same:
        lines += [
            "",
            f"For {' and '.join(same)} the product uses the same model now as before, so "
            "those rows of the two versions are the same audio.",
        ]
    lines += ["", "## By stem", "", head.replace("| |", "| Stem, version |"), rule]
    for stem in PRODUCT_STEMS:
        for version in VERSIONS:
            mine = [r for r in records if r["stem"] == stem and r["version"] == version]
            if not mine or (version == "old" and stem in same):
                continue
            model = (new_models if version == "new" else old_models).get(stem)
            label = f"{stem}, {'now' if version == 'new' else 'before'}"
            lines.append(_row(label + (f" (`{model}`)" if model else ""), summarise(mine)))
    long = sorted(
        (
            r
            for r in records
            if r["longest"] >= LONG_SECONDS and not (r["version"] == "old" and r["stem"] in same)
        ),
        key=lambda r: (-r["longest"], r["track"], r["stem"], r["version"]),
    )
    lines += ["", "## Every long dropout", ""]
    if long:
        lines += [
            "| Song | Stem | Version | True stem sounds (s) | Dropped (s) | Longest (s) | "
            "Stem under the song (dB) | The product's note |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for r in long:
            lines.append(
                f"| {r['track']} | {r['stem']} | {'now' if r['version'] == 'new' else 'before'} | "
                f"{r['sounding']:.1f} | {r['dropped']:.1f} | {r['longest']:.1f} | "
                f"{-r['below_song_db']:.1f} | {'yes' if r['noted'] else 'no'} |"
            )
        noted = sum(1 for r in long if r["noted"])
        lines += [
            "",
            f'"The product\'s note" is the one it puts on a stem that is silent or very quiet '
            f"against the whole song (more than {-run['empty_below_mix_db']:g} dB under it). "
            f"{noted} of these {len(long)} stems would carry it. The others are delivered "
            "without a word: over the whole song they are loud enough."
            if run.get("empty_below_mix_db") is not None
            else "",
        ]
    else:
        lines.append("None.")
    lines += [
        "",
        "## What this is and is not",
        "",
        f"- {run.get('tracks', '?')} songs, one dataset. It says the thing happens and how "
        "it looks; it is not a rate for music in general.",
        "- Silence in the product's stem is counted only against the true stem, which the "
        "product never has. Nothing here can be turned into a check inside the product as "
        "it is.",
        "- What the dataset files under an instrument's name is not always what a model "
        "was taught to call that instrument (a synthesiser playing the bass line, say). A "
        "dropout can be the model's miss or a difference of naming; the listener hears the "
        "same thing either way.",
        "- The opposite fault, sound in the product's stem where the true stem is silent, "
        "is bleed. It is not counted here.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from stems_common.settings import load_defaults

    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.dropouts",
        description="Where the true stem plays and the product's stem is silent.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN, help="a listening run")
    parser.add_argument("--out", help="write the report here (default: print it)")
    args = parser.parse_args(argv)
    run_dir = LISTEN_DIR / args.run_id
    defaults = load_defaults()
    limit = defaults.gate.rules.empty_below_mix_db
    try:
        records = collect(run_dir, empty_below_mix_db=limit)
    except DropoutError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    run = {
        # A run made before the kit recorded its models used the config as it is now.
        "new_models": {stem: defaults.ensembles.slots[stem].single_best for stem in PRODUCT_STEMS},
        **run,
        "run_id": args.run_id,
        "empty_below_mix_db": limit,
    }
    write_json(run_dir / "dropouts.json", {"long_seconds": LONG_SECONDS, "records": records})
    text = render(records, run)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
