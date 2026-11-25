"""What the six stems lose (Phase 14).

    python -m stems_bench.product_check run --run-id lost-v1 --tracks 20 --device mps --lost
    python -m stems_bench.lost --run-id lost-v1 --out docs/what-is-lost.md

``docs/dropouts.md`` counted, on six songs, how long the product's stem is silent while
the instrument plays. Two questions follow from it, and both can only be asked where there
are true stems to ask them against:

1. Where a stem drops out, where is the instrument? In another of the six stems (misfiled,
   and still in the user's hands), or in none of them (gone)?
2. How much of a song is in none of its six stems?

They bear on a choice that is the owner's (BLOCKED.md D15): if what a stem drops is mostly
in none of the six, making ``other`` from what is left of the song would give it back; if
it is mostly in ``other`` or another stem already, nothing is lost and nothing would change.

**The measures**, fixed here before any run:

- *Dropped* time is as in ``stems_bench.dropouts``: tenths of a second in which the true
  stem sounds and the product's stem of that name does not.
- *Found in.* Over a stem's dropped time, with ``t`` the true stem and ``e`` one of the six
  delivered stems, the share of ``t`` found in ``e`` is ``<t, e> / <t, t>`` over the samples
  of both channels. It is 1 when ``e`` holds ``t`` unchanged and nothing that resembles
  it, and 0 when it holds nothing of it. What is in no stem is ``r``, the mix minus the sum
  of the six; its share is ``<t, r> / <t, t>``. If the true stems add up to the mix and do
  not resemble one another, the seven shares add up to 1. The report shows their sum, so
  that it can be seen how far that holds.
- *A check on the share.* The same figure for a stem's own output over the time it is not
  dropped. For a well separated stem it is close to 1.
- *In no stem.* The level of ``r`` against the mix over the whole song; and the share of
  the song's sounding time in which ``r`` is not "very quiet" against the mix, by the level
  at which the product itself uses those words (``gate.rules.empty_below_mix_db``), taken
  tenth of a second by tenth of a second.
- A stem is listed by itself when its dropped time adds up to ``LONG_SECONDS`` or more.
  Shares over several stems are averages weighted by dropped time.

Nothing is decided by this. It is a set of counts, written to a report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.dropouts import LONG_SECONDS, measure
from stems_common.levels import (
    SILENCE_DBFS,
    SILENCE_WINDOW_SECONDS,
    level_below_mix_db,
    sounding_windows,
    window_power,
)
from stems_common.stems import PRODUCT_STEMS

NONE = "none"
PLACES = (*PRODUCT_STEMS, NONE)
# The commit that holds the measures above, made before any real song was measured. The
# paragraph of the report on reading the sum, its count of the rows near 100%, and the
# share found in a stem of another name (the six columns of the table, added up without
# the stem's own) came after the run: they change no figure in it.
RULES_COMMIT = "17908793fbe749c1bee41f2ff8d423469628a863"
# A row's shares are read one by one only when they add up to within this of 100%.
SUM_POINTS = 10


class LostError(ValueError):
    """The count cannot be made as asked."""


def _samples(windows: np.ndarray, samples: int, sample_rate: int) -> np.ndarray:
    """A mask over samples from a mask over the short windows of ``stems_common.levels``."""
    width = max(1, round(SILENCE_WINDOW_SECONDS * sample_rate))
    return np.repeat(windows, width)[:samples]


def shares(
    true: np.ndarray, places: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, float] | None:
    """The share of the true stem found in each place, over the masked samples.

    None when the true stem has no energy there.
    """
    t = true[..., mask].astype(np.float64)
    energy = float(np.sum(t * t))
    if energy <= 0.0:
        return None
    return {name: float(np.sum(t * audio[..., mask])) / energy for name, audio in places.items()}


def measure_track(
    mix: np.ndarray,
    references: Mapping[str, np.ndarray],
    delivered: Mapping[str, np.ndarray],
    sample_rate: int,
    *,
    empty_below_mix_db: float | None,
) -> dict[str, Any]:
    """Dropouts, where the dropped instrument is, and what is in no stem, for one song.

    ``delivered`` must hold all six product stems at the mix's length.
    """
    missing = [stem for stem in PRODUCT_STEMS if stem not in delivered]
    if missing:
        raise LostError(f"all six stems are needed; missing: {', '.join(missing)}")
    samples = mix.shape[-1]
    residual = (mix - sum(delivered[stem] for stem in PRODUCT_STEMS)).astype(np.float32)
    true_sum = sum(references[stem] for stem in PRODUCT_STEMS)
    places = {**{stem: delivered[stem] for stem in PRODUCT_STEMS}, NONE: residual}
    out: dict[str, Any] = {
        # How far the dataset's six true stems are from adding up to its mix.
        "true_stems_minus_song_db": round(level_below_mix_db(mix - true_sum, mix), 2),
        "in_no_stem": {"below_song_db": round(level_below_mix_db(residual, mix), 2)},
        "stems": {},
    }
    if empty_below_mix_db is not None:
        song = window_power(mix, sample_rate)
        left = window_power(residual, sample_rate)
        sounds = sounding_windows(mix, sample_rate)
        heard = sounds & (left >= song * 10.0 ** (empty_below_mix_db / 10.0))
        out["in_no_stem"]["share_of_time"] = (
            round(float(heard.sum()) / float(sounds.sum()), 4) if sounds.any() else None
        )
    for stem in PRODUCT_STEMS:
        true, made = references[stem], delivered[stem]
        entry: dict[str, Any] = dict(measure(true, made, sample_rate))
        sounds = sounding_windows(true, sample_rate)
        silent = ~sounding_windows(made, sample_rate)
        dropped = shares(true, places, _samples(sounds & silent, samples, sample_rate))
        if dropped is not None:
            entry["found_in"] = {name: round(value, 4) for name, value in dropped.items()}
        kept = shares(true, {stem: made}, _samples(sounds & ~silent, samples, sample_rate))
        if kept is not None:
            entry["own_share_kept"] = round(kept[stem], 4)
        out["stems"][stem] = entry
    return out


# -- the report ----------------------------------------------------------------------------


def pooled(entries: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    """Shares over several stems, each weighted by its dropped time."""
    mine = [e for e in entries if e.get("found_in") and e["dropped"] > 0]
    weight = sum(e["dropped"] for e in mine)
    if not mine or weight <= 0:
        return None
    return {
        place: sum(e["found_in"][place] * e["dropped"] for e in mine) / weight for place in PLACES
    }


def own_and_elsewhere(entries: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    """Of what is dropped: the share in the stem's own output, and in the five others.

    ``entries`` maps a stem to its entries over the songs. Weighted by dropped time.
    """
    own = elsewhere = weight = 0.0
    for stem, mine in entries.items():
        for entry in mine:
            if not entry.get("found_in") or entry["dropped"] <= 0:
                continue
            found, seconds = entry["found_in"], entry["dropped"]
            weight += seconds
            own += found[stem] * seconds
            elsewhere += sum(found[place] for place in PRODUCT_STEMS if place != stem) * seconds
    if weight <= 0:
        return {}
    return {"own": own / weight, "elsewhere": elsewhere / weight}


def analyse(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the report shows, from a run's records."""
    songs = [r for r in records if r.get("lost")]
    by_stem: dict[str, Any] = {}
    listed: list[dict[str, Any]] = []
    for stem in PRODUCT_STEMS:
        entries = [
            {**r["lost"]["stems"][stem], "track_id": r["track_id"]}
            for r in songs
            if r["lost"]["stems"][stem]["sounding"] > 0
        ]
        sounding = sum(e["sounding"] for e in entries)
        dropped = sum(e["dropped"] for e in entries)
        kept = [e["own_share_kept"] for e in entries if "own_share_kept" in e]
        by_stem[stem] = {
            "stems": len(entries),
            "sounding_minutes": sounding / 60.0,
            "dropped_minutes": dropped / 60.0,
            "share": dropped / sounding if sounding else None,
            "long": sum(1 for e in entries if e["longest"] >= LONG_SECONDS),
            "found_in": pooled(entries),
            "own_share_kept": statistics.median(kept) if kept else None,
        }
        listed += [
            {**e, "stem": stem} for e in entries if e["dropped"] >= LONG_SECONDS and "found_in" in e
        ]
    everything = [r["lost"]["stems"][stem] for r in songs for stem in PRODUCT_STEMS]
    sounding = sum(e["sounding"] for e in everything)
    dropped = sum(e["dropped"] for e in everything)
    levels = [r["lost"]["in_no_stem"]["below_song_db"] for r in songs]
    times = [
        r["lost"]["in_no_stem"]["share_of_time"]
        for r in songs
        if r["lost"]["in_no_stem"].get("share_of_time") is not None
    ]
    sums = [r["lost"]["true_stems_minus_song_db"] for r in songs]
    return {
        "songs": len(songs),
        "sounding_minutes": sounding / 60.0,
        "dropped_minutes": dropped / 60.0,
        "share": dropped / sounding if sounding else None,
        "by_stem": by_stem,
        "found_in": pooled(everything),
        **own_and_elsewhere(
            {stem: [r["lost"]["stems"][stem] for r in songs] for stem in PRODUCT_STEMS}
        ),
        "listed": sorted(listed, key=lambda e: (-e["dropped"], e["track_id"], e["stem"])),
        "in_no_stem_db": levels,
        "in_no_stem_time": times,
        "true_stems_minus_song_db": sums,
    }


def _percent(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{round(100 * value, digits) + 0.0:.{digits}f}%"  # never "-0%"


def _places(found: Mapping[str, float] | None) -> str:
    if found is None:
        return " | ".join("-" for _ in range(len(PLACES) + 1))
    cells = [_percent(found[place]) for place in PLACES]
    return " | ".join([*cells, _percent(sum(found[place] for place in PLACES))])


def _whole(value: float) -> int:
    """A share as the whole percent the tables show."""
    return round(100 * value)


def _reading(analysis: Mapping[str, Any]) -> list[str]:
    """How far the rows of the long table can be trusted, and what is in none of the six."""
    listed = analysis["listed"]
    close = [
        e
        for e in listed
        if abs(_whole(sum(e["found_in"][place] for place in PLACES)) - 100) <= SUM_POINTS
    ]
    below = sum(1 for e in listed if any(_whole(e["found_in"][place]) < 0 for place in PLACES))
    seconds = sum(e["dropped"] for e in close)
    lines = [
        "",
        "How to read the sum (this paragraph was added after the run; it changes no figure). "
        "The six delivered stems and what is in none of them add up to the song, so a row's "
        "sum is `<true, song> / <true, true>`. It leaves 100% by as much as the true stem "
        "resembles the other true stems over its dropped time, by sound they share or by "
        "chance. A row far from 100%, or with a share below zero, cannot be read share by "
        f"share. {len(close)} of the {len(listed)} stems listed are within {SUM_POINTS} "
        f"points of 100%, and they hold {seconds / 60.0:.1f} of the "
        f"{analysis['dropped_minutes']:.1f} minutes dropped; {below} have a share below zero. "
        f"A stem's own column is not zero because a stem that counts as silent (under "
        f"{SILENCE_DBFS:g} dBFS) is not empty.",
    ]
    found = analysis["found_in"]
    if found is not None:
        most = max(listed, key=lambda e: e["found_in"][NONE])
        text = (
            f"**In none of the six: {_percent(found[NONE])} of what is dropped**, over all "
            f"stems; {_percent(analysis['elsewhere'])} is in a stem of another name and "
            f"{_percent(analysis['own'])} in its own. "
            f"The most in none for one stem listed is {_percent(most['found_in'][NONE])} "
            f"(`{most['track_id'][:8]}` {most['stem']}, a row that adds up to "
            f"{_percent(sum(most['found_in'][place] for place in PLACES))})"
        )
        if close:
            best = max(close, key=lambda e: e["found_in"][NONE])
            text += (
                f"; among the rows within {SUM_POINTS} points of 100% it is "
                f"{_percent(best['found_in'][NONE])} (`{best['track_id'][:8]}` {best['stem']})"
            )
        lines += ["", text + "."]
    return lines


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    songs = analysis["songs"]
    limit = run.get("empty_below_mix_db")
    lines = [
        "# What the six stems lose",
        "",
        f"Written by `python -m stems_bench.lost` from the run `{run['run_id']}`: {songs} songs "
        f"of the frozen `{run.get('dataset', '?')}` list put through the worker's own path "
        f"with the real default model on `{run.get('device', '?')}`, all six stems asked for, "
        "and measured against the true stems before the audio was deleted. The measures were "
        f"written into `bench/stems_bench/lost.py` and committed (`{RULES_COMMIT[:7]}`) before "
        "any real song was measured"
        + (
            "; the run's own record shows it was made at that commit, with nothing uncommitted."
            if (run.get("git") or {}).get("commit") == RULES_COMMIT
            and (run.get("git") or {}).get("dirty") is False
            else "."
        ),
        "",
        "## How long a stem is silent while its instrument plays",
        "",
        "As in `docs/dropouts.md`: a tenth of a second is dropped when the true stem sounds "
        "in it and the product's stem does not; a dropout is long from "
        f"{LONG_SECONDS:g} seconds of true-stem sound inside one stretch of silence.",
        "",
        "| Stem | Songs with it | True stem sounds (min) | Dropped (min) | Share | "
        "With a long dropout |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stem, entry in analysis["by_stem"].items():
        lines.append(
            f"| {stem} | {entry['stems']} | {entry['sounding_minutes']:.1f} | "
            f"{entry['dropped_minutes']:.1f} | {_percent(entry['share'], 1)} | {entry['long']} |"
        )
    lines += [
        f"| all | | {analysis['sounding_minutes']:.1f} | {analysis['dropped_minutes']:.1f} | "
        f"{_percent(analysis['share'], 1)} | "
        f"{sum(e['long'] for e in analysis['by_stem'].values())} |",
        "",
        "## Where the instrument is when its stem is silent",
        "",
        "For the dropped time of each true stem: the share of it found in each of the six "
        "delivered stems, and in none of them (the song minus the six). A share is "
        "`<true, place> / <true, true>`; shares over several songs are weighted by dropped "
        "time. The last column is their sum, which is 100% when the true stems add up to the "
        "song and do not resemble one another.",
        "",
        "| True stem | " + " | ".join(f"in {p}" for p in PLACES[:-1]) + " | in none | Sum |",
        "|---|" + "---:|" * (len(PLACES) + 1),
    ]
    for stem, entry in analysis["by_stem"].items():
        lines.append(f"| {stem} | {_places(entry['found_in'])} |")
    lines.append(f"| all | {_places(analysis['found_in'])} |")
    lines += ["", f"Every stem with {LONG_SECONDS:g} seconds or more of dropped time:", ""]
    if analysis["listed"]:
        lines += [
            "| Song | True stem | Dropped (s) | "
            + " | ".join(f"in {p}" for p in PLACES[:-1])
            + " | in none | Sum |",
            "|---|---|---:|" + "---:|" * (len(PLACES) + 1),
        ]
        for entry in analysis["listed"]:
            lines.append(
                f"| `{entry['track_id'][:8]}` | {entry['stem']} | {entry['dropped']:.1f} | "
                f"{_places(entry['found_in'])} |"
            )
        lines += _reading(analysis)
    else:
        lines.append("None.")
    kept = {
        stem: entry["own_share_kept"]
        for stem, entry in analysis["by_stem"].items()
        if entry["own_share_kept"] is not None
    }
    if kept:
        lines += [
            "",
            "A check that the share means what it should: the same figure for each stem's own "
            "output over the time it is not dropped, median over songs. "
            + ", ".join(f"{stem} {_percent(value)}" for stem, value in kept.items())
            + ".",
        ]
    levels = analysis["in_no_stem_db"]
    lines += ["", "## How much of a song is in none of its six stems", ""]
    if levels:
        lines += [
            "What is left when the six delivered stems are taken away from the song, as a "
            "level against the song over its whole length: median "
            f"{-statistics.median(levels):.1f} dB under it (from {-max(levels):.1f} to "
            f"{-min(levels):.1f} dB under, over {len(levels)} songs).",
        ]
    times = analysis["in_no_stem_time"]
    if times and limit is not None:
        lines += [
            "",
            f"The product calls a stem very quiet when it is more than {-limit:g} dB under the "
            "song. Tenth of a second by tenth of a second, what is in no stem is louder than "
            f"that for {_percent(statistics.median(times), 1)} of a song's sounding time "
            f"(median; the most in one song is {_percent(max(times), 1)}).",
        ]
    sums = analysis["true_stems_minus_song_db"]
    lines += [
        "",
        "## What this is and is not",
        "",
        f"- {songs} songs, one dataset, one model. It is not a rate for music in general.",
        "- A share is a correlation, not a listening judgement: two instruments that play the "
        "same notes resemble one another, and then the shares do not add up to 100%.",
        *(
            [
                "- The dataset's six true stems add up to its mix: what they leave out is "
                f"{-max(sums):.1f} dB under the song or lower in every song (median "
                f"{-statistics.median(sums):.1f} dB)."
            ]
            if sums
            else []
        ),
        "- Everything here needs the true stems except the last section: what is in no stem "
        "can be worked out from the song and its six stems alone.",
        "- Nothing in the product was changed. The choice this bears on is BLOCKED.md D15.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from stems_bench.product_check import PRODUCT_DIR, read_results
    from stems_common.settings import load_defaults

    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.lost",
        description="Where a dropped instrument is, and what is in no stem.",
    )
    parser.add_argument("--run-id", required=True, help="a product_check run made with --lost")
    parser.add_argument("--out", help="write the report here (default: print it)")
    args = parser.parse_args(argv)
    run_dir = PRODUCT_DIR / args.run_id
    records = [r for r in read_results(run_dir) if r.get("lost")] if run_dir.is_dir() else []
    if not records:
        print(
            f"error: {run_dir} has no results made with --lost: run "
            "`python -m stems_bench.product_check run --lost` first",
            file=sys.stderr,
        )
        return 2
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    run = {
        **run,
        "run_id": args.run_id,
        "empty_below_mix_db": load_defaults().gate.rules.empty_below_mix_db,
    }
    text = render(analyse(records), run)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
