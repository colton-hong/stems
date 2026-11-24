"""The blind listening test, as a kit (research report section 9; task P7.5).

    python -m stems_bench.listening_kit build MANIFEST.yaml --out KIT --key KEY.json
    python -m stems_bench.listening_kit score KIT --key KEY.json --sheets DIR --out REPORT.md

``build`` takes the stems that several systems made from the same tracks and writes a
folder to hand to listeners: for each track and stem, the same 30-second excerpt from
every system, matched in level, under the names A, B, C, D in an order that differs from
one stem to the next. Nothing in the folder says which system is which. That is in the
key, which is written somewhere else and stays with whoever runs the test. Each listener
gets a scoring sheet whose rows are in an order of their own.

``score`` reads the filled-in sheets and the key, and reports mean scores per system and
whether "ours" clears the bar the report set for saying it beats the other tools.

The manifest::

    ours: v1                      # the system under test
    free_reference: uvr           # the free tool it must beat by a margin
    paid_references: [lalal, moises]
    tracks:
      - id: song-01
        mix: /path/to/song-01.wav          # only used to choose the excerpt
        systems:
          v1: /path/to/v1/song-01          # a folder holding vocals.wav, drums.wav, ...
          uvr: /path/to/uvr/song-01

Getting the other systems' stems (the paid ones need a plan bought, BLOCKED.md B8) and
finding listeners is the owner's part. This tool has only ever been run on synthetic
audio: no listening test has taken place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stems_bench.clef.clips import loudest_window
from stems_common.levels import peak_dbfs, rms_dbfs
from stems_common.stems import PRODUCT_STEMS

KIT_VERSION = 1
EXCERPT_SECONDS = 30.0
SAMPLE_RATE = 44100
LABELS = "ABCDEFGH"
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif", ".ogg", ".opus")
# A stem quieter than this in the excerpt is treated as silent and not level-matched.
SILENT_DBFS = -60.0
# Every excerpt is kept at or under this peak; if one would exceed it, all systems'
# versions of that stem are turned down together, so they stay matched.
PEAK_CEILING_DBFS = -1.0
RATINGS = ("overall", "bleed", "artifacts")
RATING_HELP = {
    "overall": "overall quality, 1 (bad) to 5 (excellent)",
    "bleed": "bleed from other instruments, 1 (heavy bleed) to 5 (clean)",
    "artifacts": "artifacts, 1 (heavy artifacts) to 5 (clean)",
}

# The report's bar for "beats existing tools" (section 9).
MARGIN_OVER_FREE = 0.3  # points of mean overall quality, per track
TRACKS_WON_SHARE = 0.7  # "at least 7 of 10 tracks"
MAX_LOSS_ON_A_STEM = 0.5  # points
MIN_LISTENERS = 5
PLANNED_TRACKS = 10


class KitError(ValueError):
    """The kit cannot be built or scored as asked."""


# -- reading the manifest --------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list) or not data["tracks"]:
        raise KitError("the manifest needs a list of tracks")
    ours = data.get("ours")
    systems: list[str] | None = None
    seen: set[str] = set()
    for track in data["tracks"]:
        if not isinstance(track, dict) or not {"id", "mix", "systems"} <= set(track):
            raise KitError("every track needs an id, a mix and its systems")
        track_id = str(track["id"])
        if track_id in seen:
            raise KitError(f"track {track_id} is listed twice")
        seen.add(track_id)
        names = sorted(track["systems"])
        if systems is None:
            systems = names
        elif names != systems:
            raise KitError(
                f"track {track_id} has systems {names}; every track must have the same "
                f"systems ({systems})"
            )
    assert systems is not None
    if len(systems) < 2:
        raise KitError("a comparison needs at least two systems")
    if len(systems) > len(LABELS):
        raise KitError(f"at most {len(LABELS)} systems can be compared")
    if ours not in systems:
        raise KitError(f"'ours' must name one of the systems {systems}")
    references = [data.get("free_reference"), *(data.get("paid_references") or [])]
    for name in references:
        if name is not None and name not in systems:
            raise KitError(f"reference {name!r} is not one of the systems {systems}")
    return {
        "ours": ours,
        "free_reference": data.get("free_reference"),
        "paid_references": list(data.get("paid_references") or []),
        "systems": systems,
        "stems": [str(s) for s in data.get("stems") or PRODUCT_STEMS],
        "tracks": data["tracks"],
    }


def find_stem(folder: Path, stem: str) -> Path | None:
    for extension in AUDIO_EXTENSIONS:
        candidate = Path(folder) / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


# -- building --------------------------------------------------------------------------------


def _order(seed: int, *parts: str) -> np.random.Generator:
    """A random source that depends on the seed and on what is being shuffled."""
    digest = hashlib.sha256("\x1f".join([str(seed), *parts]).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def match_levels(
    excerpts: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Bring every system's excerpt of one stem to the same RMS level.

    The target is the mean level (in dB) of the versions that are not silent. A silent
    version is left alone: turning silence up would only raise its noise. If any matched
    version would peak above the ceiling, all are turned down by the same amount.
    Returns the excerpts and the gain applied to each, in dB.
    """
    levels = {name: rms_dbfs(audio) for name, audio in excerpts.items()}
    audible = [level for level in levels.values() if level > SILENT_DBFS]
    if not audible:
        return dict(excerpts), dict.fromkeys(excerpts, 0.0)
    target = float(np.mean(audible))
    gains = {n: (target - lv if lv > SILENT_DBFS else 0.0) for n, lv in levels.items()}
    worst = max(peak_dbfs(excerpts[n]) + gains[n] for n in excerpts if levels[n] > SILENT_DBFS)
    trim = min(0.0, PEAK_CEILING_DBFS - worst)
    out, applied = {}, {}
    for name, audio in excerpts.items():
        gain = gains[name] + (trim if levels[name] > SILENT_DBFS else 0.0)
        out[name] = (audio * np.float32(10.0 ** (gain / 20.0))).astype(np.float32)
        applied[name] = round(gain, 3)
    return out, applied


def build_kit(
    manifest: Mapping[str, Any],
    out_dir: Path,
    key_path: Path,
    *,
    listeners: int = MIN_LISTENERS,
    seed: int = 0,
    seconds: float = EXCERPT_SECONDS,
) -> dict[str, Any]:
    """Write the kit to ``out_dir`` and the key to ``key_path``. Returns the key."""
    from stems_worker.audio import load_audio, save_audio

    out_dir, key_path = Path(out_dir), Path(key_path)
    if out_dir.resolve() in key_path.resolve().parents or key_path.resolve() == out_dir.resolve():
        raise KitError("the key must not be inside the kit: the kit is what listeners get")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise KitError(f"{out_dir} is not empty; build into a new folder")
    if listeners < 1:
        raise KitError("at least one listener is needed")
    systems: list[str] = manifest["systems"]
    items: list[dict[str, Any]] = []
    key_items: dict[str, Any] = {}
    for number, track in enumerate(manifest["tracks"], start=1):
        track_id, code = str(track["id"]), f"T{number:02d}"
        mix = load_audio(track["mix"], SAMPLE_RATE)
        window = loudest_window(mix.samples, SAMPLE_RATE, seconds)
        for stem in manifest["stems"]:
            files = {name: find_stem(Path(track["systems"][name]), stem) for name in systems}
            if not any(files.values()):
                continue  # nobody made this stem for this track
            missing = [name for name, file in files.items() if file is None]
            if missing:
                raise KitError(
                    f"track {track_id}: {stem} is missing for {', '.join(missing)}; a stem "
                    "is compared across every system or not at all"
                )
            excerpts = {}
            for name, file in files.items():
                audio = load_audio(file, SAMPLE_RATE).samples
                if abs(audio.shape[1] - mix.samples.shape[1]) > SAMPLE_RATE // 10:
                    theirs, whole = audio.shape[1] / SAMPLE_RATE, mix.samples.shape[1] / SAMPLE_RATE
                    raise KitError(
                        f"track {track_id}: {name}'s {stem} is {theirs:.1f} s "
                        f"long but the mix is {whole:.1f} s; the same "
                        "excerpt cannot be cut from both"
                    )
                excerpts[name] = window.cut(audio)
            matched, gains = match_levels(excerpts)
            order = list(systems)
            _order(seed, track_id, stem).shuffle(order)
            labels = {LABELS[i]: name for i, name in enumerate(order)}
            for label, name in labels.items():
                save_audio(
                    out_dir / "audio" / code / stem / f"{label}.wav", matched[name], SAMPLE_RATE
                )
            items.append({"track": code, "stem": stem, "samples": list(labels)})
            key_items[f"{code}/{stem}"] = {
                "labels": labels,
                "gains_db": {label: gains[name] for label, name in labels.items()},
            }
    if not items:
        raise KitError("no stem was found for any track")
    kit = {
        "kit_version": KIT_VERSION,
        "excerpt_seconds": seconds,
        "listeners": listeners,
        "ratings": list(RATINGS),
        "items": items,
    }
    (out_dir / "kit.json").write_text(json.dumps(kit, indent=2) + "\n", encoding="utf-8")
    key = {
        "kit_version": KIT_VERSION,
        "seed": seed,
        "ours": manifest["ours"],
        "free_reference": manifest["free_reference"],
        "paid_references": manifest["paid_references"],
        "systems": systems,
        "tracks": {
            f"T{n:02d}": {
                "id": str(track["id"]),
            }
            for n, track in enumerate(manifest["tracks"], start=1)
        },
        "items": key_items,
    }
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    sheets = out_dir / "sheets"
    sheets.mkdir()
    for listener in range(1, listeners + 1):
        rows = [(item["track"], item["stem"]) for item in items]
        _order(seed, "listener", str(listener)).shuffle(rows)
        with (sheets / f"listener-{listener:02d}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track", "stem", "sample", *RATINGS, "would_use", "cleanest"])
            for track_code, stem in rows:
                samples = next(
                    i["samples"] for i in items if (i["track"], i["stem"]) == (track_code, stem)
                )
                for index, sample in enumerate(samples):
                    # "cleanest" is asked once per stem: on its first row.
                    writer.writerow(
                        [track_code, stem, sample, "", "", "", "", "" if index == 0 else "-"]
                    )
    (out_dir / "README.md").write_text(instructions(kit), encoding="utf-8")
    return key


def instructions(kit: Mapping[str, Any]) -> str:
    stems = len(kit["items"])
    samples = sum(len(item["samples"]) for item in kit["items"])
    return f"""# Listening test

Thank you for listening. This takes a while: there are {stems} stems to judge, {samples}
short clips in all, each {kit["excerpt_seconds"]:.0f} seconds long.

## Before you start

- Use the same headphones or speakers for the whole test, at one volume. Do not change
  the volume between clips of the same stem.
- You have your own sheet in `sheets/` (`listener-01.csv` and so on). Open it in any
  spreadsheet program. Work down it in the order it is in: everyone's order is different
  on purpose.
- The clips are in `audio/<track>/<stem>/`. Each stem has clips named A, B, C and so on.
  They are the same 30 seconds of the same song, separated by different tools. The
  letters mean nothing and change from stem to stem.
- Do not look at file sizes, dates or anything else to guess which is which. There is
  nothing in this folder that says.

## What to fill in

For every clip, three scores from 1 to 5:

- **overall**: {RATING_HELP["overall"]}
- **bleed**: {RATING_HELP["bleed"]}
- **artifacts**: {RATING_HELP["artifacts"]}

and **would_use**: `y` or `n`. Would you use this stem in a mix?

For every stem, once, on its first row: **cleanest**: the letter of the clip that sounds
cleanest to you. Rows marked `-` in that column need nothing there.

Listen as often as you like. When you are done, save the sheet as CSV and send it back.
"""


# -- scoring ---------------------------------------------------------------------------------


def read_sheet(path: Path, kit: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The ratings in one sheet, and what is wrong with it (missing or invalid cells)."""
    expected = {
        (item["track"], item["stem"], sample) for item in kit["items"] for sample in item["samples"]
    }
    samples_of = {(i["track"], i["stem"]): i["samples"] for i in kit["items"]}
    rows, problems, seen = [], [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            ident = (
                row.get("track", "").strip(),
                row.get("stem", "").strip(),
                row.get("sample", "").strip(),
            )
            where = f"{path.name} line {line}"
            if ident not in expected:
                problems.append(f"{where}: {'/'.join(ident)} is not a clip of this kit")
                continue
            if ident in seen:
                problems.append(f"{where}: {'/'.join(ident)} appears twice")
                continue
            seen.add(ident)
            entry: dict[str, Any] = {"track": ident[0], "stem": ident[1], "sample": ident[2]}
            for rating in RATINGS:
                cell = (row.get(rating) or "").strip()
                if cell not in {"1", "2", "3", "4", "5"}:
                    problems.append(f"{where}: {rating} must be 1 to 5, not {cell!r}")
                    break
                entry[rating] = int(cell)
            else:
                use = (row.get("would_use") or "").strip().lower()
                if use not in {"y", "n", "yes", "no"}:
                    problems.append(f"{where}: would_use must be y or n, not {use!r}")
                    continue
                entry["would_use"] = use.startswith("y")
                cleanest = (row.get("cleanest") or "").strip().upper()
                if cleanest not in {"", "-"}:
                    if cleanest not in samples_of[ident[:2]]:
                        problems.append(
                            f"{where}: cleanest names {cleanest!r}, not a clip of this stem"
                        )
                        continue
                    entry["cleanest"] = cleanest
                rows.append(entry)
    for missing in sorted(expected - seen):
        problems.append(f"{path.name}: no row for {'/'.join(missing)}")
    return rows, problems


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def score_kit(
    kit: Mapping[str, Any],
    key: Mapping[str, Any],
    sheets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Unblind the sheets and judge the pass bar. ``sheets`` maps listener to rows."""
    systems: list[str] = list(key["systems"])
    ratings: list[dict[str, Any]] = []
    cleanest: dict[str, int] = dict.fromkeys(systems, 0)
    votes = 0
    for listener, rows in sheets.items():
        for row in rows:
            labels = key["items"][f"{row['track']}/{row['stem']}"]["labels"]
            ratings.append({**row, "listener": listener, "system": labels[row["sample"]]})
            if "cleanest" in row:
                cleanest[labels[row["cleanest"]]] += 1
                votes += 1

    def mean_of(system: str, rating: str = "overall", **where: str) -> float | None:
        return _mean(
            [
                r[rating]
                for r in ratings
                if r["system"] == system and all(r[k] == v for k, v in where.items())
            ]
        )

    tracks = sorted({r["track"] for r in ratings})
    stems = list(dict.fromkeys(i["stem"] for i in kit["items"]))
    per_system = {
        system: {
            **{rating: mean_of(system, rating) for rating in RATINGS},
            "would_use": _mean([float(r["would_use"]) for r in ratings if r["system"] == system]),
            "cleanest_votes": cleanest[system],
            "ratings": sum(1 for r in ratings if r["system"] == system),
        }
        for system in systems
    }
    per_track = {t: {s: mean_of(s, track=t) for s in systems} for t in tracks}
    per_stem = {st: {s: mean_of(s, stem=st) for s in systems} for st in stems}

    ours, free, paid = (
        key["ours"],
        key.get("free_reference"),
        list(key.get("paid_references") or []),
    )
    checks: dict[str, Any] = {}
    if free is not None:
        won = [
            t
            for t in tracks
            if per_track[t][ours] is not None
            and per_track[t][free] is not None
            and per_track[t][ours] - per_track[t][free] >= MARGIN_OVER_FREE - 1e-9
        ]
        needed = int(np.ceil(TRACKS_WON_SHARE * len(tracks) - 1e-9))
        checks["beats_free_reference"] = {
            "tracks_won": len(won),
            "tracks": len(tracks),
            "needed": needed,
            "met": len(won) >= needed and bool(tracks),
        }
    if paid:
        gaps = {
            p: (per_system[ours]["overall"] or 0) - (per_system[p]["overall"] or 0) for p in paid
        }
        checks["beats_paid_references"] = {
            "gaps": gaps,
            "met": all(gap > 0 for gap in gaps.values()),
        }
    others = [s for s in systems if s != ours]
    worst: tuple[float, str, str] | None = None
    for stem in stems:
        for other in others:
            mine, theirs = per_stem[stem][ours], per_stem[stem][other]
            if mine is None or theirs is None:
                continue
            if worst is None or theirs - mine > worst[0]:
                worst = (theirs - mine, stem, other)
    if worst is not None:
        checks["no_stem_lost_badly"] = {
            "worst_loss": max(0.0, worst[0]),
            "stem": worst[1],
            "against": worst[2],
            "met": worst[0] <= MAX_LOSS_ON_A_STEM + 1e-9,
        }
    complete = free is not None and bool(paid)
    return {
        "listeners": len(sheets),
        "ratings": len(ratings),
        "cleanest_votes": votes,
        "tracks": len(tracks),
        "per_system": per_system,
        "per_track": per_track,
        "per_stem": per_stem,
        "checks": checks,
        "panel_big_enough": len(sheets) >= MIN_LISTENERS,
        "all_references_present": complete,
        # The claim needs every check, every reference, and a panel of the planned size.
        "passes": complete
        and len(sheets) >= MIN_LISTENERS
        and bool(checks)
        and all(check["met"] for check in checks.values()),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render(result: Mapping[str, Any], key: Mapping[str, Any], problems: Sequence[str]) -> str:
    ours = key["ours"]
    lines = [
        "# Listening test: results",
        "",
        f"{result['listeners']} listeners, {result['tracks']} tracks, {result['ratings']} clip "
        f"ratings. The system under test is `{ours}`.",
        "",
        "## Mean scores per system (1 to 5, higher is better)",
        "",
        "| System | Overall | Bleed | Artifacts | Would use | Voted cleanest |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system, s in result["per_system"].items():
        use = "n/a" if s["would_use"] is None else f"{100 * s['would_use']:.0f}%"
        lines.append(
            f"| `{system}` | {_fmt(s['overall'])} | {_fmt(s['bleed'])} | {_fmt(s['artifacts'])} | "
            f"{use} | {s['cleanest_votes']} of {result['cleanest_votes']} |"
        )
    systems = list(result["per_system"])
    for title, table in (("stem", result["per_stem"]), ("track", result["per_track"])):
        lines += [
            "",
            f"## Mean overall quality per {title}",
            "",
            f"| {title.capitalize()} | " + " | ".join(f"`{s}`" for s in systems) + " |",
            "|---|" + "---:|" * len(systems),
        ]
        for name, row in table.items():
            label = key["tracks"][name]["id"] if title == "track" else name
            lines.append(f"| {label} | " + " | ".join(_fmt(row[s]) for s in systems) + " |")
    checks = result["checks"]
    lines += [
        "",
        "## The bar for saying it beats existing tools",
        "",
        "| Check | Measured | Result |",
        "|---|---|---|",
    ]
    if "beats_free_reference" in checks:
        c = checks["beats_free_reference"]
        lines.append(
            f"| Beats `{key['free_reference']}` by {MARGIN_OVER_FREE} points or more on at least "
            f"{c['needed']} of {c['tracks']} tracks | {c['tracks_won']} tracks | "
            f"{'pass' if c['met'] else 'fail'} |"
        )
    if "beats_paid_references" in checks:
        c = checks["beats_paid_references"]
        gaps = ", ".join(f"`{name}` {gap:+.2f}" for name, gap in c["gaps"].items())
        verdict = "pass" if c["met"] else "fail"
        lines.append(
            f"| Higher mean overall quality than each paid reference | {gaps} | {verdict} |"
        )
    if "no_stem_lost_badly" in checks:
        c = checks["no_stem_lost_badly"]
        lines.append(
            f"| Loses on no stem by more than {MAX_LOSS_ON_A_STEM} points | "
            f"worst: {c['worst_loss']:.2f} on {c['stem']} against `{c['against']}` | "
            f"{'pass' if c['met'] else 'fail'} |"
        )
    lines += ["", "## Verdict", ""]
    if result["passes"]:
        lines.append(f"`{ours}` clears the bar on this test.")
    else:
        reasons = [name.replace("_", " ") for name, c in checks.items() if not c["met"]]
        if not result["all_references_present"]:
            reasons.append("the free and the paid references were not all in the test")
        if not result["panel_big_enough"]:
            reasons.append(f"fewer than {MIN_LISTENERS} listeners")
        lines.append(
            f"`{ours}` does **not** clear the bar: {'; '.join(reasons)}. The claim that it beats "
            "existing tools is not supported by this test."
        )
    if result["tracks"] != PLANNED_TRACKS:
        lines += [
            "",
            f"The report planned {PLANNED_TRACKS} tracks; this test had {result['tracks']}.",
        ]
    if problems:
        lines += ["", "## Problems in the sheets", "", "Rows with a problem were left out.", ""]
        lines += [f"- {problem}" for problem in problems]
    lines.append("")
    return "\n".join(lines)


# -- commands --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.listening_kit",
        description="Build a blind listening test from several systems' stems, and score it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write the kit for listeners and the key for you")
    build.add_argument("manifest", type=Path)
    build.add_argument("--out", type=Path, required=True, help="the folder listeners get")
    build.add_argument(
        "--key", type=Path, required=True, help="where the key goes (not in the kit)"
    )
    build.add_argument("--listeners", type=int, default=MIN_LISTENERS)
    build.add_argument("--seed", type=int, default=0)
    score = commands.add_parser("score", help="unblind the returned sheets and report")
    score.add_argument("kit", type=Path)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--sheets", type=Path, required=True, help="folder of filled-in CSV sheets")
    score.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            key = build_kit(
                load_manifest(args.manifest),
                args.out,
                args.key,
                listeners=args.listeners,
                seed=args.seed,
            )
            print(f"kit: {args.out} ({len(key['items'])} stems, {args.listeners} sheets)")
            print(f"key: {args.key} (keep it away from the listeners)")
            return 0
        kit = json.loads((args.kit / "kit.json").read_text(encoding="utf-8"))
        key = json.loads(args.key.read_text(encoding="utf-8"))
        sheets, problems = {}, []
        for path in sorted(args.sheets.glob("*.csv")):
            rows, found = read_sheet(path, kit)
            sheets[path.stem] = rows
            problems += found
        if not sheets:
            raise KitError(f"no CSV sheets in {args.sheets}")
        text = render(score_kit(kit, key, sheets), key, problems)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0
    except (KitError, FileNotFoundError) as exc:
        print(f"listening_kit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
