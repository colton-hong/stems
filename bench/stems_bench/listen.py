"""A listening test one person can do alone, at their own machine (Phase 13).

    python -m stems_bench.listen make --tracks 6 --device mps   # the audio (long; resumable)
    python -m stems_bench.listen serve                          # the page, on 127.0.0.1
    python -m stems_bench.listen report --out docs/listening-owner.md

The benchmark says which model scores best. It cannot say which one sounds best, and the
listening test the research report asks for needs a panel and two paid tools (BLOCKED.md
B8). This is the part of it that needs neither: the owner, blind, comparing what the
product delivers now with what it delivered before and with the true stems.

**What is compared**, for each stem of each song:

- ``reference``  the stem as it was recorded: the dataset's own;
- ``new``        what the product makes now, with the config as it is;
- ``old``        what it made with the single models it used before the recommendation of
                 Phase 10 (``docs/benchmarks/defaults-before-recommendation.json``).

``make`` runs the songs (the first N of the frozen MoisesDB test subset) through the
product's own job code for ``new`` and ``old``, and builds a kit with the existing kit
builder: the same 30 seconds of every version, matched in level, under letters whose
meaning differs from stem to stem. The key that says which letter is which stays outside
the folder the page is served from. ``serve`` shows one stem at a time, plays the versions
in step so that switching does not lose the place, and saves the marks after every step.
``report`` unblinds them.

The audio is real music. It lives only in the run's folder under ``runs/``, which is not
in the repository, and nothing here uploads anything: the server listens on this machine
only.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from stems_bench import listening_kit as kit_tools
from stems_bench.gate_outputs import IN_SONG_SECONDS
from stems_bench.listen_page import PAGE
from stems_bench.run import write_json
from stems_common.levels import silence_fraction
from stems_common.paths import REPO_ROOT, RUNS_DIR
from stems_common.stems import PRODUCT_STEMS

LISTEN_DIR = RUNS_DIR / "listening"
DEFAULT_RUN = "owner-v1"
DATASET = "moisesdb_subset"
SYSTEMS = ("reference", "new", "old")
OLD_MODELS_PATH = REPO_ROOT / "docs" / "benchmarks" / "defaults-before-recommendation.json"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 64 * 1024
MAX_NOTE = 2000
RATINGS = kit_tools.RATINGS
REQUIRED = "overall"


class ListenError(ValueError):
    """The test cannot be made, served or reported as asked."""


# -- making the audio ----------------------------------------------------------------------


def old_defaults(defaults: Any, old_models: Mapping[str, str]) -> Any:
    """The config with each stem's single model set back to what it was."""
    slots = dict(defaults.ensembles.slots)
    for stem, model in old_models.items():
        slots[stem] = slots[stem].model_copy(update={"single_best": model})
    ensembles = defaults.ensembles.model_copy(update={"slots": slots})
    return defaults.model_copy(update={"ensembles": ensembles})


def make_track(
    track: Any,
    stems_dir: Path,
    *,
    configs: Mapping[str, Any],
    device: str | None,
    pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    """Every system's stems for one song, as files. Returns which stems the song has.

    A stem whose instrument is not in the song is left out for every system: there is
    nothing to compare. A song that was already made is not made again.
    """
    from stems_worker.audio import save_audio
    from stems_worker.decision.job import JobRequest, run_job

    rate = track.sample_rate
    seconds = track.inputs["mix"].shape[-1] / rate
    marker = stems_dir / "done" / f"{track.track_id}.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    present = tuple(
        stem
        for stem in PRODUCT_STEMS
        if (1.0 - silence_fraction(track.references[stem], rate)) * seconds >= IN_SONG_SECONDS
    )
    mix_path = stems_dir / "mix" / f"{track.track_id}.wav"
    save_audio(mix_path, track.inputs["mix"], rate, "wav")
    folder = stems_dir / "reference" / track.track_id
    for stem in present:
        save_audio(folder / f"{stem}.flac", track.references[stem], rate, "flac")
    for name, defaults in configs.items():
        out = stems_dir / name / track.track_id
        if out.exists():
            shutil.rmtree(out)  # a half-made song from an interrupted run
        with tempfile.TemporaryDirectory(prefix="stems-listen-") as work:
            run_job(
                mix_path,
                Path(work) / "out",
                JobRequest(
                    stems=present, fmt="flac", device=device, cache_dir=None, timings_path=None
                ),
                defaults=defaults,
                # No Jev and no tagger, unless the caller's pipeline says otherwise.
                **{"client": None, "tagger": None, **pipeline},
            )
            out.mkdir(parents=True)
            for stem in present:
                shutil.move(str(Path(work) / "out" / f"{stem}.flac"), out / f"{stem}.flac")
    made = {"track_id": track.track_id, "stems": list(present), "seconds": round(seconds, 2)}
    write_json(marker, made)
    return made


def build_kit(run_dir: Path, tracks: Sequence[Mapping[str, Any]], *, seed: int = 0) -> None:
    """The kit the page is served from, and the key beside it (never inside it)."""
    from stems_worker.audio import load_audio, save_audio

    stems_dir, kit_dir = run_dir / "stems", run_dir / "kit"
    if kit_dir.exists():
        shutil.rmtree(kit_dir)
    manifest = {
        "systems": list(SYSTEMS),
        "ours": "new",
        "free_reference": "old",
        "paid_references": [],
        "stems": list(PRODUCT_STEMS),
        "tracks": [
            {
                "id": track["track_id"],
                "mix": str(stems_dir / "mix" / f"{track['track_id']}.wav"),
                "systems": {name: str(stems_dir / name / track["track_id"]) for name in SYSTEMS},
            }
            for track in tracks
        ],
    }
    kit_tools.build_kit(manifest, kit_dir, run_dir / "key.json", listeners=1, seed=seed)
    # The same excerpt of the full song, for reference while listening. Not marked.
    for number, _track in enumerate(tracks, start=1):
        mix = load_audio(manifest["tracks"][number - 1]["mix"], kit_tools.SAMPLE_RATE).samples
        window = kit_tools.loudest_window(mix, kit_tools.SAMPLE_RATE, kit_tools.EXCERPT_SECONDS)
        save_audio(
            kit_dir / "audio" / f"T{number:02d}" / "mix.wav",
            window.cut(mix),
            kit_tools.SAMPLE_RATE,
            "wav",
        )


def make(
    run_dir: Path,
    dataset: Any,
    track_ids: Sequence[str],
    *,
    defaults: Any,
    old_models: Mapping[str, str],
    device: str | None,
    pipeline: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Make every system's stems for the songs and build the kit. Resumable."""
    configs = {"new": defaults, "old": old_defaults(defaults, old_models)}
    made = []
    for number, track_id in enumerate(track_ids, start=1):
        if progress:
            progress(f"song {number} of {len(track_ids)}: {track_id}")
        made.append(
            make_track(
                dataset.load(track_id),
                run_dir / "stems",
                configs=configs,
                device=device,
                pipeline=pipeline or {},
            )
        )
    if progress:
        progress("building the kit")
    build_kit(run_dir, made)
    return made


# -- the marks -----------------------------------------------------------------------------


def presentation_order(kit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The kit's items in the order its one listener's sheet lists them, each with an id."""
    by_key = {(item["track"], item["stem"]): item for item in kit["items"]}
    order: list[tuple[str, str]] = []
    sheet = kit.get("_sheet_rows") or []
    for track, stem in sheet:
        if (track, stem) in by_key and (track, stem) not in order:
            order.append((track, stem))
    order += [key for key in by_key if key not in order]
    return [{"id": f"{track}/{stem}", **by_key[(track, stem)]} for track, stem in order]


def load_kit(kit_dir: Path) -> dict[str, Any]:
    """``kit.json`` with the order of the listener's sheet attached."""
    import csv

    path = kit_dir / "kit.json"
    if not path.is_file():
        raise ListenError(f"{path} is missing: run `make` first")
    kit = json.loads(path.read_text(encoding="utf-8"))
    sheet = kit_dir / "sheets" / "listener-01.csv"
    rows: list[tuple[str, str]] = []
    if sheet.is_file():
        with sheet.open(newline="", encoding="utf-8") as handle:
            rows = [(row["track"], row["stem"]) for row in csv.DictReader(handle)]
    kit["_sheet_rows"] = rows
    return kit


def read_ratings(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"items": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def check_marks(item: Mapping[str, Any], body: Any) -> dict[str, Any]:
    """The marks for one step, checked. Raises ``ListenError`` with what is wrong."""
    if not isinstance(body, dict) or not isinstance(body.get("ratings"), dict):
        raise ListenError("the marks must be an object with a 'ratings' object")
    if set(body["ratings"]) != set(item["samples"]):
        raise ListenError(f"marks are needed for exactly the versions {item['samples']}")
    clean: dict[str, dict[str, int | None]] = {}
    for sample in item["samples"]:
        given = body["ratings"][sample]
        if not isinstance(given, dict) or set(given) - set(RATINGS):
            raise ListenError(f"version {sample}: the marks are {list(RATINGS)}")
        clean[sample] = {}
        for rating in RATINGS:
            value = given.get(rating)
            if value is None and rating != REQUIRED:
                clean[sample][rating] = None
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ListenError(f"version {sample}: {rating} must be a whole number from 1 to 5")
            clean[sample][rating] = value
    note = body.get("note") or ""
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise ListenError(f"the note must be text of at most {MAX_NOTE} characters")
    return {"ratings": clean, "note": note}


def save_marks(path: Path, item_id: str, marks: Mapping[str, Any]) -> None:
    """Add one step's marks to the file, replacing any earlier ones for it. Atomic."""
    data = read_ratings(path)
    data["items"][item_id] = {**marks, "saved_at": datetime.now(UTC).isoformat(timespec="seconds")}
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


# -- the server ----------------------------------------------------------------------------


def make_server(run_dir: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """The listening page on 127.0.0.1. Serves the kit folder and nothing else of the run."""
    kit_dir = (run_dir / "kit").resolve()
    kit = load_kit(kit_dir)
    items = presentation_order(kit)
    by_id = {item["id"]: item for item in items}
    ratings_path = run_dir / "ratings.json"
    audio_root = (kit_dir / "audio").resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "StemsListening/1"

        def log_message(self, format: str, *args: Any) -> None:
            return  # the terminal stays quiet; nothing here needs a log

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            self._send(status, json.dumps(value).encode(), "application/json")

        def _local(self) -> bool:
            """Only this machine's own browser, addressed as this machine."""
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in ("127.0.0.1", "localhost"):
                return False
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return origin.split("://")[-1].split(":")[0] in ("127.0.0.1", "localhost")

        def do_GET(self) -> None:
            if not self._local():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "this page is local only"})
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/kit":
                return self._json(
                    HTTPStatus.OK,
                    {
                        "items": items,
                        "ratings": list(RATINGS),
                        "help": kit_tools.RATING_HELP,
                        "excerpt_seconds": kit["excerpt_seconds"],
                    },
                )
            if path == "/api/ratings":
                return self._json(HTTPStatus.OK, read_ratings(ratings_path))
            if path.startswith("/audio/") and path.endswith(".wav"):
                target = (audio_root / path[len("/audio/") :]).resolve()
                if audio_root in target.parents and target.is_file():
                    return self._send(HTTPStatus.OK, target.read_bytes(), "audio/wav")
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._local():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "this page is local only"})
            prefix = "/api/ratings/"
            item = by_id.get(self.path[len(prefix) :]) if self.path.startswith(prefix) else None
            if item is None:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "no such step"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY_BYTES:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "the marks are too large"})
            try:
                marks = check_marks(item, json.loads(self.rfile.read(length)))
            except (ListenError, json.JSONDecodeError, UnicodeDecodeError) as error:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            save_marks(ratings_path, item["id"], marks)
            return self._json(HTTPStatus.OK, {"saved": item["id"]})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


# -- the report ----------------------------------------------------------------------------


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test: the chance of a split at least this uneven if neither
    version were better. None when there is nothing to test."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def analyse(
    kit: Mapping[str, Any], key: Mapping[str, Any], ratings: Mapping[str, Any]
) -> dict[str, Any]:
    """The marks, unblinded: per system, per stem, and the new version against the old."""
    rows: list[dict[str, Any]] = []
    for item in presentation_order(kit):
        marks = ratings["items"].get(item["id"])
        if marks is None:
            continue
        labels = key["items"][item["id"]]["labels"]
        entry = {"id": item["id"], "stem": item["stem"], "note": marks.get("note") or ""}
        for label, system in labels.items():
            entry[system] = marks["ratings"][label]
        rows.append(entry)
    systems = list(key["systems"])

    def mean(values: Sequence[float | None]) -> float | None:
        kept = [v for v in values if v is not None]
        return statistics.fmean(kept) if kept else None

    by_system = {
        system: {rating: mean([row[system][rating] for row in rows]) for rating in RATINGS}
        for system in systems
    }
    by_stem = {
        stem: {
            "steps": len(mine),
            **{system: mean([row[system][REQUIRED] for row in mine]) for system in systems},
        }
        for stem in PRODUCT_STEMS
        if (mine := [row for row in rows if row["stem"] == stem])
    }

    def versus(a: str, b: str) -> dict[str, Any]:
        wins = sum(1 for row in rows if row[a][REQUIRED] > row[b][REQUIRED])
        losses = sum(1 for row in rows if row[a][REQUIRED] < row[b][REQUIRED])
        return {
            "higher": wins,
            "lower": losses,
            "same": len(rows) - wins - losses,
            "p": sign_test(wins, losses),
        }

    top = sum(
        1
        for row in rows
        if "reference" in row
        and row["reference"][REQUIRED] == max(row[s][REQUIRED] for s in systems)
    )
    return {
        "steps": len(kit["items"]),
        "marked": len(rows),
        "systems": systems,
        "by_system": by_system,
        "by_stem": by_stem,
        "new_vs_old": versus("new", "old") if {"new", "old"} <= set(systems) else None,
        "new_vs_reference": versus("new", "reference") if "reference" in systems else None,
        "reference_top": top,
        "notes": [{"id": row["id"], "note": row["note"]} for row in rows if row["note"].strip()],
    }


SYSTEM_NAMES = {
    "reference": "The true stem",
    "new": "The product now",
    "old": "The product before",
}


def _mark(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    systems = analysis["systems"]
    marked = analysis["marked"]
    lines = [
        "# A listening test by one listener",
        "",
        f"Written by `python -m stems_bench.listen report` from "
        f"`runs/listening/{run['run_id']}/ratings.json`. One person, the owner of this "
        "project, listened blind on their own machine: for each stem of each song, 30 seconds "
        "of every version, matched in level, under letters whose meaning changed from stem "
        "to stem. Marks are from 1 (bad) to 5 (excellent).",
        "",
        f"Songs: {run.get('tracks', '?')}, the first of the frozen `{DATASET}` list. "
        f"Steps marked: {marked} of {analysis['steps']}."
        + ("" if marked == analysis["steps"] else " **The test was not finished.**"),
        "",
        "| Version | Overall | Bleed | Artifacts |",
        "|---|---:|---:|---:|",
    ]
    for system in systems:
        scores = analysis["by_system"][system]
        lines.append(
            f"| {SYSTEM_NAMES.get(system, system)} | {_mark(scores['overall'])} | "
            f"{_mark(scores['bleed'])} | {_mark(scores['artifacts'])} |"
        )
    lines += [
        "",
        "Bleed and artifacts were optional and are means over the steps where they were "
        "given. Higher is better in every column.",
        "",
    ]
    if marked == 0:
        return "\n".join([*lines, "Nothing has been marked yet.", ""])
    versus = analysis["new_vs_old"]
    if versus:
        p = versus["p"]
        chance = (
            "nothing to test"
            if p is None
            else f"a split at least this uneven would happen by chance {100 * p:.0f}% of the time"
        )
        lines += [
            "## Now against before",
            "",
            f"Over {marked} steps the version the product makes now was marked higher than the "
            f"one it made before in {versus['higher']}, lower in {versus['lower']} and the same "
            f"in {versus['same']} ({chance}, by a sign test on the steps that differ).",
            "",
        ]
    lines += [
        "## By stem",
        "",
        "Mean overall mark.",
        "",
        "| Stem | Steps | " + " | ".join(SYSTEM_NAMES.get(s, s) for s in systems) + " |",
        "|---|---:|" + "---:|" * len(systems),
    ]
    for stem, entry in analysis["by_stem"].items():
        lines.append(
            f"| {stem} | {entry['steps']} | "
            + " | ".join(_mark(entry[system]) for system in systems)
            + " |"
        )
    reference = analysis["new_vs_reference"]
    if reference:
        lines += [
            "",
            "## Could the true stem be told from the separated ones?",
            "",
            f"The true stem had the highest mark, alone or shared, in "
            f"{analysis['reference_top']} of {marked} steps. The product's version was marked "
            f"below it in {reference['lower']}, the same in {reference['same']} and above it in "
            f"{reference['higher']}.",
        ]
    if analysis["notes"]:
        lines += ["", "## Notes written while listening", ""]
        lines += [f"- `{note['id']}`: {note['note'].strip()}" for note in analysis["notes"]]
    lines += [
        "",
        "## What this is and is not",
        "",
        "- One listener, who knows the project and what was being compared, though not which "
        "letter was which.",
        "- Songs from one dataset; 30 seconds of each, chosen where the song is loudest.",
        "- Not the test the research report describes (five listeners, two paid tools). "
        "That one still waits (BLOCKED.md B8).",
        "",
    ]
    return "\n".join(lines)


# -- command line --------------------------------------------------------------------------


def cmd_make(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset, load_frozen
    from stems_common.settings import load_defaults

    run_dir = LISTEN_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen(REPO_ROOT / "bench" / "frozen" / f"{DATASET}.json", DATASET)
    track_ids = [track["id"] for track in frozen.tracks][: args.tracks]
    old_models = json.loads(OLD_MODELS_PATH.read_text(encoding="utf-8"))
    status_path = run_dir / "status.json"
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(status_path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    try:
        made = make(
            run_dir,
            get_dataset(DATASET),
            track_ids,
            defaults=load_defaults(),
            old_models=old_models,
            device=args.device,
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
            "tracks": len(made),
            "device": args.device,
            "old_models": old_models,
            "steps": sum(len(track["stems"]) for track in made),
        },
    )
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    print(
        f"the kit is in {run_dir / 'kit'}; start the page with: python -m stems_bench.listen serve"
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    run_dir = LISTEN_DIR / args.run_id
    server = make_server(run_dir, args.port)
    print(f"Listening test: open http://127.0.0.1:{server.server_address[1]} in a browser.")
    print("Marks are saved after every step. Press Ctrl+C here to stop the page.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = LISTEN_DIR / args.run_id
    key_path = run_dir / "key.json"
    if not key_path.is_file():
        raise ListenError(f"{key_path} is missing: run `make` first")
    kit = load_kit(run_dir / "kit")
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    analysis = analyse(
        kit,
        json.loads(key_path.read_text(encoding="utf-8")),
        read_ratings(run_dir / "ratings.json"),
    )
    text = render(analysis, {"run_id": args.run_id, **run})
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.listen",
        description="A blind listening test one person can do alone.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    make_parser = sub.add_parser("make", help="make the audio and the kit (long; resumable)")
    make_parser.add_argument("--run-id", default=DEFAULT_RUN)
    make_parser.add_argument("--tracks", type=int, default=6)
    make_parser.add_argument("--device", default=None, help="default: the best this machine has")
    make_parser.set_defaults(run=cmd_make)
    serve_parser = sub.add_parser("serve", help="serve the listening page on this machine")
    serve_parser.add_argument("--run-id", default=DEFAULT_RUN)
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.set_defaults(run=cmd_serve)
    report_parser = sub.add_parser("report", help="unblind the marks and write the report")
    report_parser.add_argument("--run-id", default=DEFAULT_RUN)
    report_parser.add_argument("--out")
    report_parser.set_defaults(run=cmd_report)
    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except ListenError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
