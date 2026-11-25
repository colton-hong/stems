"""A listening test one person can do alone, at their own machine (Phase 13).

    python -m stems_bench.listen make --tracks 6 --device mps   # the audio (long; resumable)
    python -m stems_bench.listen serve                          # the page, on 127.0.0.1
    python -m stems_bench.listen pack --out DIR                 # the test for a phone
    python -m stems_bench.listen take MARKS --heard "on a phone, from ..."   # marks made elsewhere
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
meaning differs from stem to stem. Each stem's 30 seconds are where its true recording is
loudest, so the instrument is playing in every step. Where the product makes a stem with
the same model now as before, ``new`` and ``old`` are the same audio and it is played
once: the step then has two versions, not three. The key that says which letter is which
stays outside the folder the page is served from. ``serve`` shows one stem at a time,
plays the versions in step so that switching does not lose the place, and saves the marks
after every step. ``report`` unblinds them.

The audio is real music. It lives only in the run's folder under ``runs/``, which is not
in the repository, and nothing here uploads anything: the server listens on this machine
only.

``pack`` writes the test into a folder for a host that serves files, so that it can be
taken on a phone: another page (``listen_phone_page``) and the clips as 320 kbps MP3,
which every phone plays and which is a seventh of the size. That copy is not lossless,
and putting the folder anywhere is the owner's to do, since the clips are real music.
``take`` brings in marks the listener made somewhere else, on a copy of the page that
keeps its own marks (a phone, say). They get the checks the server gives its own, and
each is stored with where it came from, so that the report can say how the clips were
heard: lossless on this machine, or however ``--heard`` says.
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

import numpy as np

from stems_bench import listening_kit as kit_tools
from stems_bench.gate_outputs import IN_SONG_SECONDS
from stems_bench.listen_page import PAGE
from stems_bench.run import write_json
from stems_common.levels import level_below_mix_db, rms_dbfs, silence_fraction
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
# Two versions are "the same audio" when what differs between them is at least this far
# below the stem itself: the same model run twice, give or take arithmetic.
SAME_AUDIO_DB = 60.0
RATINGS = kit_tools.RATINGS
REQUIRED = "overall"
# The phone's copy of a clip: MP3 at libsndfile's highest constant bitrate, 320 kbps.
PHONE_SUFFIX = "mp3"
PHONE_SHORT_HELP = {
    "overall": "1 bad · 5 excellent",
    "bleed": "1 heavy · 5 clean",
    "artifacts": "1 heavy · 5 clean",
}
# Where a step was marked. Marks saved by this machine's own page carry no name: they are
# "here", and were heard from the kit's own lossless clips.
VIA_HERE = "here"


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


def same_models(configs: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """The stems the product makes with the same single model now as before."""
    new, old = configs["new"].ensembles.slots, configs["old"].ensembles.slots
    return {
        stem: {"old": "new"}
        for stem in PRODUCT_STEMS
        if new[stem].single_best == old[stem].single_best
    }


def same_audio(a: Path, b: Path) -> bool:
    """Whether two files hold the same audio, give or take arithmetic."""
    from stems_worker.audio import load_audio

    first = load_audio(a, kit_tools.SAMPLE_RATE).samples
    second = load_audio(b, kit_tools.SAMPLE_RATE).samples
    if first.shape != second.shape:
        return False
    if np.array_equal(first, second):
        return True
    return rms_dbfs(first - second) <= max(rms_dbfs(first), rms_dbfs(second)) - SAME_AUDIO_DB


def build_kit(
    run_dir: Path,
    tracks: Sequence[Mapping[str, Any]],
    *,
    same_as: Mapping[str, Mapping[str, str]] | None = None,
    empty_below_mix_db: float | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """The kit the page is served from, and the key beside it (never inside it).

    ``same_as`` names the stems that two versions make with the same model. It is believed
    only where the files agree for every song: otherwise both versions are played.

    ``empty_below_mix_db`` is the level, against the whole song, under which the product
    itself calls a stem empty (``gate.rules.empty_below_mix_db``). A stem whose true
    recording is that quiet is left out of the kit: there is next to nothing to hear, and a
    mark for "silence" against "a little bleed" would say nothing about quality. Returns
    what was left out, which is also written beside the key.
    """
    from stems_bench.clef.clips import Window
    from stems_worker.audio import load_audio, save_audio

    stems_dir, kit_dir = run_dir / "stems", run_dir / "kit"
    if (run_dir / "ratings.json").exists():
        raise ListenError(
            f"{run_dir / 'ratings.json'} holds marks for the kit that is there; a new kit "
            "would change what the letters mean. Move the marks away first, or use another "
            "--run-id"
        )
    if kit_dir.exists():
        shutil.rmtree(kit_dir)
    judged: dict[str, list[str]] = {}
    left_out: list[dict[str, Any]] = []
    for number, track in enumerate(tracks, start=1):
        track_id = track["track_id"]
        judged[track_id] = list(track["stems"])
        if empty_below_mix_db is None:
            continue
        mix = load_audio(stems_dir / "mix" / f"{track_id}.wav", kit_tools.SAMPLE_RATE).samples
        for stem in track["stems"]:
            true = load_audio(
                stems_dir / "reference" / track_id / f"{stem}.flac", kit_tools.SAMPLE_RATE
            ).samples
            below = level_below_mix_db(true, mix)
            if below < empty_below_mix_db:
                judged[track_id].remove(stem)
                left_out.append(
                    {"track": f"T{number:02d}", "stem": stem, "below_song_db": round(below, 2)}
                )
    write_json(
        run_dir / "left_out.json", {"empty_below_mix_db": empty_below_mix_db, "steps": left_out}
    )
    played_once = {
        stem: dict(twins)
        for stem, twins in (same_as or {}).items()
        if all(
            same_audio(
                stems_dir / name / track["track_id"] / f"{stem}.flac",
                stems_dir / other / track["track_id"] / f"{stem}.flac",
            )
            for track in tracks
            if stem in judged[track["track_id"]]
            for name, other in twins.items()
        )
    }
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
                "stems": judged[track["track_id"]],
            }
            for track in tracks
        ],
    }
    key = kit_tools.build_kit(
        manifest,
        kit_dir,
        run_dir / "key.json",
        listeners=1,
        seed=seed,
        window_by="reference",
        same_as=played_once,
    )
    # The same half minute of the full song beside each step's versions. Not marked.
    length = round(kit_tools.EXCERPT_SECONDS * kit_tools.SAMPLE_RATE)
    for number, entry in enumerate(manifest["tracks"], start=1):
        code = f"T{number:02d}"
        mix = load_audio(entry["mix"], kit_tools.SAMPLE_RATE).samples
        for item_id, item in key["items"].items():
            track_code, stem = item_id.split("/")
            if track_code != code:
                continue
            window = Window(round(item["start_seconds"] * kit_tools.SAMPLE_RATE), length)
            save_audio(
                kit_dir / "audio" / code / stem / "song.wav",
                window.cut(mix),
                kit_tools.SAMPLE_RATE,
                "wav",
            )
    return left_out


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
    build_kit(
        run_dir,
        made,
        same_as=same_models(configs),
        empty_below_mix_db=configs["new"].gate.rules.empty_below_mix_db,
    )
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
    _write_ratings(path, data)


def _write_ratings(path: Path, data: Mapping[str, Any]) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


# -- the test in a folder, for a phone --------------------------------------------------------


def pack(run_dir: Path, out_dir: Path, *, run_id: str) -> dict[str, Any]:
    """Write the test as a folder a file host can serve: the page and the clips as MP3.

    ``page.html`` is the page as a host that wraps pages takes it; ``index.html`` is the
    same inside a whole document. Nothing of the key goes into the folder. Clips already
    there are not made again. Returns the number of clips and their size.
    """
    import soundfile as sf

    from stems_bench.listen_phone_page import PAGE as PHONE_PAGE

    if "MP3" not in sf.available_formats():
        raise ListenError("the audio library here cannot write MP3 (libsndfile 1.1 or later)")
    kit_dir = run_dir / "kit"
    items = presentation_order(load_kit(kit_dir))
    sizes: dict[str, int] = {}
    for item in items:
        for name in ("song", *item["samples"]):
            source = kit_dir / "audio" / item["track"] / item["stem"] / f"{name}.wav"
            if not source.is_file():
                raise ListenError(f"{source} is missing: the kit is not whole")
            place = Path("audio") / item["track"] / item["stem"] / f"{name}.{PHONE_SUFFIX}"
            target = out_dir / place
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                audio, rate = sf.read(str(source), dtype="float64", always_2d=True)
                partial = target.with_name(target.name + ".partial")
                sf.write(
                    str(partial),
                    audio,
                    rate,
                    format="MP3",
                    compression_level=0.0,
                    bitrate_mode="CONSTANT",
                )
                partial.replace(target)
            sizes[place.as_posix()] = target.stat().st_size
    # Two seconds of silence for the media element that keeps an iPhone's sound on.
    sf.write(str(out_dir / "silence.wav"), np.zeros((16000, 1)), 8000, subtype="PCM_16")
    total = sum(sizes.values())
    data = {
        "run": run_id,
        "items": [{k: item[k] for k in ("id", "track", "stem", "samples")} for item in items],
        "ratings": list(RATINGS),
        "help": kit_tools.RATING_HELP,
        "short": PHONE_SHORT_HELP,
        "clip_format": PHONE_SUFFIX,
        "song_format": PHONE_SUFFIX,
        "size_note": (
            f"Each step loads about {total / len(items) / 1e6:.0f} MB of audio, about "
            f"{total / 1e6:.0f} MB over the {len(items)} steps, so Wi-Fi is kinder than "
            "mobile data."
        ),
    }
    # JSON inside a script element: a "<" must not be able to end the element.
    page = PHONE_PAGE.replace("__KIT__", json.dumps(data).replace("<", "\\u003c"))
    (out_dir / "page.html").write_text(page, encoding="utf-8")
    (out_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}'
        "[hidden]{display:none!important}</style></head><body>\n" + page + "</body></html>\n",
        encoding="utf-8",
    )
    return {"steps": len(items), "clips": len(sizes), "bytes": total, "files": sorted(sizes)}


# -- marks made somewhere else --------------------------------------------------------------


def read_taken(source: Path) -> dict[str, Any]:
    """Marks as a page elsewhere gives them out: step id -> what was saved for the step.

    ``source`` is either the text such a page copies out, saved as a file
    (``{"items": {step: marks}}``), or a folder holding the documents of its store, one
    JSON file a step (the marks themselves or ``{"data": marks}``, naming their step as
    ``item``). Nothing is checked here beyond the shape; ``take_marks`` checks the marks.
    """

    def load(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ListenError(f"{path} is not a JSON file that can be read") from error

    if source.is_dir():
        taken: dict[str, Any] = {}
        for path in sorted(source.rglob("*.json")):
            document = load(path)
            if isinstance(document, dict) and isinstance(document.get("data"), dict):
                document = document["data"]
            if not isinstance(document, dict) or not isinstance(document.get("item"), str):
                raise ListenError(f"{path} does not say which step its marks are for")
            taken[document["item"]] = document
        if not taken:
            raise ListenError(f"{source} holds no marks")
        return taken
    if not source.is_file():
        raise ListenError(f"{source} is missing")
    data = load(source)
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict) or not data["items"]:
        raise ListenError(f"{source} holds no marks: it needs an 'items' object")
    return dict(data["items"])


def _when(value: Any) -> datetime | None:
    """A saved time as a moment, or None if it is not one. A time with no zone is UTC."""
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def take_marks(run_dir: Path, taken: Mapping[str, Any], *, heard: str, via: str) -> dict[str, int]:
    """Write marks made elsewhere into the run's ``ratings.json``.

    Every step gets the checks the server gives its own page, and all of them are checked
    before anything is written. Of two sets of marks for a step the later one stays.
    ``via`` names where these were made and ``heard`` says, for the report, how the clips
    were heard there. Returns how many steps were new, replaced, unchanged or older.
    """
    if not via.isidentifier() or via == VIA_HERE:
        raise ListenError(f"--via must be one word other than '{VIA_HERE}', not {via!r}")
    heard = " ".join(heard.split())
    if not heard or len(heard) > 300:
        raise ListenError("--heard must say how the clips were heard, in at most 300 characters")
    kit = load_kit(run_dir / "kit")
    items = {item["id"]: item for item in presentation_order(kit)}
    unknown = sorted(set(taken) - set(items))
    if unknown:
        raise ListenError(f"the kit has no step {', '.join(unknown)}")
    checked: dict[str, dict[str, Any]] = {}
    for item_id, body in taken.items():
        try:
            marks = check_marks(items[item_id], body)
        except ListenError as error:
            raise ListenError(f"{item_id}: {error}") from None
        moment = _when(body.get("saved_at"))
        if moment is None:
            raise ListenError(f"{item_id}: the marks do not say when they were saved")
        saved_at = moment.astimezone(UTC).isoformat(timespec="seconds")
        checked[item_id] = {**marks, "saved_at": saved_at, "via": via}
    path = run_dir / "ratings.json"
    data = read_ratings(path)
    counts = {"new": 0, "replaced": 0, "unchanged": 0, "older": 0}
    for item_id, entry in checked.items():
        before = data["items"].get(item_id)
        if before is None:
            counts["new"] += 1
        elif before == entry:
            counts["unchanged"] += 1
            continue
        elif (there := _when(before.get("saved_at"))) and there >= _when(entry["saved_at"]):
            counts["older"] += 1  # what is there was saved later: it stays
            continue
        else:
            counts["replaced"] += 1
        data["items"][item_id] = entry
    data.setdefault("heard", {})[via] = heard
    _write_ratings(path, data)
    return counts


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
        twins = key["items"][item["id"]].get("same_as", {})
        entry = {
            "id": item["id"],
            "stem": item["stem"],
            "note": marks.get("note") or "",
            "twins": twins,
            "via": marks.get("via") or VIA_HERE,
        }
        for label, system in labels.items():
            entry[system] = marks["ratings"][label]
        for system, other in twins.items():
            entry[system] = entry[other]  # the same audio: one clip, one mark
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
        # Only the steps where the two were two clips: one clip cannot differ from itself.
        mine = [row for row in rows if row["twins"].get(a) != b and row["twins"].get(b) != a]
        wins = sum(1 for row in mine if row[a][REQUIRED] > row[b][REQUIRED])
        losses = sum(1 for row in mine if row[a][REQUIRED] < row[b][REQUIRED])
        return {
            "steps": len(mine),
            "higher": wins,
            "lower": losses,
            "same": len(mine) - wins - losses,
            "p": sign_test(wins, losses),
        }

    one_clip = [row for row in rows if row["twins"].get("old") == "new"]

    top = sum(
        1
        for row in rows
        if "reference" in row
        and row["reference"][REQUIRED] == max(row[s][REQUIRED] for s in systems)
    )
    # Where the steps were marked: here first, then the others as they come.
    places = sorted(dict.fromkeys(row["via"] for row in rows), key=lambda via: via != VIA_HERE)
    return {
        "steps": len(kit["items"]),
        "marked": len(rows),
        "heard": [
            {
                "via": via,
                "steps": sum(1 for row in rows if row["via"] == via),
                "how": (ratings.get("heard") or {}).get(via),
            }
            for via in places
        ],
        "systems": systems,
        "by_system": by_system,
        "by_stem": by_stem,
        "new_vs_old": versus("new", "old") if {"new", "old"} <= set(systems) else None,
        "same_model": {
            "steps": len(one_clip),
            "stems": [stem for stem in PRODUCT_STEMS if any(r["stem"] == stem for r in one_clip)],
        },
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


def _heard(analysis: Mapping[str, Any]) -> list[str]:
    """How the clips were heard, when any step was marked away from this machine."""
    heard = analysis.get("heard") or []
    if all(entry["via"] == VIA_HERE for entry in heard):
        return []
    parts = []
    for entry in heard:
        steps = f"{entry['steps']} {'step was' if entry['steps'] == 1 else 'steps were'} marked"
        if entry["via"] == VIA_HERE:
            parts.append(f"{steps} on the listener's own machine, from the kit's lossless clips.")
        elif entry["how"]:
            parts.append(f"{steps} {entry['how']}.")
        else:
            parts.append(f"{steps} somewhere else, and how the clips were heard is not recorded.")
    return ["**How the clips were heard.** " + " ".join(parts), ""]


def render(analysis: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    systems = analysis["systems"]
    marked = analysis["marked"]
    heard = _heard(analysis)
    lines = [
        "# A listening test by one listener",
        "",
        f"Written by `python -m stems_bench.listen report` from "
        f"`runs/listening/{run['run_id']}/ratings.json`. One person, the owner of this "
        f"project, listened blind{'' if heard else ' on their own machine'}: for each stem of "
        "each song, 30 seconds of every version, matched in level, under letters whose "
        "meaning changed from stem to stem. Marks are from 1 (bad) to 5 (excellent).",
        "",
        f"Songs: {run.get('tracks', '?')}, the first of the frozen `{DATASET}` list. "
        f"Steps marked: {marked} of {analysis['steps']}."
        + ("" if marked == analysis["steps"] else " **The test was not finished.**"),
        "",
        *heard,
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
    same = analysis["same_model"]
    if same["steps"]:
        lines += [
            f"In {same['steps']} of the marked steps ({', '.join(same['stems'])}) the product "
            "makes the stem with the same model now as before. The two are then the same "
            "audio; it was played once, and its mark counts for both versions above and in "
            "the table by stem.",
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
            f"Over the {versus['steps']} steps where the two come from different models, the "
            f"version the product makes now was marked higher than the "
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
        "- Songs from one dataset; 30 seconds of each stem, chosen where its true recording "
        "is loudest.",
        *(
            [
                "- Left out, because the true recording is so quiet that the product itself "
                f"would call such a stem empty (more than {-run['empty_below_mix_db']:g} dB "
                "under the song): "
                + "; ".join(
                    f"`{step['track']}/{step['stem']}` ({-step['below_song_db']:.1f} dB under)"
                    for step in run["left_out"]
                )
                + "."
            ]
            if run.get("left_out")
            else []
        ),
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

    defaults = load_defaults()
    try:
        made = make(
            run_dir,
            get_dataset(DATASET),
            track_ids,
            defaults=defaults,
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
    left = json.loads((run_dir / "left_out.json").read_text(encoding="utf-8"))
    write_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "finished": finished,
            "tracks": len(made),
            "device": args.device,
            "old_models": old_models,
            "new_models": {
                stem: defaults.ensembles.slots[stem].single_best for stem in PRODUCT_STEMS
            },
            "steps": len(load_kit(run_dir / "kit")["items"]),
            "left_out": left["steps"],
            "empty_below_mix_db": left["empty_below_mix_db"],
        },
    )
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    for step in left["steps"]:
        print(
            f"left out: {step['track']} {step['stem']} (its true recording is "
            f"{-step['below_song_db']:.1f} dB under the song)"
        )
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


def cmd_pack(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    made = pack(LISTEN_DIR / args.run_id, out_dir, run_id=args.run_id)
    print(
        f"{made['steps']} steps, {made['clips']} clips, {made['bytes'] / 1e6:.1f} MB as MP3, "
        f"in {out_dir}: page.html for a host that wraps pages, index.html for any other."
    )
    print("The clips are real music: where the folder goes is yours to decide.")
    return 0


def cmd_take(args: argparse.Namespace) -> int:
    run_dir = LISTEN_DIR / args.run_id
    counts = take_marks(run_dir, read_taken(Path(args.source)), heard=args.heard, via=args.via)
    print(
        f"{counts['new']} new, {counts['replaced']} replaced, {counts['unchanged']} unchanged, "
        f"{counts['older']} left as they were (saved later here): {run_dir / 'ratings.json'}"
    )
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
    pack_parser = sub.add_parser("pack", help="write the test as a folder for a phone")
    pack_parser.add_argument("--run-id", default=DEFAULT_RUN)
    pack_parser.add_argument("--out", required=True, help="the folder to write")
    pack_parser.set_defaults(run=cmd_pack)
    take_parser = sub.add_parser("take", help="bring in marks made somewhere else")
    take_parser.add_argument(
        "source", help="the text the page copied out, in a file, or a folder of its documents"
    )
    take_parser.add_argument("--run-id", default=DEFAULT_RUN)
    take_parser.add_argument("--via", default="phone", help="one word for where they were made")
    take_parser.add_argument(
        "--heard",
        required=True,
        help="how the clips were heard there; it finishes the sentence 'N steps were marked ...'",
    )
    take_parser.set_defaults(run=cmd_take)
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
