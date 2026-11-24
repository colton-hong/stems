"""Real songs through the product's own path, set beside the benchmark (P9.2).

    python -m stems_bench.product_check run --tracks 3 --device cpu     # long; resumable
    python -m stems_bench.product_check report --out docs/product-vs-benchmark.md

The benchmark runs each model directly on a track. A user's song takes a longer road: it
is uploaded as a file, the worker reads it from storage, plans, separates, lets the gate
retry what looks wrong, and writes FLAC files back to storage
(``stems_worker.jobs.run_job``). Every test of that road so far used stand-in models or a
synthetic signal. This sends dataset tracks down it with the real models and scores what
a user would download, with the benchmark's own scorer and the same references.

**Fixed here, before any track is run:**

- The tracks are the first N of the frozen ``moisesdb_subset`` list, in the list's order.
- Each job asks for all six stems as FLAC, with no request text and no quality named, so
  the product's defaults decide. No Jev key is used: the rules decide. The tagger runs if
  ``listening.enabled`` says so, as in a real job.
- A delivered stem is set beside the benchmark's result for the same model and track when
  one model made it and the benchmark scored that model's output against the same stem.
  The benchmark ran on this Mac's GPU; the device used here is in the report, and the
  two agree when they differ by at most ``AGREE_DB``.
- An instrument is "in the song" if its reference sounds for at least five seconds in
  total (``stems_bench.gate_outputs.IN_SONG_SECONDS``).

The uploads and the delivered stems are real music. They live only in the run's storage
folder under ``runs/`` (never in the repository) and are deleted as soon as a track has
been scored; what is kept is numbers.
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stems_bench.gate_outputs import IN_SONG_SECONDS
from stems_bench.run import BENCH_DIR, git_state, write_json
from stems_common.levels import silence_fraction
from stems_common.paths import RUNS_DIR
from stems_common.stems import PRODUCT_STEMS

PRODUCT_DIR = RUNS_DIR / "product"
DEFAULT_RUN = "product-v1"
DATASET = "moisesdb_subset"
# Product and benchmark "agree" on a stem when their SDR differs by at most this.
AGREE_DB = 0.5
EMPTY_REASON = "mostly_silent"
_NAMESPACE = uuid.UUID("6f0f3f0e-5a55-4a4e-9c57-2d8d6c1f7b10")


class ProductCheckError(ValueError):
    """The check cannot be run or reported as asked."""


def job_id_for(run_id: str, track_id: str) -> str:
    """The same job id for the same track of the same run, so a rerun finds its work."""
    return str(uuid.uuid5(_NAMESPACE, f"{run_id}:{track_id}"))


def benchmark_scores(
    bench_dir: Path, dataset: str, track_id: str, model_id: str
) -> dict[str, dict[str, Any]]:
    """What the benchmark measured for one model on one track: output -> target, SDR, device.

    Empty when the benchmark has no result for the pair. Of several results (repeat runs)
    the first by file name is used.
    """
    folder = bench_dir / "results" / dataset / model_id
    if not folder.is_dir():
        return {}
    for path in sorted(folder.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("kind") != "model" or result.get("track_id") != track_id:
            continue
        return {
            name: {
                "target": scored.get("target"),
                "sdr_db": (scored.get("bss") or {}).get("sdr"),
                "device": result.get("device"),
            }
            for name, scored in result.get("outputs", {}).items()
        }
    return {}


def compare_stem(
    stem: Mapping[str, Any],
    sdr_db: float | None,
    benchmark: Callable[[str], Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    """The benchmark's figure for a delivered stem, when there is one to compare with.

    None when more than one model made the stem (a blend), or the benchmark did not score
    that model's output of this name against this stem (a four-stem model's "other").
    """
    models = list(stem["model_ids"])
    if len(models) != 1:
        return None
    measured = benchmark(models[0]).get(stem["name"])
    if not measured or measured["target"] != stem["name"] or measured["sdr_db"] is None:
        return None
    difference = None if sdr_db is None else round(sdr_db - measured["sdr_db"], 3)
    return {
        "model_id": models[0],
        "sdr_db": round(measured["sdr_db"], 3),
        "device": measured["device"],
        "difference_db": difference,
        "agrees": None if difference is None else abs(difference) <= AGREE_DB,
    }


def run_track(
    track: Any,
    *,
    run_id: str,
    storage: Any,
    defaults: Any,
    device: str,
    work: Path,
    benchmark: Callable[[str], Mapping[str, Mapping[str, Any]]],
    pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    """One track through the worker's storage entry point; scored; its audio deleted."""
    from stems_api.storage import put_json, read_bytes
    from stems_bench.run import score_estimates
    from stems_common import jobs as contract
    from stems_worker import jobs as worker
    from stems_worker.audio import load_audio, save_audio

    job_id = job_id_for(run_id, track.track_id)
    rate = track.sample_rate
    mix = track.inputs["mix"]
    seconds = mix.shape[-1] / rate
    upload = work / "upload.wav"
    save_audio(upload, mix, rate, "wav")
    spec = contract.build_spec(
        job_id, created_at=datetime.now(UTC), stems=list(PRODUCT_STEMS), fmt="flac"
    )
    with upload.open("rb") as handle:
        storage.put(contract.source_key(job_id), handle, "application/octet-stream")
    upload.unlink()
    put_json(storage, contract.spec_key(job_id), spec)
    try:
        started = time.perf_counter()
        result = worker.run_job(
            job_id, storage, defaults=defaults, device=device, timings_path=None, **pipeline
        )
        wall = time.perf_counter() - started
        record: dict[str, Any] = {
            "track_id": track.track_id,
            "job_id": job_id,
            "state": result["state"],
            "error": result.get("error"),
            "device": result.get("device"),
            "audio_seconds": round(seconds, 2),
            "wall_seconds": round(wall, 1),
            "timings": result["timings"],
            "stems": {},
        }
        if result["state"] != "ready":
            return record
        delivered: dict[str, Any] = {}
        for stem in result["stems"]:
            path = work / f"{stem['name']}.flac"
            path.write_bytes(read_bytes(storage, stem["key"]))
            delivered[stem["name"]] = load_audio(path).samples
            path.unlink()
        scorable = {
            name: audio
            for name, audio in delivered.items()
            if audio.shape[-1] == track.references[name].shape[-1]
        }
        scored, _ = score_estimates(track, "mix", scorable, {name: name for name in scorable})
        for stem in result["stems"]:
            name = stem["name"]
            bss = scored.get(name, {}).get("bss") or {}
            sdr = bss.get("sdr")
            assessment = (stem.get("assessment") or {}).get("assessment") or {}
            sounding = (1.0 - silence_fraction(track.references[name], rate)) * seconds
            record["stems"][name] = {
                "model_ids": stem["model_ids"],
                "mode": stem.get("ensemble_mode"),
                "attempts": stem["attempts"],
                "warning": stem.get("warning"),
                "reason_code": assessment.get("reason_code"),
                "flagged_empty": assessment.get("reason_code") == EMPTY_REASON,
                "reference_sounding_seconds": round(sounding, 2),
                "in_song": sounding >= IN_SONG_SECONDS,
                "sdr_db": None if sdr is None else round(sdr, 3),
                "sir_db": None if bss.get("sir") is None else round(bss["sir"], 3),
                "length_matches": name in scorable,
                "benchmark": compare_stem(stem, sdr, benchmark),
            }
        return record
    finally:
        # Real music: nothing of it outlives the track's scoring.
        storage.delete_prefix(contract.job_prefix(job_id))


def run(
    dataset: Any,
    track_ids: Sequence[str],
    run_dir: Path,
    *,
    run_id: str,
    defaults: Any,
    device: str,
    bench_dir: Path = BENCH_DIR,
    pipeline: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Run and score each track that has no result yet. Returns how many were run."""
    from stems_api.storage import LocalStorage

    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    storage = LocalStorage(run_dir / "storage", secrets.token_bytes(32))
    ran = 0
    for number, track_id in enumerate(track_ids, start=1):
        out = results / f"{track_id}.json"
        if out.is_file():
            continue
        if progress:
            progress(f"track {number} of {len(track_ids)}: {track_id}")
        track = dataset.load(track_id)
        cache: dict[str, dict[str, dict[str, Any]]] = {}

        def benchmark(model_id: str, track_id: str = track_id, cache: dict = cache) -> dict:
            if model_id not in cache:
                cache[model_id] = benchmark_scores(bench_dir, dataset.name, track_id, model_id)
            return cache[model_id]

        with tempfile.TemporaryDirectory(prefix="stems-product-") as work:
            record = run_track(
                track,
                run_id=run_id,
                storage=storage,
                defaults=defaults,
                device=device,
                work=Path(work),
                benchmark=benchmark,
                pipeline=pipeline or {},
            )
        write_json(out, record)
        ran += 1
    return ran


# -- analysis and report -------------------------------------------------------------------


def read_results(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "results").glob("*.json"))
    ]


def seconds_by_stage(record: Mapping[str, Any]) -> dict[str, float]:
    """Model seconds actually spent (not reused), by stage."""
    out: dict[str, float] = {}
    for timing in record["timings"]:
        if not timing["cached"]:
            out[timing["stage"]] = out.get(timing["stage"], 0.0) + timing["wall_seconds"]
    return {stage: round(value, 1) for stage, value in out.items()}


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the report states beyond the per-track rows."""
    ready = [record for record in records if record["state"] == "ready"]
    stems = [
        {"track_id": record["track_id"], "name": name, **stem}
        for record in ready
        for name, stem in record["stems"].items()
    ]
    compared = [s for s in stems if s["benchmark"] and s["benchmark"]["difference_db"] is not None]
    differences = [s["benchmark"]["difference_db"] for s in compared]
    absent = [s for s in stems if not s["in_song"]]
    present = [s for s in stems if s["in_song"]]
    return {
        "tracks": len(records),
        "ready": len(ready),
        "failed": [record["track_id"] for record in records if record["state"] != "ready"],
        "stems": len(stems),
        "compared": len(compared),
        "agree": sum(1 for s in compared if s["benchmark"]["agrees"]),
        "median_difference_db": round(statistics.median(differences), 3) if differences else None,
        "largest_difference_db": max(differences, key=abs) if differences else None,
        "not_in_song": len(absent),
        "not_in_song_flagged": sum(1 for s in absent if s["flagged_empty"]),
        "not_in_song_unflagged": [
            {
                "track_id": s["track_id"],
                "name": s["name"],
                "attempts": s["attempts"],
                "model_ids": list(s["model_ids"]),
            }
            for s in absent
            if not s["flagged_empty"]
        ],
        "in_song": len(present),
        "in_song_flagged": [
            {"track_id": s["track_id"], "name": s["name"], "sdr_db": s["sdr_db"]}
            for s in present
            if s["flagged_empty"]
        ],
        "audio_seconds": round(sum(record["audio_seconds"] for record in ready), 1),
        "wall_seconds": round(sum(record["wall_seconds"] for record in ready), 1),
    }


def _num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(records: Sequence[Mapping[str, Any]], run: Mapping[str, Any]) -> str:
    summary = summarise(records)
    git = run.get("git") or {}
    commit = str(git.get("commit") or "unknown")[:12] + (
        ", with uncommitted changes" if git.get("dirty") else ""
    )
    devices = sorted({str(record["device"]) for record in records if record.get("device")})
    lines = [
        "# The product beside the benchmark",
        "",
        f"Generated from `runs/product/{run['run_id']}/results/` (finished {run['finished']}, "
        f"commit `{commit}`). Reproduce with `python -m stems_bench.product_check run --tracks "
        f"{run['tracks']} --device {run['device']}` and then `report`; that needs MoisesDB and "
        "the model weights.",
        "",
        "The benchmark runs each model directly on a track. This sends the same tracks down "
        "the road a user's upload takes (a file in storage, the worker, its plan, the gate "
        "and its retries, FLAC files back in storage) with the real models, and scores what "
        "comes out with the benchmark's own scorer against the same references. No Jev key "
        "was used, so the rules decided. The audio was deleted after each track was scored.",
        "",
        f"Tracks: the first {run['tracks']} of the frozen `{DATASET}` list. Device here: "
        f"{', '.join(devices) or 'unknown'}; the benchmark's figures are from this Mac's GPU "
        "(`docs/benchmarks/moisesdb.md`), so a small difference between the two columns is "
        "expected (`docs/benchmarks/device-spread-synthetic.md`).",
        "",
        "## Stem by stem",
        "",
        'SDR in dB, the median over one-second windows, higher is better. "Benchmark" is '
        "the benchmark's figure for the same model on the same track, where one model made "
        'the stem and the benchmark scored that output against the same stem. "In song" '
        f"is whether the reference sounds for at least {IN_SONG_SECONDS:g} seconds.",
        "",
        "| Track | Stem | Made by | Attempts | In song | Product SDR | Benchmark SDR | "
        "Difference | Gate's note |",
        "|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for record in records:
        short = record["track_id"][:8]
        if record["state"] != "ready":
            lines.append(
                f"| `{short}` | - | - | - | - | - | - | - | job {record['state']}: "
                f"{record.get('error') or 'no reason given'} |"
            )
            continue
        for name in PRODUCT_STEMS:
            stem = record["stems"].get(name)
            if stem is None:
                continue
            bench = stem["benchmark"]
            note = (
                "probably empty"
                if stem["flagged_empty"]
                else "not checked"
                if stem["warning"]
                else "passed"
            )
            lines.append(
                f"| `{short}` | {name} | {' + '.join(f'`{m}`' for m in stem['model_ids'])} | "
                f"{stem['attempts']} | {'yes' if stem['in_song'] else 'no'} | "
                f"{_num(stem['sdr_db'])} | {_num(bench['sdr_db']) if bench else '-'} | "
                f"{_num(bench['difference_db']) if bench else '-'} | {note} |"
            )
    lines += [
        "",
        "A stem with no SDR has a reference that is silent throughout: there is nothing to "
        "score it against.",
        "",
        "## Does the product deliver what the benchmark measured?",
        "",
    ]
    if summary["compared"]:
        lines.append(
            f"{summary['compared']} delivered stems could be set beside a benchmark figure. "
            f"{summary['agree']} of them are within {AGREE_DB:g} dB of it. The median "
            f"difference (product minus benchmark) is {_num(summary['median_difference_db'])} dB "
            f"and the largest {_num(summary['largest_difference_db'])} dB."
        )
    else:
        lines.append("No delivered stem could be set beside a benchmark figure.")
    lines += [
        "",
        "## Did the gate flag the right stems?",
        "",
        f"Of {summary['not_in_song']} stems whose instrument is not in the song, the gate "
        f"flagged {summary['not_in_song_flagged']} as probably empty. Of {summary['in_song']} "
        f"stems whose instrument is in the song, it flagged {len(summary['in_song_flagged'])}.",
    ]
    for missed in summary["not_in_song_unflagged"]:
        made_by = " + ".join(f"`{m}`" for m in missed["model_ids"])
        how = (
            f"an earlier attempt was found empty, the retry with {made_by} was not, and the "
            "retry is what was delivered, without the note. The song has no such instrument, "
            "so what the retry found is something else"
            if missed["attempts"] > 1
            else f"the first attempt, by {made_by}, was not found empty"
        )
        lines.append(
            f"- `{missed['track_id'][:8]}` {missed['name']}: not in the song, not flagged: {how}."
        )
    for flagged in summary["in_song_flagged"]:
        lines.append(
            f"- `{flagged['track_id'][:8]}` {flagged['name']}: in the song, flagged; its SDR "
            f"is {_num(flagged['sdr_db'])} dB."
        )
    lines += [
        "",
        "## Time",
        "",
        "| Track | Audio | Wall time | Models | Tagger | Everything else |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        if record["state"] != "ready":
            continue
        stages = seconds_by_stage(record)
        listening = stages.get("listening", 0.0)
        models = round(sum(v for stage, v in stages.items() if stage != "listening"), 1)
        rest = round(record["wall_seconds"] - models - listening, 1)
        lines.append(
            f"| `{record['track_id'][:8]}` | {record['audio_seconds']:.0f} s | "
            f"{record['wall_seconds']:.0f} s | {models:.0f} s | {listening:.0f} s | {rest:.0f} s |"
        )
    lines += [
        "",
        f"All on {', '.join(devices) or 'an unknown device'}, on a machine that was also "
        "running a benchmark and at times this project's tests; these are single runs, not "
        'speed measurements. "Models" '
        "counts each model once (a blend that reuses an output costs nothing more). "
        '"Everything else" is decoding, the gate\'s level checks, writing FLAC and peaks, '
        "and the zip. Scoring is not included.",
        "",
        "## Limits",
        "",
        f"- {summary['tracks']} tracks. That is enough to see whether the road works and "
        "far too few to say how often it does.",
        "- One dataset, the product's default models, no Jev.",
        "- The worker's entry point was called directly. The API, the browser and the "
        "download links are not in this path; `make web-e2e-real` covers those on a "
        "synthetic signal.",
        "",
    ]
    if summary["failed"]:
        lines.insert(
            lines.index("## Stem by stem"),
            f"**{len(summary['failed'])} of {summary['tracks']} jobs did not finish.**\n",
        )
    return "\n".join(lines)


# -- command line --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from stems_bench.datasets import get_dataset, load_frozen
    from stems_common.paths import REPO_ROOT
    from stems_common.settings import load_defaults
    from stems_worker.listening.beats import BeatsTagger

    if args.tracks < 1:
        raise ProductCheckError("--tracks must be at least 1")
    run_dir = PRODUCT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen(REPO_ROOT / "bench" / "frozen" / f"{DATASET}.json", DATASET)
    track_ids = [track["id"] for track in frozen.tracks][: args.tracks]
    defaults = load_defaults()
    status_path = run_dir / "status.json"
    started = datetime.now(UTC).isoformat(timespec="seconds")

    def progress(message: str) -> None:
        write_json(status_path, {"state": "running", "started": started, "now": message})
        print(message, flush=True)

    tagger = None
    if defaults.listening.enabled:
        tagger = BeatsTagger.from_registry()
        tagger.load("cpu")
    try:
        run(
            get_dataset(DATASET),
            track_ids,
            run_dir,
            run_id=args.run_id,
            defaults=defaults,
            device=args.device,
            pipeline={"client": None, "tagger": tagger},
            progress=progress,
        )
    except BaseException as error:
        write_json(
            status_path, {"state": "failed", "started": started, "error": type(error).__name__}
        )
        raise
    finally:
        if tagger is not None:
            tagger.unload()
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "finished": finished,
            "tracks": args.tracks,
            "device": args.device,
            "tagger": None if tagger is None else tagger.name,
            "git": git_state(),
        },
    )
    write_json(status_path, {"state": "done", "started": started, "finished": finished})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = PRODUCT_DIR / args.run_id
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ProductCheckError(f"{run_path} is missing: the run has not finished")
    run_info = json.loads(run_path.read_text(encoding="utf-8"))
    records = read_results(run_dir)[: run_info["tracks"]]
    text = render(records, run_info)
    write_json(run_dir / "summary.json", summarise(records))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.product_check",
        description="Real songs through the product's own path, beside the benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run and score the tracks (long; resumable)")
    run_parser.add_argument("--run-id", default=DEFAULT_RUN)
    run_parser.add_argument("--tracks", type=int, default=3)
    run_parser.add_argument("--device", default="cpu")
    run_parser.set_defaults(run=cmd_run)
    report_parser = sub.add_parser("report", help="write the report from a finished run")
    report_parser.add_argument("--run-id", default=DEFAULT_RUN)
    report_parser.add_argument("--out")
    report_parser.set_defaults(run=cmd_report)
    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except ProductCheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
