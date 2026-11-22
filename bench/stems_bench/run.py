"""Run models over a dataset's frozen track list and score every output. Resumable.

The unit of work is one model on one track. Its result is a JSON file under
``runs/bench/results/``, named by a hash of everything that could change it: the protocol
version, the dataset and its frozen list, the track, the model and its pinned weights, the
device asked for and the chunk overlap. A job whose result file exists is never run again,
by this run or any other, so a killed run picks up where it stopped.

A run is a list of such jobs with its own folder, ``runs/bench/<run-id>/``:

- ``run.json``: what was asked for, on which commit, with which package versions;
- ``status.json``: rewritten after every job (state, counts, the job in hand, failures);
- ``results.jsonl``: one line per job with its outcome and the path of its result file.

Models are taken one at a time and loaded once for all of their pending tracks.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.datasets import Dataset, DatasetError, TrackAudio
from stems_bench.metrics import bss_eval_stem
from stems_bench.protocol import PROTOCOL_VERSION, input_for, target_for
from stems_bench.signal_metrics import compare
from stems_common.paths import REPO_ROOT, RUNS_DIR
from stems_models.registry import ModelEntry, Registry
from stems_worker.cache import CACHE_VERSION, OutputCache, audio_sha256
from stems_worker.separator import DEFAULT_OVERLAP, Separator, run_separator
from stems_worker.timing import TimingLog

# Results live in BENCH_DIR/results, kept estimates in BENCH_DIR/estimates.
BENCH_DIR = RUNS_DIR / "bench"
# Bump when the content or meaning of a result file changes.
RESULT_VERSION = 1
_PACKAGES = ("torch", "numpy", "scipy", "museval", "onnxruntime", "demucs")

Factory = Callable[[str, tuple[str, ...]], Sequence[Separator]]


class BenchError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model_ids: tuple[str, ...]
    device: str
    overlap: float = DEFAULT_OVERLAP
    batch_size: int = 1
    # None means every track of the frozen list.
    track_ids: tuple[str, ...] | None = None
    # Models whose separated audio is kept (for blending later). Scores are always kept.
    keep_estimates: tuple[str, ...] = ()


@dataclass
class JobOutcome:
    model_id: str
    track_id: str
    outcome: str  # computed | cached | failed | not_applicable
    result: str | None = None  # path of the result file, relative to the repository
    detail: str | None = None


@dataclass
class RunSummary:
    run_id: str
    run_dir: Path
    state: str
    jobs: list[JobOutcome] = field(default_factory=list)

    def count(self, outcome: str) -> int:
        return sum(job.outcome == outcome for job in self.jobs)

    @property
    def ok(self) -> bool:
        return self.state == "done" and self.count("failed") == 0


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_state() -> dict[str, Any]:
    """The commit of this checkout and whether tracked files differ from it."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_json(path: Path, data: Any) -> None:
    """Write atomically: a reader sees the old file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def result_key(
    *,
    dataset: str,
    frozen_sha256: str,
    track_id: str,
    model_id: str,
    fingerprint: str,
    device: str,
    overlap: float,
) -> str:
    payload = {
        "result_version": RESULT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "inference_version": CACHE_VERSION,
        "dataset": dataset,
        "frozen_sha256": frozen_sha256,
        "track_id": track_id,
        "model_id": model_id,
        "fingerprint": fingerprint,
        "device": device,
        "overlap": overlap,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def applicability(entry: ModelEntry, dataset: Dataset) -> str | None:
    """Why a model cannot be scored on a dataset, or None if it can."""
    needs = input_for(entry.stems)
    if needs not in dataset.inputs:
        return f"needs the {needs!r} input, which {dataset.name} does not have"
    targets = {
        target_for(o, other_contains_guitar_piano=entry.other_contains_guitar_piano)
        for o in entry.stems
    }
    if {"guitar", "piano", "other"} <= set(entry.stems):
        targets.add("other_4stem")
    if not targets & set(dataset.targets):
        return f"{dataset.name} has no reference for any of its outputs ({sorted(entry.stems)})"
    return None


def score_outputs(
    entry: ModelEntry, track: TrackAudio, estimates: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Score each estimate against its reference. Returns (scored, reasons for the rest)."""
    mix = track.inputs[input_for(entry.stems)]
    estimates = dict(estimates)
    derived = set()
    if {"guitar", "piano", "other"} <= set(estimates) and "other_4stem" in track.references:
        # A six-stem model's answer to the four-stem "other" is the sum of the three.
        estimates["other_4stem"] = estimates["guitar"] + estimates["piano"] + estimates["other"]
        derived.add("other_4stem")
    scored: dict[str, Any] = {}
    unscored: dict[str, str] = {}
    for name, estimate in estimates.items():
        target = (
            name
            if name in derived
            else target_for(name, other_contains_guitar_piano=entry.other_contains_guitar_piano)
        )
        reference = track.references.get(target)
        if reference is None:
            unscored[name] = f"no {target!r} reference in this dataset"
            continue
        scored[name] = {
            "target": target,
            "derived": name in derived,
            "reference_present": bool(reference.any()),
            "bss": bss_eval_stem(reference, estimate, mix, track.sample_rate).as_dict(),
            "signal": compare(estimate, reference, track.sample_rate),
        }
    return scored, unscored


class _Status:
    def __init__(self, path: Path, base: dict[str, Any], total: int) -> None:
        self.path = path
        self.data = {
            **base,
            "state": "running",
            "pid": os.getpid(),
            "started_at": now(),
            "updated_at": now(),
            "total": total,
            "finished": 0,
            "computed": 0,
            "cached": 0,
            "failed": 0,
            "not_applicable": 0,
            "current": None,
            "failures": [],
        }
        self.write()

    def write(self) -> None:
        self.data["updated_at"] = now()
        write_json(self.path, self.data)

    def working_on(self, model_id: str, track_id: str) -> None:
        self.data["current"] = {"model_id": model_id, "track_id": track_id}
        self.write()

    def finished(self, job: JobOutcome) -> None:
        self.data["finished"] += 1
        self.data[job.outcome] += 1
        if job.outcome == "failed":
            self.data["failures"].append(
                {"model_id": job.model_id, "track_id": job.track_id, "error": job.detail}
            )
        self.data["current"] = None
        self.write()

    def end(self, state: str, error: str | None = None) -> None:
        self.data["state"] = state
        self.data["current"] = None
        if error:
            self.data["error"] = error
        self.write()


class _Runner:
    """The state of one invocation of ``run_benchmark``."""

    def __init__(
        self,
        spec: RunSpec,
        *,
        dataset: Dataset,
        factory: Factory,
        fingerprint: Callable[[str], str],
        identity: dict[str, Any],
        bench_dir: Path,
        summary: RunSummary,
        status: _Status,
        manifest: Any,
        timings: TimingLog | None,
        weights_sha256: Callable[[str], dict[str, str]] | None,
        say: Callable[[str], None],
    ) -> None:
        self.spec = spec
        self.dataset = dataset
        self.factory = factory
        self.fingerprint = fingerprint
        self.identity = identity
        self.bench_dir = bench_dir
        self.results_dir = bench_dir / "results" / dataset.name
        self.summary = summary
        self.status = status
        self.manifest = manifest
        self.timings = timings
        self.weights_sha256 = weights_sha256
        self.say = say
        self._track: TrackAudio | None = None

    # --- bookkeeping ---------------------------------------------------------------------

    def record(self, job: JobOutcome) -> None:
        self.summary.jobs.append(job)
        self.manifest.write(json.dumps(job.__dict__, sort_keys=True) + "\n")
        self.manifest.flush()
        self.status.finished(job)
        detail = f" ({job.detail})" if job.detail else ""
        self.say(f"{job.model_id} / {job.track_id}: {job.outcome}{detail}")

    def result_path(self, entry: ModelEntry, track_id: str) -> tuple[str, Path]:
        key = result_key(
            dataset=self.dataset.name,
            frozen_sha256=self.dataset.frozen.sha256,
            track_id=track_id,
            model_id=entry.id,
            fingerprint=self.fingerprint(entry.id),
            device=self.spec.device,
            overlap=self.spec.overlap,
        )
        return key, self.results_dir / entry.id / f"{key}.json"

    def load_track(self, track_id: str) -> TrackAudio:
        """One track in memory at a time."""
        if self._track is None or self._track.track_id != track_id:
            self._track = None
            self._track = self.dataset.load(track_id)
        return self._track

    # --- one model -----------------------------------------------------------------------

    def run_model(self, entry: ModelEntry, track_ids: Sequence[str]) -> None:
        reason = applicability(entry, self.dataset)
        pending: list[str] = []
        for track_id in track_ids:
            if reason is not None:
                self.record(JobOutcome(entry.id, track_id, "not_applicable", detail=reason))
                continue
            _, path = self.result_path(entry, track_id)
            if path.exists():
                self.record(JobOutcome(entry.id, track_id, "cached", _relative(path)))
            else:
                pending.append(track_id)
        if not pending:
            return

        parts: dict[str, dict[str, Any]] = {t: _new_part() for t in pending}
        failed: dict[str, str] = {}
        finished: set[str] = set()
        try:
            per_stem = tuple(f.stem for f in entry.weights if f.stem is not None)
            separators = list(self.factory(entry.id, per_stem or tuple(entry.stems)))
        except Exception as e:
            separators = []
            failed = dict.fromkeys(pending, f"{type(e).__name__}: {e}")

        for index, separator in enumerate(separators):
            last = index == len(separators) - 1
            started = time.perf_counter()
            try:
                separator.load(self.spec.device)
            except Exception as e:
                for track_id in pending:
                    failed.setdefault(track_id, f"{type(e).__name__}: {e}")
                separator.unload()
                continue
            load_seconds = time.perf_counter() - started
            try:
                for track_id in pending:
                    if track_id not in failed:
                        self.status.working_on(entry.id, track_id)
                        try:
                            self.separate_and_score(entry, separator, track_id, parts[track_id])
                            parts[track_id]["load"] += load_seconds
                        except Exception as e:  # one track failing must not stop the others
                            failed[track_id] = f"{type(e).__name__}: {e}"
                    if last:
                        self.finish(entry, track_id, parts[track_id], failed.get(track_id))
                        finished.add(track_id)
            finally:
                separator.unload()
        # Whatever the loop did not reach: the model was not built, or its last part did
        # not load.
        for track_id in pending:
            if track_id not in finished:
                error = failed.get(track_id, "the model did not run")
                self.finish(entry, track_id, parts[track_id], error)

    def separate_and_score(
        self, entry: ModelEntry, separator: Separator, track_id: str, part: dict[str, Any]
    ) -> None:
        spec = self.spec
        track = self.load_track(track_id)
        audio = track.inputs[input_for(entry.stems)]
        cache = None
        if entry.id in spec.keep_estimates:
            cache = OutputCache(
                self.bench_dir / "estimates",
                audio_hash=audio_sha256(audio, track.sample_rate),
                overlap=spec.overlap,
                device=spec.device,
                fingerprint=self.fingerprint,
            )
        hit = cache.get(entry.id, tuple(separator.outputs)) if cache is not None else None
        if hit is not None:
            stems = dict(hit.stems)
            wall = float(hit.meta.get("wall_seconds", 0.0))
            chunks = int(hit.meta.get("num_chunks", 0))
            device = str(hit.meta.get("device", spec.device))
            part["from_cache"] = True
        else:
            run = run_separator(
                separator,
                audio,
                track.sample_rate,
                overlap=spec.overlap,
                batch_size=spec.batch_size,
            )
            stems, wall, chunks, device = run.stems, run.wall_seconds, run.num_chunks, run.device
            if self.timings is not None:
                self.timings.record(
                    model_id=entry.id,
                    device=device,
                    outputs=list(stems),
                    audio_seconds=run.audio_seconds,
                    wall_seconds=wall,
                    num_chunks=chunks,
                    stage="bench",
                )
            if cache is not None:
                meta = {"device": device, "wall_seconds": wall, "num_chunks": chunks}
                cache.put(entry.id, stems, meta)
        bad = [name for name, x in stems.items() if not np.isfinite(x).all()]
        if bad:
            raise ValueError(f"non-finite samples in {bad}")
        started = time.perf_counter()
        scored, unscored = score_outputs(entry, track, stems)
        part["eval"] += time.perf_counter() - started
        part["outputs"].update(scored)
        part["unscored"].update(unscored)
        part["wall"] += wall
        part["chunks"] += chunks
        part["device"] = device
        part["audio_seconds"] = track.duration_seconds

    def finish(
        self, entry: ModelEntry, track_id: str, part: dict[str, Any], error: str | None
    ) -> None:
        if error is not None:
            self.record(JobOutcome(entry.id, track_id, "failed", detail=error))
            return
        key, path = self.result_path(entry, track_id)
        audio_seconds = part["audio_seconds"]
        write_json(
            path,
            {
                "result_version": RESULT_VERSION,
                "key": key,
                **{k: v for k, v in self.identity.items() if k != "device"},
                "track_id": track_id,
                "model_id": entry.id,
                "fingerprint": self.fingerprint(entry.id),
                "weights_sha256": self.weights_sha256(entry.id) if self.weights_sha256 else {},
                "device_requested": self.spec.device,
                "device": part["device"],
                "batch_size": self.spec.batch_size,
                "input": input_for(entry.stems),
                "audio_seconds": round(audio_seconds, 3),
                "wall_seconds": round(part["wall"], 4),
                "seconds_per_audio_second": round(part["wall"] / audio_seconds, 5),
                "load_seconds": round(part["load"], 3),
                "num_chunks": part["chunks"],
                "estimates_from_cache": part["from_cache"],
                "eval_seconds": round(part["eval"], 3),
                "outputs": part["outputs"],
                "unscored_outputs": part["unscored"],
                "at": now(),
                "git": git_state(),
                "packages": package_versions(),
            },
        )
        self.record(JobOutcome(entry.id, track_id, "computed", _relative(path)))


def _new_part() -> dict[str, Any]:
    return {
        "outputs": {},
        "unscored": {},
        "wall": 0.0,
        "load": 0.0,
        "chunks": 0,
        "eval": 0.0,
        "from_cache": False,
    }


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_benchmark(
    spec: RunSpec,
    *,
    dataset: Dataset,
    registry: Registry,
    factory: Factory,
    fingerprint: Callable[[str], str],
    run_id: str,
    bench_dir: Path = BENCH_DIR,
    command: str = "",
    timings: TimingLog | None = None,
    weights_sha256: Callable[[str], dict[str, str]] | None = None,
    log: Callable[[str], None] | None = None,
) -> RunSummary:
    """Do every job of ``spec`` that has no result yet, and list them all.

    Raises ``DatasetError`` before any work if the dataset's frozen list is not committed,
    and ``BenchError`` if ``run_id`` already exists for a different dataset, list or device.
    A model or track that fails is recorded and the rest still run.
    """
    dataset.check_ready()
    frozen = dataset.frozen
    track_ids = list(spec.track_ids) if spec.track_ids is not None else dataset.track_ids()
    unknown = [t for t in track_ids if t not in frozen.track_ids]
    if unknown:
        raise DatasetError(f"not in {frozen.path.name}: {unknown}")
    entries = []
    for model_id in dict.fromkeys(spec.model_ids):
        entry = registry.get(model_id)
        if entry.kind != "separator" or not entry.is_loadable:
            raise BenchError(f"{model_id} is not a loadable separator")
        entries.append(entry)

    run_dir = bench_dir / run_id
    identity = {
        "run_id": run_id,
        "dataset": dataset.name,
        "frozen_sha256": frozen.sha256,
        "protocol_version": PROTOCOL_VERSION,
        "device": spec.device,
        "overlap": spec.overlap,
    }
    run_file = run_dir / "run.json"
    if run_file.exists():
        earlier = json.loads(run_file.read_text(encoding="utf-8"))
        changed = [k for k, v in identity.items() if earlier.get(k) != v]
        if changed:
            raise BenchError(
                f"run {run_id!r} already exists with a different {', '.join(changed)}; "
                "resume it with the same settings or choose another run id"
            )
    write_json(
        run_file,
        {
            **identity,
            "batch_size": spec.batch_size,
            "models": [e.id for e in entries],
            "tracks": track_ids,
            "keep_estimates": list(spec.keep_estimates),
            "command": command,
            "git": git_state(),
            "packages": package_versions(),
            "started_at": now(),
        },
    )
    summary = RunSummary(run_id=run_id, run_dir=run_dir, state="running")
    status = _Status(run_dir / "status.json", identity, total=len(entries) * len(track_ids))
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as manifest:
        runner = _Runner(
            spec,
            dataset=dataset,
            factory=factory,
            fingerprint=fingerprint,
            identity=identity,
            bench_dir=bench_dir,
            summary=summary,
            status=status,
            manifest=manifest,
            timings=timings,
            weights_sha256=weights_sha256,
            say=log or (lambda _message: None),
        )
        try:
            for entry in entries:
                runner.run_model(entry, track_ids)
        except KeyboardInterrupt:
            summary.state = "interrupted"
            status.end("interrupted")
            raise
        except BaseException as e:
            summary.state = "crashed"
            status.end("crashed", f"{type(e).__name__}: {e}")
            raise
    summary.state = "done"
    status.end("done")
    return summary


def read_status(run_dir: Path) -> dict[str, Any]:
    """A run's status, with ``state`` corrected to ``dead`` if its process is gone."""
    data = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if data.get("state") == "running":
        try:
            os.kill(int(data["pid"]), 0)
        except ProcessLookupError:
            data["state"] = "dead"
        except (PermissionError, KeyError, ValueError):
            pass
    return data


def latest_run(bench_dir: Path = BENCH_DIR) -> Path | None:
    runs = [p.parent for p in bench_dir.glob("*/status.json")]
    return max(runs, key=lambda p: (p / "status.json").stat().st_mtime) if runs else None
