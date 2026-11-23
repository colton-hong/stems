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

Blends are jobs too. A system is a slot of ``config/defaults.yaml`` in one of its three
modes (single best, two-model blend, full ensemble); its members' separated audio is kept
under ``runs/bench/estimates/`` while the models run, then blended and scored track by
track. A system's time is the sum of its members' times.
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
from stems_common.settings import ENSEMBLE_SLOTS, Defaults
from stems_common.stems import SLOT_REQUIRED_OUTPUTS
from stems_models.registry import ModelEntry, Registry
from stems_worker.cache import CACHE_VERSION
from stems_worker.ensemble import blend_stems
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
class EnsembleSystem:
    """One slot in one mode: which models are blended, how, and what comes out."""

    slot: str
    mode: str
    members: tuple[str, ...]
    weights: tuple[float, ...]
    method: str
    outputs: tuple[str, ...]
    # The reference each output is scored against.
    targets: tuple[str, ...]
    input: str
    # Systems whose blended outputs are subtracted from this one's (the six-stem "other"
    # built from a four-stem "other" minus guitar and piano).
    minus: tuple[EnsembleSystem, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.slot}:{self.mode}"

    @property
    def all_members(self) -> tuple[str, ...]:
        ids = [*self.members, *(m for sub in self.minus for m in sub.members)]
        return tuple(dict.fromkeys(ids))

    def needed(self) -> dict[str, set[str]]:
        """Which outputs of which models this system reads."""
        needed: dict[str, set[str]] = {}
        for system in (self, *self.minus):
            for member in system.members:
                needed.setdefault(member, set()).update(system.outputs)
        return needed


def ensemble_systems(
    defaults: Defaults, registry: Registry, slots: Sequence[str] | None = None
) -> list[EnsembleSystem]:
    """Every slot of the config in each of its three modes, plus the composed "other"."""
    ensembles = defaults.ensembles
    modes = ("single_best", "two_model_blend", "full_ensemble")
    built: dict[tuple[str, str], EnsembleSystem] = {}
    for slot in ENSEMBLE_SLOTS:
        outputs = SLOT_REQUIRED_OUTPUTS[slot]
        for mode in modes:
            members = tuple(ensembles.slots[slot].for_mode(mode))
            entries = [registry.get(m) for m in members]
            for entry in entries:
                missing = [o for o in outputs if o not in entry.stems]
                if missing:
                    raise BenchError(f"{slot}: {entry.id} does not emit {missing}")
            if len({input_for(e.stems) for e in entries}) != 1:
                raise BenchError(f"{slot}:{mode}: its members take different inputs")
            kinds = {e.other_contains_guitar_piano for e in entries}
            if "other" in outputs and len(kinds) != 1:
                raise BenchError(
                    f"{slot}:{mode} mixes four-stem and six-stem 'other' models: {members}"
                )
            targets = tuple(
                target_for(o, other_contains_guitar_piano=entries[0].other_contains_guitar_piano)
                for o in outputs
            )
            built[slot, mode] = EnsembleSystem(
                slot=slot,
                mode=mode,
                members=members,
                weights=tuple(ensembles.slots[slot].weights_for(mode)),
                method=ensembles.method_for(slot),
                outputs=tuple(outputs),
                targets=targets,
                input=input_for(entries[0].stems),
            )
    for mode in modes:
        other = built["other", mode]
        if other.targets == ("other_4stem",):
            # What the product delivers as "other": the four-stem blend with the same
            # mode's guitar and piano taken out.
            built["other_composed", mode] = EnsembleSystem(
                slot="other_composed",
                mode=mode,
                members=other.members,
                weights=other.weights,
                method=other.method,
                outputs=("other",),
                targets=("other",),
                input=other.input,
                minus=(built["guitar", mode], built["piano", mode]),
            )
    wanted = None if slots is None else set(slots)
    return [s for (slot, _), s in built.items() if wanted is None or slot in wanted]


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model_ids: tuple[str, ...]
    device: str
    overlap: float = DEFAULT_OVERLAP
    batch_size: int = 1
    # None means every track of the frozen list.
    track_ids: tuple[str, ...] | None = None
    # Models whose separated audio is kept, all outputs. Scores are always kept.
    keep_estimates: tuple[str, ...] = ()
    # Blends to score after the models. What they read from their members is kept too.
    ensembles: tuple[EnsembleSystem, ...] = ()
    # Above zero, the same jobs are done afresh under their own keys: a repeat, to see
    # how much scores move from one run to the next.
    trial: int = 0


@dataclass
class JobOutcome:
    model_id: str  # a registry id, or a system id such as "vocals:two_model_blend"
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
    trial: int = 0,
) -> str:
    payload: dict[str, Any] = {
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
    if trial:
        payload["trial"] = trial  # absent for trial 0, so first-run keys never change
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class EstimateStore:
    """Separated audio kept on disk, one folder per (model, track) job key.

    Float32 ``.npy`` files exactly as the model produced them, with the time the run took.
    It exists so blends can be scored without running their members again; deleting it
    loses nothing but that time.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _directory(self, model_id: str, key: str) -> Path:
        return self.root / model_id / key

    def _meta(self, model_id: str, key: str) -> dict[str, Any]:
        try:
            text = (self._directory(model_id, key) / "meta.json").read_text(encoding="utf-8")
            return json.loads(text)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def has(self, model_id: str, key: str, outputs: Sequence[str]) -> bool:
        return set(outputs) <= set(self._meta(model_id, key))

    def get(
        self, model_id: str, key: str, outputs: Sequence[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
        """The stored outputs and the timing of the run that made the first of them."""
        meta = self._meta(model_id, key)
        if not outputs or not set(outputs) <= set(meta):
            return None
        directory = self._directory(model_id, key)
        try:
            stems = {name: np.load(directory / f"{name}.npy") for name in outputs}
        except (OSError, ValueError):
            return None
        return stems, meta[outputs[0]]

    def put(
        self, model_id: str, key: str, stems: dict[str, np.ndarray], timing: dict[str, Any]
    ) -> None:
        directory = self._directory(model_id, key)
        directory.mkdir(parents=True, exist_ok=True)
        for name, samples in stems.items():
            partial = directory / f"{name}.{os.getpid()}.tmp.npy"
            np.save(partial, np.ascontiguousarray(samples, dtype=np.float32))
            os.replace(partial, directory / f"{name}.npy")
        meta = {**self._meta(model_id, key), **dict.fromkeys(stems, timing)}
        write_json(directory / "meta.json", meta)  # last: marks the outputs usable


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


def score_estimates(
    track: TrackAudio,
    input_name: str,
    estimates: dict[str, np.ndarray],
    targets: dict[str, str],
    derived: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Score each estimate against its target. Returns (scored, reasons for the rest)."""
    mix = track.inputs[input_name]
    scored: dict[str, Any] = {}
    unscored: dict[str, str] = {}
    for name, estimate in estimates.items():
        target = targets[name]
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


def score_outputs(
    entry: ModelEntry, track: TrackAudio, estimates: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Score one model's outputs, each against the reference the protocol gives it."""
    estimates = dict(estimates)
    targets = {
        name: target_for(name, other_contains_guitar_piano=entry.other_contains_guitar_piano)
        for name in estimates
    }
    derived = []
    if {"guitar", "piano", "other"} <= set(estimates) and "other_4stem" in track.references:
        # A six-stem model's answer to the four-stem "other" is the sum of the three.
        estimates["other_4stem"] = estimates["guitar"] + estimates["piano"] + estimates["other"]
        targets["other_4stem"] = "other_4stem"
        derived.append("other_4stem")
    return score_estimates(track, input_for(entry.stems), estimates, targets, derived)


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


_ALL = None  # in the keep table: keep every output of the model


class _Runner:
    """The state of one invocation of ``run_benchmark``."""

    def __init__(
        self,
        spec: RunSpec,
        *,
        dataset: Dataset,
        registry: Registry,
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
        self.registry = registry
        self.factory = factory
        self.fingerprint = fingerprint
        self.identity = identity
        self.results_dir = bench_dir / "results" / dataset.name
        self.store = EstimateStore(bench_dir / "estimates" / dataset.name)
        self.summary = summary
        self.status = status
        self.manifest = manifest
        self.timings = timings
        self.weights_sha256 = weights_sha256
        self.say = say
        # The code that produces every result of this run is the code loaded now.
        self.provenance = {"git": git_state(), "packages": package_versions()}
        self._track: TrackAudio | None = None
        self.keep: dict[str, set[str] | None] = dict.fromkeys(spec.keep_estimates, _ALL)
        for system in spec.ensembles:
            if self.ensemble_applicability(system) is not None:
                continue  # never scored on this dataset: keep nothing for it
            for model_id, outputs in system.needed().items():
                if model_id not in self.keep:
                    self.keep[model_id] = set()
                kept = self.keep[model_id]
                if kept is not _ALL:
                    kept.update(outputs)

    # --- bookkeeping ---------------------------------------------------------------------

    def record(self, job: JobOutcome) -> None:
        self.summary.jobs.append(job)
        self.manifest.write(json.dumps(job.__dict__, sort_keys=True) + "\n")
        self.manifest.flush()
        self.status.finished(job)
        detail = f" ({job.detail})" if job.detail else ""
        self.say(f"{job.model_id} / {job.track_id}: {job.outcome}{detail}")

    def _key(self, track_id: str, system_id: str, fingerprint: str) -> str:
        return result_key(
            dataset=self.dataset.name,
            frozen_sha256=self.dataset.frozen.sha256,
            track_id=track_id,
            model_id=system_id,
            fingerprint=fingerprint,
            device=self.spec.device,
            overlap=self.spec.overlap,
            trial=self.spec.trial,
        )

    def result_path(self, entry: ModelEntry, track_id: str) -> tuple[str, Path]:
        key = self._key(track_id, entry.id, self.fingerprint(entry.id))
        return key, self.results_dir / entry.id / f"{key}.json"

    def ensemble_path(self, system: EnsembleSystem, track_id: str) -> tuple[str, Path]:
        """Keyed on everything that defines the blend, its members' weights included."""
        definition = [
            [sub.slot, sub.members, sub.weights, sub.method, sub.outputs]
            + [self.fingerprint(m) for m in sub.members]
            for sub in (system, *system.minus)
        ]
        digest = hashlib.sha256(json.dumps(definition).encode()).hexdigest()
        key = self._key(track_id, system.id, digest)
        folder = self.results_dir / "_ensembles" / system.id.replace(":", ".")
        return key, folder / f"{key}.json"

    def load_track(self, track_id: str) -> TrackAudio:
        """One track in memory at a time."""
        if self._track is None or self._track.track_id != track_id:
            self._track = None
            self._track = self.dataset.load(track_id)
        return self._track

    def kept_outputs(self, entry: ModelEntry) -> tuple[str, ...]:
        if entry.id not in self.keep:
            return ()
        kept = self.keep[entry.id]
        return tuple(s for s in entry.stems if kept is _ALL or s in kept)

    # --- one model -----------------------------------------------------------------------

    def run_model(self, entry: ModelEntry, track_ids: Sequence[str]) -> None:
        reason = applicability(entry, self.dataset)
        kept = self.kept_outputs(entry)
        pending: list[str] = []
        parts: dict[str, dict[str, Any]] = {}
        for track_id in track_ids:
            if reason is not None:
                self.record(JobOutcome(entry.id, track_id, "not_applicable", detail=reason))
                continue
            key, path = self.result_path(entry, track_id)
            scored = path.exists()
            if scored and (not kept or self.store.has(entry.id, key, kept)):
                self.record(JobOutcome(entry.id, track_id, "cached", _relative(path)))
                continue
            pending.append(track_id)
            # A result without the audio a blend needs: separate again, keep the scores.
            parts[track_id] = _new_part(key, path, score=not scored)
        if not pending:
            return

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
        kept = [name for name in separator.outputs if name in self.kept_outputs(entry)]
        hit = self.store.get(entry.id, part["key"], tuple(separator.outputs)) if kept else None
        if hit is not None:
            stems, timing = hit
            part["from_cache"] = True
        else:
            run = run_separator(
                separator,
                audio,
                track.sample_rate,
                overlap=spec.overlap,
                batch_size=spec.batch_size,
            )
            stems = run.stems
            timing = {
                "wall_seconds": run.wall_seconds,
                "num_chunks": run.num_chunks,
                "device": run.device,
            }
            if self.timings is not None:
                self.timings.record(
                    model_id=entry.id,
                    device=run.device,
                    outputs=list(stems),
                    audio_seconds=run.audio_seconds,
                    wall_seconds=run.wall_seconds,
                    num_chunks=run.num_chunks,
                    stage="bench",
                )
        bad = [name for name, x in stems.items() if not np.isfinite(x).all()]
        if bad:
            raise ValueError(f"non-finite samples in {bad}")
        if kept and hit is None:
            self.store.put(entry.id, part["key"], {n: stems[n] for n in kept}, timing)
        if part["score"]:
            started = time.perf_counter()
            scored, unscored = score_outputs(entry, track, stems)
            part["eval"] += time.perf_counter() - started
            part["outputs"].update(scored)
            part["unscored"].update(unscored)
        part["wall"] += float(timing["wall_seconds"])
        part["chunks"] += int(timing["num_chunks"])
        part["device"] = str(timing["device"])
        part["audio_seconds"] = track.duration_seconds

    def finish(
        self, entry: ModelEntry, track_id: str, part: dict[str, Any], error: str | None
    ) -> None:
        if error is not None:
            self.record(JobOutcome(entry.id, track_id, "failed", detail=error))
            return
        path = part["path"]
        if not part["score"]:
            detail = "scores kept; audio separated again for the blends"
            self.record(JobOutcome(entry.id, track_id, "cached", _relative(path), detail))
            return
        audio_seconds = part["audio_seconds"]
        write_json(
            path,
            {
                "result_version": RESULT_VERSION,
                "kind": "model",
                "key": part["key"],
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
                **self.provenance,
            },
        )
        self.record(JobOutcome(entry.id, track_id, "computed", _relative(path)))

    # --- blends --------------------------------------------------------------------------

    def run_ensembles(self, track_ids: Sequence[str]) -> None:
        for track_id in track_ids:
            for system in self.spec.ensembles:
                reason = self.ensemble_applicability(system)
                if reason is not None:
                    self.record(JobOutcome(system.id, track_id, "not_applicable", detail=reason))
                    continue
                key, path = self.ensemble_path(system, track_id)
                if path.exists():
                    self.record(JobOutcome(system.id, track_id, "cached", _relative(path)))
                    continue
                self.status.working_on(system.id, track_id)
                try:
                    write_json(path, self.score_ensemble(system, track_id, key))
                except Exception as e:  # one blend failing must not stop the others
                    detail = f"{type(e).__name__}: {e}"
                    self.record(JobOutcome(system.id, track_id, "failed", detail=detail))
                    continue
                self.record(JobOutcome(system.id, track_id, "computed", _relative(path)))

    def ensemble_applicability(self, system: EnsembleSystem) -> str | None:
        if system.input not in self.dataset.inputs:
            return f"needs the {system.input!r} input, which {self.dataset.name} does not have"
        if not set(system.targets) & set(self.dataset.targets):
            return f"{self.dataset.name} has no {' or '.join(system.targets)} reference"
        return None

    def _blend(self, system: EnsembleSystem, track_id: str) -> dict[str, np.ndarray]:
        per_member = []
        for member in system.members:
            key, _ = self.result_path(self.registry.get(member), track_id)
            hit = self.store.get(member, key, system.outputs)
            if hit is None:
                raise BenchError(
                    f"{member} has no separated audio for this track "
                    "(it failed, or was not part of this run)"
                )
            per_member.append(hit[0])
        return blend_stems(per_member, list(system.weights), system.method)

    def score_ensemble(self, system: EnsembleSystem, track_id: str, key: str) -> dict[str, Any]:
        track = self.load_track(track_id)
        started = time.perf_counter()
        blended = self._blend(system, track_id)
        for sub in system.minus:
            taken = sum(self._blend(sub, track_id).values())
            blended = {name: x - taken for name, x in blended.items()}
        blend_seconds = time.perf_counter() - started
        # The time a blend costs is the time of every model it needs.
        wall = blend_seconds
        devices = set()
        for member in system.all_members:
            _, member_path = self.result_path(self.registry.get(member), track_id)
            member_result = json.loads(member_path.read_text(encoding="utf-8"))
            wall += member_result["wall_seconds"]
            devices.add(member_result["device"])
        started = time.perf_counter()
        targets = dict(zip(system.outputs, system.targets, strict=True))
        scored, unscored = score_estimates(track, system.input, blended, targets)
        return {
            "result_version": RESULT_VERSION,
            "kind": "ensemble",
            "key": key,
            **{k: v for k, v in self.identity.items() if k != "device"},
            "track_id": track_id,
            "model_id": system.id,
            "slot": system.slot,
            "mode": system.mode,
            "members": list(system.members),
            "weights": list(system.weights),
            "method": system.method,
            "minus": [
                {"slot": s.slot, "members": list(s.members), "weights": list(s.weights)}
                for s in system.minus
            ],
            "all_members": list(system.all_members),
            "device_requested": self.spec.device,
            "device": "+".join(sorted(devices)),
            "input": system.input,
            "audio_seconds": round(track.duration_seconds, 3),
            "wall_seconds": round(wall, 4),
            "seconds_per_audio_second": round(wall / track.duration_seconds, 5),
            "blend_seconds": round(blend_seconds, 4),
            "eval_seconds": round(time.perf_counter() - started, 3),
            "outputs": scored,
            "unscored_outputs": unscored,
            "at": now(),
            **self.provenance,
        }


def _new_part(key: str, path: Path, *, score: bool) -> dict[str, Any]:
    return {
        "key": key,
        "path": path,
        "score": score,
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
    A model, track or blend that fails is recorded and the rest still run.
    """
    dataset.check_ready()
    frozen = dataset.frozen
    track_ids = list(spec.track_ids) if spec.track_ids is not None else dataset.track_ids()
    unknown = [t for t in track_ids if t not in frozen.track_ids]
    if unknown:
        raise DatasetError(f"not in {frozen.path.name}: {unknown}")
    member_ids = [m for system in spec.ensembles for m in system.all_members]
    entries = []
    for model_id in dict.fromkeys([*spec.model_ids, *member_ids]):
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
    if spec.trial:
        identity["trial"] = spec.trial
    run_file = run_dir / "run.json"
    if run_file.exists():
        earlier = json.loads(run_file.read_text(encoding="utf-8"))
        keys = [*identity, *(["trial"] if "trial" in earlier and not spec.trial else [])]
        changed = [k for k in keys if earlier.get(k) != identity.get(k)]
        if changed:
            raise BenchError(
                f"run {run_id!r} already exists with a different {', '.join(changed)}; "
                "resume it with the same settings or choose another run id"
            )
    summary = RunSummary(run_id=run_id, run_dir=run_dir, state="running")
    total = (len(entries) + len(spec.ensembles)) * len(track_ids)
    status = _Status(run_dir / "status.json", identity, total=total)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as manifest:
        runner = _Runner(
            spec,
            dataset=dataset,
            registry=registry,
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
        write_json(
            run_file,
            {
                **identity,
                "batch_size": spec.batch_size,
                "models": [e.id for e in entries],
                "ensembles": [system.id for system in spec.ensembles],
                "tracks": track_ids,
                "keep_estimates": sorted(runner.keep),
                "reliable_metrics": list(dataset.reliable_metrics),
                "command": command,
                **runner.provenance,
                "started_at": now(),
            },
        )
        try:
            for entry in entries:
                runner.run_model(entry, track_ids)
            runner.run_ensembles(track_ids)
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
