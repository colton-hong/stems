"""One job from request to stems: parse, listen, plan, separate, assess, gate, retry, write.

This is the worker's code path for a job (Phase 5 wraps it): every decision goes through
the decision layer, so it runs the same with Jev, with a recorded mock, or with no key at
all, when the rule-based fallbacks decide. The output folder gets the stems and a
``job.json`` that holds the parsed request, the track features, the plan, every attempt at
every stem with its assessment, and the decision log: one record per decision with its
provider, model, token usage, answers and fallback reason.

    python -m stems_worker.decision.job SONG --request "vocals and drums as flac" --out DIR

A request in which no stem can be recognised is not guessed at: the job stops with status
``needs_review`` and writes no stems.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_common.levels import level_below_mix_db
from stems_common.paths import RUNS_DIR
from stems_common.settings import Defaults, load_defaults
from stems_common.stems import PRODUCT_STEMS
from stems_models.registry import Registry, load_registry
from stems_worker.audio import extension_for, load_audio, save_stems
from stems_worker.cache import OutputCache, audio_sha256, registry_fingerprint
from stems_worker.compose import (
    Blend,
    SeparatorFactory,
    as_blend,
    compose,
    plan_composition,
)
from stems_worker.decision import retry
from stems_worker.decision.assess import assess_stem
from stems_worker.decision.jev import JevClient
from stems_worker.decision.parse import from_selection, parse_request, wanted_stems
from stems_worker.decision.plan import assignment_for, plan_ensemble
from stems_worker.decision.record import Decision
from stems_worker.device import pick_device
from stems_worker.factory import build_separators
from stems_worker.listening.features import (
    PromptScorer,
    Tagger,
    build_track_features,
    stem_signals,
)
from stems_worker.pipeline import STAGES, default_assignment, run_stage
from stems_worker.timing import DEFAULT_TIMINGS_PATH, TimingLog

JOB_VERSION = 1
DEFAULT_CACHE_DIR = RUNS_DIR / "cache" / "separation"
READY = "ready"
NEEDS_REVIEW = "needs_review"


class JobError(ValueError):
    """The job cannot be run as asked."""


@dataclass(frozen=True)
class JobRequest:
    # Free text to parse, or ticked stems. Exactly one of the two.
    text: str | None = None
    stems: tuple[str, ...] | None = None
    # Options that are not part of the text. With ticked stems they are the whole request.
    split_vocals: bool = False
    dereverb: bool = False
    quality_mode: str = "unspecified"
    # A format given here wins over one named in the text.
    fmt: str | None = None
    device: str | None = None
    cache_dir: Path | None = DEFAULT_CACHE_DIR
    timings_path: Path | None = DEFAULT_TIMINGS_PATH

    def __post_init__(self) -> None:
        if (self.text is None) == (self.stems is None):
            raise JobError("a job needs either a request text or a list of stems, not both")


class _Runner:
    """Runs models on one track, with the cache and timing log every run shares."""

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        request: JobRequest,
        defaults: Defaults,
        registry: Registry,
        factory: SeparatorFactory,
        fingerprint: Callable[[str], str],
        device: str,
        progress: Callable[[str, int, int], None] | None,
    ) -> None:
        self.samples, self.sample_rate = samples, sample_rate
        self.defaults, self.registry, self.factory = defaults, registry, factory
        self.device, self.progress = device, progress
        self._cache_dir, self._fingerprint = request.cache_dir, fingerprint
        self.overlap = defaults.separation.overlap
        self.batch_size = defaults.separation.batch_size
        self.timings = None
        if request.timings_path is not None:
            context = {"overlap": self.overlap, "batch_size": self.batch_size, "source": "job"}
            self.timings = TimingLog(request.timings_path, context)
        self.runs: list[dict[str, Any]] = []

    def _cache(self, audio: np.ndarray) -> OutputCache | None:
        if self._cache_dir is None:
            return None
        return OutputCache(
            self._cache_dir,
            audio_hash=audio_sha256(audio, self.sample_rate),
            overlap=self.overlap,
            device=self.device,
            fingerprint=self._fingerprint,
        )

    def _note(self, records: Any, stage: str) -> None:
        for r in records:
            self.runs.append(
                {
                    "model_id": r.model_id,
                    "stage": stage,
                    "outputs": list(r.outputs),
                    "device": r.device,
                    "audio_seconds": round(r.audio_seconds, 3),
                    "wall_seconds": round(r.wall_seconds, 3),
                    "cached": r.cached,
                }
            )

    def separate(self, stems: tuple[str, ...], assignment: dict[str, str | Blend], also=()):
        plan = plan_composition(
            stems, assignment, self.registry, self.defaults.separation.other_method, also
        )
        composition = compose(
            self.samples,
            self.sample_rate,
            plan,
            self.factory,
            device=self.device,
            overlap=self.overlap,
            batch_size=self.batch_size,
            residual_warn_db=self.defaults.separation.residual_warn_db,
            silent_stem_db=self.defaults.separation.silent_stem_db,
            progress=self.progress,
            cache=self._cache(self.samples),
            timings=self.timings,
        )
        self._note(composition.runs, "separation")
        return composition

    def stage(self, stage: str, source: str | Blend, vocals: np.ndarray) -> dict[str, np.ndarray]:
        outputs, records = run_stage(
            stage,
            as_blend(source),
            vocals,
            self.sample_rate,
            self.factory,
            registry=self.registry,
            device=self.device,
            overlap=self.overlap,
            batch_size=self.batch_size,
            progress=self.progress,
            cache=self._cache(vocals),
            timings=self.timings,
        )
        self._note(records, stage)
        return outputs


def _write(output_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / f"job.{os.getpid()}.tmp"
    partial.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, output_dir / "job.json")  # last: its presence marks the job done
    return result


def run_job(
    input_path: Path | str,
    output_dir: Path | str,
    request: JobRequest,
    *,
    client: JevClient | None = None,
    tagger: Tagger | None = None,
    clap: PromptScorer | None = None,
    registry: Registry | None = None,
    defaults: Defaults | None = None,
    factory: SeparatorFactory | None = None,
    fingerprint: Callable[[str], str] | None = None,
    rates: Mapping[tuple[str, str], float] | None = None,
    job_id: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    on_stage: Callable[[str, Mapping[str, Any]], None] | None = None,
    max_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one job and write its stems and ``job.json`` into ``output_dir``. Returns the json.

    ``client`` is the Jev client (real or mock); None means the fallbacks decide.
    ``tagger`` is the listening model; None means nothing is heard, the gate says so, and
    stems are delivered with a warning that they were not checked. ``factory`` and
    ``fingerprint`` let tests substitute weightless models.

    ``on_stage(name, info)`` is told where the job is: ``transcoding``, ``analysing``,
    ``planning``, ``separating`` (with ``stems``), then per stem ``assessing``, ``retrying``
    and ``delivered`` (with ``stem``, ``attempt`` and for the last ``state``), and
    ``packaging``. Audio longer than ``max_duration_seconds`` is refused while decoding.
    """
    input_path, output_dir = Path(input_path), Path(output_dir)
    registry = registry or load_registry()
    defaults = defaults or load_defaults()
    job_id = job_id or str(uuid.uuid4())
    if request.fmt is not None:
        extension_for(request.fmt)  # reject an unknown format before doing any work
    decisions: list[dict[str, Any]] = []

    def tell(name: str, **info: Any) -> None:
        if on_stage is not None:
            on_stage(name, info)

    def log(decision: Decision, **extra: Any) -> Decision:
        decisions.append({**decision.record, **extra})
        return decision

    # 1. What is being asked for.
    if request.text is not None:
        parsed = log(
            parse_request(
                request.text, client=client, settings=defaults.decision, request_id=job_id
            )
        ).value
    else:
        parsed = log(
            from_selection(
                request.stems or (),
                settings=defaults.decision,
                output_format=request.fmt or "unspecified",
                quality_mode=request.quality_mode,
                split_vocals=request.split_vocals,
                request_id=job_id,
            )
        ).value
    requested = wanted_stems(parsed)
    result: dict[str, Any] = {
        "job_version": JOB_VERSION,
        "job_id": job_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": READY,
        "input": {"name": input_path.name},
        "request": {
            "text": request.text,
            "stems": None if request.stems is None else list(request.stems),
            "split_vocals": request.split_vocals,
            "dereverb": request.dereverb,
            "format": request.fmt,
        },
        "parsed_request": parsed,
        "decisions": decisions,
    }
    if not requested:
        # Never assume all six: the user is asked to confirm instead.
        return _write(output_dir, {**result, "status": NEEDS_REVIEW, "stems": {}, "extras": {}})

    fmt = request.fmt or parsed["output_format"]
    fmt = defaults.audio.output_format if fmt == "unspecified" else fmt
    split = request.split_vocals or parsed["vocal_split"] == "lead_and_backing"
    parsed_for_plan = {**parsed, "vocal_split": "lead_and_backing" if split else "single_vocals"}

    # 2. What the track is.
    tell("transcoding")
    loaded = load_audio(
        input_path, defaults.audio.sample_rate_hz, max_duration_seconds=max_duration_seconds
    )
    device = pick_device(request.device)
    # What the tagger cost: one line per thing it listened to, beside the models' lines.
    listened: list[dict[str, Any]] = []

    def listening(what: str, call: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        value = call()
        if tagger is not None:
            listened.append(
                {
                    "model_id": tagger.name,
                    "stage": "listening",
                    "outputs": [what],
                    "device": str(getattr(tagger, "device", None) or "cpu"),
                    "audio_seconds": round(loaded.duration_seconds, 3),
                    "wall_seconds": round(time.perf_counter() - started, 3),
                    "cached": False,
                }
            )
        return value

    tell("analysing")
    features = listening(
        "mix",
        lambda: build_track_features(
            loaded.samples,
            loaded.sample_rate,
            tagger=tagger,
            settings=defaults.listening,
            source_sample_rate=loaded.source.sample_rate,
            source_channels=loaded.source.channels,
        ),
    )

    # 3. Which models.
    tell("planning")
    plan = log(
        plan_ensemble(
            parsed_for_plan,
            features,
            client=client,
            defaults=defaults,
            registry=registry,
            job_id=job_id,
            dereverb=request.dereverb,
            device=device,
            rates=rates,
        )
    ).value
    assignment = assignment_for(plan, defaults)
    mode = plan["ensemble_mode"]
    stages = [s for s, on in (("vocal_split", split), ("dereverb", request.dereverb)) if on]

    if factory is None:

        def factory(model_id: str, stems: tuple[str, ...]):
            return build_separators(model_id, stems, registry=registry)

    runner = _Runner(
        loaded.samples,
        loaded.sample_rate,
        request=request,
        defaults=defaults,
        registry=registry,
        factory=factory,
        fingerprint=fingerprint or registry_fingerprint(registry),
        device=device,
        progress=progress,
    )

    # 4. Separate.
    tell("separating", stems=list(requested))
    first = runner.separate(requested, assignment, ("vocals",) if stages else ())
    slots = defaults.ensembles.slots
    dereverb_model = slots["dereverb"].single_best if registry.candidates_for("dereverb") else None

    def produce(spec: retry.Spec) -> np.ndarray:
        """The audio of a retry: the stem again, from the attempt's models."""
        retried = dict(assignment)
        retried[spec.stem] = (
            spec.models[0]
            if spec.mode == "single_best"
            else default_assignment(defaults, spec.mode)[spec.stem]
        )
        audio = runner.separate((spec.stem,), retried).stems[spec.stem]
        if spec.dereverb is not None:
            audio = runner.stage("dereverb", spec.dereverb, audio)["dry"]
        return audio

    # 5. Assess each stem, and retry or warn as the gate says.
    delivered: dict[str, np.ndarray] = {}
    stems_report: dict[str, Any] = {}
    for stem in requested:
        machine = retry.StemGate(
            retry.Spec(stem, as_blend(assignment[stem]).models, mode),
            settings=defaults.gate,
            fallback_model=plan["plan"][stem]["fallback_model"],
            blends={m: slots[stem].for_mode(m) for m in retry.BLEND_MODES},
            dereverb_model=dereverb_model,
        )
        audio = first.stems[stem]
        attempts: list[np.ndarray] = []
        while machine.state == retry.RUNNING:
            tell("assessing", stem=stem, attempt=len(attempts))
            signals = listening(
                stem,
                lambda audio=audio, stem=stem: stem_signals(
                    stem,
                    audio,
                    loaded.sample_rate,
                    tagger=tagger,
                    clap=clap,
                    settings=defaults.listening,
                ),
            )
            decision = log(
                assess_stem(
                    stem,
                    machine.current.model_used,
                    signals.metrics(),
                    client=client,
                    defaults=defaults,
                    job_id=job_id,
                    present=signals.present,
                    below_mix_db=level_below_mix_db(audio, loaded.samples),
                    retries_used=machine.retries_used,
                    available=machine.available(),
                ),
                attempt=len(attempts),
            )
            attempts.append(audio)
            step = machine.step(decision.value)
            if step.next is not None:
                tell("retrying", stem=stem, attempt=len(attempts))
                audio = produce(step.next)
        assert machine.chosen is not None
        tell("delivered", stem=stem, state=machine.state, attempt=len(attempts) - 1)
        delivered[stem] = attempts[machine.chosen]
        chosen = machine.delivered.spec
        stems_report[stem] = {
            "state": machine.state,
            "model_used": chosen.model_used,
            "model_ids": [*chosen.models, *([chosen.dereverb] if chosen.dereverb else [])],
            "mode": chosen.mode,
            "delivered_attempt": machine.chosen,
            "retries": len(machine.attempts) - 1,
            "warning": machine.warning,
            "attempts": [
                {"model_used": a.spec.model_used, "mode": a.spec.mode, **a.assessment}
                for a in machine.attempts
            ],
        }

    # 6. Optional extras, made from the vocals that are delivered.
    extras: dict[str, np.ndarray] = {}
    extras_report: dict[str, Any] = {}
    if stages:
        vocals = delivered.get("vocals", first.dependencies.get("vocals"))
        assert vocals is not None  # computed for the stages even when not requested
        for stage in stages:
            outputs = runner.stage(stage, assignment[stage], vocals)
            for output, name in STAGES[stage].items():
                extras[name] = outputs[output]
                extras_report[name] = {
                    "stage": stage,
                    "model_ids": list(as_blend(assignment[stage]).models),
                    "source": f"{as_blend(assignment[stage]).describe()} on the vocals stem",
                }

    # 7. Write.
    tell("packaging")
    written = save_stems(output_dir, {**delivered, **extras}, loaded.sample_rate, fmt)
    for name, report in (*stems_report.items(), *extras_report.items()):
        report["file"] = written.paths[name].name
    residual = first.residual
    result.update(
        {
            "input": {
                "name": input_path.name,
                "audio_sha256": audio_sha256(loaded.samples, loaded.sample_rate),
                "duration_seconds": round(loaded.duration_seconds, 3),
                "source_sample_rate_hz": loaded.source.sample_rate,
                "source_channels": loaded.source.channels,
            },
            "format": fmt,
            "device": device,
            "listening": {
                "tagger": None if tagger is None else tagger.name,
                "calibrated": defaults.listening.calibrated.model_dump(),
            },
            "track_features": features,
            "plan": plan,
            "stems": stems_report,
            "extras": extras_report,
            "warnings": {s: r["warning"] for s, r in stems_report.items() if r["warning"]},
            "models": [*runner.runs, *listened],
            "output_gain": round(written.gain, 6),
            "residual": {
                "computed": residual.computed,
                "level_db": None if residual.level_db is None else round(residual.level_db, 2),
                "ok": residual.ok,
                "note": residual.note,
            },
        }
    )
    return _write(output_dir, result)


def providers(result: Mapping[str, Any]) -> dict[str, set[str]]:
    """Who decided, per kind of decision, over a whole job."""
    found: dict[str, set[str]] = {}
    for entry in result["decisions"]:
        found.setdefault(entry["kind"], set()).add(entry["provider"])
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_worker.decision.job",
        description="Run one job through the decision layer and write stems and job.json.",
    )
    parser.add_argument("file", type=Path)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--request", help="what you want, in words")
    what.add_argument("--stems", help="comma-separated stem names, or 'all'")
    parser.add_argument("--out", type=Path, help="output folder (default: <file name>_job)")
    parser.add_argument("--split-vocals", action="store_true")
    parser.add_argument("--dereverb", action="store_true")
    parser.add_argument("--format", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-tagger", action="store_true", help="do not load BEATs")
    args = parser.parse_args(argv)
    stems = None
    if args.stems is not None:
        names = PRODUCT_STEMS if args.stems == "all" else tuple(args.stems.split(","))
        stems = tuple(n.strip() for n in names)
    defaults = load_defaults()
    output_dir = args.out or args.file.with_name(f"{args.file.stem}_job")
    client = JevClient.from_env(defaults.decision, log_path=output_dir / "jev-calls.jsonl")
    tagger = None
    if not args.no_tagger and defaults.listening.enabled:
        from stems_models.download import DownloadError
        from stems_worker.listening.beats import BeatsError, BeatsTagger

        try:
            tagger = BeatsTagger.from_registry()
            tagger.load("cpu")
        except (BeatsError, DownloadError, OSError) as e:
            print(f"job: no tagger ({e}); stems will not be quality-checked", file=sys.stderr)
            tagger = None
    try:
        result = run_job(
            args.file,
            output_dir,
            JobRequest(
                text=args.request,
                stems=stems,
                split_vocals=args.split_vocals,
                dereverb=args.dereverb,
                fmt=args.format,
                device=args.device,
            ),
            client=client,
            tagger=tagger,
            defaults=defaults,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"job: {e}", file=sys.stderr)
        return 1
    finally:
        if tagger is not None:
            tagger.unload()
    print(f"status: {result['status']}")
    for kind, who in providers(result).items():
        print(f"{kind}: decided by {', '.join(sorted(who))}")
    for stem, report in result["stems"].items():
        note = f"  warning: {report['warning']}" if report["warning"] else ""
        print(f"{output_dir / report['file']}  [{stem}, {report['model_used']}]{note}")
    print(f"log: {output_dir / 'job.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
