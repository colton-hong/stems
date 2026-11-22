"""Separate one audio file into stems and describe what was done.

This is the single code path behind the ``separate`` command, and later behind the job
worker: decode, pick a model per stem, run them, write the stems, write a manifest.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from stems_common.paths import REPO_ROOT, RUNS_DIR
from stems_common.settings import Defaults, load_defaults
from stems_models.registry import Registry, load_registry
from stems_worker.audio import extension_for, load_audio, save_stems
from stems_worker.cache import OutputCache, audio_sha256, registry_fingerprint
from stems_worker.compose import (
    Blend,
    CompositionError,
    ModelRun,
    SeparatorFactory,
    as_blend,
    compose,
    plan_composition,
    run_model,
)
from stems_worker.device import pick_device
from stems_worker.ensemble import blend_stems
from stems_worker.factory import build_separators
from stems_worker.timing import DEFAULT_TIMINGS_PATH, TimingLog

MANIFEST_VERSION = 1
DEFAULT_CACHE_DIR = RUNS_DIR / "cache" / "separation"


class PipelineError(ValueError):
    """The request is understood but cannot be carried out as asked."""


@dataclass(frozen=True)
class SeparateOptions:
    stems: tuple[str, ...]
    fmt: str = "wav"
    # None means: STEMS_DEVICE if set, otherwise the best usable device.
    device: str | None = None
    mode: str = "single_best"
    # None means: take the value from config/defaults.yaml.
    other_method: str | None = None
    overlap: float | None = None
    batch_size: int | None = None
    # Per-slot overrides of the default model, as {stem or stage: registry id}.
    models: dict[str, str] = field(default_factory=dict)
    # Optional stages, both run on the vocals stem. The vocals are computed for them even
    # if the vocals stem itself was not requested.
    split_vocals: bool = False  # adds lead_vocals and backing_vocals
    dereverb: bool = False  # adds vocals_dry
    # Where model run times are appended. None disables the log.
    timings_path: Path | None = DEFAULT_TIMINGS_PATH
    # None disables the cache.
    cache_dir: Path | None = DEFAULT_CACHE_DIR


MODES: tuple[str, ...] = ("single_best", "two_model_blend", "full_ensemble")


def default_assignment(defaults: Defaults, mode: str = "single_best") -> dict[str, str | Blend]:
    """What serves each slot in ``mode``: a model id, or a blend with the config's weights."""
    if mode not in MODES:
        raise PipelineError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    ensembles = defaults.ensembles
    if mode == "single_best":
        return {slot: models.single_best for slot, models in ensembles.slots.items()}
    return {
        slot: Blend(
            tuple(models.for_mode(mode)),
            tuple(models.weights_for(mode)),
            ensembles.method_for(slot),
        )
        for slot, models in ensembles.slots.items()
    }


def _members(blend: Blend) -> dict[str, Any]:
    """The members of a blended stem, for the manifest. Nothing for a single model."""
    if not blend.is_blend:
        return {}
    return {
        "models": list(blend.models),
        "weights": [round(w, 6) for w in blend.weights],
        "blend_method": blend.method,
    }


# What each optional stage takes from its model, and what the files are called.
STAGES: dict[str, dict[str, str]] = {
    "vocal_split": {"lead_vocals": "lead_vocals", "backing_vocals": "backing_vocals"},
    "dereverb": {"dry": "vocals_dry"},
}


def _stage_blend(stage: str, source: str | Blend, registry: Registry) -> Blend:
    blend = as_blend(source)
    for model_id in blend.models:
        if model_id not in registry:
            raise CompositionError(f"{stage}: {model_id!r} is not in the registry")
        entry = registry.get(model_id)
        lacking = [o for o in STAGES[stage] if o not in entry.stems]
        if not entry.is_loadable or lacking:
            raise CompositionError(
                f"{stage}: {model_id} cannot serve it (status {entry.status}, emits {entry.stems})"
            )
    return blend


def _level_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return round(20.0 * float(np.log10(max(rms, 1e-12))), 2)


def _code_version() -> str | None:
    """The git commit of this checkout, so a manifest can be traced to the code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def separate_file(
    input_path: Path | str,
    output_dir: Path | str,
    options: SeparateOptions,
    *,
    registry: Registry | None = None,
    defaults: Defaults | None = None,
    factory: SeparatorFactory | None = None,
    fingerprint: Callable[[str], str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Separate ``input_path`` and write stems plus ``manifest.json`` into ``output_dir``.

    Returns the manifest. ``factory`` and ``fingerprint`` exist so tests can substitute
    weightless models; by default real models are built from the registry.
    """
    input_path, output_dir = Path(input_path), Path(output_dir)
    registry = registry or load_registry()
    defaults = defaults or load_defaults()
    extension_for(options.fmt)  # reject an unknown format before doing any work

    assignment = {**default_assignment(defaults, options.mode), **options.models}
    other_method = options.other_method or defaults.separation.other_method
    overlap = defaults.separation.overlap if options.overlap is None else options.overlap
    batch_size = options.batch_size or defaults.separation.batch_size
    stages = [
        stage
        for stage, wanted in (("vocal_split", options.split_vocals), ("dereverb", options.dereverb))
        if wanted
    ]
    stage_blends = {stage: _stage_blend(stage, assignment[stage], registry) for stage in stages}
    plan = plan_composition(
        options.stems,
        assignment,
        registry,
        other_method,
        also_compute=("vocals",) if stages else (),
    )

    loaded = load_audio(input_path, defaults.audio.sample_rate_hz)
    device = pick_device(options.device)
    audio_hash = audio_sha256(loaded.samples, loaded.sample_rate)

    if factory is None:

        def factory(model_id: str, stems: tuple[str, ...]):
            return build_separators(model_id, stems, registry=registry)

    cache = None
    if options.cache_dir is not None:
        cache = OutputCache(
            options.cache_dir,
            audio_hash=audio_hash,
            overlap=overlap,
            device=device,
            fingerprint=fingerprint or registry_fingerprint(registry),
        )

    code_version = _code_version()
    timings = None
    if options.timings_path is not None:
        timings = TimingLog(
            options.timings_path,
            {
                "audio_sha256": audio_hash,
                "overlap": overlap,
                "batch_size": batch_size,
                "code_version": code_version,
                "torch": _version("torch"),
                "platform": platform.platform(),
            },
        )

    composition = compose(
        loaded.samples,
        loaded.sample_rate,
        plan,
        factory,
        device=device,
        overlap=overlap,
        batch_size=batch_size,
        residual_warn_db=defaults.separation.residual_warn_db,
        silent_stem_db=defaults.separation.silent_stem_db,
        progress=progress,
        cache=cache,
        timings=timings,
    )
    # Optional stages work on the vocals stem and have their own cache entries, keyed on
    # that stem's audio.
    extras: dict[str, np.ndarray] = {}
    stage_runs: list[tuple[str, Any]] = []
    if stages:
        vocals = composition.stems.get("vocals", composition.dependencies.get("vocals"))
        assert vocals is not None  # also_compute guarantees it
        stage_cache = None
        if options.cache_dir is not None:
            stage_cache = OutputCache(
                options.cache_dir,
                audio_hash=audio_sha256(vocals, loaded.sample_rate),
                overlap=overlap,
                device=device,
                fingerprint=fingerprint or registry_fingerprint(registry),
            )
        for stage in stages:
            source = stage_blends[stage]
            per_member = []
            for model_id in source.models:
                outputs, records = run_model(
                    ModelRun(model_id, tuple(STAGES[stage])),
                    vocals,
                    loaded.sample_rate,
                    factory,
                    device=device,
                    overlap=overlap,
                    batch_size=batch_size,
                    progress=progress,
                    cache=stage_cache,
                    timings=timings,
                    stage=stage,
                )
                per_member.append({output: outputs[output] for output in STAGES[stage]})
                stage_runs += [(stage, record) for record in records]
            blended = blend_stems(per_member, list(source.weights), source.method)
            for output, name in STAGES[stage].items():
                extras[name] = blended[output]

    written = save_stems(
        output_dir, {**composition.stems, **extras}, loaded.sample_rate, options.fmt
    )

    def describe(stem: str) -> dict[str, Any]:
        delivered = composition.stems[stem] * np.float32(written.gain)
        if stem == "other" and plan.other_method == "subtract":
            source = f"{plan.sources['other']} minus guitar and piano"
        elif stem == "other" and plan.other_method == "residual":
            source = "mix minus the other five stems"
        else:
            source = plan.sources[stem]
        described = {
            "file": written.paths[stem].name,
            "source": source,
            "rms_dbfs": _level_dbfs(delivered),
            "peak": round(float(np.abs(delivered).max()), 6),
        }
        if stem in plan.blends:
            described |= _members(plan.blends[stem])
        return described

    def describe_extra(name: str) -> dict[str, Any]:
        delivered = extras[name] * np.float32(written.gain)
        stage = next(s for s, names in STAGES.items() if name in names.values())
        return {
            "file": written.paths[name].name,
            "stage": stage,
            "source": f"{stage_blends[stage].describe()} on the vocals stem",
            "rms_dbfs": _level_dbfs(delivered),
            "peak": round(float(np.abs(delivered).max()), 6),
            **_members(stage_blends[stage]),
        }

    def describe_run(record: Any, stage: str) -> dict[str, Any]:
        return {
            "model_id": record.model_id,
            "stage": stage,
            "input": "mix" if stage == "separation" else "vocals stem",
            "outputs": list(record.outputs),
            "device": record.device,
            "audio_seconds": round(record.audio_seconds, 3),
            "wall_seconds": round(record.wall_seconds, 3),
            "realtime_factor": round(record.wall_seconds / record.audio_seconds, 4),
            "num_chunks": record.num_chunks,
            "cached": record.cached,
        }

    residual = composition.residual
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": code_version,
        "input": {
            "name": input_path.name,
            "audio_sha256": audio_hash,
            "codec": loaded.source.codec,
            "source_sample_rate_hz": loaded.source.sample_rate,
            "source_channels": loaded.source.channels,
            "duration_seconds": round(loaded.duration_seconds, 3),
        },
        "request": {
            "stems": list(plan.requested),
            "format": options.fmt,
            "mode": options.mode,
            "other_method": other_method,
            "overlap": overlap,
            "batch_size": batch_size,
            "model_overrides": dict(options.models),
            "split_vocals": options.split_vocals,
            "dereverb": options.dereverb,
        },
        "device": device,
        "sample_rate_hz": loaded.sample_rate,
        "models": [
            *(describe_run(r, "separation") for r in composition.runs),
            *(describe_run(r, stage) for stage, r in stage_runs),
        ],
        "stems": {stem: describe(stem) for stem in plan.requested},
        # Files produced by the optional stages.
        "extras": {name: describe_extra(name) for name in extras},
        "other_method": plan.other_method,
        "dependencies": list(plan.dependencies),
        # One gain applied to every stem, below 1.0 only if a stem would have clipped.
        "output_gain": round(written.gain, 6),
        "residual": {
            "computed": residual.computed,
            "level_db": None if residual.level_db is None else round(residual.level_db, 2),
            "silent_stems": list(residual.silent_stems),
            "ok": residual.ok,
            "note": residual.note,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _version("torch"),
            "onnxruntime": _version("onnxruntime"),
        },
    }
    partial = output_dir / f"manifest.{os.getpid()}.tmp"
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, output_dir / "manifest.json")  # last: its presence marks success
    return manifest
