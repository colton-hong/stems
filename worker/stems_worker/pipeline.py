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
from stems_common.stems import PRODUCT_STEMS
from stems_models.registry import Registry, load_registry
from stems_worker.audio import extension_for, load_audio, save_stems
from stems_worker.cache import OutputCache, audio_sha256, registry_fingerprint
from stems_worker.compose import SeparatorFactory, compose, plan_composition
from stems_worker.device import pick_device
from stems_worker.factory import build_separators

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
    # Per-stem overrides of the default model, as {stem: registry id}.
    models: dict[str, str] = field(default_factory=dict)
    # None disables the cache.
    cache_dir: Path | None = DEFAULT_CACHE_DIR


def default_assignment(defaults: Defaults, mode: str = "single_best") -> dict[str, str]:
    if mode != "single_best":
        raise PipelineError(
            f"mode {mode!r} is not available yet: blends and ensembles are added with the "
            f"benchmark that chooses them (Phase 2). Use single_best."
        )
    return {stem: defaults.ensembles.slots[stem].single_best for stem in PRODUCT_STEMS}


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
    plan = plan_composition(options.stems, assignment, registry, other_method)

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
    )
    written = save_stems(output_dir, composition.stems, loaded.sample_rate, options.fmt)

    def describe(stem: str) -> dict[str, Any]:
        delivered = composition.stems[stem] * np.float32(written.gain)
        if stem == "other" and plan.other_method == "subtract":
            source = f"{plan.sources['other']} minus guitar and piano"
        elif stem == "other" and plan.other_method == "residual":
            source = "mix minus the other five stems"
        else:
            source = plan.sources[stem]
        return {
            "file": written.paths[stem].name,
            "source": source,
            "rms_dbfs": _level_dbfs(delivered),
            "peak": round(float(np.abs(delivered).max()), 6),
        }

    residual = composition.residual
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": _code_version(),
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
        },
        "device": device,
        "sample_rate_hz": loaded.sample_rate,
        "models": [
            {
                "model_id": r.model_id,
                "outputs": list(r.outputs),
                "device": r.device,
                "audio_seconds": round(r.audio_seconds, 3),
                "wall_seconds": round(r.wall_seconds, 3),
                "realtime_factor": round(r.wall_seconds / r.audio_seconds, 4),
                "num_chunks": r.num_chunks,
                "cached": r.cached,
            }
            for r in composition.runs
        ],
        "stems": {stem: describe(stem) for stem in plan.requested},
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
