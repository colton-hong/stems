"""Describe the machine the worker runs on, and write it to runs/env/worker.json.

Timings are only meaningful next to the device that produced them, so benchmark and
timing code reads this description instead of guessing.

    python -m stems_worker.env
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from stems_common.paths import RUNS_DIR

ENV_PATH = RUNS_DIR / "env" / "worker.json"

# Distributions whose versions affect results. Missing ones are recorded as None.
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "numpy",
    "scipy",
    "soundfile",
    "soxr",
    "av",
    "onnxruntime",
    "einops",
    "beartype",
    "rotary-embedding-torch",
    "librosa",
    "packaging",
    "PoPE-pytorch",
    "demucs",
    "audioread",
)

DEVICE_PREFERENCE: tuple[str, ...] = ("cuda", "mps", "cpu")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _smoke(device: str) -> str:
    """Run a tiny computation on ``device``. "ok", or the error text if it fails.

    A backend can report itself available and still fail on first use, so availability
    alone is not trusted.
    """
    import torch

    try:
        x = torch.arange(8, dtype=torch.float32, device=device)
        value = float((x * 2).sum().cpu())
    except Exception as e:  # any backend failure is a result to record, not to raise
        return f"{type(e).__name__}: {e}"
    return "ok" if value == 56.0 else f"wrong result {value}"


def torch_devices() -> dict[str, Any]:
    """Which torch devices exist and work here."""
    import torch

    cuda = bool(torch.cuda.is_available())
    mps_built = bool(torch.backends.mps.is_built())
    mps = bool(torch.backends.mps.is_available())
    info: dict[str, Any] = {
        "cuda": {
            "available": cuda,
            "name": torch.cuda.get_device_name(0) if cuda else None,
            "smoke": _smoke("cuda") if cuda else None,
        },
        "mps": {
            "built": mps_built,
            "available": mps,
            "smoke": _smoke("mps") if mps else None,
        },
        "cpu": {"available": True, "threads": torch.get_num_threads(), "smoke": _smoke("cpu")},
    }
    usable = [d for d in DEVICE_PREFERENCE if info[d]["available"] and info[d]["smoke"] == "ok"]
    info["usable"] = usable
    info["default"] = usable[0] if usable else None
    return info


def describe_environment() -> dict[str, Any]:
    """Everything about this machine that a timing or benchmark result should carry."""
    description: dict[str, Any] = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
    }
    if description["packages"]["torch"] is None:
        description["devices"] = None
    else:
        description["devices"] = torch_devices()
    return description


def write_environment(path: Path | None = None) -> Path:
    path = ENV_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(describe_environment(), indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    path = write_environment()
    print(path.read_text(encoding="utf-8"), end="")
    print(f"written to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
