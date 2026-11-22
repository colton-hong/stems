"""Cache of model outputs, so repeating a run does not repeat the expensive part.

An entry is one model's outputs for one piece of audio. Its key covers everything that
can change those outputs: the decoded audio itself (not the file name), the model and the
exact weights it was loaded from, the overlap, the device asked for (the same model gives
slightly different numbers on different devices) and a version number that is bumped
whenever the inference code changes.

Outputs are stored as float32 ``.npy`` files, exactly as the model produced them, before
any stem arithmetic or gain. The cache lives under ``runs/`` and can be deleted at any time.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stems_models.download import ChecksumLock
from stems_models.registry import Registry

# Bump when chunking, stitching or an adapter changes in a way that alters outputs.
CACHE_VERSION = 1


def audio_sha256(samples: np.ndarray, sample_rate: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{samples.shape}:{samples.dtype}:".encode())
    digest.update(np.ascontiguousarray(samples).tobytes())
    return digest.hexdigest()


def registry_fingerprint(
    registry: Registry, lock: ChecksumLock | None = None
) -> Callable[[str], str]:
    """A function giving, for a model id, a hash of everything that defines that model.

    It covers the registry entry's loader settings and each file's pinned SHA-256, so
    replacing a checkpoint or editing its config invalidates cached outputs.
    """
    lock = lock or ChecksumLock()

    def fingerprint(model_id: str) -> str:
        entry = registry.get(model_id)
        files = []
        for f in entry.files:
            pinned = lock.get(f.url)
            files.append([f.filename, f.url, pinned["sha256"] if pinned else "unpinned"])
        payload = {"id": entry.id, "loader": entry.loader, "config": entry.config, "files": files}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    return fingerprint


@dataclass(frozen=True)
class CachedRun:
    stems: dict[str, np.ndarray]
    meta: dict[str, Any]


class OutputCache:
    def __init__(
        self,
        root: Path | str,
        *,
        audio_hash: str,
        overlap: float,
        device: str,
        fingerprint: Callable[[str], str],
    ) -> None:
        self.root = Path(root)
        self._base = {
            "version": CACHE_VERSION,
            "audio": audio_hash,
            "overlap": overlap,
            "device": device,
        }
        self._fingerprint = fingerprint

    def key(self, model_id: str) -> str:
        payload = {**self._base, "model": model_id, "fingerprint": self._fingerprint(model_id)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _directory(self, model_id: str) -> Path:
        key = self.key(model_id)
        return self.root / key[:2] / key

    def get(self, model_id: str, outputs: tuple[str, ...]) -> CachedRun | None:
        """The cached outputs, or None unless every one of ``outputs`` is there."""
        directory = self._directory(model_id)
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if not set(outputs) <= set(meta["outputs"]):
                return None
            stems = {name: np.load(directory / f"{name}.npy") for name in outputs}
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError):
            return None
        return CachedRun(stems, meta)

    def put(self, model_id: str, stems: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        """Store outputs, adding to whatever this model already has for this audio."""
        directory = self._directory(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        known: list[str] = []
        with contextlib.suppress(FileNotFoundError, json.JSONDecodeError, KeyError):
            meta_text = (directory / "meta.json").read_text(encoding="utf-8")
            known = list(json.loads(meta_text)["outputs"])
        for name, samples in stems.items():
            partial = directory / f"{name}.{os.getpid()}.tmp.npy"
            np.save(partial, np.ascontiguousarray(samples, dtype=np.float32))
            os.replace(partial, directory / f"{name}.npy")
        merged = {**meta, "model_id": model_id, "outputs": sorted({*known, *stems})}
        partial_meta = directory / f"meta.{os.getpid()}.tmp"
        partial_meta.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        os.replace(partial_meta, directory / "meta.json")  # written last: marks the entry usable
