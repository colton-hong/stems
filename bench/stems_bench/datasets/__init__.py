"""Benchmark datasets: a frozen list of tracks, and the audio for each one.

A dataset never decides at run time which tracks to use. The list lives in a committed
file under ``bench/frozen/``, written once, before any model is scored on it; the hash of
that file goes into every result. A dataset whose audio no longer matches its frozen list
refuses to load.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.protocol import PROTOCOL_VERSION
from stems_common.paths import REPO_ROOT

FROZEN_DIR = REPO_ROOT / "bench" / "frozen"


class DatasetError(RuntimeError):
    pass


class DatasetUnavailableError(DatasetError):
    """The dataset's audio is not on this machine (the owner supplies it)."""


@dataclass(frozen=True)
class TrackAudio:
    """One track, decoded. Arrays are float32 ``(2, samples)`` at ``sample_rate``.

    ``references`` holds every target the track has ground truth for; a target that the
    dataset covers but this track lacks is present as silence. ``inputs`` holds what models
    are fed: always ``mix``, plus ``vocals`` and ``vocals_wet`` where the dataset has them.
    """

    track_id: str
    sample_rate: int
    inputs: dict[str, np.ndarray]
    references: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mix(self) -> np.ndarray:
        return self.inputs["mix"]

    @property
    def duration_seconds(self) -> float:
        return self.mix.shape[-1] / self.sample_rate


@dataclass(frozen=True)
class FrozenList:
    """A committed track list. ``sha256`` is of the file's exact bytes."""

    path: Path
    sha256: str
    data: dict[str, Any]

    @property
    def tracks(self) -> list[dict[str, Any]]:
        return self.data["tracks"]

    @property
    def track_ids(self) -> list[str]:
        return [t["id"] for t in self.tracks]

    def track(self, track_id: str) -> dict[str, Any]:
        for entry in self.tracks:
            if entry["id"] == track_id:
                return entry
        raise DatasetError(f"{track_id!r} is not in {self.path.name}")


def load_frozen(path: Path, dataset: str) -> FrozenList:
    """Read a frozen list, refusing one for another dataset or another protocol version."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise DatasetError(
            f"{path} does not exist: the track list for {dataset!r} has not been frozen"
        ) from None
    data = json.loads(raw)
    if data.get("dataset") != dataset:
        raise DatasetError(f"{path.name} is for {data.get('dataset')!r}, not {dataset!r}")
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise DatasetError(
            f"{path.name} was frozen under protocol {data.get('protocol_version')}; "
            f"this code is protocol {PROTOCOL_VERSION}"
        )
    ids = [t["id"] for t in data.get("tracks", [])]
    if not ids or len(set(ids)) != len(ids):
        raise DatasetError(f"{path.name} must list at least one track and no id twice")
    return FrozenList(path=path, sha256=hashlib.sha256(raw).hexdigest(), data=data)


def write_frozen(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def is_committed(path: Path) -> bool:
    """True if ``path`` is tracked by git and has no uncommitted changes."""
    git = ["git", "-C", str(path.parent)]
    try:
        status = subprocess.run(
            [*git, "status", "--porcelain", "--", path.name],
            capture_output=True,
            text=True,
            check=False,
        )
        tracked = subprocess.run(
            [*git, "ls-files", "--error-unmatch", "--", path.name],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return status.returncode == 0 and not status.stdout.strip() and tracked.returncode == 0


class Dataset(ABC):
    """A frozen list of tracks and a way to load each one."""

    name: str
    # Targets the dataset has references for.
    targets: tuple[str, ...]
    # Inputs it can feed a model (see stems_bench.protocol).
    inputs: tuple[str, ...] = ("mix",)
    # BSS Eval figures worth reporting on this data.
    reliable_metrics: tuple[str, ...] = ("sdr", "sir", "sar")

    @property
    @abstractmethod
    def frozen(self) -> FrozenList: ...

    def track_ids(self) -> list[str]:
        return self.frozen.track_ids

    def check_ready(self) -> None:
        """Refuse to be benchmarked on unless the frozen list is committed as it stands."""
        path = self.frozen.path
        if not is_committed(path):
            raise DatasetError(
                f"{path} is not committed (or has uncommitted changes). Commit the frozen "
                "track list before benchmarking, so every result can name the list it used."
            )

    @abstractmethod
    def load(self, track_id: str) -> TrackAudio:
        """Decode one track of the frozen list. Raises ``DatasetError`` if it has changed."""


# Lists that blends may be tuned on, and the test lists each must share no track with.
TUNING_LISTS: dict[str, tuple[str, ...]] = {
    "synthetic_tuning": ("synthetic",),
    "musdb18hq_train": ("musdb18hq_test",),
    "moisesdb_tuning": ("moisesdb_subset",),
}


def get_dataset(name: str) -> Dataset:
    """The dataset registered under ``name``."""
    if name in ("synthetic", "synthetic_tuning"):
        from stems_bench.datasets.synthetic import SyntheticDataset

        return SyntheticDataset(name=name)
    if name in ("musdb18hq_test", "musdb18hq_train"):
        from stems_bench.datasets.musdb import Musdb18HqDataset

        return Musdb18HqDataset(name.removeprefix("musdb18hq_"))
    if name in ("moisesdb_subset", "moisesdb_tuning"):
        from stems_bench.datasets.moisesdb import MoisesDbDataset

        return MoisesDbDataset(name=name)
    raise DatasetError(f"unknown dataset {name!r}")
