"""MUSDB18-HQ, read from the folder the owner points ``MUSDB18HQ_PATH`` at.

The folder holds ``train/`` and ``test/``, each with one sub-folder per track containing
``mixture.wav``, ``vocals.wav``, ``drums.wav``, ``bass.wav`` and ``other.wav`` (44.1 kHz
stereo). This project never downloads the dataset.

The first time the real dataset is seen, the track list is frozen to
``bench/frozen/musdb18hq_<split>.json`` with a SHA-256 of every file. Nothing is
benchmarked until that file is committed, and a track whose files no longer match it is
refused.

    python -m stems_bench.datasets.musdb            # freeze or check the test split
    python -m stems_bench.datasets.musdb --split train
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.datasets import (
    FROZEN_DIR,
    Dataset,
    DatasetError,
    DatasetUnavailableError,
    FrozenList,
    TrackAudio,
    load_frozen,
    write_frozen,
)
from stems_bench.protocol import PROTOCOL_VERSION, SAMPLE_RATE
from stems_common.env import get_env

ENV_VAR = "MUSDB18HQ_PATH"
STEM_FILES: tuple[str, ...] = ("vocals", "drums", "bass", "other")
FILES: tuple[str, ...] = ("mixture.wav", *(f"{stem}.wav" for stem in STEM_FILES))
# The published split sizes. A folder with another count is not MUSDB18-HQ as released.
SPLIT_SIZES = {"test": 50, "train": 100}


def _decode(raw: bytes, label: str) -> np.ndarray:
    import soundfile as sf

    try:
        samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except sf.SoundFileError as e:
        raise DatasetError(f"{label}: cannot decode ({e})") from None
    if rate != SAMPLE_RATE or samples.shape[1] != 2:
        raise DatasetError(
            f"{label}: expected {SAMPLE_RATE} Hz stereo, found {rate} Hz, "
            f"{samples.shape[1]} channel(s)"
        )
    return np.ascontiguousarray(samples.T)


class Musdb18HqDataset(Dataset):
    targets = ("vocals", "drums", "bass", "other_4stem", "instrumental")
    inputs = ("mix",)

    def __init__(
        self,
        split: str = "test",
        *,
        root: Path | str | None = None,
        frozen_path: Path | None = None,
        expected_tracks: int | None = None,
    ) -> None:
        if split not in SPLIT_SIZES:
            raise DatasetError(f"unknown MUSDB18-HQ split {split!r}")
        self.split = split
        self.name = f"musdb18hq_{split}"
        self._root = Path(root) if root is not None else None
        self._frozen_path = frozen_path or FROZEN_DIR / f"{self.name}.json"
        self._expected = SPLIT_SIZES[split] if expected_tracks is None else expected_tracks
        self._frozen: FrozenList | None = None

    @property
    def root(self) -> Path:
        """The split's folder. Raises if the dataset is not on this machine."""
        base = self._root
        if base is None:
            configured = get_env(ENV_VAR)
            if not configured:
                raise DatasetUnavailableError(
                    f"{ENV_VAR} is not set. MUSDB18-HQ is obtained by the owner (BLOCKED.md B2)."
                )
            base = Path(configured).expanduser()
        folder = base / self.split
        if not folder.is_dir():
            raise DatasetUnavailableError(f"{ENV_VAR} has no {self.split}/ folder")
        return folder

    def _build_frozen(self) -> dict[str, Any]:
        folders = sorted(p for p in self.root.iterdir() if (p / "mixture.wav").is_file())
        if len(folders) != self._expected:
            raise DatasetError(
                f"found {len(folders)} tracks in {self.split}/, expected {self._expected}"
            )
        tracks = []
        for folder in folders:
            files = {}
            for name in FILES:
                try:
                    files[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
                except FileNotFoundError:
                    raise DatasetError(f"{folder.name}: {name} is missing") from None
            tracks.append({"id": folder.name, "files": files})
        return {
            "dataset": self.name,
            "protocol_version": PROTOCOL_VERSION,
            "sample_rate": SAMPLE_RATE,
            "split": self.split,
            "tracks": tracks,
        }

    @property
    def frozen(self) -> FrozenList:
        if self._frozen is None:
            if not self._frozen_path.exists():
                # First contact with the dataset: fix the list before anything is scored.
                write_frozen(self._frozen_path, self._build_frozen())
            self._frozen = load_frozen(self._frozen_path, self.name)
        return self._frozen

    def load(self, track_id: str) -> TrackAudio:
        entry = self.frozen.track(track_id)
        folder = self.root / track_id
        decoded: dict[str, np.ndarray] = {}
        for name, want in entry["files"].items():
            try:
                raw = (folder / name).read_bytes()
            except FileNotFoundError:
                raise DatasetError(f"{track_id}: {name} is missing") from None
            if hashlib.sha256(raw).hexdigest() != want:
                raise DatasetError(
                    f"{track_id}: {name} does not match {self.frozen.path.name}; "
                    "this is not the audio the list was frozen from"
                )
            decoded[name.removesuffix(".wav")] = _decode(raw, f"{track_id}/{name}")
        lengths = {x.shape[1] for x in decoded.values()}
        if len(lengths) != 1:
            raise DatasetError(f"{track_id}: files differ in length ({sorted(lengths)})")
        mix = decoded["mixture"]
        references = {
            "vocals": decoded["vocals"],
            "drums": decoded["drums"],
            "bass": decoded["bass"],
            # MUSDB's "other" still contains guitars and keyboards.
            "other_4stem": decoded["other"],
            "instrumental": decoded["drums"] + decoded["bass"] + decoded["other"],
        }
        return TrackAudio(
            track_id=track_id,
            sample_rate=SAMPLE_RATE,
            inputs={"mix": mix},
            references=references,
            meta={"split": self.split},
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or check a MUSDB18-HQ track list.")
    parser.add_argument("--split", choices=sorted(SPLIT_SIZES), default="test")
    args = parser.parse_args(argv)
    dataset = Musdb18HqDataset(args.split)
    try:
        ids = dataset.track_ids()
        dataset.check_ready()
    except DatasetError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{dataset.name}: {len(ids)} tracks, list sha256 {dataset.frozen.sha256[:12]}, committed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
