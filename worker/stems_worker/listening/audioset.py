"""AudioSet's class list, and the committed table that maps its classes to stems.

python -m stems_worker.listening.audioset     # validate and print the mapping
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from stems_common.paths import REPO_ROOT
from stems_common.stems import PRODUCT_STEMS

LISTENING_DIR = REPO_ROOT / "models" / "listening"
STEM_MAP_PATH = LISTENING_DIR / "audioset_stem_map.yaml"
NUM_CLASSES = 527


class StemMapError(ValueError):
    pass


@dataclass(frozen=True)
class AudioSetLabels:
    """The 527 classes in AudioSet's own order."""

    mids: tuple[str, ...]
    names: tuple[str, ...]
    sha256: str

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise StemMapError(f"{name!r} is not an AudioSet class") from None


@dataclass(frozen=True)
class StemMap:
    version: int
    labels: AudioSetLabels
    # stem -> indices into the label list
    stems: dict[str, tuple[int, ...]]
    unmapped: dict[str, str]
    # SHA-256 of the mapping file's bytes, for the pre-registration record.
    sha256: str

    def classes(self, stem: str) -> list[str]:
        return [self.labels.names[i] for i in self.stems[stem]]

    def stem_scores(self, probabilities: Sequence[float]) -> dict[str, float]:
        """Six presence scores from 527 class probabilities given in AudioSet's order.

        A stem's score is the largest probability among its classes.
        """
        if len(probabilities) != len(self.labels.names):
            raise StemMapError(
                f"expected {len(self.labels.names)} probabilities, got {len(probabilities)}"
            )
        return {
            stem: float(max(probabilities[i] for i in indices))
            for stem, indices in self.stems.items()
        }

    def stem_scores_by_mid(self, probabilities: Mapping[str, float]) -> dict[str, float]:
        """The same, for a tagger that names its outputs by AudioSet id in its own order."""
        missing = [mid for mid in self.labels.mids if mid not in probabilities]
        if missing:
            raise StemMapError(f"{len(missing)} AudioSet classes are missing, e.g. {missing[:3]}")
        return self.stem_scores([probabilities[mid] for mid in self.labels.mids])


def load_labels(path: Path) -> AudioSetLabels:
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    if [int(r["index"]) for r in rows] != list(range(len(rows))):
        raise StemMapError(f"{path.name}: indices are not 0..n-1 in order")
    if len(rows) != NUM_CLASSES:
        raise StemMapError(f"{path.name}: {len(rows)} classes, expected {NUM_CLASSES}")
    return AudioSetLabels(
        mids=tuple(r["mid"] for r in rows),
        names=tuple(r["display_name"] for r in rows),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_stem_map(path: Path = STEM_MAP_PATH) -> StemMap:
    """Read and check the mapping. Raises ``StemMapError`` on anything inconsistent."""
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    labels = load_labels(path.parent / data["labels_file"])
    if labels.sha256 != data["labels_sha256"]:
        raise StemMapError(f"{data['labels_file']} is not the file the mapping was written for")
    if data.get("combine") != "max":
        raise StemMapError(f"unknown combine rule {data.get('combine')!r}")
    if set(data["stems"]) != set(PRODUCT_STEMS):
        raise StemMapError(f"stems must be exactly {PRODUCT_STEMS}")
    stems: dict[str, tuple[int, ...]] = {}
    seen: dict[str, str] = {}
    for stem in PRODUCT_STEMS:
        names = data["stems"][stem]
        if not names:
            raise StemMapError(f"{stem} has no classes")
        for name in names:
            if name in seen:
                raise StemMapError(f"{name!r} is mapped to both {seen[name]} and {stem}")
            seen[name] = stem
        stems[stem] = tuple(labels.index_of(name) for name in names)
    unmapped = dict(data.get("unmapped") or {})
    for name in unmapped:
        labels.index_of(name)
        if name in seen:
            raise StemMapError(f"{name!r} is both mapped to {seen[name]} and listed as unmapped")
    return StemMap(
        version=int(data["version"]),
        labels=labels,
        stems=stems,
        unmapped=unmapped,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


@cache
def default_stem_map() -> StemMap:
    return load_stem_map()


def main() -> int:
    try:
        mapping = load_stem_map()
    except StemMapError as e:
        print(f"invalid mapping: {e}", file=sys.stderr)
        return 1
    print(f"AudioSet stem map version {mapping.version}, sha256 {mapping.sha256[:12]}")
    for stem in PRODUCT_STEMS:
        first = ", ".join(mapping.classes(stem)[:4])
        print(f"  {stem:7s} {len(mapping.stems[stem]):2d} classes: {first}, ...")
    print(f"  {len(mapping.unmapped)} instrument classes deliberately unmapped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
