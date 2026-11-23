"""MoisesDB, read from the folder the owner points ``MOISESDB_PATH`` at, and the frozen
stratified 50-track subset the benchmark and the Clef-omni test both use.

Layout, as read by the dataset's own loader (github.com/moises-ai/moises-db, commit
1fbc5c78): ``<root>/<provider>/<track id>/data.json`` names the genre and lists stems, each
with source files at ``<stem name>/<source id>.<extension>``. There is no mixture file; the
mix is the sum of the sources. This project never downloads the dataset.

Grouping into our stems follows the dataset's own six-stem mapping: vocals, bass, drums,
guitar and piano are themselves, and everything else (percussion, other keys, plucked and
bowed strings, wind, effects) is ``other``. Lead and backing vocals come from the source
types inside the vocals stem.

One deliberate difference from the official loader: sources of unequal length are padded
with silence to the longest, never trimmed, so the mix is exactly the sum of the six stems.

    python -m stems_bench.datasets.moisesdb --freeze   # scan, select, write the list (once)
    python -m stems_bench.datasets.moisesdb            # check the list against the audio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
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
from stems_common.levels import rms_dbfs
from stems_common.stems import PRODUCT_STEMS

ENV_VAR = "MOISESDB_PATH"
NAME = "moisesdb_subset"
FROZEN_PATH = FROZEN_DIR / f"{NAME}.json"
TUNING_NAME = "moisesdb_tuning"
TUNING_FROZEN_PATH = FROZEN_DIR / f"{TUNING_NAME}.json"
TUNING_SIZE = 30
RELEASED_TRACKS = 240

# MoisesDB top-level stem -> our stem (the dataset's own ``mix_6_stems``).
STEM_GROUPS: dict[str, str] = {
    "vocals": "vocals",
    "bass": "bass",
    "drums": "drums",
    "guitar": "guitar",
    "piano": "piano",
    "other": "other",
    "other_plucked": "other",
    "percussion": "other",
    "other_keys": "other",
    "bowed_strings": "other",
    "wind": "other",
}
# Source types inside the vocals stem. The dataset's loader code writes these with spaces
# ("lead male singer"); the data.json files of the release write them with underscores
# ("lead_male_singer"). Both are accepted: see ``_type``.
LEAD_TYPES = frozenset({"lead male singer", "lead female singer"})
BACKING_TYPES = frozenset({"background vocals", "human choir"})


def _type(track_type: str) -> str:
    return track_type.strip().lower().replace("_", " ")


# --- Selection (report section 5). Fixed before the dataset was ever seen.
SELECTION_SEED = 2307
SUBSET_SIZE = 50
MIN_GENRES = 6
# stratum -> the least number of selected tracks that must have it
QUOTAS: dict[str, int] = {
    "lead_vocals": 8,
    "instrumental": 8,
    "guitar": 10,
    "piano": 10,
    "prominent_drums_bass": 10,
    # For the lead-versus-backing comparisons: tracks annotated with both.
    "lead_and_backing": 8,
}
# Strata the dataset turned out not to have enough of: the selection takes all there are.
# MoisesDB has one track without vocals (docs/corrections.md C15), against a quota of 8.
SHORTFALL_OK: tuple[str, ...] = ("instrumental",)
# A stem is in a track if it has sources and its whole-track RMS is above this.
PRESENT_DBFS = -60.0
# Drums and bass are "prominent" when each is within this many dB of the mix's RMS.
PROMINENT_WITHIN_DB = 12.0


@dataclass(frozen=True)
class Source:
    stem: str  # MoisesDB stem name, underscores for spaces
    track_type: str
    path: str  # relative to the track folder


@dataclass(frozen=True)
class TrackMeta:
    track_id: str
    provider: str
    genre: str
    sources: tuple[Source, ...]

    @property
    def relative_dir(self) -> str:
        return f"{self.provider}/{self.track_id}" if self.provider else self.track_id


@dataclass(frozen=True)
class TrackInfo:
    """What selection needs to know about a track."""

    track_id: str
    genre: str
    strata: tuple[str, ...]
    levels_dbfs: dict[str, float] = field(default_factory=dict, compare=False)


def read_meta(track_dir: Path, provider: str) -> TrackMeta:
    data = json.loads((track_dir / "data.json").read_text(encoding="utf-8"))
    sources = []
    for stem in data.get("stems", []):
        name = str(stem.get("stemName", "")).strip().replace(" ", "_")
        if name not in STEM_GROUPS:
            raise DatasetError(f"{track_dir.name}: unknown MoisesDB stem {name!r}")
        for source in stem.get("tracks", []):
            sources.append(
                Source(
                    stem=name,
                    track_type=_type(str(source.get("trackType", ""))),
                    path=f"{stem.get('stemName')}/{source['id']}.{source.get('extension', 'wav')}",
                )
            )
    if not sources:
        raise DatasetError(f"{track_dir.name}: data.json lists no audio")
    return TrackMeta(
        track_id=track_dir.name,
        provider=provider,
        genre=str(data.get("genre") or "undefined"),
        sources=tuple(sorted(sources, key=lambda s: s.path)),
    )


def scan(root: Path) -> list[TrackMeta]:
    """Every track under ``root``, whether it holds provider folders or tracks directly."""
    found: list[TrackMeta] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "data.json").is_file():
            found.append(read_meta(child, provider=""))
            continue
        for track_dir in sorted(p for p in child.iterdir() if (p / "data.json").is_file()):
            found.append(read_meta(track_dir, provider=child.name))
    ids = [m.track_id for m in found]
    if len(set(ids)) != len(ids):
        raise DatasetError("the same track id appears under two providers")
    return found


def _pad(audio: np.ndarray, samples: int) -> np.ndarray:
    return np.pad(audio, ((0, 0), (0, samples - audio.shape[1])))


def decode(meta: TrackMeta, root: Path) -> dict[str, np.ndarray]:
    """The track's mix, six stems, and lead and backing vocals, all of one length."""
    from stems_worker.audio import AudioError, load_audio

    loaded: list[tuple[Source, np.ndarray]] = []
    for source in meta.sources:
        path = root / meta.relative_dir / source.path
        try:
            loaded.append((source, load_audio(path, SAMPLE_RATE).samples))
        except FileNotFoundError:
            raise DatasetError(f"{meta.track_id}: {source.path} is missing") from None
        except AudioError as e:
            raise DatasetError(f"{meta.track_id}: {source.path}: {e}") from None
    samples = max(audio.shape[1] for _, audio in loaded)
    out = {
        name: np.zeros((2, samples), dtype=np.float32)
        for name in (*PRODUCT_STEMS, "lead_vocals", "backing_vocals")
    }
    for source, audio in loaded:
        padded = _pad(audio, samples)
        out[STEM_GROUPS[source.stem]] += padded
        if source.stem == "vocals" and source.track_type in LEAD_TYPES:
            out["lead_vocals"] += padded
        elif source.stem == "vocals" and source.track_type in BACKING_TYPES:
            out["backing_vocals"] += padded
    out["mix"] = np.sum([out[s] for s in PRODUCT_STEMS], axis=0, dtype=np.float32)
    return out


def strata_of(levels: dict[str, float]) -> tuple[str, ...]:
    """Which selection strata a track belongs to, from its stems' whole-track levels."""

    def present(name: str) -> bool:
        return levels[name] > PRESENT_DBFS

    def prominent(name: str) -> bool:
        return present(name) and levels[name] >= levels["mix"] - PROMINENT_WITHIN_DB

    flags = {
        "lead_vocals": present("lead_vocals"),
        "instrumental": not present("vocals"),
        "guitar": present("guitar"),
        "piano": present("piano"),
        "prominent_drums_bass": prominent("drums") and prominent("bass"),
        "lead_and_backing": present("lead_vocals") and present("backing_vocals"),
    }
    return tuple(name for name in QUOTAS if flags[name])


def analyse(meta: TrackMeta, root: Path) -> TrackInfo:
    audio = decode(meta, root)
    levels = {name: round(rms_dbfs(x), 2) for name, x in audio.items()}
    return TrackInfo(meta.track_id, meta.genre, strata_of(levels), levels)


def select_subset(
    candidates: Sequence[TrackInfo],
    *,
    seed: int = SELECTION_SEED,
    size: int = SUBSET_SIZE,
    quotas: dict[str, int] | None = None,
    min_genres: int = MIN_GENRES,
    shortfall_ok: Sequence[str] = (),
) -> list[TrackInfo]:
    """Choose ``size`` tracks at random, subject to the quotas. Same input, same answer.

    A stratum named in ``shortfall_ok`` that has fewer tracks than its quota contributes
    every track it has; any other shortfall is an error.

    Candidates are put in a seeded random order. Each stratum, scarcest first, takes
    tracks in that order until its quota is met; then genres; then the rest of the order
    fills the subset. Raises if the quotas cannot be met within ``size`` tracks.
    """
    quotas = QUOTAS if quotas is None else quotas
    order = sorted(candidates, key=lambda t: t.track_id)
    if len({t.track_id for t in order}) != len(order):
        raise DatasetError("duplicate track ids among the candidates")
    random.Random(seed).shuffle(order)
    chosen: dict[str, TrackInfo] = {}

    def have(stratum: str) -> int:
        return sum(stratum in t.strata for t in chosen.values())

    eligible = {s: [t for t in order if s in t.strata] for s in quotas}
    for stratum in sorted(quotas, key=lambda s: (len(eligible[s]), s)):
        if len(eligible[stratum]) < quotas[stratum] and stratum not in shortfall_ok:
            raise DatasetError(
                f"only {len(eligible[stratum])} tracks are {stratum!r}; "
                f"the subset needs {quotas[stratum]}"
            )
        for track in eligible[stratum]:
            if have(stratum) >= quotas[stratum]:
                break
            chosen.setdefault(track.track_id, track)
    if len({t.genre for t in order}) < min_genres:
        raise DatasetError(f"the candidates span fewer than {min_genres} genres")
    for track in order:
        if len({t.genre for t in chosen.values()}) >= min_genres:
            break
        if track.genre not in {t.genre for t in chosen.values()}:
            chosen[track.track_id] = track
    if len(chosen) > size:
        raise DatasetError(f"meeting the quotas takes {len(chosen)} tracks, more than {size}")
    for track in order:
        if len(chosen) >= size:
            break
        chosen.setdefault(track.track_id, track)
    if len(chosen) < size:
        raise DatasetError(f"only {len(chosen)} candidates, the subset needs {size}")
    return sorted(chosen.values(), key=lambda t: t.track_id)


def _file_hashes(meta: TrackMeta, root: Path) -> dict[str, str]:
    base = root / meta.relative_dir
    return {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in ("data.json", *(s.path for s in meta.sources))
    }


def build_frozen(
    root: Path,
    *,
    seed: int = SELECTION_SEED,
    size: int = SUBSET_SIZE,
    quotas: dict[str, int] | None = None,
    min_genres: int = MIN_GENRES,
    expected_tracks: int | None = RELEASED_TRACKS,
    shortfall_ok: Sequence[str] = SHORTFALL_OK,
    progress: Any = None,
) -> dict[str, Any]:
    """Scan the dataset, select the subset and describe it. Decodes every track once."""
    quotas = QUOTAS if quotas is None else quotas
    metas = scan(root)
    if expected_tracks is not None and len(metas) != expected_tracks:
        raise DatasetError(f"found {len(metas)} tracks, expected {expected_tracks}")
    infos = []
    for i, meta in enumerate(metas):
        infos.append(analyse(meta, root))
        if progress is not None:
            progress(i + 1, len(metas))
    selected = select_subset(
        infos,
        seed=seed,
        size=size,
        quotas=quotas,
        min_genres=min_genres,
        shortfall_ok=shortfall_ok,
    )
    available = {s: sum(s in info.strata for info in infos) for s in quotas}
    by_id = {m.track_id: m for m in metas}
    all_ids = "\n".join(sorted(by_id)).encode()
    return {
        "dataset": NAME,
        "protocol_version": PROTOCOL_VERSION,
        "sample_rate": SAMPLE_RATE,
        "selection": {
            "seed": seed,
            "size": size,
            "quotas": quotas,
            "min_genres": min_genres,
            "present_dbfs": PRESENT_DBFS,
            "prominent_within_db": PROMINENT_WITHIN_DB,
            "candidates": len(metas),
            "candidate_ids_sha256": hashlib.sha256(all_ids).hexdigest(),
            # How many of all the candidates fall in each stratum, and which quotas the
            # dataset could not fill.
            "available": available,
            "shortfalls": {s: n for s, n in available.items() if n < quotas[s]},
        },
        "tracks": [
            {
                "id": info.track_id,
                "provider": by_id[info.track_id].provider,
                "genre": info.genre,
                "strata": list(info.strata),
                "levels_dbfs": info.levels_dbfs,
                "files": _file_hashes(by_id[info.track_id], root),
            }
            for info in selected
        ],
    }


def build_tuning_frozen(
    root: Path,
    subset: dict[str, Any],
    *,
    seed: int = SELECTION_SEED,
    size: int = TUNING_SIZE,
) -> dict[str, Any]:
    """A seeded random list of tracks that are not in the frozen subset, for tuning blends."""
    excluded = {t["id"] for t in subset["tracks"]}
    outside = sorted(
        (m for m in scan(root) if m.track_id not in excluded), key=lambda m: m.track_id
    )
    if len(outside) < size:
        raise DatasetError(f"only {len(outside)} tracks outside the subset, tuning needs {size}")
    random.Random(seed).shuffle(outside)
    chosen = sorted(outside[:size], key=lambda m: m.track_id)
    return {
        "dataset": TUNING_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "sample_rate": SAMPLE_RATE,
        "selection": {"seed": seed, "size": size, "excludes": NAME},
        "tracks": [
            {
                "id": meta.track_id,
                "provider": meta.provider,
                "genre": meta.genre,
                "strata": [],
                "files": _file_hashes(meta, root),
            }
            for meta in chosen
        ],
    }


class MoisesDbDataset(Dataset):
    targets = (*PRODUCT_STEMS, "other_4stem", "instrumental", "lead_vocals", "backing_vocals")
    inputs = ("mix", "vocals")

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        frozen_path: Path | None = None,
        name: str = NAME,
    ) -> None:
        if name not in (NAME, TUNING_NAME):
            raise DatasetError(f"unknown MoisesDB list {name!r}")
        self.name = name
        self._root = Path(root) if root is not None else None
        self._frozen_path = frozen_path or (FROZEN_PATH if name == NAME else TUNING_FROZEN_PATH)
        self._frozen: FrozenList | None = None

    @property
    def root(self) -> Path:
        if self._root is not None:
            return self._root
        configured = get_env(ENV_VAR)
        if not configured:
            raise DatasetUnavailableError(
                f"{ENV_VAR} is not set. MoisesDB is obtained by the owner (BLOCKED.md B3)."
            )
        root = Path(configured).expanduser()
        if not root.is_dir():
            raise DatasetUnavailableError(f"{ENV_VAR} is not a folder")
        return root

    @property
    def frozen(self) -> FrozenList:
        if self._frozen is None:
            if not self._frozen_path.exists():
                self.root  # noqa: B018 - say "not on this machine" before "not frozen"
            self._frozen = load_frozen(self._frozen_path, self.name)
        return self._frozen

    def load(self, track_id: str) -> TrackAudio:
        entry = self.frozen.track(track_id)
        track_dir = self.root / entry["provider"] / track_id
        for name, want in entry["files"].items():
            try:
                digest = hashlib.sha256((track_dir / name).read_bytes()).hexdigest()
            except FileNotFoundError:
                raise DatasetError(f"{track_id}: {name} is missing") from None
            if digest != want:
                raise DatasetError(
                    f"{track_id}: {name} does not match {self.frozen.path.name}; "
                    "this is not the audio the list was frozen from"
                )
        meta = read_meta(track_dir, entry["provider"])
        if {"data.json", *(s.path for s in meta.sources)} != set(entry["files"]):
            raise DatasetError(f"{track_id}: its files are not the ones in the frozen list")
        audio = decode(meta, self.root)
        mix = audio.pop("mix")
        references = {
            **audio,
            "other_4stem": audio["guitar"] + audio["piano"] + audio["other"],
            "instrumental": mix - audio["vocals"],
        }
        return TrackAudio(
            track_id=track_id,
            sample_rate=SAMPLE_RATE,
            inputs={"mix": mix, "vocals": audio["vocals"]},
            references=references,
            meta={"genre": entry["genre"], "strata": list(entry["strata"])},
        )


def summarise(data: dict[str, Any]) -> str:
    tracks = data["tracks"]
    counts = {s: sum(s in t["strata"] for t in tracks) for s in data["selection"]["quotas"]}
    genres = sorted({t["genre"] for t in tracks})
    parts = [f"{len(tracks)} tracks", f"{len(genres)} genres"]
    parts += [f"{name} {count}" for name, count in counts.items()]
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or check the MoisesDB subset.")
    parser.add_argument("--freeze", action="store_true", help="select and write the subset")
    parser.add_argument(
        "--freeze-tuning",
        action="store_true",
        help="write the tuning list: tracks outside the subset (the subset must exist first)",
    )
    args = parser.parse_args(argv)
    dataset = MoisesDbDataset()
    try:
        if args.freeze_tuning:
            if TUNING_FROZEN_PATH.exists():
                print(f"{TUNING_FROZEN_PATH} already exists; a frozen list is not rewritten")
                return 1
            data = build_tuning_frozen(dataset.root, dataset.frozen.data)
            write_frozen(TUNING_FROZEN_PATH, data)
            print(f"wrote {TUNING_FROZEN_PATH}: {len(data['tracks'])} tracks outside the subset")
            print("Commit this file before tuning.")
            return 0
        if args.freeze:
            if FROZEN_PATH.exists():
                print(f"{FROZEN_PATH} already exists; a frozen list is not rewritten")
                return 1

            def progress(done: int, total: int) -> None:
                print(f"\rscanning {done} of {total}", end="", file=sys.stderr, flush=True)

            data = build_frozen(dataset.root, progress=progress)
            print(file=sys.stderr)
            write_frozen(FROZEN_PATH, data)
            print(f"wrote {FROZEN_PATH}: {summarise(data)}")
            print("Commit this file before benchmarking.")
            return 0
        for track_id in dataset.track_ids():
            dataset.load(track_id)
        dataset.check_ready()
    except DatasetError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{dataset.name}: {summarise(dataset.frozen.data)}; audio matches; list is committed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
