"""The synthetic dataset: generated tracks with exactly known stems.

It exists so the whole benchmark can run with no music on disk. Its scores say whether the
harness works, not how good a model is: the tracks are sine tones and noise, which most
models were never trained on, and only SDR is meaningful on them (see
``stems_bench.metrics``).

    python -m stems_bench.datasets.synthetic --freeze    # write bench/frozen/synthetic.json
    python -m stems_bench.datasets.synthetic             # check the frozen list still holds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.datasets import (
    FROZEN_DIR,
    Dataset,
    DatasetError,
    FrozenList,
    TrackAudio,
    load_frozen,
    write_frozen,
)
from stems_bench.protocol import PROTOCOL_VERSION, SAMPLE_RATE, TARGETS
from stems_common.stems import PRODUCT_STEMS
from stems_common.synthetic import SyntheticTrack, generate_track

FROZEN_PATH = FROZEN_DIR / "synthetic.json"
TUNING_FROZEN_PATH = FROZEN_DIR / "synthetic_tuning.json"
# Generated audio is compared with the frozen fingerprints to this absolute tolerance. It
# is far above the last-bit differences between platforms and far below any real change.
FINGERPRINT_TOLERANCE = 1e-6
DURATION_SECONDS = 12.0

# The tracks to freeze. Two are deliberately incomplete, so "this stem is not in the track"
# is exercised on every run.
SPEC: tuple[dict[str, Any], ...] = (
    *({"id": f"synth-{i:02d}", "seed": 100 + i, "active_stems": list(PRODUCT_STEMS)}
      for i in range(1, 7)),
    {"id": "synth-07", "seed": 107, "active_stems": ["vocals", "drums", "bass", "other"]},
    {"id": "synth-08", "seed": 108, "active_stems": ["drums", "bass", "guitar", "piano", "other"]},
)  # fmt: skip
# A second list with other seeds, for tuning blends: nothing tuned on it is scored on it.
TUNING_SPEC: tuple[dict[str, Any], ...] = tuple(
    {"id": f"tune-{i:02d}", "seed": 200 + i, "active_stems": list(PRODUCT_STEMS)}
    for i in range(1, 5)
)


def fingerprint(audio: np.ndarray) -> list[float]:
    """Three numbers that pin down a signal without being sensitive to its last bit."""
    x = audio.astype(np.float64).ravel()
    index = np.arange(x.size, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    chirp = float(np.dot(x, np.cos(1e-7 * index * index)) / x.size)
    ramp = float(np.dot(x, index / x.size) / x.size)
    return [round(rms, 9), round(chirp, 9), round(ramp, 9)]


def _signals(track: SyntheticTrack) -> dict[str, np.ndarray]:
    return {
        "mix": track.mix,
        **track.stems,
        "lead_vocals": track.lead_vocals,
        "backing_vocals": track.backing_vocals,
        "vocals_wet": track.vocals_wet,
    }


def _generate(entry: dict[str, Any], sample_rate: int) -> SyntheticTrack:
    return generate_track(
        seed=entry["seed"],
        duration_seconds=entry["duration_seconds"],
        sample_rate=sample_rate,
        active_stems=entry["active_stems"],
    )


def build_frozen(name: str = "synthetic") -> dict[str, Any]:
    """The frozen list for a spec, with fingerprints of the audio it generates today."""
    tracks = []
    for spec in SPEC if name == "synthetic" else TUNING_SPEC:
        entry = {**spec, "duration_seconds": DURATION_SECONDS}
        signals = _signals(_generate(entry, SAMPLE_RATE))
        entry["fingerprint"] = {name: fingerprint(x) for name, x in signals.items()}
        tracks.append(entry)
    return {
        "dataset": name,
        "protocol_version": PROTOCOL_VERSION,
        "sample_rate": SAMPLE_RATE,
        "generator": "stems_common.synthetic.generate_track",
        "tracks": tracks,
    }


class SyntheticDataset(Dataset):
    name = "synthetic"
    targets = TARGETS
    inputs = ("mix", "vocals", "vocals_wet")
    reliable_metrics = ("sdr",)

    def __init__(self, frozen_path: Path | None = None, name: str = "synthetic") -> None:
        self.name = name
        default = FROZEN_PATH if name == "synthetic" else TUNING_FROZEN_PATH
        self._frozen_path = frozen_path or default
        self._frozen: FrozenList | None = None

    @property
    def frozen(self) -> FrozenList:
        if self._frozen is None:
            self._frozen = load_frozen(self._frozen_path, self.name)
        return self._frozen

    def load(self, track_id: str) -> TrackAudio:
        entry = self.frozen.track(track_id)
        track = _generate(entry, self.frozen.data["sample_rate"])
        for name, signal in _signals(track).items():
            want = entry["fingerprint"].get(name)
            got = fingerprint(signal)
            if want is None or not np.allclose(got, want, rtol=0, atol=FINGERPRINT_TOLERANCE):
                raise DatasetError(
                    f"{track_id}: generated {name} does not match {self.frozen.path.name} "
                    f"(fingerprint {got}, frozen {want}). The generator or the list has "
                    "changed; results would not be comparable with earlier runs."
                )
        stems = track.stems
        mix = track.mix
        references = {
            **stems,
            "other_4stem": (stems["guitar"] + stems["piano"] + stems["other"]).astype(np.float32),
            "instrumental": (mix - stems["vocals"]).astype(np.float32),
            "lead_vocals": track.lead_vocals,
            "backing_vocals": track.backing_vocals,
            "dry": stems["vocals"],
        }
        return TrackAudio(
            track_id=track_id,
            sample_rate=track.sample_rate,
            inputs={"mix": mix, "vocals": stems["vocals"], "vocals_wet": track.vocals_wet},
            references=references,
            meta={"seed": entry["seed"], "active_stems": list(entry["active_stems"])},
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or check the synthetic track list.")
    parser.add_argument("--freeze", action="store_true", help="write the list if there is none")
    parser.add_argument("--name", choices=("synthetic", "synthetic_tuning"), default="synthetic")
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args(argv)
    dataset = SyntheticDataset(args.path, args.name)
    args.path = dataset._frozen_path
    if args.freeze:
        if args.path.exists():
            print(f"{args.path} already exists; a frozen list is not rewritten", file=sys.stderr)
            return 1
        write_frozen(args.path, build_frozen(args.name))
        print(f"wrote {args.path}")
    try:
        for track_id in dataset.track_ids():
            dataset.load(track_id)
    except DatasetError as e:
        print(e, file=sys.stderr)
        return 1
    count = len(dataset.track_ids())
    print(f"{count} tracks match {args.path.name} (sha256 {dataset.frozen.sha256[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
