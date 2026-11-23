"""From a dataset to the test's plan, and from the plan to clip files.

``make_plan`` decides everything that has a known answer: each track's excerpt, which
stems are present in it, which clean stems Part B uses and what is mixed into them. The
plan is plain data and goes into the pre-registration file. ``build_clips`` then cuts and
mixes exactly what the plan says and writes the clips with a manifest that records, for
every clip, its answer and (for bleed clips) the SIR measured from the two parts.

Clips are real music when the dataset is: they are written under ``runs/`` only, which git
ignores, and are never committed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stems_bench.clef import clips as c
from stems_bench.datasets import Dataset, TrackAudio
from stems_common.stems import PRODUCT_STEMS
from stems_common.synthetic import add_reverb

REVERB_CLIPS = 5
BLEED_LEVELS = {None: 0, 30.0: 1, 20.0: 2, 12.0: 3, 6.0: 4}


def _window(entry: dict[str, Any]) -> c.Window:
    return c.Window(entry["window"]["start_sample"], entry["window"]["num_samples"])


def _has_both_vocal_parts(track: TrackAudio) -> bool:
    refs = track.references
    return all(
        name in refs and c.is_present(refs[name], track.sample_rate)
        for name in ("lead_vocals", "backing_vocals")
    )


def make_plan(
    dataset: Dataset,
    seed: int,
    *,
    window_seconds: float = c.WINDOW_SECONDS,
    part_b_targets: Sequence[str] = c.PART_B_TARGETS,
    reverb_clips: int = REVERB_CLIPS,
) -> dict[str, Any]:
    """Excerpts, presence, and the Part B, reverb and lead-or-backing selections."""
    tracks: dict[str, Any] = {}
    lead_backing: list[str] = []
    sample_rate = None
    for track_id in dataset.track_ids():
        track = dataset.load(track_id)
        sample_rate = track.sample_rate
        window = c.loudest_window(track.inputs["mix"], track.sample_rate, window_seconds)
        entry: dict[str, Any] = {
            "window": {
                "start_sample": window.start,
                "num_samples": window.length,
                "start_seconds": window.start / track.sample_rate,
                "seconds": window.length / track.sample_rate,
            },
            "present": c.presence(track.references, window, track.sample_rate),
        }
        if _has_both_vocal_parts(track):
            entry["lead_backing"] = c.lead_backing_answer(
                track.references["lead_vocals"],
                track.references["backing_vocals"],
                window,
                track.sample_rate,
            )
            lead_backing.append(track_id)
        tracks[track_id] = entry
    part_b = c.plan_part_b({t: e["present"] for t, e in tracks.items()}, seed, part_b_targets)
    vocal_items = [item for item in part_b if item.target == "vocals"]
    return {
        "dataset": dataset.name,
        "sample_rate": sample_rate,
        "seed": seed,
        "tracks": tracks,
        "lead_backing": lead_backing,
        "part_b": [
            {"track_id": i.track_id, "target": i.target, "contaminant": i.contaminant}
            for i in part_b
        ],
        # The reverb probe reuses Part B's clean vocal stems.
        "reverb": [
            {"track_id": item.track_id, "target": item.target, "seed": seed + n}
            for n, item in enumerate(vocal_items[:reverb_clips])
        ],
    }


def build_clips(dataset: Dataset, plan: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write every clip of ``plan`` into ``out_dir`` and return the manifest."""
    if plan["dataset"] != dataset.name:
        raise c.ClipError(f"the plan is for {plan['dataset']}, not {dataset.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    cache: dict[str, TrackAudio] = {}

    def load(track_id: str) -> TrackAudio:
        if track_id not in cache:
            cache.clear()  # one track in memory at a time
            cache[track_id] = dataset.load(track_id)
        return cache[track_id]

    def add(clip_id: str, part: str, track_id: str, audio, rate: int, **facts: Any) -> None:
        encoded = c.encode_clip(audio, rate)
        (out_dir / f"{clip_id}.wav").write_bytes(encoded.data)
        entries.append(
            {
                "id": clip_id,
                "part": part,
                "track_id": track_id,
                "file": f"{clip_id}.wav",
                "sha256": hashlib.sha256(encoded.data).hexdigest(),
                "bytes": len(encoded.data),
                "seconds": encoded.seconds,
                "gain": encoded.gain,
                **facts,
            }
        )

    for n, (track_id, entry) in enumerate(plan["tracks"].items(), start=1):
        track = load(track_id)
        window = _window(entry)
        add(f"a{n:02d}", "A", track_id, window.cut(track.inputs["mix"]), track.sample_rate,
            present=entry["present"])  # fmt: skip
    for n, track_id in enumerate(plan["lead_backing"], start=1):
        track = load(track_id)
        entry = plan["tracks"][track_id]
        add(f"lb{n:02d}", "lead_backing", track_id, _window(entry).cut(track.references["vocals"]),
            track.sample_rate, answer=entry["lead_backing"])  # fmt: skip
    for n, item in enumerate(plan["part_b"], start=1):
        track = load(item["track_id"])
        window = _window(plan["tracks"][item["track_id"]])
        target = window.cut(track.references[item["target"]])
        contaminant = window.cut(track.references[item["contaminant"]])
        facts = {"target": item["target"], "contaminant": item["contaminant"]}
        add(f"b{n:02d}-clean", "B", item["track_id"], target, track.sample_rate,
            **facts, sir_db=None, measured_sir_db=None, bleed_level=0)  # fmt: skip
        for sir in c.SIR_LEVELS_DB:
            mixture, scaled = c.mix_at_sir(target, contaminant, sir)
            add(f"b{n:02d}-sir{sir:02.0f}", "B", item["track_id"], mixture, track.sample_rate,
                **facts, sir_db=sir, measured_sir_db=c.measured_sir_db(target, scaled),
                bleed_level=BLEED_LEVELS[sir])  # fmt: skip
    for n, item in enumerate(plan["reverb"], start=1):
        track = load(item["track_id"])
        window = _window(plan["tracks"][item["track_id"]])
        wet = add_reverb(window.cut(track.references[item["target"]]), track.sample_rate,
                         seed=item["seed"], wet_db=c.REVERB_WET_DB)  # fmt: skip
        add(f"r{n:02d}", "reverb", item["track_id"], wet, track.sample_rate,
            target=item["target"], wet_db=c.REVERB_WET_DB)  # fmt: skip
    manifest = {
        "dataset": dataset.name,
        "clip_sample_rate": c.CLIP_SAMPLE_RATE,
        "stems": list(PRODUCT_STEMS),
        "clips": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
