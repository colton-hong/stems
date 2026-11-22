"""Choose how each slot's blend is combined: the method, and a weight per model.

Tuning happens on a list of tracks that shares none with the test lists, and that is
checked before anything runs: a blend tuned on the tracks it is later scored on would
flatter itself. What the tuner chooses is then scored on the test lists by an ordinary
benchmark run.

For every slot, in its two-model and full-ensemble modes, every candidate is tried on every
tuning track: each method, and for the weighted methods a grid of weights (which includes
"all the weight on one model"). A candidate's score is the median over tracks of the
track's median SDR, averaged over the slot's outputs. SDR here is ``metrics.sdr_frames``,
the plain definition, which is what museval's SDR is; it is used to choose, not to report.

A slot gets one method for both modes (that is what the config can say), the one with the
best summed score, and each mode gets its best weights under that method.

    bench tune --dataset synthetic_tuning --out docs/benchmarks/tuning-synthetic.md
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.datasets import TUNING_LISTS, Dataset, get_dataset
from stems_bench.metrics import median_sdr
from stems_bench.run import (
    BENCH_DIR,
    BenchError,
    EnsembleSystem,
    EstimateStore,
    Factory,
    RunSpec,
    ensemble_systems,
    git_state,
    now,
    result_key,
    run_benchmark,
    write_json,
)
from stems_common.settings import Defaults
from stems_models.registry import Registry
from stems_worker.ensemble import Blender
from stems_worker.separator import DEFAULT_OVERLAP

WEIGHTED_METHODS: tuple[str, ...] = ("waveform", "stft_magnitude")
CHOICE_METHODS: tuple[str, ...] = ("stft_min", "stft_max")
# Bump when the candidates or the scoring change, so cached scores are not reused.
TUNE_VERSION = 1
# Weights move in steps of 1/GRID_STEPS[number of models].
GRID_STEPS = {2: 10, 3: 5}
DEFAULT_GRID_STEPS = 4

Candidate = tuple[str, tuple[float, ...] | None]


def weight_grid(count: int, steps: int | None = None) -> list[tuple[float, ...]]:
    """Every way to share a weight of 1 between ``count`` models in steps of 1/``steps``."""
    steps = steps or GRID_STEPS.get(count, DEFAULT_GRID_STEPS)
    grid = []
    for parts in itertools.product(range(steps + 1), repeat=count):
        if sum(parts) == steps:
            grid.append(tuple(p / steps for p in parts))
    return grid


def candidates(count: int) -> list[Candidate]:
    """What is tried for a blend of ``count`` models. Equal-weight waveform comes first."""
    equal = tuple(1.0 / count for _ in range(count))
    out: list[Candidate] = [("waveform", equal)]
    for method in WEIGHTED_METHODS:
        for weights in weight_grid(count):
            single = sum(w > 0 for w in weights) == 1
            if (method, weights) in out or (single and method != "waveform"):
                continue  # one model alone is the same whatever the method
            out.append((method, weights))
    if ("stft_magnitude", equal) not in out:
        out.append(("stft_magnitude", equal))
    out += [(method, None) for method in CHOICE_METHODS]
    return out


def check_disjoint(dataset: Dataset, test_lists: Sequence[Dataset] | None = None) -> list[dict]:
    """Refuse a list that is not a tuning list, or that shares a track with a test list."""
    if dataset.name not in TUNING_LISTS:
        raise BenchError(
            f"{dataset.name} is not a tuning list (those are {', '.join(TUNING_LISTS)}); "
            "blends are never tuned on the tracks they are scored on"
        )
    if test_lists is None:
        test_lists = [get_dataset(name) for name in TUNING_LISTS[dataset.name]]
    tuning = set(dataset.track_ids())
    checked = []
    for test in test_lists:
        shared = sorted(tuning & set(test.track_ids()))
        if shared:
            raise BenchError(f"{dataset.name} shares tracks with {test.name}: {shared[:5]}")
        checked.append(
            {
                "dataset": test.name,
                "frozen_sha256": test.frozen.sha256,
                "tracks": len(test.track_ids()),
            }
        )
    return checked


def _score_track(
    system: EnsembleSystem,
    track_id: str,
    *,
    dataset: Dataset,
    store: EstimateStore,
    member_key: Callable[[str, str], str],
) -> list[dict[str, Any]] | None:
    """Every candidate's SDR per output on one track, or None if a member's audio is missing."""
    track = dataset.load(track_id)
    blenders = {}
    for output in system.outputs:
        estimates = []
        for member in system.members:
            hit = store.get(member, member_key(member, track_id), (output,))
            if hit is None:
                return None
            estimates.append(hit[0][output])
        blenders[output] = Blender(estimates)
    scored = []
    for method, weights in candidates(len(system.members)):
        sdr = {}
        for output, target in zip(system.outputs, system.targets, strict=True):
            reference = track.references[target]
            blended = blenders[output].blend(weights, method)
            value = median_sdr(reference, blended, track.sample_rate)
            sdr[output] = None if np.isnan(value) else round(value, 4)
        scored.append({"method": method, "weights": weights, "sdr": sdr})
    return scored


def _aggregate(
    per_track: list[list[dict[str, Any]]], outputs: Sequence[str]
) -> list[dict[str, Any]]:
    """Per candidate: the mean over outputs of the median over tracks."""
    out = []
    for index, candidate in enumerate(per_track[0]):
        medians = []
        for output in outputs:
            values = [t[index]["sdr"][output] for t in per_track]
            values = [v for v in values if v is not None]
            medians.append(statistics.median(values) if values else None)
        score = None if None in medians else statistics.fmean(medians)
        out.append({"method": candidate["method"], "weights": candidate["weights"], "score": score})
    return out


# A candidate has to beat an earlier one by more than this many dB to displace it. Earlier
# means simpler: equal weights before unequal, waveform before the STFT methods.
TIE_DB = 0.001


def _best(scores: list[dict[str, Any]], method: str | None = None) -> dict[str, Any] | None:
    usable = [s for s in scores if s["score"] is not None and method in (None, s["method"])]
    if not usable:
        return None
    top = max(s["score"] for s in usable)
    return next(s for s in usable if s["score"] >= top - TIE_DB)


def tune(
    dataset: Dataset,
    *,
    defaults: Defaults,
    registry: Registry,
    factory: Factory,
    fingerprint: Callable[[str], str],
    device: str,
    run_id: str,
    overlap: float = DEFAULT_OVERLAP,
    slots: Sequence[str] | None = None,
    bench_dir: Path = BENCH_DIR,
    test_lists: Sequence[Dataset] | None = None,
    command: str = "",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the members on the tuning list, try every candidate, and write ``tuning.json``."""
    say = log or (lambda _message: None)
    checked = check_disjoint(dataset, test_lists)
    systems = [
        s
        for s in ensemble_systems(defaults, registry, slots)
        if s.mode != "single_best" and not s.minus
    ]
    spec = RunSpec(
        dataset=dataset.name,
        model_ids=(),
        device=device,
        overlap=overlap,
        ensembles=tuple(systems),
    )
    summary = run_benchmark(
        spec,
        dataset=dataset,
        registry=registry,
        factory=factory,
        fingerprint=fingerprint,
        run_id=run_id,
        bench_dir=bench_dir,
        command=command,
        log=log,
    )
    store = EstimateStore(bench_dir / "estimates" / dataset.name)
    frozen = dataset.frozen

    def member_key(model_id: str, track_id: str) -> str:
        return result_key(
            dataset=dataset.name,
            frozen_sha256=frozen.sha256,
            track_id=track_id,
            model_id=model_id,
            fingerprint=fingerprint(model_id),
            device=device,
            overlap=overlap,
        )

    tune_dir = bench_dir / "tune" / run_id
    by_slot: dict[str, dict[str, Any]] = {}
    for system in systems:
        per_track = []
        skipped = []
        for track_id in dataset.track_ids():
            definition = [TUNE_VERSION, system.outputs, system.targets, track_id]
            definition += [member_key(m, track_id) for m in system.members]
            digest = hashlib.sha256(json.dumps(definition).encode()).hexdigest()
            path = tune_dir / "scores" / system.id.replace(":", ".") / f"{digest}.json"
            if path.exists():
                scored = json.loads(path.read_text(encoding="utf-8"))["candidates"]
                scored = [{**c, "weights": c["weights"] and tuple(c["weights"])} for c in scored]
            else:
                scored = _score_track(
                    system, track_id, dataset=dataset, store=store, member_key=member_key
                )
                if scored is None:
                    skipped.append(track_id)
                    continue
                write_json(path, {"system": system.id, "track_id": track_id, "candidates": scored})
            per_track.append(scored)
        say(f"{system.id}: {len(per_track)} tracks tried, {len(skipped)} without audio")
        entry = by_slot.setdefault(system.slot, {"outputs": list(system.outputs), "modes": {}})
        entry["modes"][system.mode] = {
            "members": list(system.members),
            "tracks": len(per_track),
            "skipped_tracks": skipped,
            "scores": _aggregate(per_track, system.outputs) if per_track else [],
        }

    for slot, entry in by_slot.items():
        modes = entry["modes"]
        totals = {}
        for method in (*WEIGHTED_METHODS, *CHOICE_METHODS):
            bests = [_best(m["scores"], method) for m in modes.values()]
            if all(b is not None for b in bests):
                totals[method] = sum(b["score"] for b in bests)
        # The first method within a tie wins, and waveform is first.
        top = max(totals.values(), default=None)
        entry["method"] = next((m for m, t in totals.items() if t >= top - TIE_DB), None)
        for mode in modes.values():
            scores = mode.pop("scores")
            mode["baseline"] = scores[0] if scores else None
            mode["best_any_method"] = _best(scores)
            mode["chosen"] = _best(scores, entry["method"]) if entry["method"] else None
            ranked = sorted(
                (s for s in scores if s["score"] is not None), key=lambda s: -s["score"]
            )
            mode["top"] = ranked[:5]
            mode["candidates_tried"] = len(scores)
        say(f"{slot}: {entry['method']}")

    result = {
        "tune_version": TUNE_VERSION,
        "run_id": run_id,
        "dataset": dataset.name,
        "frozen_sha256": frozen.sha256,
        "tracks": dataset.track_ids(),
        "disjoint_from": checked,
        "device": device,
        "overlap": overlap,
        "command": command,
        "git": git_state(),
        "at": now(),
        "member_failures": [
            {"model_id": j.model_id, "track_id": j.track_id, "error": j.detail}
            for j in summary.jobs
            if j.outcome == "failed" and ":" not in j.model_id
        ],
        "slots": by_slot,
    }
    write_json(tune_dir / "tuning.json", result)
    return result


def _weights(weights: Sequence[float] | None) -> str:
    return "none" if weights is None else ", ".join(f"{w:.2f}" for w in weights)


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def config_snippet(result: dict[str, Any]) -> str:
    """The chosen settings as they would be written in ``config/defaults.yaml``."""
    keys = {"two_model_blend": "two_model_weights", "full_ensemble": "full_ensemble_weights"}
    lines = ["ensembles:", "  slots:"]
    for slot, entry in result["slots"].items():
        if entry["method"] is None:
            continue
        lines += [f"    {slot}:", f"      blend_method: {entry['method']}"]
        for mode, key in keys.items():
            chosen = entry["modes"].get(mode, {}).get("chosen")
            if chosen and chosen["weights"] is not None:
                lines.append(f"      {key}: [{', '.join(f'{w:g}' for w in chosen['weights'])}]")
    return "\n".join(lines)


def render(result: dict[str, Any]) -> str:
    dirty = " with uncommitted changes" if result["git"]["dirty"] else ""
    lines = [
        f"# Blend tuning: {result['dataset']}",
        "",
        "Which method and weights give each slot's blend the best SDR on the tuning tracks.",
        "Every figure is computed from the separated audio of the run named here. Nothing",
        "here is a test score: what is chosen below is scored on the test lists by an",
        "ordinary benchmark run.",
        "",
        f"- Run: `{result['run_id']}`, {result['at']}",
        f"- Command: `{result['command']}`",
        f"- Code: commit `{result['git']['commit']}`{dirty}",
        f"- Device asked for: `{result['device']}`; chunk overlap {result['overlap']}",
        f"- Tuning list: `{result['dataset']}`, sha256 `{result['frozen_sha256']}`, "
        f"{len(result['tracks'])} tracks: " + ", ".join(f"`{t}`" for t in result["tracks"]),
    ]
    for test in result["disjoint_from"]:
        lines.append(
            f"- Shares no track with `{test['dataset']}` ({test['tracks']} tracks, "
            f"sha256 `{test['frozen_sha256']}`): checked before the run"
        )
    lines += [
        "",
        "A candidate's score is the median over tracks of the track's median SDR on 1-second",
        'windows, averaged over the slot\'s outputs, in dB. "Equal" is the plain average of the',
        "waveforms with equal weights, which is what the config holds until it is tuned.",
        "",
        "| slot | mode | models | equal | method | weights | score | gain | best of any method |",
        "|---|---|---|---:|---|---|---:|---:|---|",
    ]
    labels = {"two_model_blend": "two-model blend", "full_ensemble": "full ensemble"}
    for slot, entry in result["slots"].items():
        for mode, data in entry["modes"].items():
            base, chosen, any_method = data["baseline"], data["chosen"], data["best_any_method"]
            if not base or not chosen or base["score"] is None:
                lines.append(
                    f"| {slot} | {labels[mode]} | {', '.join(data['members'])} | n/a | | | | | |"
                )
                continue
            other = ""
            if any_method and (any_method["method"], any_method["weights"]) != (
                chosen["method"],
                chosen["weights"],
            ):
                other = (
                    f"{any_method['method']} ({_weights(any_method['weights'])}): "
                    f"{_score(any_method['score'])}"
                )
            cells = [
                slot,
                labels[mode],
                ", ".join(f"`{m}`" for m in data["members"]),
                _score(base["score"]),
                chosen["method"],
                _weights(chosen["weights"]),
                _score(chosen["score"]),
                f"{round(chosen['score'] - base['score'], 2) + 0.0:+.2f}",
                other,
            ]
            lines.append("| " + " | ".join(cells) + " |")
    if result["member_failures"]:
        lines += ["", "## Models that failed", ""]
        lines += [
            f"- `{f['model_id']}` on `{f['track_id']}`: {f['error']}"
            for f in result["member_failures"]
        ]
    lines += [
        "",
        "## As config",
        "",
        "Not applied. To use it, put these under the matching slots of",
        "`config/defaults.yaml`, then score the result on the test lists.",
        "",
        "```yaml",
        config_snippet(result),
        "```",
        "",
    ]
    return "\n".join(lines)
