"""Load and run every separator in the registry on the synthetic fixture.

    python -m stems_worker.smoke                 # every loadable separator
    python -m stems_worker.smoke htdemucs scnet_large --device cpu

Writes ``runs/smoke/report.md`` and ``runs/smoke/report.json``. The point is to prove each
model loads, runs, and emits the outputs the registry promises; one model failing never
stops the rest.

What it does not prove is quality. The fixture is sine tones and noise, which most models
were never trained on. The "where each fixture stem went" table is a sanity check on output
naming: drums and bass are recognisable even as synthetic tones, voices and pianos mostly
are not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_common.paths import RUNS_DIR
from stems_common.synthetic import SyntheticTrack, generate_track
from stems_models.registry import ModelEntry, Registry, load_registry
from stems_worker.device import pick_device
from stems_worker.factory import build_separators
from stems_worker.separator import DEFAULT_OVERLAP, Separator, run_separator
from stems_worker.timing import TimingLog

SMOKE_DIR = RUNS_DIR / "smoke"
FIXTURE_SEED = 1

Factory = Callable[[str, tuple[str, ...]], Sequence[Separator]]


@dataclass
class SmokeResult:
    model_id: str
    ok: bool
    # What the model was fed: the mix, or the vocals stem for the vocal stages.
    input: str
    outputs: list[str] = field(default_factory=list)
    device: str | None = None
    load_seconds: float | None = None
    run_seconds: float | None = None
    audio_seconds: float | None = None
    # outputs -> {fixture stem -> fraction of that stem's energy found in the output}
    routing: dict[str, dict[str, float]] = field(default_factory=dict)
    error: str | None = None


def shares(output: np.ndarray, references: dict[str, np.ndarray]) -> dict[str, float]:
    """How much of each known signal is in ``output``.

    The output is fitted as a weighted sum of all the reference signals at once (least
    squares), and the weights are returned: 1.0 means the whole of that signal is there,
    0.0 none of it. Fitting them jointly matters because the fixture's signals are not
    independent; two voices in the same key overlap, and measuring each on its own would
    credit one with the other's energy.
    """
    names = list(references)
    design = np.stack([references[n].astype(np.float64).ravel() for n in names], axis=1)
    active = [i for i in range(len(names)) if design[:, i].any()]
    weights = np.zeros(len(names))
    if active:
        target = output.astype(np.float64).ravel()
        weights[active] = np.linalg.lstsq(design[:, active], target, rcond=None)[0]
    return {name: round(float(w), 2) for name, w in zip(names, weights, strict=True)}


def _input_for(
    entry: ModelEntry, track: SyntheticTrack
) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    """The audio to feed a model, and the known signals to compare its outputs with."""
    if "lead_vocals" in entry.stems or "backing_vocals" in entry.stems:
        parts = {"lead": track.lead_vocals, "backing": track.backing_vocals}
        return "vocals stem", track.stems["vocals"], parts
    if "dry" in entry.stems:
        tail = track.vocals_wet - track.stems["vocals"]
        return (
            "vocals stem with reverb",
            track.vocals_wet,
            {"dry": track.stems["vocals"], "reverb": tail},
        )
    return "mix", track.mix, dict(track.stems)


def smoke_model(
    entry: ModelEntry,
    track: SyntheticTrack,
    factory: Factory,
    device: str,
    *,
    overlap: float = DEFAULT_OVERLAP,
    timings: TimingLog | None = None,
) -> SmokeResult:
    label, audio, references = _input_for(entry, track)
    result = SmokeResult(model_id=entry.id, ok=False, input=label)
    try:
        per_stem = tuple(f.stem for f in entry.weights if f.stem is not None)
        load_seconds = run_seconds = 0.0
        stems: dict[str, np.ndarray] = {}
        for separator in factory(entry.id, per_stem or tuple(entry.stems)):
            started = time.perf_counter()
            separator.load(device)
            load_seconds += time.perf_counter() - started
            try:
                run = run_separator(separator, audio, track.sample_rate, overlap=overlap)
            finally:
                separator.unload()
            run_seconds += run.wall_seconds
            result.device = run.device
            stems.update(run.stems)
            if timings is not None:
                timings.record(
                    model_id=entry.id,
                    device=run.device,
                    outputs=list(run.stems),
                    audio_seconds=run.audio_seconds,
                    wall_seconds=run.wall_seconds,
                    num_chunks=run.num_chunks,
                    stage="smoke",
                )
        if set(stems) != set(entry.stems):
            raise ValueError(
                f"emitted {sorted(stems)} but the registry lists {sorted(entry.stems)}"
            )
        bad = [
            name for name, x in stems.items() if x.shape != audio.shape or not np.isfinite(x).all()
        ]
        if bad:
            raise ValueError(f"outputs with the wrong shape or non-finite samples: {bad}")
        result.outputs = list(stems)
        result.load_seconds = round(load_seconds, 2)
        result.run_seconds = round(run_seconds, 2)
        result.audio_seconds = round(audio.shape[1] / track.sample_rate, 2)
        result.routing = {name: shares(x, references) for name, x in stems.items()}
        result.ok = True
    except Exception as e:  # one model failing must not stop the others
        last = traceback.extract_tb(e.__traceback__)[-1]
        result.error = f"{type(e).__name__}: {e} ({Path(last.filename).name}:{last.lineno})"
    return result


def run_smoke(
    registry: Registry,
    factory: Factory,
    device: str,
    *,
    model_ids: Sequence[str] = (),
    seconds: float = 10.0,
    timings: TimingLog | None = None,
    progress: Callable[[SmokeResult], None] | None = None,
) -> list[SmokeResult]:
    track = generate_track(seed=FIXTURE_SEED, duration_seconds=seconds)
    entries = [m for m in registry.models if m.kind == "separator" and m.is_loadable]
    if model_ids:
        unknown = [m for m in model_ids if m not in {e.id for e in entries}]
        if unknown:
            raise KeyError(f"not loadable separators: {unknown}")
        entries = [e for e in entries if e.id in model_ids]
    results = []
    for entry in entries:
        results.append(smoke_model(entry, track, factory, device, timings=timings))
        if progress is not None:
            progress(results[-1])
    return results


def render_report(results: Sequence[SmokeResult], header: dict[str, Any]) -> str:
    passed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lines = [
        "# Separator smoke test",
        "",
        f"- Command: `{header['command']}`",
        f"- Run at: {header['at']}",
        f"- Device requested: {header['device']}",
        f"- Fixture: synthetic, seed {header['seed']}, {header['seconds']} s, 44.1 kHz stereo",
        f"- Result: **{len(passed)} of {len(results)} models loaded and ran**"
        + (f", {len(failed)} failed" if failed else ""),
        "",
        "Timings are from this run, on this machine, on a short synthetic clip. They show",
        "that a model runs, not how fast it is on real songs.",
        "",
        "| model | result | input | device | load s | run s | outputs |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.ok:
            lines.append(
                f"| `{r.model_id}` | ran | {r.input} | {r.device} | {r.load_seconds} "
                f"| {r.run_seconds} | {', '.join(r.outputs)} |"
            )
        else:
            lines.append(f"| `{r.model_id}` | **FAILED** | {r.input} | | | | {r.error} |")
    lines += [
        "",
        "## Where each fixture signal went",
        "",
        "Each number is the share of one known fixture signal found in one model output",
        "(1.00 is all of it), from fitting the output as a weighted sum of all the known",
        "signals at once. This checks output naming, not quality: the fixture is sine",
        "tones and noise, which most models were never trained on.",
        "",
    ]
    for r in passed:
        references = list(next(iter(r.routing.values())))
        lines += [
            f"### `{r.model_id}` ({r.input})",
            "",
            "| output | " + " | ".join(references) + " |",
            "|---|" + "---|" * len(references),
        ]
        for output, row in r.routing.items():
            lines.append(
                f"| {output} | " + " | ".join(f"{row[ref]:+.2f}" for ref in references) + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    *,
    factory: Factory | None = None,
    registry: Registry | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Load and run every separator on the fixture.")
    parser.add_argument("ids", nargs="*", help="registry ids (default: every loadable separator)")
    parser.add_argument("--device", default=None, help="auto, cpu, mps or cuda")
    parser.add_argument("--seconds", type=float, default=10.0, help="length of the fixture clip")
    parser.add_argument("--out", type=Path, default=SMOKE_DIR)
    parser.add_argument("--no-timings", action="store_true")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*window was not provided.*", category=UserWarning)

    registry = registry or load_registry()
    if factory is None:

        def factory(model_id: str, stems: tuple[str, ...]):
            return build_separators(model_id, stems, registry=registry)

    device = pick_device(args.device)
    timings = (
        None if args.no_timings else TimingLog(RUNS_DIR / "timings.jsonl", {"source": "smoke"})
    )

    def progress(result: SmokeResult) -> None:
        status = (
            f"ran in {result.run_seconds} s on {result.device}"
            if result.ok
            else f"FAILED: {result.error}"
        )
        print(f"{result.model_id}: {status}", flush=True)

    try:
        results = run_smoke(
            registry, factory, device, model_ids=args.ids, seconds=args.seconds,
            timings=timings, progress=progress,
        )  # fmt: skip
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2

    given = " ".join(argv if argv is not None else sys.argv[1:])
    header = {
        "command": f"python -m stems_worker.smoke {given}".strip(),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": device,
        "seed": FIXTURE_SEED,
        "seconds": args.seconds,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(render_report(results, header) + "\n", encoding="utf-8")
    (args.out / "report.json").write_text(
        json.dumps({"header": header, "results": [asdict(r) for r in results]}, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = [r for r in results if not r.ok]
    print(f"{len(results) - len(failed)} of {len(results)} ran; report: {args.out / 'report.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
