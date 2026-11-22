"""Compare the scores of runs that did the same jobs under different conditions.

Used to answer two questions with numbers from this repository: how much do scores move
when the same model runs the same tracks a second time, and how much do they move on
another device? If the second figure is no bigger than the first, devices can be mixed in
one comparison; if it is bigger, they cannot.

Like the report, this reads only run folders and the result files they list.

    bench compare --pair synthetic-v1 synthetic-mps-again --pair synthetic-v1 synthetic-cpu \\
        --out docs/benchmarks/device-spread-synthetic.md
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stems_bench.report import ReportError, RunData, _number, _signed, load_run
from stems_common.paths import REPO_ROOT


@dataclass
class Difference:
    """One system's output, scored on the same tracks in two runs."""

    system: str
    output: str
    target: str
    tracks: int
    sdr_a: float
    sdr_b: float
    # Per track: SDR in the second run minus SDR in the first.
    deltas: dict[str, float]

    @property
    def largest(self) -> float:
        return max(abs(d) for d in self.deltas.values())

    @property
    def worst_track(self) -> str:
        return max(self.deltas, key=lambda t: abs(self.deltas[t]))

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "largest": self.largest, "worst_track": self.worst_track}


@dataclass
class Comparison:
    a: RunData
    b: RunData
    differences: list[Difference]

    @property
    def same_device(self) -> bool:
        return self.a.run["device"] == self.b.run["device"]

    @property
    def all_deltas(self) -> list[float]:
        return [abs(d) for diff in self.differences for d in diff.deltas.values()]

    @property
    def largest(self) -> float | None:
        return max(self.all_deltas) if self.all_deltas else None

    @property
    def median(self) -> float | None:
        return statistics.median(self.all_deltas) if self.all_deltas else None


def _label(data: RunData) -> str:
    trial = data.run.get("trial")
    return f"`{data.run['run_id']}` ({data.run['device']}" + (f", trial {trial})" if trial else ")")


def _scores(data: RunData) -> dict[tuple[str, str], dict[str, tuple[str, float]]]:
    """(system, output) -> track -> (target, that track's median SDR), where scored."""
    table: dict[tuple[str, str], dict[str, tuple[str, float]]] = {}
    for _, _, result in data.results:
        for output, scored in result["outputs"].items():
            sdr = scored["bss"]["sdr"]
            if sdr is not None:
                key = (result["model_id"], output)
                table.setdefault(key, {})[result["track_id"]] = (scored["target"], sdr)
    return table


def compare_runs(a: RunData, b: RunData) -> Comparison:
    """Every (system, output, track) scored in both runs, side by side."""
    for key in ("dataset", "frozen_sha256", "protocol_version", "overlap"):
        if a.run[key] != b.run[key]:
            raise ReportError(
                f"{a.run['run_id']} and {b.run['run_id']} differ in {key}: they did not "
                "do the same jobs"
            )
    first, second = _scores(a), _scores(b)
    differences = []
    for key in sorted(set(first) & set(second)):
        tracks = sorted(set(first[key]) & set(second[key]))
        if not tracks:
            continue
        differences.append(
            Difference(
                system=key[0],
                output=key[1],
                target=first[key][tracks[0]][0],
                tracks=len(tracks),
                sdr_a=statistics.median(first[key][t][1] for t in tracks),
                sdr_b=statistics.median(second[key][t][1] for t in tracks),
                deltas={t: second[key][t][1] - first[key][t][1] for t in tracks},
            )
        )
    return Comparison(a, b, differences)


def _describe_run(data: RunData) -> str:
    run = data.run
    commit = run["git"]["commit"]
    dirty = " with uncommitted changes" if run["git"]["dirty"] else ""
    return f"- {_label(data)}: `{run['command']}`, commit `{commit}`{dirty}"


def render(comparisons: list[Comparison]) -> str:
    if not comparisons:
        raise ReportError("nothing to compare")
    first = comparisons[0].a.run
    lines = [
        f"# Score differences between runs: {first['dataset']}",
        "",
        "The same models on the same tracks, run more than once. Each figure is a difference",
        "between SDR values in the result files of the runs named here; nothing else is",
        "measured. A track's SDR is its median over 1-second windows.",
        "",
        f"- Frozen track list: sha256 `{first['frozen_sha256']}`",
        f"- Protocol version: {first['protocol_version']}; chunk overlap {first['overlap']}",
    ]
    seen: dict[str, RunData] = {}
    for comparison in comparisons:
        for data in (comparison.a, comparison.b):
            seen.setdefault(data.run["run_id"], data)
    lines += [_describe_run(data) for data in seen.values()]
    lines += [
        "",
        "## Summary",
        "",
        "| second run | against | kind | scores compared | median difference | largest |",
        "|---|---|---|---:|---:|---:|",
    ]
    for c in comparisons:
        kind = "same device, run again" if c.same_device else "another device"
        cells = [_label(c.b), _label(c.a), kind, str(len(c.all_deltas))]
        cells += [_number(c.median, 3), _number(c.largest, 3)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Differences are absolute values in dB, one per model output per track.", ""]

    repeats = [c.largest for c in comparisons if c.same_device and c.largest is not None]
    devices = [c.largest for c in comparisons if not c.same_device and c.largest is not None]
    if repeats and devices:
        noise, spread = max(repeats), max(devices)
        lines += ["## What follows from it", ""]
        if spread > noise:
            lines += [
                f"Moving to another device changes a score by up to {_number(spread, 3)} dB; "
                f"running again on the same device changes it by up to {_number(noise, 3)} dB.",
                "The device effect is larger than run-to-run noise, so scores from different",
                "devices must not be compared with each other. Every benchmark report is built",
                "from one run, and a run asks for one device.",
                "",
            ]
        else:
            lines += [
                f"Moving to another device changes a score by up to {_number(spread, 3)} dB, "
                f"no more than running again on the same device ({_number(noise, 3)} dB).",
                "On this data the device does not matter beyond run-to-run noise.",
                "",
            ]
    for c in comparisons:
        lines += [
            f"## {_label(c.b)} against {_label(c.a)}",
            "",
            "| model | output | tracks | SDR first | SDR second | second minus first "
            "| largest on one track | track |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for diff in sorted(c.differences, key=lambda d: -d.largest):
            cells = [f"`{diff.system}`", diff.output, str(diff.tracks)]
            cells += [_number(diff.sdr_a, 3), _number(diff.sdr_b, 3)]
            cells += [_signed(diff.sdr_b - diff.sdr_a, 3), _number(diff.largest, 3)]
            cells.append(f"`{diff.worst_track}`")
            lines.append("| " + " | ".join(cells) + " |")
        if not c.differences:
            lines.append("| (no output was scored in both runs) | | | | | | | |")
        lines.append("")
    return "\n".join(lines)


def write_comparison(
    pairs: list[tuple[Path, Path]], out: Path, *, repo_root: Path = REPO_ROOT
) -> Path:
    """Write the Markdown comparison and a JSON file of the same figures beside it."""
    loaded: dict[Path, RunData] = {}
    for run_dir in {d for pair in pairs for d in pair}:
        loaded[run_dir] = load_run(run_dir, repo_root=repo_root)
    comparisons = [compare_runs(loaded[a], loaded[b]) for a, b in pairs]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(comparisons) + "\n", encoding="utf-8")
    sidecar = {
        "comparisons": [
            {
                "first": c.a.run["run_id"],
                "second": c.b.run["run_id"],
                "same_device": c.same_device,
                "median": c.median,
                "largest": c.largest,
                "differences": [d.as_dict() for d in c.differences],
            }
            for c in comparisons
        ],
        "result_files": {
            data.run["run_id"]: [{"path": p, "sha256": h} for p, h, _ in data.results]
            for data in loaded.values()
        },
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return out
