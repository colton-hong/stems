"""Turn a finished benchmark run into a Markdown report.

The report is a pure function of the run folder: ``run.json``, ``status.json``,
``results.jsonl`` and the result files that list names. Every figure in it is either copied
from those files or is a median or count of values in them, and ``report.json``, written
next to it, holds the same figures with the hash of every result file they came from.
Nothing is measured, estimated or looked up while writing a report.

    bench report <run-id>                                  # runs/bench/<run-id>/report.md
    bench report <run-id> --out docs/benchmarks/synthetic.md
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stems_common.paths import REPO_ROOT

METRIC_LABELS = {"sdr": "SDR", "sir": "SIR", "sar": "SAR", "isr": "ISR"}
# The order targets are reported in; anything else follows alphabetically.
TARGET_ORDER = (
    "vocals",
    "drums",
    "bass",
    "guitar",
    "piano",
    "other",
    "other_4stem",
    "instrumental",
    "lead_vocals",
    "backing_vocals",
    "dry",
)
TARGET_NOTES = {
    "other": "six-stem other: what is left without vocals, drums, bass, guitar and piano",
    "other_4stem": "four-stem other: guitar, piano and everything else that is not vocals, "
    "drums or bass",
    "instrumental": "everything except the vocals",
    "dry": "the vocals without reverb; the model is fed the vocals with reverb",
    "lead_vocals": "the model is fed the reference vocals stem",
    "backing_vocals": "the model is fed the reference vocals stem",
}


class ReportError(RuntimeError):
    pass


@dataclass
class RunData:
    run_dir: Path
    run: dict[str, Any]
    status: dict[str, Any]
    jobs: list[dict[str, Any]]
    # (path as listed, sha256 of the file, parsed content), for computed and cached jobs
    results: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)


@dataclass
class Row:
    """One system's figures for one target, over the run's tracks."""

    system: str
    output: str
    target: str
    derived: bool
    tracks: int  # tracks the system has a result for
    tracks_scored: int  # of those, tracks where the target is present and was scored
    medians: dict[str, float | None]
    seconds_per_track: float
    seconds_per_audio_minute: float
    devices: list[str]
    high_band_ratio_db: float | None
    silence_fraction: float | None
    duration_mismatches: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_run(run_dir: Path, *, repo_root: Path = REPO_ROOT) -> RunData:
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        raise ReportError(f"{run_dir} is not a benchmark run: {e.filename} is missing") from None
    data = RunData(run_dir, run, status, [json.loads(line) for line in lines if line.strip()])
    for job in data.jobs:
        if job["outcome"] not in ("computed", "cached"):
            continue
        path = Path(job["result"])
        raw = (path if path.is_absolute() else repo_root / path).read_bytes()
        result = json.loads(raw)
        for key in ("dataset", "frozen_sha256", "protocol_version"):
            if result[key] != run[key]:
                raise ReportError(f"{job['result']}: its {key} is not the run's")
        data.results.append((job["result"], hashlib.sha256(raw).hexdigest(), result))
    return data


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate(results: list[dict[str, Any]]) -> list[Row]:
    """Per system and output: the median over tracks of each track's own median.

    A track where the target is absent, or where nothing could be scored, has no value and
    is left out of the median; ``tracks_scored`` says how many were in it.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        for output in result["outputs"]:
            grouped.setdefault((result["model_id"], output), []).append(result)
    rows = []
    for (system, output), items in grouped.items():
        scores = [r["outputs"][output] for r in items]
        medians = {}
        for metric in METRIC_LABELS:
            values = [s["bss"][metric] for s in scores if s["bss"][metric] is not None]
            medians[metric] = _median(values)
        present = [s for s in scores if s["reference_present"]]
        ratios = [
            s["signal"]["high_band_ratio_db"]
            for s in present
            if s["signal"]["high_band_ratio_db"] is not None
        ]
        rows.append(
            Row(
                system=system,
                output=output,
                target=scores[0]["target"],
                derived=bool(scores[0]["derived"]),
                tracks=len(items),
                tracks_scored=sum(s["bss"]["sdr"] is not None for s in scores),
                medians=medians,
                seconds_per_track=statistics.median(r["wall_seconds"] for r in items),
                seconds_per_audio_minute=60.0
                * statistics.median(r["seconds_per_audio_second"] for r in items),
                devices=sorted({r["device"] for r in items}),
                high_band_ratio_db=_median(ratios),
                silence_fraction=_median(
                    [s["signal"]["estimate"]["silence_fraction"] for s in present]
                ),
                duration_mismatches=sum(not s["signal"]["duration_matches"] for s in scores),
            )
        )
    return rows


def _number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{round(value, digits) + 0.0:.{digits}f}"  # + 0.0 turns -0.0 into 0.0


def _target_key(target: str) -> tuple[int, str]:
    return (TARGET_ORDER.index(target) if target in TARGET_ORDER else len(TARGET_ORDER), target)


def _sort_rows(rows: list[Row]) -> list[Row]:
    def key(row: Row) -> tuple[int, float, str]:
        sdr = row.medians["sdr"]
        return (0 if sdr is not None else 1, -(sdr or 0.0), row.system)

    return sorted(rows, key=key)


def render(data: RunData, rows: list[Row]) -> str:
    run, status = data.run, data.status
    metrics = [m for m in ("sdr", "sir", "sar") if m in run.get("reliable_metrics", ["sdr"])]
    commits = sorted({r["git"]["commit"] or "unknown" for _, _, r in data.results})
    dirty = any(r["git"]["dirty"] for _, _, r in data.results)
    tracks = run["tracks"]
    audio = {r["track_id"]: r["audio_seconds"] for _, _, r in data.results}
    outcomes = ("computed", "cached", "failed", "not_applicable")
    counts = {o: sum(j["outcome"] == o for j in data.jobs) for o in outcomes}
    lines = [
        f"# Benchmark: {run['dataset']}",
        "",
        "Generated from the result files of one run. Every figure here is copied from those",
        "files or is a median or count of values in them; `report.json` next to the run lists",
        "each file with its hash.",
        "",
        f"- Run: `{run['run_id']}`, state `{status['state']}`, started {run['started_at']}",
        f"- Command: `{run['command']}`",
        f"- Code: commit `{run['git']['commit']}`"
        + (" with uncommitted changes" if run["git"]["dirty"] else ""),
        f"- Results computed on commit(s): {', '.join(f'`{c}`' for c in commits)}"
        + (" (some with uncommitted changes)" if dirty else ""),
        f"- Protocol version: {run['protocol_version']}",
        f"- Frozen track list: sha256 `{run['frozen_sha256']}`, {len(tracks)} tracks in this run",
        f"- Device asked for: `{run['device']}`; chunk overlap {run['overlap']}; "
        f"batch size {run['batch_size']}",
        "- Packages: "
        + ", ".join(f"{name} {version}" for name, version in run["packages"].items() if version),
        f"- Jobs (one model on one track): {counts['computed']} computed in this run, "
        f"{counts['cached']} taken from earlier runs, {counts['failed']} failed, "
        f"{counts['not_applicable']} not applicable",
    ]
    if audio:
        total = sum(audio.values())
        lines.append(
            f"- Audio: {len(audio)} tracks with results, {total:.1f} s in total, "
            f"median {statistics.median(audio.values()):.1f} s per track"
        )
    lines += [
        "",
        "## How to read the tables",
        "",
        "- Scores are in dB, higher is better. A model's score for a stem is the median over",
        "  tracks of each track's median over 1-second windows (museval BSS Eval v4).",
        "- Every stem is scored as the pair (target, input minus target), so figures are",
        "  comparable between models with different numbers of outputs.",
        '- "tracks" is how many tracks the score rests on, out of those the model ran on. A',
        "  track where the stem is absent is not scored.",
        "- Time is for the whole model on one track, all of its outputs together, on the",
        "  device shown. It does not include loading the model.",
        '- "high band" is the estimate\'s energy at and above 8 kHz relative to the',
        "  reference's, in dB: 0 is a match, negative means lost top end.",
    ]
    if set(metrics) != {"sdr", "sir", "sar"}:
        lines += [
            "- **Only " + ", ".join(METRIC_LABELS[m] for m in metrics) + " is reported for this "
            "dataset.** Its tracks are sine tones and",
            "  noise, for which museval's SIR and SAR are numerical noise. The scores show that",
            "  the harness works; they do not show how good a model is on music.",
        ]
    lines.append("")

    systems = {r["model_id"]: r for _, _, r in data.results if r.get("kind") == "ensemble"}
    by_target: dict[str, list[Row]] = {}
    for row in rows:
        if row.system not in systems:
            by_target.setdefault(row.target, []).append(row)
    for target in sorted(by_target, key=_target_key):
        lines += [f"## {target}", ""]
        if target in TARGET_NOTES:
            lines += [TARGET_NOTES[target][0].upper() + TARGET_NOTES[target][1:] + ".", ""]
        header = ["model", *(METRIC_LABELS[m] for m in metrics), "tracks", "s per track"]
        header += ["s per min of audio", "device", "high band", "note"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|" + "---:|" * (len(metrics) + 3) + "---|---:|---|")
        for row in _sort_rows(by_target[target]):
            notes = []
            if row.derived:
                notes.append("sum of its guitar, piano and other")
            if row.duration_mismatches:
                notes.append(f"wrong length on {row.duration_mismatches} track(s)")
            if row.silence_fraction is not None and row.silence_fraction >= 0.5:
                notes.append(f"output is {row.silence_fraction:.0%} silence")
            cells = [
                f"`{row.system}`",
                *(_number(row.medians[m]) for m in metrics),
                f"{row.tracks_scored} of {row.tracks}",
                _number(row.seconds_per_track),
                _number(row.seconds_per_audio_minute),
                ", ".join(row.devices),
                _number(row.high_band_ratio_db, 1),
                "; ".join(notes),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += _ensemble_section(rows, systems, metrics)
    failed = [j for j in data.jobs if j["outcome"] == "failed"]
    skipped = [j for j in data.jobs if j["outcome"] == "not_applicable"]
    if failed:
        lines += ["## Failures", "", "These jobs have no result and are in no table.", ""]
        lines += [f"- `{j['model_id']}` on `{j['track_id']}`: {j['detail']}" for j in failed]
        lines.append("")
    if skipped:
        reasons: dict[str, str] = {}
        for job in skipped:
            reasons.setdefault(job["model_id"], job["detail"])
        lines += ["## Not applicable to this dataset", ""]
        lines += [f"- `{model}`: {reason}" for model, reason in reasons.items()]
        lines.append("")
    if status["state"] != "done":
        lines += [f"**This run did not finish (state `{status['state']}`).**", ""]
    return "\n".join(lines)


MODE_ORDER = ("single_best", "two_model_blend", "full_ensemble")
MODE_LABELS = {
    "single_best": "single best",
    "two_model_blend": "two-model blend",
    "full_ensemble": "full ensemble",
}


def _ensemble_section(
    rows: list[Row], systems: dict[str, dict[str, Any]], metrics: list[str]
) -> list[str]:
    """Per slot: its three modes side by side, with what each costs in time."""
    if not systems:
        return []
    lines = [
        "## Single best, two-model blend, full ensemble",
        "",
        "Each slot of `config/defaults.yaml` in its three modes, on the same tracks. A blend's",
        "time is the sum of the times of every model it needs, as if that stem alone were",
        "asked for. \"vs single best\" is the blend's median SDR minus the single best's.",
        "",
    ]
    by_slot: dict[str, list[Row]] = {}
    for row in rows:
        if row.system in systems:
            by_slot.setdefault(systems[row.system]["slot"], []).append(row)
    for slot, slot_rows in by_slot.items():
        example = systems[slot_rows[0].system]
        lines += [f"### {slot}", ""]
        if example["minus"]:
            taken = " and ".join(sub["slot"] for sub in example["minus"])
            lines += [
                f"The four-stem blend with the same mode's {taken} taken out: what the "
                "pipeline delivers as the six-stem other.",
                "",
            ]
        header = ["mode", "output", "scored against", *(METRIC_LABELS[m] for m in metrics)]
        header += ["vs single best", "tracks", "s per track", "s per min of audio", "device"]
        header += ["models (weight)", "method"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|---|---|" + "---:|" * (len(metrics) + 4) + "---|---|---|")
        baseline = {
            row.output: row.medians["sdr"]
            for row in slot_rows
            if systems[row.system]["mode"] == "single_best"
        }
        slot_rows.sort(key=lambda r: (r.output, MODE_ORDER.index(systems[r.system]["mode"])))
        for row in slot_rows:
            system = systems[row.system]
            sdr, base = row.medians["sdr"], baseline.get(row.output)
            gain = None if sdr is None or base is None else sdr - base
            members = [
                f"`{model}` ({weight:.2f})"
                for model, weight in zip(system["members"], system["weights"], strict=True)
            ]
            for sub in system["minus"]:
                members += [f"minus {sub['slot']}: `{model}`" for model in sub["members"]]
            cells = [
                MODE_LABELS[system["mode"]],
                row.output,
                row.target,
                *(_number(row.medians[m]) for m in metrics),
                "" if system["mode"] == "single_best" else _signed(gain),
                f"{row.tracks_scored} of {row.tracks}",
                _number(row.seconds_per_track),
                _number(row.seconds_per_audio_minute),
                ", ".join(row.devices),
                ", ".join(members),
                system["method"] if len(system["members"]) > 1 else "",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _signed(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{round(value, digits) + 0.0:+.{digits}f}"


def write_report(run_dir: Path, out: Path | None = None, *, repo_root: Path = REPO_ROOT) -> Path:
    """Write the Markdown report and ``report.json``. Returns the Markdown path."""
    data = load_run(run_dir, repo_root=repo_root)
    rows = aggregate([r for _, _, r in data.results])
    out = out or run_dir / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, rows) + "\n", encoding="utf-8")
    sidecar = {
        "run_id": data.run["run_id"],
        "dataset": data.run["dataset"],
        "frozen_sha256": data.run["frozen_sha256"],
        "protocol_version": data.run["protocol_version"],
        "rows": [row.as_dict() for row in rows],
        "result_files": [{"path": path, "sha256": digest} for path, digest, _ in data.results],
    }
    (out.with_suffix(".json")).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return out
