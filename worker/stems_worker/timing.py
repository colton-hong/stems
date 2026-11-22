"""Append-only log of how long each model run took.

One JSON object per line in ``runs/timings.jsonl``, written whenever a model actually runs
(never for a cache hit). Cost-per-song and "is the ensemble worth it" both come from this
file, so each line carries what is needed to interpret it later: the device, the audio
length, the settings, and the versions of the code and of torch.

The audio is identified by a hash of its samples. File names are never logged.

    python -m stems_worker.timing            # summary per model and device
"""

from __future__ import annotations

import fcntl
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stems_common.paths import RUNS_DIR

DEFAULT_TIMINGS_PATH = RUNS_DIR / "timings.jsonl"


class TimingLog:
    def __init__(self, path: Path | str, context: dict[str, Any] | None = None) -> None:
        """``context`` is added to every line: things constant for the whole job."""
        self.path = Path(path)
        self.context = dict(context or {})

    def record(
        self,
        *,
        model_id: str,
        device: str,
        outputs: Sequence[str],
        audio_seconds: float,
        wall_seconds: float,
        num_chunks: int,
        stage: str = "separation",
    ) -> dict[str, Any]:
        line = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_id": model_id,
            "device": device,
            "stage": stage,
            "outputs": list(outputs),
            "audio_seconds": round(audio_seconds, 3),
            "wall_seconds": round(wall_seconds, 4),
            "realtime_factor": round(wall_seconds / audio_seconds, 5) if audio_seconds else None,
            "num_chunks": num_chunks,
            **self.context,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # several jobs may log at once
            f.write(json.dumps(line, sort_keys=True) + "\n")
        return line


def read_timings(path: Path | str = DEFAULT_TIMINGS_PATH) -> list[dict[str, Any]]:
    """Every well-formed line. A line cut short by a crash is skipped, not fatal."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    lines = []
    for raw in text.splitlines():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "model_id" in parsed and "wall_seconds" in parsed:
            lines.append(parsed)
    return lines


def summarise(lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs grouped by model and device, with the median seconds per second of audio."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in lines:
        if line.get("realtime_factor") is not None:
            groups.setdefault((line["model_id"], line["device"]), []).append(line)
    summary = []
    for (model_id, device), runs in sorted(groups.items()):
        factors = [r["realtime_factor"] for r in runs]
        summary.append(
            {
                "model_id": model_id,
                "device": device,
                "runs": len(runs),
                "audio_seconds": round(sum(r["audio_seconds"] for r in runs), 1),
                "median_realtime_factor": round(statistics.median(factors), 4),
                "min_realtime_factor": round(min(factors), 4),
                "max_realtime_factor": round(max(factors), 4),
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_TIMINGS_PATH
    rows = summarise(read_timings(path))
    if not rows:
        print(f"no timings in {path}")
        return 0
    print(f"{'model':40s} {'device':6s} {'runs':>4s} {'audio s':>8s}  seconds per second of audio")
    print(f"{'':40s} {'':6s} {'':>4s} {'':>8s}  {'median':>7s} {'min':>7s} {'max':>7s}")
    for r in rows:
        print(
            f"{r['model_id']:40s} {r['device']:6s} {r['runs']:4d} {r['audio_seconds']:8.1f}  "
            f"{r['median_realtime_factor']:7.3f} {r['min_realtime_factor']:7.3f} "
            f"{r['max_realtime_factor']:7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
