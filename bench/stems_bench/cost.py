"""What one song costs, from measured usage and prices checked on a stated day (P7.3).

    python -m stems_bench.cost --out docs/cost-per-song.md

Three costs make up a song: the decision layer (Jev, per token), the separation (GPU
time, per second) and keeping the files for a week (storage, per gigabyte-month). Each
is a measured quantity times a price. The prices are in ``config/prices.yaml`` with the
page and the day they were read. The quantities come from files of runs in this repo,
named in the report next to every figure.

Where a quantity has not been measured the report says so and gives no dollar figure.
That is the case for GPU time: the only timings here are from this machine (Apple MPS
or CPU), and how long a rented T4 takes is unknown until a job is run on one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from stems_common.paths import REPO_ROOT
from stems_common.settings import Defaults, load_defaults
from stems_common.stems import PRODUCT_STEMS

PRICES_PATH = REPO_ROOT / "config" / "prices.yaml"
RECORDINGS_DIR = REPO_ROOT / "tests" / "fixtures" / "jev"
BENCHMARK_PATH = REPO_ROOT / "docs" / "benchmarks" / "moisesdb.json"
SONG_MINUTES = 4.0
MODES = ("single_best", "two_model_blend", "full_ensemble")
# 24-bit stereo WAV at 44.1 kHz: what the product writes by default.
WAV_BYTES_PER_SECOND = 44100 * 2 * 3
RETENTION_DAYS_PER_MONTH = 30.0


class CostError(ValueError):
    """The cost report cannot be made from the files given."""


def load_prices(path: Path = PRICES_PATH) -> dict[str, Any]:
    prices = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for section in ("jev", "modal", "r2"):
        if "source" not in prices.get(section, {}):
            raise CostError(f"{path}: the {section} prices need a source")
    if "checked" not in prices:
        raise CostError(f"{path}: say on which day the prices were checked")
    return prices


def recorded_usage(directory: Path = RECORDINGS_DIR) -> dict[str, dict[str, Any]]:
    """Tokens of the one real call per decision kind (the P4.9 live check)."""
    usage = {}
    for kind in ("parse", "plan", "gate"):
        path = directory / f"{kind}.recorded.jsonl"
        if not path.is_file():
            raise CostError(f"no recording of a real {kind} call: {path}")
        entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        used = entry["response"]["usage"]
        usage[kind] = {
            "input_tokens": int(used["input_tokens"]),
            "output_tokens": int(used.get("output_tokens") or 0),
            "file": str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT)
            else str(path),
        }
    return usage


def decision_tokens(
    usage: Mapping[str, Mapping[str, Any]], stems: int, *, parse: bool, retries_per_stem: int = 0
) -> tuple[int, int]:
    """(input, output) tokens for one job: one plan, one gate call per stem attempt."""
    calls = {"plan": 1, "gate": stems * (1 + retries_per_stem), "parse": 1 if parse else 0}
    return (
        sum(usage[kind]["input_tokens"] * n for kind, n in calls.items()),
        sum(usage[kind]["output_tokens"] * n for kind, n in calls.items()),
    )


def jev_cost(tokens: tuple[int, int], prices: Mapping[str, Any]) -> float:
    jev = prices["jev"]
    return (
        tokens[0] * jev["usd_per_million_input_tokens"]
        + tokens[1] * jev["usd_per_million_output_tokens"]
    ) / 1_000_000


def model_rates(benchmark: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Seconds of work per minute of audio for each model in a benchmark report."""
    rates: dict[str, dict[str, Any]] = {}
    for row in benchmark["rows"]:
        if ":" in row["system"] or row.get("seconds_per_audio_minute") is None:
            continue  # a blend's row: its time is the sum of its members'
        rates[row["system"]] = {
            "seconds_per_audio_minute": float(row["seconds_per_audio_minute"]),
            "devices": list(row["devices"]),
        }
    return rates


def models_for(defaults: Defaults, mode: str, stems: Sequence[str] = PRODUCT_STEMS) -> list[str]:
    """The distinct models a job runs for ``stems`` in ``mode``, in first-use order."""
    models: list[str] = []
    for stem in stems:
        for model in defaults.ensembles.slots[stem].for_mode(mode):
            if model not in models:
                models.append(model)
    return models


def separation_seconds(
    models: Sequence[str], rates: Mapping[str, Mapping[str, Any]], minutes: float
) -> tuple[float | None, list[str]]:
    """Seconds to run ``models`` on ``minutes`` of audio, and the models with no timing."""
    missing = [model for model in models if model not in rates]
    if missing:
        return None, missing
    return sum(rates[m]["seconds_per_audio_minute"] for m in models) * minutes, []


def storage_gb(stems: int, minutes: float) -> float:
    """Upper bound on what one job stores: the upload, each stem, and a zip of the stems,
    all taken as 24-bit WAV. By arithmetic from the format, not measured."""
    one = WAV_BYTES_PER_SECOND * minutes * 60
    return (one + stems * one + stems * one) / 1e9


def build(
    prices: Mapping[str, Any],
    usage: Mapping[str, Mapping[str, Any]],
    benchmark: Mapping[str, Any],
    defaults: Defaults,
    *,
    benchmark_file: str,
    minutes: float = SONG_MINUTES,
) -> dict[str, Any]:
    stems = len(PRODUCT_STEMS)
    rates = model_rates(benchmark)
    decisions = {}
    for label, parse, retries in (
        ("six ticked stems, no retries", False, 0),
        ("a request in words, six stems, no retries", True, 0),
        ("six ticked stems, every stem retried twice", False, 2),
    ):
        tokens = decision_tokens(usage, stems, parse=parse, retries_per_stem=retries)
        decisions[label] = {"tokens": tokens, "usd": jev_cost(tokens, prices)}
    separation = {}
    for mode in MODES:
        models = models_for(defaults, mode)
        seconds, missing = separation_seconds(models, rates, minutes)
        separation[mode] = {"models": models, "seconds": seconds, "unmeasured": missing}
    gigabytes = storage_gb(stems, minutes)
    r2 = prices["r2"]
    return {
        "checked": str(prices["checked"]),
        "minutes": minutes,
        "usage": usage,
        "decisions": decisions,
        "separation": separation,
        "benchmark": {
            "file": benchmark_file,
            "run_id": benchmark["run_id"],
            "devices": sorted({d for r in rates.values() for d in r["devices"]}),
        },
        "storage": {
            "gigabytes": gigabytes,
            "days": defaults.retention.days,
            "usd": gigabytes
            * r2["usd_per_gb_month"]
            * defaults.retention.days
            / RETENTION_DAYS_PER_MONTH,
        },
        "provisional_defaults": defaults.ensembles.provisional,
    }


def _usd(value: float) -> str:
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def render(report: Mapping[str, Any], prices: Mapping[str, Any]) -> str:
    jev, modal, r2 = prices["jev"], prices["modal"], prices["r2"]
    minutes = report["minutes"]
    lines = [
        "# Cost per song",
        "",
        f"Generated by `python -m stems_bench.cost`. Prices were read on {report['checked']} from "
        "the pages named below and are kept in `config/prices.yaml`. Every quantity is from a "
        f"file in this repo, named beside it. A song here is {minutes:.0f} minutes long and all "
        "six stems are asked for.",
        "",
        "## Summary",
        "",
        "| Part | Per song | Status |",
        "|---|---:|---|",
    ]
    first = next(iter(report["decisions"].values()))
    lines += [
        f"| Decisions (Jev) | {_usd(first['usd'])} | measured tokens, one call of each kind |",
        "| Separation on a rented GPU | not known | no GPU timing exists (BLOCKED.md B4) |",
        f"| Storage for {report['storage']['days']} days (Cloudflare R2) | "
        f"{_usd(report['storage']['usd'])} at most | sizes by arithmetic, not measured |",
        "| Downloads (R2) | $0 | R2 does not charge for egress |",
        "",
        "There is no total: the largest part, GPU time, has not been measured.",
        "",
        "## Decisions (Jev)",
        "",
        f"Price: ${jev['usd_per_million_input_tokens']} per million input tokens; output tokens "
        f"cost ${jev['usd_per_million_output_tokens']:.0f} ({jev['source']}, read "
        f"{report['checked']}).",
        "",
        "Tokens used by one real call of each kind (the live check of 2026-11-24):",
        "",
        "| Call | Input tokens | Output tokens | From |",
        "|---|---:|---:|---|",
    ]
    for kind in ("parse", "plan", "gate"):
        used = report["usage"][kind]
        lines.append(
            f"| {kind} | {used['input_tokens']:,} | {used['output_tokens']:,} | `{used['file']}` |"
        )
    lines += [
        "",
        "A job makes one plan call and one gate call per stem, per attempt; a request in words "
        "adds one parse call.",
        "",
        "| Job | Input tokens | Cost |",
        "|---|---:|---:|",
    ]
    for label, entry in report["decisions"].items():
        lines.append(f"| {label} | {entry['tokens'][0]:,} | {_usd(entry['usd'])} |")
    lines += [
        "",
        "Each token count is from a single call on made-up inputs. Real jobs will differ: the "
        "plan call grows with the number of stems asked for and of candidate models.",
        "",
        "## Separation",
        "",
        f"Time to separate a {minutes:.0f}-minute song into six stems on this machine, from the "
        f"median seconds per minute of audio of each model in `{report['benchmark']['file']}` "
        f"(run `{report['benchmark']['run_id']}`, device "
        f"{', '.join(report['benchmark']['devices'])}). Loading the models is not included.",
        "",
        "| Mode | Models run | Seconds | Minutes |",
        "|---|---|---:|---:|",
    ]
    for mode, entry in report["separation"].items():
        models = ", ".join(f"`{m}`" for m in entry["models"])
        if entry["seconds"] is None:
            missing = ", ".join(f"`{m}`" for m in entry["unmeasured"])
            lines.append(f"| {mode} | {models} | not known: no timing for {missing} | |")
        else:
            lines.append(
                f"| {mode} | {models} | {entry['seconds']:.0f} | {entry['seconds'] / 60:.1f} |"
            )
    if report["provisional_defaults"]:
        lines += [
            "",
            "These are the provisional defaults of `config/defaults.yaml`. The benchmark's "
            "recommendation (P2.13) will change which models run, and with them these times.",
        ]
    t4, l4 = modal["usd_per_second"]["T4"], modal["usd_per_second"]["L4"]
    lines += [
        "",
        f"**What this costs on a rented GPU is not known.** Modal charges ${t4} per second for "
        f"a T4 and ${l4} for an L4 ({modal['source']}, read {report['checked']}; CPU and memory "
        "are charged on top, and non-preemptible runs at three times the list price). The cost "
        "of a song is that price times the seconds a T4 or L4 takes. Those seconds have not "
        "been measured: every timing in this repo is from Apple MPS or CPU, and a different "
        "GPU can be faster or slower by an amount nobody here has observed. One job on Modal "
        "(BLOCKED.md B4) gives the number.",
        "",
        "## Storage and downloads",
        "",
        f"Cloudflare R2: ${r2['usd_per_gb_month']} per GB-month, no charge for downloads, "
        f"{r2['free_gb_month']} GB-month free each month ({r2['source']}, read "
        f"{report['checked']}).",
        "",
        f"One job keeps at most {report['storage']['gigabytes']:.2f} GB for "
        f"{report['storage']['days']} days: the upload, six stems and a zip of them, all "
        "counted as 24-bit 44.1 kHz stereo WAV. That is arithmetic from the format, not a "
        f"measurement, and it is the upper end: FLAC and MP3 are smaller. It comes to "
        f"{_usd(report['storage']['usd'])} per song, before the free allowance.",
        "",
        "## Not priced",
        "",
    ]
    for item in prices.get("not_found") or []:
        lines.append(f"- {item['what']}: looked for at {item['source']} and not found there.")
    lines += [
        "- The machine the API runs on, and its database.",
        "- Electricity and wear when the worker is this laptop.",
        "- Azure: nothing has run there.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.cost", description="Write the cost-per-song report."
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--minutes", type=float, default=SONG_MINUTES)
    args = parser.parse_args(argv)
    try:
        prices = load_prices()
        benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
        name = args.benchmark.resolve()
        shown = str(name.relative_to(REPO_ROOT)) if name.is_relative_to(REPO_ROOT) else str(name)
        report = build(
            prices,
            recorded_usage(),
            benchmark,
            load_defaults(),
            benchmark_file=shown,
            minutes=args.minutes,
        )
    except (CostError, FileNotFoundError) as exc:
        print(f"cost: {exc}", file=sys.stderr)
        return 1
    text = render(report, prices)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
