"""The ``bench`` command.

    bench run --dataset synthetic --all                  # every loadable separator
    bench run --dataset synthetic --models htdemucs,scnet_xl_ihf --device cpu
    bench run --dataset musdb18hq_test --all --background
    bench run --dataset synthetic --all --run-id <id>    # resume: finished jobs are skipped
    bench status [<run-id>]                              # the latest run if no id is given

``python -m stems_bench`` is the same command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from stems_bench.datasets import Dataset, DatasetError, get_dataset
from stems_bench.run import (
    BENCH_DIR,
    BenchError,
    Factory,
    RunSpec,
    latest_run,
    read_status,
    run_benchmark,
)
from stems_common.paths import REPO_ROOT
from stems_models.registry import Registry, load_registry


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="Benchmark separation models.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score models on a dataset's frozen track list")
    run.add_argument("--dataset", required=True, help="synthetic, musdb18hq_test, moisesdb_subset")
    which = run.add_mutually_exclusive_group(required=True)
    which.add_argument("--models", type=_csv, help="comma-separated registry ids")
    which.add_argument("--all", action="store_true", help="every loadable separator")
    run.add_argument("--tracks", type=_csv, default=None, help="only these track ids")
    run.add_argument("--device", default=None, help="auto, cpu, mps or cuda (default: auto)")
    run.add_argument("--overlap", type=float, default=None, help="default: from config")
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--run-id", default=None, help="name a new run, or resume an existing one")
    run.add_argument(
        "--keep-estimates",
        type=_csv,
        default=(),
        metavar="MODELS",
        help="keep the separated audio of these models under runs/bench/estimates",
    )
    run.add_argument("--background", action="store_true", help="detach and log to the run folder")
    run.add_argument("--quiet", action="store_true")
    status = commands.add_parser("status", help="show a run's progress")
    status.add_argument("run_id", nargs="?", help="default: the most recently updated run")
    return parser


def _print_status(run_dir: Path) -> int:
    data = read_status(run_dir)
    print(f"run {data['run_id']}: {data['state']}")
    print(f"  dataset {data['dataset']}, device {data['device']}, overlap {data['overlap']}")
    print(
        f"  {data['finished']} of {data['total']} jobs: {data['computed']} computed, "
        f"{data['cached']} already done, {data['failed']} failed, "
        f"{data['not_applicable']} not applicable"
    )
    if data.get("current"):
        print(f"  now: {data['current']['model_id']} on {data['current']['track_id']}")
    for failure in data["failures"][:20]:
        print(f"  FAILED {failure['model_id']} / {failure['track_id']}: {failure['error']}")
    print(f"  updated {data['updated_at']}")
    return 0 if data["state"] in ("running", "done") and not data["failed"] else 1


def main(
    argv: list[str] | None = None,
    *,
    factory: Factory | None = None,
    registry: Registry | None = None,
    dataset: Dataset | None = None,
    bench_dir: Path = BENCH_DIR,
) -> int:
    """Run the command. The keyword arguments let tests swap in weightless models."""
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)

    if args.command == "status":
        run_dir = bench_dir / args.run_id if args.run_id else latest_run(bench_dir)
        if run_dir is None or not (run_dir / "status.json").exists():
            print("bench: no such run", file=sys.stderr)
            return 2
        return _print_status(run_dir)

    from stems_common.settings import load_defaults
    from stems_models.download import ChecksumLock
    from stems_worker.cache import registry_fingerprint
    from stems_worker.device import DeviceError, pick_device
    from stems_worker.timing import TimingLog

    defaults = load_defaults()
    registry = registry or load_registry()
    run_id = args.run_id or f"{args.dataset}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"

    if args.background:
        run_dir = bench_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "stems_bench", *(a for a in argv if a != "--background")]
        if not args.run_id:
            command += ["--run-id", run_id]
        with (run_dir / "log.txt").open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                cwd=REPO_ROOT,
            )
        print(f"started run {run_id} (pid {process.pid}); log: {run_dir / 'log.txt'}")
        print(f"check it with: bench status {run_id}")
        return 0

    try:
        dataset = dataset or get_dataset(args.dataset)
        device = pick_device(args.device)
        if args.all:
            models = tuple(m.id for m in registry.models if m.kind == "separator" and m.is_loadable)
        else:
            models = args.models
            unknown = [m for m in models if m not in registry]
            if unknown:
                raise BenchError(f"not in the registry: {unknown}")
        if factory is None:
            from stems_worker.factory import build_separators

            def factory(model_id: str, stems: tuple[str, ...]):
                return build_separators(model_id, stems, registry=registry)

        lock = ChecksumLock()

        def weights_sha256(model_id: str) -> dict[str, str]:
            pinned = {f.filename: lock.get(f.url) for f in registry.get(model_id).files}
            return {name: entry["sha256"] for name, entry in pinned.items() if entry}

        spec = RunSpec(
            dataset=dataset.name,
            model_ids=models,
            device=device,
            overlap=defaults.separation.overlap if args.overlap is None else args.overlap,
            batch_size=args.batch_size or defaults.separation.batch_size,
            track_ids=args.tracks,
            keep_estimates=args.keep_estimates,
        )
        summary = run_benchmark(
            spec,
            dataset=dataset,
            registry=registry,
            factory=factory,
            fingerprint=registry_fingerprint(registry, lock),
            run_id=run_id,
            bench_dir=bench_dir,
            command="bench " + " ".join(argv),
            timings=TimingLog(
                bench_dir.parent / "timings.jsonl", {"source": "bench", "run_id": run_id}
            ),
            weights_sha256=weights_sha256,
            log=None if args.quiet else lambda message: print(message, flush=True),
        )
    except (DatasetError, BenchError, DeviceError) as e:
        print(f"bench: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"bench: interrupted; resume with --run-id {run_id}", file=sys.stderr)
        return 130

    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "computed": summary.count("computed"),
                "already_done": summary.count("cached"),
                "failed": summary.count("failed"),
                "not_applicable": summary.count("not_applicable"),
                "run_dir": str(summary.run_dir),
            }
        )
    )
    return 0 if summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())
