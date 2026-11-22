"""The ``separate`` command.

separate song.wav --stems vocals,drums,bass,guitar,piano,other
separate song.mp3 --stems all --format flac --out ./stems
separate song.wav --stems bass --model bass=scnet_xl_ihf --device cpu
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from stems_common.stems import PRODUCT_STEMS
from stems_models.download import DownloadError
from stems_worker.audio import AudioError
from stems_worker.compose import CompositionError, SeparatorFactory
from stems_worker.device import DeviceError
from stems_worker.pipeline import DEFAULT_CACHE_DIR, PipelineError, SeparateOptions, separate_file

FORMATS = ("wav", "flac", "mp3_320", "mp3_192")


def _stems(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return PRODUCT_STEMS
    names = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = [n for n in names if n not in PRODUCT_STEMS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown stems {unknown or value!r}; choose from {', '.join(PRODUCT_STEMS)} or 'all'"
        )
    return tuple(dict.fromkeys(names))


def _override(value: str) -> tuple[str, str]:
    stem, _, model_id = value.partition("=")
    if stem not in PRODUCT_STEMS or not model_id:
        raise argparse.ArgumentTypeError(f"expected STEM=MODEL_ID with a known stem, got {value!r}")
    return stem, model_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="separate", description="Split an audio file into stems.")
    parser.add_argument("file", type=Path, help="audio file to separate")
    parser.add_argument(
        "--stems", type=_stems, required=True, help="comma-separated stem names, or 'all'"
    )
    parser.add_argument("--out", type=Path, help="output folder (default: <file name>_stems)")
    parser.add_argument("--format", choices=FORMATS, default=None, help="default: from config")
    parser.add_argument("--device", default=None, help="auto, cpu, mps or cuda (default: auto)")
    parser.add_argument(
        "--mode", default="single_best", help="single_best (blends arrive in Phase 2)"
    )
    parser.add_argument(
        "--model",
        type=_override,
        action="append",
        default=[],
        metavar="STEM=MODEL_ID",
        help="use a specific registry model for one stem (repeatable)",
    )
    parser.add_argument("--other-method", choices=("auto", "residual"), default=None)
    parser.add_argument("--overlap", type=float, default=None, help="chunk overlap, 0 to <1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true", help="always run every model")
    parser.add_argument("--quiet", action="store_true", help="no progress on stderr")
    return parser


def main(
    argv: list[str] | None = None, *, factory: SeparatorFactory | None = None, **inject
) -> int:
    """Run the command. ``factory`` and ``inject`` let tests swap in weightless models."""
    args = build_parser().parse_args(argv)
    # Raised inside third-party model code on every run; they say nothing about this run.
    warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*window was not provided.*", category=UserWarning)
    from stems_common.settings import load_defaults

    defaults = inject.pop("defaults", None) or load_defaults()
    options = SeparateOptions(
        stems=args.stems,
        fmt=args.format or defaults.audio.output_format,
        device=args.device,
        mode=args.mode,
        other_method=args.other_method,
        overlap=args.overlap,
        batch_size=args.batch_size,
        models=dict(args.model),
        cache_dir=None if args.no_cache else args.cache_dir,
    )
    output_dir = args.out or args.file.with_name(f"{args.file.stem}_stems")

    last: dict[str, int] = {}

    def progress(model_id: str, done: int, total: int) -> None:
        if last.get(model_id) != done:
            last[model_id] = done
            end = "\n" if done == total else "\r"
            print(f"  {model_id}: chunk {done} of {total}", end=end, file=sys.stderr, flush=True)

    try:
        manifest = separate_file(
            args.file,
            output_dir,
            options,
            defaults=defaults,
            factory=factory,
            progress=None if args.quiet else progress,
            **inject,
        )
    except FileNotFoundError as e:
        print(f"separate: file not found: {e}", file=sys.stderr)
        return 1
    except (AudioError, CompositionError, DeviceError, DownloadError, PipelineError) as e:
        print(f"separate: {e}", file=sys.stderr)
        return 1
    except ValueError as e:  # bad option values (overlap out of range, unknown model id)
        print(f"separate: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"separate: {e.args[0]}", file=sys.stderr)
        return 1

    for run in manifest["models"]:
        how = "cached" if run["cached"] else f"{run['wall_seconds']:.1f} s on {run['device']}"
        print(f"{run['model_id']}: {', '.join(run['outputs'])} ({how})")
    for stem, info in manifest["stems"].items():
        print(f"{output_dir / info['file']}  [{stem}]")
    residual = manifest["residual"]
    if residual["computed"]:
        print(f"residual: {residual['level_db']} dB relative to the mix ({residual['note']})")
    if manifest["output_gain"] != 1.0:
        print(
            f"note: all stems were lowered by a gain of {manifest['output_gain']} to avoid clipping"
        )
    print(f"manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
