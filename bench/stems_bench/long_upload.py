"""The longest upload the limits allow, through the worker as the API starts it (P12.2).

    python -m stems_bench.long_upload --device mps      # about as long as the audio is short

Every job so far was a few minutes of audio. The limit is ``uploads.max_duration_seconds``
(twenty minutes), and a job that long holds the whole mix and all six stems in memory at
once. This makes a synthetic signal of exactly that length (a one-minute signal of tones
and noise, repeated; never real music), puts it in storage as an upload, and starts the
worker the way the local provider does: ``python -m stems_worker.jobs JOB_ID`` in a
process of its own, with the real default model. It records whether the job finished,
whether every stem has the right length, how long it took, and the most memory the worker
process held, as the operating system reports it.

The audio is deleted afterwards; what is kept is ``result.json`` in the run's folder.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import resource
import secrets
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.run import git_state, write_json
from stems_common.paths import REPO_ROOT, RUNS_DIR
from stems_common.stems import PRODUCT_STEMS

LONG_DIR = RUNS_DIR / "long-upload"
DEFAULT_RUN = "long-upload-v1"
# The signal that is repeated: one minute from the synthetic generator.
UNIT_SECONDS = 60.0
SEED = 41


class LongUploadError(RuntimeError):
    """The check could not be run."""


def make_signal(seconds: float) -> tuple[np.ndarray, int]:
    """``seconds`` of synthetic stereo audio: a one-minute signal, repeated and cut."""
    from stems_common.synthetic import generate_track

    unit = generate_track(seed=SEED, duration_seconds=min(UNIT_SECONDS, seconds))
    rate = unit.sample_rate
    wanted = round(seconds * rate)
    repeats = -(-wanted // unit.mix.shape[-1])
    return np.tile(unit.mix, (1, repeats))[:, :wanted].astype(np.float32), rate


def peak_rss_bytes(usage: resource.struct_rusage) -> int:
    """``ru_maxrss`` in bytes: macOS reports bytes, Linux kilobytes."""
    return int(usage.ru_maxrss) if sys.platform == "darwin" else int(usage.ru_maxrss) * 1024


def run(
    run_dir: Path,
    *,
    seconds: float,
    device: str | None,
    fake_models: bool = False,
    timeout: float = 4 * 3600,
) -> dict[str, Any]:
    """Run one job of ``seconds`` of audio in a worker process and say what happened."""
    import soundfile as sf

    from stems_api.storage import LocalStorage, put_json, read_bytes, read_json
    from stems_common import jobs as contract
    from stems_worker.audio import save_audio

    run_dir.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(32)
    storage = LocalStorage(run_dir / "storage", secret.encode())
    job_id = str(uuid.uuid4())
    audio, rate = make_signal(seconds)
    frames = audio.shape[-1]
    upload = run_dir / "upload.wav"
    save_audio(upload, audio, rate, "wav")
    del audio
    upload_bytes = upload.stat().st_size
    with upload.open("rb") as handle:
        storage.put(contract.source_key(job_id), handle, "application/octet-stream")
    upload.unlink()
    spec = contract.build_spec(
        job_id, created_at=datetime.now(UTC), stems=list(PRODUCT_STEMS), fmt="flac"
    )
    put_json(storage, contract.spec_key(job_id), spec)

    env = {
        **os.environ,
        "STEMS_STORAGE_BACKEND": "local",
        "STEMS_STORAGE_LOCAL_DIR": str(run_dir / "storage"),
        "STEMS_SIGNING_SECRET": secret,
        # No .env: no key is used, so the rules decide, as in every other check.
        "STEMS_DOTENV_PATH": str(run_dir / "no-such-file.env"),
        "STEMS_WORKER_FAKE_MODELS": "1" if fake_models else "",
    }
    command = [sys.executable, "-m", "stems_worker.jobs", job_id]
    if device:
        command += ["--device", device]
    record: dict[str, Any] = {
        "job_id": job_id,
        "audio_seconds": round(frames / rate, 3),
        "upload_bytes": upload_bytes,
        "device_asked": device,
        "fake_models": fake_models,
        "command": " ".join(command[1:]),
    }
    try:
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
            done = subprocess.run(
                command, cwd=REPO_ROOT, env=env, stdout=log, stderr=log, timeout=timeout
            )
        wall = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(
            exit_code=done.returncode,
            wall_seconds=round(wall, 1),
            # The most any child of this process held; the worker is the only large one.
            worker_peak_rss_bytes=max(peak_rss_bytes(after), peak_rss_bytes(before)),
            worker_cpu_seconds=round(
                (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime), 1
            ),
        )
        if not storage.exists(contract.result_key(job_id)):
            record.update(state="no result", error="the worker wrote no result file")
            return record
        result = read_json(storage, contract.result_key(job_id))
        record.update(
            state=result["state"],
            error=result.get("error"),
            device=result.get("device"),
            timings=result["timings"],
            stems={},
        )
        for stem in result.get("stems", []):
            info = sf.info(io.BytesIO(read_bytes(storage, stem["key"])))
            record["stems"][stem["name"]] = {
                "bytes": stem["bytes"],
                "frames": info.frames,
                "right_length": info.frames == frames and info.samplerate == rate,
                "warning": stem.get("warning"),
                "attempts": stem["attempts"],
            }
        return record
    finally:
        storage.delete_prefix(contract.job_prefix(job_id))


def summary(record: dict[str, Any]) -> str:
    lines = [
        f"job {record['job_id']}: {record.get('state', 'not run')}"
        + (f" ({record['error']})" if record.get("error") else ""),
        f"  audio: {record['audio_seconds']:.0f} s, upload {record['upload_bytes'] / 1e6:.0f} MB",
    ]
    if "wall_seconds" in record:
        lines += [
            f"  wall time: {record['wall_seconds']:.0f} s on {record.get('device') or 'unknown'}",
            f"  worker's peak memory: {record['worker_peak_rss_bytes'] / 1e9:.2f} GB",
        ]
    for name, stem in (record.get("stems") or {}).items():
        lines.append(
            f"  {name}: {stem['bytes'] / 1e6:.0f} MB, "
            f"{'right length' if stem['right_length'] else 'WRONG LENGTH'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from stems_common.settings import load_defaults

    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.long_upload",
        description="The longest upload the limits allow, through a worker process.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--device", default=None, help="default: the worker's own choice")
    parser.add_argument(
        "--seconds", type=float, default=None, help="default: uploads.max_duration_seconds"
    )
    parser.add_argument("--fake-models", action="store_true", help="stand-in models, no weights")
    args = parser.parse_args(argv)
    defaults = load_defaults()
    seconds = args.seconds or float(defaults.uploads.max_duration_seconds)
    run_dir = LONG_DIR / args.run_id
    started = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(run_dir / "status.json", {"state": "running", "started": started})
    try:
        record = run(run_dir, seconds=seconds, device=args.device, fake_models=args.fake_models)
    except BaseException as error:
        write_json(
            run_dir / "status.json",
            {"state": "failed", "started": started, "error": type(error).__name__},
        )
        raise
    record.update(
        run_id=args.run_id,
        started=started,
        finished=datetime.now(UTC).isoformat(timespec="seconds"),
        limit_seconds=defaults.uploads.max_duration_seconds,
        git=git_state(),
    )
    write_json(run_dir / "result.json", record)
    write_json(run_dir / "status.json", {"state": "done", "started": started})
    print(summary(record))
    print(json.dumps({"result": str(run_dir / "result.json")}))
    ok = record.get("state") == "ready" and all(
        stem["right_length"] for stem in record.get("stems", {}).values()
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
