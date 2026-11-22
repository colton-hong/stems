"""Nothing that must stay out of the repository is tracked by git."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from stems_common.paths import REPO_ROOT

AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif"}
WEIGHTS = {".ckpt", ".pth", ".th", ".pt", ".onnx", ".pb", ".safetensors", ".bin"}


def _tracked_files() -> list[str]:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


def test_no_audio_or_model_weights_are_tracked() -> None:
    forbidden = AUDIO | WEIGHTS
    offenders = [f for f in _tracked_files() if any(f.lower().endswith(ext) for ext in forbidden)]
    assert offenders == []


def test_no_env_file_or_run_output_is_tracked() -> None:
    offenders = [
        f
        for f in _tracked_files()
        if f == ".env"
        or (f.startswith(".env.") and f != ".env.example")
        or (f.startswith("runs/") and f != "runs/.gitkeep")
        or f.startswith("models/weights/")
    ]
    assert offenders == []
