"""Fetch third-party model code that is not installable as a package here.

Most checkpoints in the registry were trained with ZFTurbo's Music-Source-Separation-
Training (MSST) and need its model classes to load. Its PyPI package (``msst`` 0.1.0)
requires Python below 3.14 and torch below 2.12, which rules it out on this project's
interpreter and pins, so the source is fetched at one pinned commit instead:

- the commit hash is fixed here, in code;
- the archive's SHA-256 is pinned in ``models/checksums.lock.json`` like any model file;
- only the ``models/`` tree and the licence are unpacked, into ``third_party/``, which is
  git-ignored.

Only model definitions are used from it. Config parsing and weight loading are ours
(safe YAML, ``weights_only=True``); see ``stems_worker.adapters.msst``.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from stems_common.paths import REPO_ROOT
from stems_models.download import ChecksumLock, fetch_file
from stems_models.registry import ModelFile

MSST_REPOSITORY = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
MSST_COMMIT = "1523206d12a521bdb20673726562524cf0f9635b"  # main on 2026-11-21, MIT licence
MSST_KEEP: tuple[str, ...] = ("models/", "LICENSE")

THIRD_PARTY_DIR = REPO_ROOT / "third_party"


def msst_root(directory: Path | None = None) -> Path:
    return (directory or THIRD_PARTY_DIR) / f"msst-{MSST_COMMIT[:12]}"


def extract_subset(archive: Path, destination: Path, keep: tuple[str, ...]) -> int:
    """Unpack the regular files under ``keep`` from a GitHub source archive.

    The archive's single top-level directory is dropped. Links, devices and anything that
    would land outside ``destination`` are skipped, so a hostile archive cannot write
    elsewhere. Returns the number of files written.
    """
    destination = destination.resolve()
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            _, _, relative = member.name.partition("/")
            if ".." in PurePosixPath(relative).parts or relative.startswith("/"):
                continue
            if not relative or not any(
                relative == k or (k.endswith("/") and relative.startswith(k)) for k in keep
            ):
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            written += 1
    return written


def ensure_msst(directory: Path | None = None, lock: ChecksumLock | None = None) -> Path:
    """Return the directory holding the pinned MSST ``models/`` tree, fetching it once."""
    root = msst_root(directory)
    if (root / ".complete").exists():
        return root

    base = root.parent
    archive_file = ModelFile.model_construct(
        role="config",
        filename=f"msst-{MSST_COMMIT[:12]}.tar.gz",
        url=f"{MSST_REPOSITORY}/archive/{MSST_COMMIT}.tar.gz",
        size_bytes=None,
        stem=None,
    )
    archive, _ = fetch_file(archive_file, base / "_archives", lock=lock or ChecksumLock())

    staging = base / f"{root.name}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if extract_subset(archive, staging, MSST_KEEP) == 0:
            raise RuntimeError(f"{archive} contained none of {MSST_KEEP}")
        (staging / ".complete").write_text(MSST_COMMIT + "\n", encoding="utf-8")
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root


# --- UVR's VR-architecture code ------------------------------------------------------------
#
# The VR models (UVR karaoke, de-echo, de-noise) need UVR's network definitions and its
# multi-band spectrogram routines. The maintained copy of those lives inside the
# audio-separator package (MIT). Installing that package would pull in a large dependency
# tree and its import runs all of it, so only the files below are unpacked from the pinned
# wheel, with empty package files around them.

AUDIO_SEPARATOR_VERSION = "0.48.0"
AUDIO_SEPARATOR_WHEEL = (
    "https://files.pythonhosted.org/packages/83/50/"
    "29d83017f8efc049a9db683b20467083e3130256a98ea24237eb3eb6d016/"
    "audio_separator-0.48.0-py3-none-any.whl"
)
UVR_KEEP: tuple[str, ...] = (
    "audio_separator/separator/uvr_lib_v5/spec_utils.py",
    "audio_separator/separator/uvr_lib_v5/vr_network/",
    "audio_separator/separator/exceptions.py",
    f"audio_separator-{AUDIO_SEPARATOR_VERSION}.dist-info/licenses/LICENSE",
)
# Files written in place of the package's own. The real __init__ files import the whole
# package; pyrb wraps an external time-stretch program we never call.
UVR_STUBS: dict[str, str] = {
    "audio_separator/__init__.py": "",
    "audio_separator/separator/__init__.py": "",
    "audio_separator/separator/uvr_lib_v5/__init__.py": "",
    "audio_separator/separator/uvr_lib_v5/pyrb.py": "# Unused here; lets spec_utils import.\n",
}


def uvr_lib_root(directory: Path | None = None) -> Path:
    return (directory or THIRD_PARTY_DIR) / f"audio_separator-{AUDIO_SEPARATOR_VERSION}"


def extract_zip_subset(archive: Path, destination: Path, keep: tuple[str, ...]) -> int:
    """Unpack the regular files under ``keep`` from a zip (a wheel). Same rules as above."""
    destination = destination.resolve()
    written = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename
            if info.is_dir() or ".." in PurePosixPath(name).parts or name.startswith("/"):
                continue
            if not any(name == k or (k.endswith("/") and name.startswith(k)) for k in keep):
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            written += 1
    return written


def ensure_uvr_lib(directory: Path | None = None, lock: ChecksumLock | None = None) -> Path:
    """Return the directory holding the pinned UVR VR code, fetching it once."""
    root = uvr_lib_root(directory)
    if (root / ".complete").exists():
        return root

    base = root.parent
    wheel_file = ModelFile.model_construct(
        role="config",
        filename=AUDIO_SEPARATOR_WHEEL.rsplit("/", 1)[1],
        url=AUDIO_SEPARATOR_WHEEL,
        size_bytes=None,
        stem=None,
    )
    wheel, _ = fetch_file(wheel_file, base / "_archives", lock=lock or ChecksumLock())

    staging = base / f"{root.name}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if extract_zip_subset(wheel, staging, UVR_KEEP) == 0:
            raise RuntimeError(f"{wheel} contained none of {UVR_KEEP}")
        for name, content in UVR_STUBS.items():
            (staging / name).write_text(content, encoding="utf-8")
        (staging / ".complete").write_text(AUDIO_SEPARATOR_VERSION + "\n", encoding="utf-8")
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root
