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
