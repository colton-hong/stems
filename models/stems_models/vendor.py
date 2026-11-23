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


# --- LAION-CLAP's model code ---------------------------------------------------------------
#
# The `laion_clap` package pins numpy 1.x and installs a training stack (wandb, webdataset,
# h5py, torchvision). Only its model definition is needed to run the released checkpoint,
# so, as with UVR above, those files are taken from the pinned wheel and the rest is left.
# The licence (CC0-1.0) is unpacked beside them.

LAION_CLAP_VERSION = "1.1.7"
LAION_CLAP_WHEEL = (
    "https://files.pythonhosted.org/packages/ab/ea/"
    "6fb3d90bc1d85e6039506c248009e7f195456b81efa0a7aa413b3e13eb04/"
    "laion_clap-1.1.7-py3-none-any.whl"
)
CLAP_KEEP: tuple[str, ...] = (
    "laion_clap/clap_module/model.py",
    "laion_clap/clap_module/htsat.py",
    "laion_clap/clap_module/feature_fusion.py",
    "laion_clap/clap_module/model_configs/HTSAT-base.json",
    f"laion_clap-{LAION_CLAP_VERSION}.dist-info/licenses/LICENSE",
)
# Written in place of the package's own files. The real __init__ files import the training
# code; the image and PANN towers are never built here; and the real utils module imports
# torchvision and h5py for helpers this project does not call. The three helpers the model
# code does import are small enough to state here.
CLAP_STUBS: dict[str, str] = {
    "laion_clap/__init__.py": "",
    "laion_clap/clap_module/__init__.py": "",
    "laion_clap/clap_module/timm_model.py": (
        "class TimmModel:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise NotImplementedError('image towers are not vendored')\n"
    ),
    "laion_clap/clap_module/pann_model.py": (
        "def create_pann_model(*args, **kwargs):\n"
        "    raise NotImplementedError('PANN audio towers are not vendored')\n"
    ),
    "laion_clap/clap_module/utils.py": (
        "def interpolate(x, ratio):\n"
        "    # Repeat each time step `ratio` times along the time axis.\n"
        "    batch, steps, classes = x.shape\n"
        "    repeated = x[:, :, None, :].repeat(1, 1, ratio, 1)\n"
        "    return repeated.reshape(batch, steps * ratio, classes)\n"
        "\n\n"
        "def do_mixup(x, mixup_lambda):\n"
        "    raise NotImplementedError('mixup is a training-time augmentation')\n"
        "\n\n"
        "def freeze_batch_norm_2d(module, module_match={}, name=''):\n"
        "    raise NotImplementedError('only used when fine-tuning image towers')\n"
    ),
}


def clap_lib_root(directory: Path | None = None) -> Path:
    return (directory or THIRD_PARTY_DIR) / f"laion_clap-{LAION_CLAP_VERSION}"


def ensure_clap_lib(directory: Path | None = None, lock: ChecksumLock | None = None) -> Path:
    """Return the directory holding the pinned CLAP model code, fetching it once."""
    root = clap_lib_root(directory)
    if (root / ".complete").exists():
        return root

    base = root.parent
    wheel_file = ModelFile.model_construct(
        role="config",
        filename=LAION_CLAP_WHEEL.rsplit("/", 1)[1],
        url=LAION_CLAP_WHEEL,
        size_bytes=None,
        stem=None,
    )
    wheel, _ = fetch_file(wheel_file, base / "_archives", lock=lock or ChecksumLock())

    staging = base / f"{root.name}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if extract_zip_subset(wheel, staging, CLAP_KEEP) != len(CLAP_KEEP):
            raise RuntimeError(f"{wheel} does not contain all of {CLAP_KEEP}")
        for name, content in CLAP_STUBS.items():
            (staging / name).write_text(content, encoding="utf-8")
        (staging / ".complete").write_text(LAION_CLAP_VERSION + "\n", encoding="utf-8")
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root


# --- BEATs model code ----------------------------------------------------------------------
#
# BEATs is not a package: its three model files live in microsoft/unilm (MIT) and expect to
# be run from their own folder, and they call torchaudio's Kaldi filterbank, which has no
# build for this project's torch. So each file is fetched at one pinned revision, its
# SHA-256 pinned in the lock like any model file, and written into a package of its own
# with the imports rewritten to stay inside it. The filterbank is torchaudio's
# `compliance/kaldi.py` (BSD-2-Clause), which is plain torch code; its one use of the
# rest of torchaudio is in `mfcc`, which is not called here.

BEATS_COMMIT = "732d834db70ee0fc3886b4bcbcfb4ce7fb829be2"  # beats/ last changed 2024-04-05
TORCHAUDIO_TAG = "v2.5.1"
_UNILM_RAW = f"https://raw.githubusercontent.com/microsoft/unilm/{BEATS_COMMIT}"
_TORCHAUDIO_RAW = f"https://raw.githubusercontent.com/pytorch/audio/{TORCHAUDIO_TAG}"
# local name -> (url, {text to replace: replacement}); every replacement must apply once.
BEATS_SOURCES: dict[str, tuple[str, dict[str, str]]] = {
    "BEATs.py": (
        f"{_UNILM_RAW}/beats/BEATs.py",
        {
            "import torchaudio.compliance.kaldi as ta_kaldi\n": "from . import kaldi as ta_kaldi\n",
            "from backbone import (": "from .backbone import (",
        },
    ),
    "backbone.py": (
        f"{_UNILM_RAW}/beats/backbone.py",
        {"from modules import (": "from .modules import ("},
    ),
    "modules.py": (f"{_UNILM_RAW}/beats/modules.py", {}),
    "LICENSE": (f"{_UNILM_RAW}/LICENSE", {}),
    "kaldi.py": (
        f"{_TORCHAUDIO_RAW}/src/torchaudio/compliance/kaldi.py",
        {"import torchaudio\n": "torchaudio = None  # only mfcc() uses it; not vendored\n"},
    ),
    "LICENSE.torchaudio": (f"{_TORCHAUDIO_RAW}/LICENSE", {}),
}
BEATS_PACKAGE = "beats_vendor"


def beats_lib_root(directory: Path | None = None) -> Path:
    return (directory or THIRD_PARTY_DIR) / f"beats-{BEATS_COMMIT[:12]}"


def rewrite_imports(name: str, text: str, replacements: dict[str, str]) -> str:
    """Apply each replacement, insisting that it matches exactly once."""
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"{name}: expected one {old.strip()!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


def ensure_beats_lib(directory: Path | None = None, lock: ChecksumLock | None = None) -> Path:
    """Return the directory holding the pinned BEATs model code, fetching it once.

    The code is importable from there as the package ``beats_vendor``.
    """
    root = beats_lib_root(directory)
    if (root / ".complete").exists():
        return root

    base = root.parent
    lock = lock or ChecksumLock()
    staging = base / f"{root.name}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        package = staging / BEATS_PACKAGE
        package.mkdir(parents=True)
        for name, (url, replacements) in BEATS_SOURCES.items():
            source = ModelFile.model_construct(
                role="config", filename=name, url=url, size_bytes=None, stem=None
            )
            fetched, _ = fetch_file(source, base / "_archives" / root.name, lock=lock)
            text = fetched.read_text(encoding="utf-8")
            (package / name).write_text(rewrite_imports(name, text, replacements), encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (staging / ".complete").write_text(BEATS_COMMIT + "\n", encoding="utf-8")
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root
