"""Choose the storage backend from the environment.

    STEMS_STORAGE_BACKEND     local (default) | s3 | azure_blob
    STEMS_STORAGE_LOCAL_DIR   where the local backend keeps its files (default runs/dev/storage)
    STEMS_SIGNING_SECRET      signs the local backend's download links

Nothing has to be set for local use. With no ``STEMS_SIGNING_SECRET`` the local backend
makes a random secret once and keeps it in ``<dir>/signing-secret`` (owner-only), so the
API and the worker on one machine agree and links survive a restart. Set the variable
instead when the API runs on more than one machine.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from pathlib import Path

from stems_api.storage.base import Storage, StorageError
from stems_api.storage.local import MIN_SECRET_BYTES, LocalStorage
from stems_common.env import get_env
from stems_common.paths import RUNS_DIR

BACKEND_ENV = "STEMS_STORAGE_BACKEND"
LOCAL_DIR_ENV = "STEMS_STORAGE_LOCAL_DIR"
SECRET_ENV = "STEMS_SIGNING_SECRET"
DEFAULT_LOCAL_DIR = RUNS_DIR / "dev" / "storage"
SECRET_FILE = "signing-secret"
BACKENDS = ("local", "s3", "azure_blob")

Getter = Callable[[str], str | None]


class StorageConfigError(StorageError):
    """The environment does not describe a usable storage backend."""


def local_secret(root: Path, getenv: Getter = get_env) -> bytes:
    """The link-signing secret: from the environment, else a file made on first use."""
    given = getenv(SECRET_ENV)
    if given:
        raw = given.encode("utf-8")
        if len(raw) < MIN_SECRET_BYTES:
            raise StorageConfigError(
                f"{SECRET_ENV} must be at least {MIN_SECRET_BYTES} characters; generate one "
                'with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        return raw
    path = root / SECRET_FILE
    root.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL: when two processes start together, one creates the file and the other reads it.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(secrets.token_urlsafe(48))
    raw = path.read_text(encoding="ascii").strip().encode("ascii")
    if len(raw) < MIN_SECRET_BYTES:
        raise StorageConfigError(f"{path} is damaged; delete it and a new secret is made")
    return raw


def local_storage(getenv: Getter = get_env, *, base_url: str = "/api/files") -> LocalStorage:
    root = Path(getenv(LOCAL_DIR_ENV) or DEFAULT_LOCAL_DIR).expanduser()
    return LocalStorage(root, local_secret(root, getenv), base_url=base_url)


def storage_from_env(getenv: Getter = get_env, *, base_url: str = "/api/files") -> Storage:
    """The backend named by ``STEMS_STORAGE_BACKEND``. ``base_url`` is where the API serves
    the local backend's links; the other backends link straight to the bucket."""
    backend = (getenv(BACKEND_ENV) or "local").strip().lower()
    if backend == "local":
        return local_storage(getenv, base_url=base_url)
    if backend == "s3":
        from stems_api.storage.s3 import s3_from_env

        return s3_from_env(getenv)
    if backend == "azure_blob":
        from stems_api.storage.azure_blob import azure_from_env

        return azure_from_env(getenv)
    raise StorageConfigError(
        f"{BACKEND_ENV}={backend!r} is not a storage backend; choose one of {', '.join(BACKENDS)}"
    )
