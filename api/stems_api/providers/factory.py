"""Choose the worker provider from the environment.

    STEMS_WORKER_PROVIDER   local (default) | modal | azure_container_apps

``modal`` needs ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` and a storage backend the
Modal container can reach (``s3`` or ``azure_blob``). ``azure_container_apps`` is a seam
that is not implemented: selecting it fails here, at startup, with instructions.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from stems_api.providers.base import ProviderError, WorkerProvider
from stems_api.providers.local import LocalProvider
from stems_common.env import get_env

PROVIDER_ENV = "STEMS_WORKER_PROVIDER"
STORAGE_ENV = "STEMS_STORAGE_BACKEND"
MODAL_TOKEN_ENVS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
PROVIDERS = ("local", "modal", "azure_container_apps")


class ProviderConfigError(ProviderError):
    """The environment does not describe a usable worker provider."""


def provider_from_env(getenv: Callable[[str], str | None] = get_env) -> WorkerProvider:
    name = (getenv(PROVIDER_ENV) or "local").strip().lower()
    if name == "local":
        return LocalProvider()
    if name == "modal":
        return _modal(getenv)
    if name == "azure_container_apps":
        from stems_api.providers.azure_aca import NOT_BUILT

        raise ProviderConfigError(NOT_BUILT)
    raise ProviderConfigError(
        f"{PROVIDER_ENV}={name!r} is not a worker provider; choose one of {', '.join(PROVIDERS)}"
    )


def _modal(getenv: Callable[[str], str | None]) -> WorkerProvider:
    from stems_api.providers.modal import ModalProvider

    missing = [name for name in MODAL_TOKEN_ENVS if not getenv(name)]
    if missing:
        raise ProviderConfigError(f"the modal worker provider needs {' and '.join(missing)}")
    storage = (getenv(STORAGE_ENV) or "local").strip().lower()
    if storage == "local":
        raise ProviderConfigError(
            "the modal worker provider needs shared storage: set STEMS_STORAGE_BACKEND to s3 "
            "or azure_blob (a Modal container cannot read this machine's disk)"
        )
    # The SDK reads its token from the process environment; .env values are put there.
    for name in MODAL_TOKEN_ENVS:
        os.environ.setdefault(name, getenv(name) or "")
    return ModalProvider()
