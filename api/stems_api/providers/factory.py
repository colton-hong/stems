"""Choose the worker provider from the environment.

STEMS_WORKER_PROVIDER   local (default) | modal | azure_container_apps
"""

from __future__ import annotations

from collections.abc import Callable

from stems_api.providers.base import ProviderError, WorkerProvider
from stems_api.providers.local import LocalProvider
from stems_common.env import get_env

PROVIDER_ENV = "STEMS_WORKER_PROVIDER"
PROVIDERS = ("local", "modal", "azure_container_apps")


class ProviderConfigError(ProviderError):
    """The environment does not describe a usable worker provider."""


def provider_from_env(getenv: Callable[[str], str | None] = get_env) -> WorkerProvider:
    name = (getenv(PROVIDER_ENV) or "local").strip().lower()
    if name == "local":
        return LocalProvider()
    if name in PROVIDERS:
        raise ProviderConfigError(f"the {name!r} worker provider is not built yet")
    raise ProviderConfigError(
        f"{PROVIDER_ENV}={name!r} is not a worker provider; choose one of {', '.join(PROVIDERS)}"
    )
