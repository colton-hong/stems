"""Where jobs run: the local machine, Modal, or (as a seam) Azure Container Apps.

The Modal and Azure modules are imported only when selected.
"""

from stems_api.providers.base import (
    JobSpec,
    ProviderError,
    ProviderHandle,
    ProviderStatus,
    WorkerProvider,
)
from stems_api.providers.factory import ProviderConfigError, provider_from_env
from stems_api.providers.local import LocalProvider

__all__ = [
    "JobSpec",
    "LocalProvider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderHandle",
    "ProviderStatus",
    "WorkerProvider",
    "provider_from_env",
]
