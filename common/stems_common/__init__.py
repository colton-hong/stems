"""Code shared by the API, the worker and the benchmark harness."""

from stems_common.paths import REPO_ROOT
from stems_common.stems import (
    AUX_OUTPUTS,
    PRODUCT_STEMS,
    SLOT_REQUIRED_OUTPUTS,
    SLOTS,
)

__all__ = ["AUX_OUTPUTS", "PRODUCT_STEMS", "REPO_ROOT", "SLOTS", "SLOT_REQUIRED_OUTPUTS"]
