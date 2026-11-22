"""Choose the torch device a model runs on."""

from __future__ import annotations

import os
from functools import lru_cache

from stems_worker.env import torch_devices


class DeviceError(RuntimeError):
    """The requested device does not exist or does not work on this machine."""


@lru_cache(maxsize=1)
def usable_devices() -> tuple[str, ...]:
    """Devices that exist and pass a smoke computation, best first (cuda, mps, cpu)."""
    return tuple(torch_devices()["usable"])


def pick_device(preferred: str | None = None) -> str:
    """Return the device to use.

    Order of precedence: the ``preferred`` argument, the ``STEMS_DEVICE`` environment
    variable, then the best usable device. Asking for a device that is not usable is an
    error rather than a silent fall back, because a timing recorded against the wrong
    device is worse than no timing.
    """
    wanted = preferred or os.environ.get("STEMS_DEVICE") or "auto"
    usable = usable_devices()
    if wanted == "auto":
        if not usable:
            raise DeviceError("no usable torch device found")
        return usable[0]
    if wanted not in usable:
        raise DeviceError(f"device {wanted!r} is not usable here (usable: {', '.join(usable)})")
    return wanted


def release_device_memory(device: str | None) -> None:
    """Give memory held for unloaded models back to the system.

    Dropping the last reference to a model is not enough on MPS or CUDA: torch keeps the
    freed blocks in its own cache. A job that loads several models in turn (or a benchmark
    that loads dozens) would otherwise run out of memory.
    """
    import gc

    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device is not None and device.startswith("cuda"):
        torch.cuda.empty_cache()
