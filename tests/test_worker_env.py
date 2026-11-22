"""The environment description is complete, serialisable and consistent with pyproject pins."""

from __future__ import annotations

import json
import re
import tomllib

import pytest

from stems_common.paths import REPO_ROOT
from stems_worker import env

torch = pytest.importorskip("torch")


def _worker_pins() -> dict[str, str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = {}
    for requirement in project["project"]["optional-dependencies"]["worker"]:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^;\s]+)", requirement)
        assert match, f"worker dependency is not an exact pin: {requirement!r}"
        pins[match[1]] = match[2]
    return pins


def test_description_has_every_field_and_is_json() -> None:
    description = env.describe_environment()
    assert json.loads(json.dumps(description)) == description
    assert set(description["packages"]) == set(env.TRACKED_PACKAGES)
    devices = description["devices"]
    assert devices["cpu"]["smoke"] == "ok"
    assert "cpu" in devices["usable"]
    assert devices["default"] == devices["usable"][0]


def test_default_device_follows_the_preference_order() -> None:
    devices = env.torch_devices()
    ranks = [env.DEVICE_PREFERENCE.index(d) for d in devices["usable"]]
    assert ranks == sorted(ranks)


def test_a_backend_that_fails_on_use_is_not_usable(monkeypatch) -> None:
    monkeypatch.setattr(
        env, "_smoke", lambda device: "ok" if device == "cpu" else "RuntimeError: broken"
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    devices = env.torch_devices()
    assert devices["mps"]["available"] is True
    assert "mps" not in devices["usable"]
    assert devices["default"] in {"cuda", "cpu"}


def test_tracked_packages_are_exactly_the_pinned_worker_dependencies() -> None:
    assert set(_worker_pins()) == set(env.TRACKED_PACKAGES)


def test_installed_versions_match_the_pins() -> None:
    installed = env.describe_environment()["packages"]
    for name, pinned in _worker_pins().items():
        # A wheel from another index carries a local tag (CI's CPU-only torch is
        # "2.15.0+cpu"): the same release, built differently.
        assert installed[name].split("+")[0] == pinned, name


def test_write_environment_creates_the_file(tmp_path) -> None:
    path = env.write_environment(tmp_path / "env" / "worker.json")
    assert json.loads(path.read_text(encoding="utf-8"))["packages"]["torch"]
