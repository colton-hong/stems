"""The registry size writer edits YAML text without disturbing anything else."""

from __future__ import annotations

import pytest

from stems_common.paths import REGISTRY_PATH
from stems_models import check_sources
from stems_models.check_sources import SourceResult, apply_sizes, head
from stems_models.registry import load_registry

SAMPLE = """\
models:
  - id: a
    files:
      - role: weights
        filename: a.ckpt
        url: https://example.invalid/a.ckpt
        size_bytes: null
      - role: config
        filename: a.yaml
        url: https://example.invalid/a.yaml
        size_bytes: 10
    # a comment that must survive
"""


def test_apply_sizes_fills_only_the_matching_lines() -> None:
    out, changed = apply_sizes(SAMPLE, {"https://example.invalid/a.ckpt": 123})
    assert changed == 1
    assert "        size_bytes: 123\n" in out
    assert "        size_bytes: 10\n" in out
    assert "# a comment that must survive" in out
    assert out.count("\n") == SAMPLE.count("\n")


def test_apply_sizes_is_idempotent() -> None:
    sizes = {"https://example.invalid/a.ckpt": 123, "https://example.invalid/a.yaml": 10}
    once, _ = apply_sizes(SAMPLE, sizes)
    twice, changed = apply_sizes(once, sizes)
    assert (twice, changed) == (once, 0)


def test_apply_sizes_refuses_a_registry_that_breaks_the_layout_rule() -> None:
    broken = SAMPLE.replace("        size_bytes: null\n", "")
    with pytest.raises(ValueError, match="right after 'url:'"):
        apply_sizes(broken, {"https://example.invalid/a.ckpt": 1})


def test_committed_registry_follows_the_layout_rule() -> None:
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    urls = {f.url for m in load_registry().models for f in m.files}
    # Applying a size to every URL must find a size_bytes line for each of them.
    _, changed = apply_sizes(text, dict.fromkeys(urls, 1))
    assert changed >= 1


def test_share_links_are_not_fetched() -> None:
    result = head("https://1drv.ms/u/s!example")
    assert result == SourceResult(
        "https://1drv.ms/u/s!example", True, None, None, note="not checkable"
    )


def test_main_writes_sizes_and_status_without_network(tmp_path, monkeypatch) -> None:
    registry_copy = tmp_path / "registry.yaml"
    registry_copy.write_text(REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    status = tmp_path / "status.json"

    def fake_check_all(urls: list[str], workers: int = 8) -> dict[str, SourceResult]:
        return {u: SourceResult(u, True, 200, 4242) for u in urls}

    monkeypatch.setattr(check_sources, "check_all", fake_check_all)
    code = check_sources.main(
        ["--registry", str(registry_copy), "--status-file", str(status), "--write"]
    )

    assert code == 0
    assert status.exists()
    updated = load_registry(registry_copy)
    assert all(f.size_bytes == 4242 for m in updated.models for f in m.files)
    assert updated.get("htdemucs").total_size_bytes == 4242


def test_main_fails_when_a_source_is_down(tmp_path, monkeypatch) -> None:
    def fake_check_all(urls: list[str], workers: int = 8) -> dict[str, SourceResult]:
        return {u: SourceResult(u, False, 404, None, note="Not Found") for u in urls}

    monkeypatch.setattr(check_sources, "check_all", fake_check_all)
    code = check_sources.main(["--status-file", str(tmp_path / "status.json")])
    assert code == 1
