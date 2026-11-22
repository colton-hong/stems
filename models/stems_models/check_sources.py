"""Check that every download URL in the registry answers, and record file sizes.

Sends one HTTP HEAD per unique URL (no file bodies are downloaded) and writes a status
file under runs/. With ``--write`` it also fills the ``size_bytes:`` lines in the
registry, editing the YAML text in place so comments and layout survive.

    python -m stems_models.check_sources            # report only
    python -m stems_models.check_sources --write    # also update models/registry.yaml

Idempotent: re-running only changes sizes that differ from what the server reports.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import certifi

from stems_common.paths import REGISTRY_PATH, RUNS_DIR
from stems_models.registry import load_registry

USER_AGENT = "stems-registry-check/0.1"
STATUS_PATH = RUNS_DIR / "registry" / "source_check.json"

# Share links that redirect to an interactive page; a HEAD says nothing useful about them.
UNCHECKABLE_HOSTS = ("1drv.ms",)

# python.org builds of Python ship without system CA certificates, so use certifi's bundle.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

_URL_LINE = re.compile(r"^(?P<indent>\s*)url:\s*(?P<url>\S+)\s*$")
_SIZE_LINE = re.compile(r"^(?P<indent>\s*)size_bytes:\s*(?P<value>\S+)\s*$")


@dataclass(frozen=True)
class SourceResult:
    url: str
    ok: bool
    http_status: int | None
    size_bytes: int | None
    note: str = ""


def _size_from_headers(headers) -> int | None:
    # Hugging Face reports the real size of LFS files in X-Linked-Size.
    for name in ("X-Linked-Size", "Content-Length"):
        value = headers.get(name)
        if value and value.isdigit():
            return int(value)
    return None


def head(url: str, timeout: float = 30.0) -> SourceResult:
    if any(host in url for host in UNCHECKABLE_HOSTS):
        return SourceResult(url, ok=True, http_status=None, size_bytes=None, note="not checkable")
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_SSL_CONTEXT) as response:
            return SourceResult(url, True, response.status, _size_from_headers(response.headers))
    except urllib.error.HTTPError as e:
        return SourceResult(url, False, e.code, None, note=str(e.reason))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return SourceResult(url, False, None, None, note=str(e))


def check_all(urls: list[str], workers: int = 8) -> dict[str, SourceResult]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(urls, pool.map(head, urls), strict=True))


def apply_sizes(text: str, sizes: dict[str, int]) -> tuple[str, int]:
    """Return the registry text with size_bytes filled in, and how many lines changed.

    Relies on the registry convention that ``size_bytes:`` is the line right after ``url:``.
    """
    lines = text.split("\n")
    changed = 0
    for i, line in enumerate(lines[:-1]):
        url_match = _URL_LINE.match(line)
        if not url_match or url_match["url"] not in sizes:
            continue
        size_match = _SIZE_LINE.match(lines[i + 1])
        if not size_match:
            raise ValueError(f"line {i + 2}: expected 'size_bytes:' right after 'url:'")
        new_value = str(sizes[url_match["url"]])
        if size_match["value"] != new_value:
            lines[i + 1] = f"{size_match['indent']}size_bytes: {new_value}"
            changed += 1
    return "\n".join(lines), changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--write", action="store_true", help="fill size_bytes in the registry")
    parser.add_argument("--status-file", type=Path, default=STATUS_PATH)
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    urls = sorted({f.url for m in registry.models for f in m.files})
    results = check_all(urls)

    failed = [r for r in results.values() if not r.ok]
    unsized = [r for r in results.values() if r.ok and r.size_bytes is None]
    for r in failed:
        print(f"FAIL {r.http_status or '---'} {r.url}  {r.note}")
    for r in unsized:
        print(f"NOSIZE {r.url}  {r.note}")
    print(
        f"{len(urls)} URLs: {len(urls) - len(failed)} ok, {len(failed)} failed, "
        f"{len(unsized)} ok without a size"
    )

    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    args.status_file.write_text(
        json.dumps(
            {
                "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "command": "python -m stems_models.check_sources"
                + (" --write" if args.write else ""),
                "results": [asdict(results[u]) for u in urls],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status written to {args.status_file}")

    if args.write:
        sizes = {u: r.size_bytes for u, r in results.items() if r.ok and r.size_bytes is not None}
        new_text, changed = apply_sizes(args.registry.read_text(encoding="utf-8"), sizes)
        if changed:
            args.registry.write_text(new_text, encoding="utf-8")
            load_registry(args.registry)  # fail loudly if the edit broke the file
        print(f"registry: {changed} size_bytes line(s) updated")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
