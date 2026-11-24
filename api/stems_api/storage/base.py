"""The storage interface the API and the worker share (PLAN.md section 6).

The two sides never call each other: the API writes a job's source and ``spec.json``, the
worker writes ``status.json``, the stems and ``result.json``, all under ``jobs/{id}/``.
Every backend takes the same keys and behaves the same way, which the shared tests in
``tests/test_storage.py`` check for each of them.

Keys are relative paths made of plain segments. A key is checked before any backend sees
it, so no backend has to defend against ``..`` or an absolute path on its own.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, BinaryIO, Protocol, runtime_checkable

MAX_KEY_LENGTH = 512
# A segment starts with a letter, digit or underscore, so "." and ".." and hidden names
# are all refused, and the local backend can keep its own files under dot-names.
_SEGMENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
MAX_FILENAME_LENGTH = 150


class StorageError(Exception):
    """A storage operation could not be carried out."""


class InvalidKey(StorageError, ValueError):
    """The key or prefix is not one a backend may be given."""


class ObjectNotFound(StorageError):
    """No object is stored under that key."""


@runtime_checkable
class Storage(Protocol):
    def put(self, key: str, data: BinaryIO, content_type: str) -> None:
        """Store everything ``data`` yields under ``key``, replacing what was there."""

    def open(self, key: str) -> BinaryIO:
        """The object's bytes as a readable binary file. ``ObjectNotFound`` if missing."""

    def exists(self, key: str) -> bool: ...

    def delete_prefix(self, prefix: str) -> int:
        """Delete the object named ``prefix`` and every object under ``prefix/``.

        Returns how many objects were deleted. Deleting nothing is not an error, so the
        call can be repeated.
        """

    def signed_url(self, key: str, ttl_seconds: int, filename: str | None = None) -> str:
        """A link that reads ``key`` for ``ttl_seconds`` and for nobody without the link.

        ``filename`` is the name a browser saves the download under.
        """


def check_key(key: str) -> str:
    """``key`` unchanged if it is a valid object key; ``InvalidKey`` otherwise."""
    if not isinstance(key, str) or not key:
        raise InvalidKey("a storage key must be a non-empty string")
    if len(key) > MAX_KEY_LENGTH:
        raise InvalidKey(f"storage key is longer than {MAX_KEY_LENGTH} characters")
    for segment in key.split("/"):
        if not _SEGMENT.match(segment):
            raise InvalidKey(f"storage key {key!r} has the invalid segment {segment!r}")
    return key


def check_prefix(prefix: str) -> str:
    """A folder to delete, without its trailing slash. The empty prefix is refused."""
    if not isinstance(prefix, str):
        raise InvalidKey("a storage prefix must be a string")
    trimmed = prefix[:-1] if prefix.endswith("/") else prefix
    if not trimmed:
        raise InvalidKey("refusing the empty prefix: it would name every object")
    return check_key(trimmed)


def check_ttl(ttl_seconds: int) -> int:
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be a positive whole number of seconds")
    return ttl_seconds


def clean_filename(filename: str | None) -> str | None:
    """A download name that is safe in a ``Content-Disposition`` header, or None.

    Path separators, quotes and control characters are dropped: the name comes from the
    uploader's file and is only ever a label.
    """
    if filename is None:
        return None
    kept = "".join(
        ch for ch in filename if ch.isprintable() and ch not in '/\\"\r\n\x00' and ch != "\x7f"
    )
    kept = kept.strip().lstrip(".").strip()
    return kept[:MAX_FILENAME_LENGTH] or None


def put_bytes(storage: Storage, key: str, data: bytes, content_type: str) -> None:
    storage.put(key, io.BytesIO(data), content_type)


def read_bytes(storage: Storage, key: str) -> bytes:
    with storage.open(key) as handle:
        return handle.read()


def put_json(storage: Storage, key: str, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    put_bytes(storage, key, data, "application/json")


def read_json(storage: Storage, key: str) -> Any:
    return json.loads(read_bytes(storage, key).decode("utf-8"))
