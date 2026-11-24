"""Storage on the local disk, with links signed by an HMAC over the key and the expiry.

Objects live under ``root/objects/<key>`` and each one's content type under
``root/meta/<key>.json``. A link is ``<base_url>/<token>``; the API's ``GET /files/{token}``
route calls ``verify`` and serves the file. A token is

    base64url(payload) "." base64url(HMAC-SHA256(secret, base64url(payload)))

where the payload is JSON holding the key, the expiry as Unix seconds and the download
name. Changing any character of either half makes the signature check fail, and the
expiry is inside what is signed, so it cannot be extended.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from stems_api.storage.base import (
    InvalidKey,
    ObjectNotFound,
    StorageError,
    check_key,
    check_prefix,
    check_ttl,
    clean_filename,
)

MIN_SECRET_BYTES = 32
MAX_TOKEN_LENGTH = 4096
_COPY_CHUNK = 1024 * 1024


class LinkError(StorageError):
    """A download link is not acceptable. The two subclasses say why."""


class LinkInvalid(LinkError):
    """The token is malformed, was changed, or was signed with another secret."""


class LinkExpired(LinkError):
    """The token is genuine and past its expiry."""


@dataclass(frozen=True)
class SignedLink:
    key: str
    expires_at: int
    filename: str | None


@dataclass(frozen=True)
class StoredObject:
    key: str
    path: Path
    size: int
    content_type: str


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    try:
        data = base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LinkInvalid("the link is malformed") from exc
    # Only the one spelling this module writes is accepted. The last character of a
    # base64 string can carry bits that decode to nothing, so without this a token could
    # be respelled and still verify.
    if _b64(data) != text:
        raise LinkInvalid("the link is malformed")
    return data


def _signature(secret: bytes, body: str) -> bytes:
    return hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()


def sign(secret: bytes, key: str, expires_at: int, filename: str | None = None) -> str:
    """The token for reading ``key`` until ``expires_at`` (Unix seconds)."""
    payload = {"k": check_key(key), "e": int(expires_at), "f": clean_filename(filename)}
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{body}.{_b64(_signature(secret, body))}"


def verify(secret: bytes, token: str, now: float) -> SignedLink:
    """What ``token`` grants. ``LinkInvalid`` or ``LinkExpired`` if it grants nothing."""
    if not isinstance(token, str) or len(token) > MAX_TOKEN_LENGTH or token.count(".") != 1:
        raise LinkInvalid("the link is malformed")
    body, _, given = token.partition(".")
    try:
        expected = _signature(secret, body)
    except UnicodeEncodeError as exc:
        raise LinkInvalid("the link is malformed") from exc
    # The signature is checked before anything in the payload is looked at.
    if not hmac.compare_digest(expected, _unb64(given)):
        raise LinkInvalid("the link's signature does not match")
    try:
        payload = json.loads(_unb64(body).decode("utf-8"))
        key, expires_at, filename = payload["k"], payload["e"], payload["f"]
        check_key(key)
    except (ValueError, KeyError, TypeError) as exc:
        raise LinkInvalid("the link is malformed") from exc
    if isinstance(expires_at, bool) or not isinstance(expires_at, int):
        raise LinkInvalid("the link is malformed")
    if filename is not None and not isinstance(filename, str):
        raise LinkInvalid("the link is malformed")
    if now >= expires_at:
        raise LinkExpired("the link has expired")
    return SignedLink(key=key, expires_at=expires_at, filename=filename)


class LocalStorage:
    """The ``Storage`` protocol on a directory. Safe for several processes at once:
    an object appears under its key only once it is completely written."""

    def __init__(
        self,
        root: Path | str,
        secret: bytes | str,
        *,
        base_url: str = "/api/files",
        clock: Callable[[], float] = time.time,
    ) -> None:
        raw = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(raw) < MIN_SECRET_BYTES:
            raise ValueError(f"the signing secret must be at least {MIN_SECRET_BYTES} bytes")
        self._secret = raw
        self.root = Path(root)
        self._objects = self.root / "objects"
        self._meta = self.root / "meta"
        self.base_url = base_url.rstrip("/")
        self._clock = clock
        created = not self.root.exists()
        self._objects.mkdir(parents=True, exist_ok=True)
        self._meta.mkdir(parents=True, exist_ok=True)
        if created:
            # Uploads are private to their uploader: no other user of this machine
            # should be able to read them off the disk. A folder that already existed
            # is left as its owner set it.
            os.chmod(self.root, 0o700)

    def __repr__(self) -> str:  # never shows the secret
        return f"LocalStorage(root={str(self.root)!r})"

    # -- paths ---------------------------------------------------------------------------

    def _inside(self, base: Path, key: str, suffix: str = "") -> Path:
        """The path for ``key`` under ``base``, refused if it resolves outside it.

        ``check_key`` already rules out ``..`` and absolute paths. Resolving as well
        covers a symbolic link planted inside the storage directory.
        """
        path = base.joinpath(*check_key(key).split("/"))
        if suffix:
            path = path.with_name(path.name + suffix)
        resolved_base = base.resolve()
        if not path.resolve().is_relative_to(resolved_base):
            raise InvalidKey(f"storage key {key!r} leaves the storage directory")
        return path

    def _object_path(self, key: str) -> Path:
        return self._inside(self._objects, key)

    def _meta_path(self, key: str) -> Path:
        return self._inside(self._meta, key, ".json")

    # -- Storage -------------------------------------------------------------------------

    def put(self, key: str, data: BinaryIO, content_type: str) -> None:
        path = self._object_path(key)
        meta = self._meta_path(key)
        if path.is_dir():
            raise StorageError(f"{key!r} is a folder of other objects and cannot be an object")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            meta.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, NotADirectoryError) as exc:
            raise StorageError(f"part of {key!r} is already an object") from exc
        temp = path.with_name(f".tmp-{secrets.token_hex(8)}")
        try:
            with temp.open("wb") as out:
                shutil.copyfileobj(data, out, _COPY_CHUNK)
                out.flush()
                os.fsync(out.fileno())
            meta_temp = meta.with_name(f".tmp-{secrets.token_hex(8)}")
            meta_temp.write_text(json.dumps({"content_type": content_type}), encoding="utf-8")
            os.replace(meta_temp, meta)
            os.replace(temp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()

    def open(self, key: str) -> BinaryIO:
        path = self._object_path(key)
        try:
            return path.open("rb")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
            raise ObjectNotFound(key) from exc

    def exists(self, key: str) -> bool:
        return self._object_path(key).is_file()

    def delete_prefix(self, prefix: str) -> int:
        trimmed = check_prefix(prefix)
        target = self._object_path(trimmed)
        deleted = 0
        if target.is_file():
            target.unlink()
            deleted += 1
        elif target.is_dir():
            deleted += sum(
                1 for p in target.rglob("*") if p.is_file() and not p.name.startswith(".")
            )
            shutil.rmtree(target)
        meta_file = self._meta_path(trimmed)
        with contextlib.suppress(FileNotFoundError):
            meta_file.unlink()
        meta_dir = self._inside(self._meta, trimmed)
        if meta_dir.is_dir():
            shutil.rmtree(meta_dir)
        self._prune(target.parent, self._objects)
        self._prune(meta_dir.parent, self._meta)
        return deleted

    def signed_url(self, key: str, ttl_seconds: int, filename: str | None = None) -> str:
        expires_at = int(self._clock()) + check_ttl(ttl_seconds)
        return f"{self.base_url}/{sign(self._secret, key, expires_at, filename)}"

    # -- what the API's file route needs -------------------------------------------------

    def verify(self, token: str) -> SignedLink:
        """What a link's token grants right now, or ``LinkInvalid`` / ``LinkExpired``."""
        return verify(self._secret, token, self._clock())

    def stat(self, key: str) -> StoredObject:
        path = self._object_path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        content_type = "application/octet-stream"
        with contextlib.suppress(FileNotFoundError, ValueError, KeyError, TypeError):
            stored = json.loads(self._meta_path(key).read_text(encoding="utf-8"))
            content_type = str(stored["content_type"])
        return StoredObject(key=key, path=path, size=path.stat().st_size, content_type=content_type)

    # -- helpers -------------------------------------------------------------------------

    @staticmethod
    def _prune(folder: Path, stop: Path) -> None:
        """Remove ``folder`` and its parents while they are empty, never ``stop`` itself."""
        while folder != stop and stop in folder.parents:
            try:
                folder.rmdir()
            except OSError:
                return
            folder = folder.parent
