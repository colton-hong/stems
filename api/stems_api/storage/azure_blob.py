"""Storage in an Azure Blob container.

The backend is given an ``azure.storage.blob.ContainerClient`` and a function that makes
a SAS query string, and uses five client calls: ``upload_blob``, ``download_blob``,
``exists``, ``list_blob_names`` and ``delete_blob``. A link is the blob's URL with a
read-only SAS token, so a download goes straight to the container.

The Azure SDK is imported only by ``azure_from_env``.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, BinaryIO
from urllib.parse import quote

from stems_api.storage.base import (
    ObjectNotFound,
    check_key,
    check_prefix,
    check_ttl,
    content_disposition,
)
from stems_common.env import get_env

CONNECTION_ENV = "AZURE_STORAGE_CONNECTION_STRING"
CONTAINER_ENV = "AZURE_STORAGE_CONTAINER"

Getter = Callable[[str], str | None]
# (blob name, expiry, Content-Disposition or None) -> the SAS query string, no leading "?"
SasSigner = Callable[[str, datetime, str | None], str]


def _is_missing(exc: Exception) -> bool:
    """True for azure-core's ``ResourceNotFoundError``, without importing azure-core."""
    return getattr(exc, "status_code", None) == 404


class _BlobStream(io.RawIOBase):
    """A ``StorageStreamDownloader`` as a raw readable stream."""

    def __init__(self, downloader: Any) -> None:
        self._downloader = downloader

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self._downloader.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class AzureBlobStorage:
    """The ``Storage`` protocol on one container."""

    def __init__(
        self,
        container: Any,
        sas: SasSigner,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._container = container
        self._sas = sas
        self._clock = clock

    def __repr__(self) -> str:
        return f"AzureBlobStorage(container={getattr(self._container, 'container_name', '?')!r})"

    def put(self, key: str, data: BinaryIO, content_type: str) -> None:
        from azure.storage.blob import ContentSettings

        self._container.get_blob_client(check_key(key)).upload_blob(
            data, overwrite=True, content_settings=ContentSettings(content_type=content_type)
        )

    def open(self, key: str) -> BinaryIO:
        try:
            downloader = self._container.get_blob_client(check_key(key)).download_blob()
        except Exception as exc:
            if _is_missing(exc):
                raise ObjectNotFound(key) from exc
            raise
        return io.BufferedReader(_BlobStream(downloader))

    def exists(self, key: str) -> bool:
        return bool(self._container.get_blob_client(check_key(key)).exists())

    def delete_prefix(self, prefix: str) -> int:
        trimmed = check_prefix(prefix)
        names = [trimmed] if self.exists(trimmed) else []
        # The slash keeps "jobs/abc" from matching "jobs/abcd/...".
        names.extend(self._container.list_blob_names(name_starts_with=trimmed + "/"))
        deleted = 0
        for name in names:
            try:
                self._container.delete_blob(name)
            except Exception as exc:
                if _is_missing(exc):  # deleted by someone else since it was listed
                    continue
                raise
            deleted += 1
        return deleted

    def signed_url(self, key: str, ttl_seconds: int, filename: str | None = None) -> str:
        check_key(key)
        expiry = datetime.fromtimestamp(int(self._clock()) + check_ttl(ttl_seconds), tz=UTC)
        disposition = None if filename is None else content_disposition(filename)
        base = str(self._container.url).rstrip("/")
        return f"{base}/{quote(key, safe='/')}?{self._sas(key, expiry, disposition)}"


def shared_key_signer(account_name: str, account_key: str, container_name: str) -> SasSigner:
    """Read-only, HTTPS-only blob SAS tokens signed with the storage account key."""
    from azure.storage.blob import BlobSasPermissions, generate_blob_sas

    def sign(blob_name: str, expiry: datetime, disposition: str | None) -> str:
        return generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
            protocol="https",
            content_disposition=disposition,
        )

    return sign


def azure_from_env(getenv: Getter = get_env, **client_options: Any) -> AzureBlobStorage:
    """An ``AzureBlobStorage`` from ``AZURE_STORAGE_CONNECTION_STRING`` and
    ``AZURE_STORAGE_CONTAINER``. The connection string must hold the account key: that
    key is what signs the download links. ``client_options`` go to the SDK's client."""
    from stems_api.storage.factory import StorageConfigError

    connection, container_name = getenv(CONNECTION_ENV), getenv(CONTAINER_ENV)
    missing = [
        name
        for name, value in ((CONNECTION_ENV, connection), (CONTAINER_ENV, container_name))
        if not value
    ]
    if missing:
        raise StorageConfigError(f"the azure_blob storage backend needs {' and '.join(missing)}")
    try:
        from azure.storage.blob import ContainerClient
    except ImportError as exc:
        raise StorageConfigError(
            "the azure_blob storage backend needs azure-storage-blob: pip install -e '.[api]'"
        ) from exc
    try:
        container = ContainerClient.from_connection_string(
            connection, container_name, **client_options
        )
    except ValueError:
        # The SDK's message can quote the connection string, so it is not passed on.
        raise StorageConfigError(f"{CONNECTION_ENV} is not a valid connection string") from None
    account_key = getattr(container.credential, "account_key", None)
    if not account_key:
        raise StorageConfigError(
            f"{CONNECTION_ENV} must contain AccountKey: download links are signed with it"
        )
    signer = shared_key_signer(container.account_name, account_key, container_name)
    return AzureBlobStorage(container, signer)
