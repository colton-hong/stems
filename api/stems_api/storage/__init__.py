"""Object storage behind one interface: local disk, S3-compatible, Azure Blob.

Imported by the API and by the worker. Nothing here imports the web framework.
"""

from stems_api.storage.base import (
    InvalidKey,
    ObjectNotFound,
    Storage,
    StorageError,
    check_key,
    check_prefix,
    clean_filename,
    content_disposition,
    put_bytes,
    put_json,
    read_bytes,
    read_json,
)
from stems_api.storage.factory import StorageConfigError, storage_from_env
from stems_api.storage.local import (
    LinkError,
    LinkExpired,
    LinkInvalid,
    LocalStorage,
    SignedLink,
    StoredObject,
)

__all__ = [
    "InvalidKey",
    "LinkError",
    "LinkExpired",
    "LinkInvalid",
    "LocalStorage",
    "ObjectNotFound",
    "SignedLink",
    "Storage",
    "StorageConfigError",
    "StorageError",
    "StoredObject",
    "check_key",
    "check_prefix",
    "clean_filename",
    "content_disposition",
    "put_bytes",
    "put_json",
    "read_bytes",
    "read_json",
    "storage_from_env",
]
