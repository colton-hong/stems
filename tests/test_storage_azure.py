"""The Azure Blob backend through the real azure-storage-blob client, with no network.

``tests/azure_fake.py`` answers the SDK's HTTP requests from memory, so the backend's
calls go through the SDK's real request building and response parsing. SAS tokens are
computed locally by the SDK, so the links are the real thing. What none of this proves is
that a storage account accepts the key: that is the live check (BLOCKED.md B7).
"""

from __future__ import annotations

import base64
import hashlib
from urllib.parse import parse_qs, urlsplit

import pytest

from stems_api.storage import (
    ObjectNotFound,
    StorageConfigError,
    put_bytes,
    storage_from_env,
)
from stems_api.storage.azure_blob import AzureBlobStorage, azure_from_env

pytest.importorskip("azure.storage.blob")
import azure_fake

CONTAINER = "stems-test"
KEY = "jobs/abc/stems/vocals.wav"
NOW = 1_800_000_000


def env(values: dict[str, str]):
    return lambda name: values.get(name) or None


SETTINGS = {
    "AZURE_STORAGE_CONNECTION_STRING": azure_fake.CONNECTION_STRING,
    "AZURE_STORAGE_CONTAINER": CONTAINER,
}


@pytest.fixture
def transport() -> azure_fake.FakeBlobTransport:
    return azure_fake.FakeBlobTransport()


@pytest.fixture
def storage(transport) -> AzureBlobStorage:
    return azure_from_env(env(SETTINGS), transport=transport, retry_total=0)


def test_put_sends_the_content_type_and_replaces(storage, transport) -> None:
    put_bytes(storage, KEY, b"first", "audio/wav")
    put_bytes(storage, KEY, b"second", "audio/flac")
    assert transport.blobs[CONTAINER, KEY] == (b"second", "audio/flac")


def test_open_is_a_buffered_binary_file(storage) -> None:
    data = bytes(range(256)) * 400
    put_bytes(storage, KEY, data, "audio/wav")
    with storage.open(KEY) as handle:
        assert handle.read(10) == data[:10]
        assert handle.read(1000) == data[10:1010]
        assert handle.read() == data[1010:]
        assert handle.read() == b""
    with pytest.raises(ObjectNotFound):
        storage.open("jobs/abc/stems/missing.wav")


def test_delete_prefix_lists_under_the_folder_only(storage, transport) -> None:
    for key in ("jobs/abc/source", KEY, "jobs/abcd/source"):
        put_bytes(storage, key, b"x", "audio/wav")
    transport.requests.clear()
    assert storage.delete_prefix("jobs/abc") == 2
    assert sorted(name for _, name in transport.blobs) == ["jobs/abcd/source"]
    methods = [method for method, _ in transport.requests]
    assert methods == ["HEAD", "GET", "DELETE", "DELETE"]


def test_delete_prefix_tolerates_a_blob_that_is_already_gone(storage, transport) -> None:
    for key in ("jobs/abc/source", KEY):
        put_bytes(storage, key, b"x", "audio/wav")
    transport.vanish_on_delete.add("jobs/abc/source")
    assert storage.delete_prefix("jobs/abc") == 1
    assert transport.blobs == {}


def test_link_is_a_read_only_https_sas_with_expiry_and_download_name(transport) -> None:
    storage = azure_from_env(env(SETTINGS), transport=transport)
    storage._clock = lambda: NOW  # a fixed time, so the expiry can be checked exactly
    url = storage.signed_url(KEY, 900, filename='Ma Chanson "été".wav')
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.scheme == "https"
    assert parts.netloc == f"{azure_fake.ACCOUNT}.blob.core.windows.net"
    assert parts.path == f"/{CONTAINER}/{KEY}"
    assert query["sp"] == ["r"]  # read only
    assert query["sr"] == ["b"]  # this one blob
    assert query["spr"] == ["https"]
    assert query["se"] == ["2027-01-15T08:15:00Z"]  # NOW + 900 s
    assert query["rscd"] == [
        'attachment; filename="Ma Chanson _t_.wav"; '
        "filename*=UTF-8''Ma%20Chanson%20%C3%A9t%C3%A9.wav"
    ]
    assert len(base64.b64decode(query["sig"][0])) == 32
    assert azure_fake.ACCOUNT_KEY not in url
    assert "rscd" not in storage.signed_url(KEY, 60)
    # A link to another blob, or with another expiry, has another signature.
    other = parse_qs(urlsplit(storage.signed_url("jobs/other/source", 900)).query)
    later = parse_qs(urlsplit(storage.signed_url(KEY, 901)).query)
    assert len({query["sig"][0], other["sig"][0], later["sig"][0]}) == 3


def test_the_signature_is_an_hmac_with_the_account_key(transport) -> None:
    """The token is signed locally with the account key: changing the key changes it."""
    storage = azure_from_env(env(SETTINGS), transport=transport)
    storage._clock = lambda: NOW
    other_key = base64.b64encode(hashlib.sha256(b"another key").digest()).decode()
    other_settings = {
        **SETTINGS,
        "AZURE_STORAGE_CONNECTION_STRING": azure_fake.CONNECTION_STRING.replace(
            azure_fake.ACCOUNT_KEY, other_key
        ),
    }
    other = azure_from_env(env(other_settings), transport=transport)
    other._clock = lambda: NOW
    first = parse_qs(urlsplit(storage.signed_url(KEY, 900)).query)["sig"][0]
    second = parse_qs(urlsplit(other.signed_url(KEY, 900)).query)["sig"][0]
    assert first != second
    # The same key, blob and expiry give the same token: nothing random is in it.
    assert parse_qs(urlsplit(storage.signed_url(KEY, 900)).query)["sig"][0] == first


def test_factory_reports_what_is_missing_without_showing_the_key() -> None:
    with pytest.raises(StorageConfigError, match="AZURE_STORAGE_CONNECTION_STRING and AZURE_"):
        azure_from_env(env({}))
    with pytest.raises(StorageConfigError, match="AZURE_STORAGE_CONTAINER"):
        azure_from_env(env({"AZURE_STORAGE_CONNECTION_STRING": azure_fake.CONNECTION_STRING}))
    with pytest.raises(StorageConfigError, match="not a valid connection string") as caught:
        azure_from_env(
            env({**SETTINGS, "AZURE_STORAGE_CONNECTION_STRING": "AccountKey=secret-thing"})
        )
    assert "secret-thing" not in str(caught.value)
    no_key = (
        "BlobEndpoint=https://devaccount.blob.core.windows.net/;"
        "SharedAccessSignature=sv=2026-01-01&sig=abc"
    )
    with pytest.raises(StorageConfigError, match="must contain AccountKey"):
        azure_from_env(env({**SETTINGS, "AZURE_STORAGE_CONNECTION_STRING": no_key}))


def test_storage_from_env_selects_azure_and_hides_the_key() -> None:
    storage = storage_from_env(env({**SETTINGS, "STEMS_STORAGE_BACKEND": "azure_blob"}))
    assert isinstance(storage, AzureBlobStorage)
    assert repr(storage) == "AzureBlobStorage(container='stems-test')"
    assert azure_fake.ACCOUNT_KEY not in repr(storage)
