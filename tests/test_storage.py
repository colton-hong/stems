"""The storage contract every backend must meet.

Each test runs once per backend named in ``BACKENDS``; a new backend is added by teaching
``make_storage`` to build it. Tests of one backend's own behaviour (the local backend's
signed links) live in their own files.
"""

from __future__ import annotations

import io

import pytest
import storage_fakes

from stems_api.storage import (
    InvalidKey,
    LocalStorage,
    ObjectNotFound,
    Storage,
    put_bytes,
    put_json,
    read_bytes,
    read_json,
)
from stems_api.storage.s3 import S3Storage

SECRET = b"0123456789abcdef0123456789abcdef-test-only"
BACKENDS = ["local", "s3", "azure_blob"]

BAD_KEYS = [
    "",
    "/etc/passwd",
    "../outside",
    "jobs/../../outside",
    "jobs/./a",
    "jobs//a",
    "jobs/a/",
    "jobs/.hidden",
    "jobs\\a",
    "jobs/a b",
    "jobs/a\x00b",
    "jobs/é",
    "~/file",
    "a" * 513,
]


def make_storage(backend: str, tmp_path) -> Storage:
    if backend == "local":
        return LocalStorage(tmp_path / "store", SECRET)
    if backend == "s3":
        # A small page size so that listing a job's folder takes more than one page.
        return S3Storage(storage_fakes.FakeS3Client(page_size=3), "stems-test")
    if backend == "azure_blob":
        # The real SDK client, answered from memory (tests/azure_fake.py).
        pytest.importorskip("azure.storage.blob")
        import azure_fake

        from stems_api.storage.azure_blob import azure_from_env

        settings = {
            "AZURE_STORAGE_CONNECTION_STRING": azure_fake.CONNECTION_STRING,
            "AZURE_STORAGE_CONTAINER": "stems-test",
        }
        return azure_from_env(settings.get, transport=azure_fake.FakeBlobTransport(), retry_total=0)
    raise AssertionError(f"no test double for the {backend!r} backend")


@pytest.fixture(params=BACKENDS)
def storage(request, tmp_path) -> Storage:
    return make_storage(request.param, tmp_path)


def test_every_backend_implements_the_protocol(storage) -> None:
    assert isinstance(storage, Storage)


def test_put_then_open_returns_the_same_bytes(storage) -> None:
    data = bytes(range(256)) * 5000  # 1.28 MB: more than one copy chunk
    storage.put("jobs/abc/source", io.BytesIO(data), "audio/wav")
    assert storage.exists("jobs/abc/source")
    with storage.open("jobs/abc/source") as handle:
        assert handle.read() == data


def test_put_replaces_what_was_there(storage) -> None:
    put_bytes(storage, "jobs/abc/status.json", b"first", "application/json")
    put_bytes(storage, "jobs/abc/status.json", b"second, and longer", "application/json")
    assert read_bytes(storage, "jobs/abc/status.json") == b"second, and longer"


def test_json_helpers_round_trip(storage) -> None:
    value = {"stage": "separating", "stems": ["vocals", "drums"], "progress": 0.5, "né": None}
    put_json(storage, "jobs/abc/status.json", value)
    assert read_json(storage, "jobs/abc/status.json") == value


def test_missing_object(storage) -> None:
    assert not storage.exists("jobs/abc/nothing")
    with pytest.raises(ObjectNotFound):
        storage.open("jobs/abc/nothing")
    # A folder of objects is not itself an object.
    put_bytes(storage, "jobs/abc/stems/vocals.wav", b"x", "audio/wav")
    assert not storage.exists("jobs/abc/stems")
    with pytest.raises(ObjectNotFound):
        storage.open("jobs/abc/stems")


@pytest.mark.parametrize("key", BAD_KEYS)
def test_bad_keys_are_refused_by_every_method(storage, key) -> None:
    with pytest.raises(InvalidKey):
        storage.put(key, io.BytesIO(b"x"), "text/plain")
    with pytest.raises(InvalidKey):
        storage.open(key)
    with pytest.raises(InvalidKey):
        storage.exists(key)
    with pytest.raises(InvalidKey):
        storage.signed_url(key, 60)


@pytest.mark.parametrize("prefix", ["", "/", "..", "../", "jobs/../..", "/jobs", "jobs//"])
def test_bad_prefixes_are_refused(storage, prefix) -> None:
    put_bytes(storage, "jobs/abc/source", b"x", "audio/wav")
    with pytest.raises(InvalidKey):
        storage.delete_prefix(prefix)
    assert storage.exists("jobs/abc/source")


def test_delete_prefix_removes_the_folder_and_nothing_else(storage) -> None:
    for key in (
        "jobs/abc/source",
        "jobs/abc/spec.json",
        "jobs/abc/stems/vocals.wav",
        "jobs/abc/stems/drums.wav",
        "jobs/abcd/source",  # shares the characters "jobs/abc" but is another job
        "jobs/ab/source",
    ):
        put_bytes(storage, key, b"x", "application/octet-stream")
    assert storage.delete_prefix("jobs/abc/") == 4
    for gone in ("jobs/abc/source", "jobs/abc/spec.json", "jobs/abc/stems/vocals.wav"):
        assert not storage.exists(gone)
    assert storage.exists("jobs/abcd/source")
    assert storage.exists("jobs/ab/source")
    # Repeating it deletes nothing and is not an error.
    assert storage.delete_prefix("jobs/abc/") == 0
    assert storage.delete_prefix("jobs/abc") == 0
    assert storage.delete_prefix("jobs/never-existed") == 0


def test_delete_prefix_without_the_slash_and_on_one_object(storage) -> None:
    put_bytes(storage, "jobs/abc/stems/vocals.wav", b"x", "audio/wav")
    put_bytes(storage, "jobs/abc/stems/drums.wav", b"x", "audio/wav")
    assert storage.delete_prefix("jobs/abc/stems/vocals.wav") == 1
    assert storage.exists("jobs/abc/stems/drums.wav")
    assert storage.delete_prefix("jobs/abc") == 1
    assert not storage.exists("jobs/abc/stems/drums.wav")
    # The key can be used again afterwards.
    put_bytes(storage, "jobs/abc/stems/drums.wav", b"again", "audio/wav")
    assert read_bytes(storage, "jobs/abc/stems/drums.wav") == b"again"


def test_signed_url_is_a_string_and_checks_its_arguments(storage) -> None:
    put_bytes(storage, "jobs/abc/stems/vocals.wav", b"x", "audio/wav")
    url = storage.signed_url("jobs/abc/stems/vocals.wav", 900, filename="My Song (vocals).wav")
    assert isinstance(url, str)
    assert url
    for bad in (0, -5, 1.5, True, None, "60"):
        with pytest.raises(ValueError, match="ttl_seconds"):
            storage.signed_url("jobs/abc/stems/vocals.wav", bad)
