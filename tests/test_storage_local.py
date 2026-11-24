"""The local backend's own behaviour: signed links, the storage root, the secret."""

from __future__ import annotations

import io
import os
import stat

import pytest

from stems_api.storage import (
    InvalidKey,
    LinkExpired,
    LinkInvalid,
    LocalStorage,
    ObjectNotFound,
    StorageConfigError,
    StorageError,
    put_bytes,
    read_bytes,
    storage_from_env,
)
from stems_api.storage import factory as storage_factory
from stems_api.storage import local as local_module

SECRET = b"0123456789abcdef0123456789abcdef-test-only"
KEY = "jobs/abc/stems/vocals.wav"


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def storage(tmp_path, clock) -> LocalStorage:
    store = LocalStorage(tmp_path / "store", SECRET, clock=clock)
    put_bytes(store, KEY, b"RIFF....", "audio/wav")
    return store


def token_of(url: str) -> str:
    prefix = "/api/files/"
    assert url.startswith(prefix)
    return url[len(prefix) :]


# -- signed links ------------------------------------------------------------------------


def test_a_fresh_link_verifies_and_carries_key_expiry_and_name(storage, clock) -> None:
    token = token_of(storage.signed_url(KEY, 900, filename="My Song (vocals).wav"))
    link = storage.verify(token)
    assert link.key == KEY
    assert link.expires_at == int(clock.now) + 900
    assert link.filename == "My Song (vocals).wav"
    # Only URL-safe characters, so the token can sit in a path unescaped.
    assert set(token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.")


def test_a_link_is_rejected_from_its_expiry_onwards(storage, clock) -> None:
    token = token_of(storage.signed_url(KEY, 900))
    clock.now += 899
    assert storage.verify(token).key == KEY
    clock.now += 1  # exactly at the expiry
    with pytest.raises(LinkExpired):
        storage.verify(token)
    clock.now += 86_400
    with pytest.raises(LinkExpired):
        storage.verify(token)


def test_changing_any_character_of_a_link_is_rejected(storage) -> None:
    token = token_of(storage.signed_url(KEY, 900, filename="song.wav"))
    for index, original in enumerate(token):
        if original == ".":
            continue
        replacement = "A" if original != "A" else "B"
        tampered = token[:index] + replacement + token[index + 1 :]
        with pytest.raises(LinkInvalid):
            storage.verify(tampered)


def test_a_token_has_exactly_one_spelling(storage) -> None:
    """Base64's last character can hold bits that decode to nothing. A token respelled
    that way carries the same signature bytes, and is refused all the same."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    token = token_of(storage.signed_url(KEY, 900, filename="song.wav"))
    body, _, signature = token.partition(".")
    accepted = []
    for part, rebuild in (
        (signature, lambda s: f"{body}.{s}"),
        (body, lambda b: f"{b}.{signature}"),
    ):
        for letter in alphabet:
            candidate = rebuild(part[:-1] + letter)
            try:
                storage.verify(candidate)
            except LinkInvalid:
                continue
            accepted.append(candidate)
    assert accepted == [token, token]  # only the original, once per half
    for padded in (token + "=", f"{body}=.{signature}", f"{body}.{signature}=="):
        with pytest.raises(LinkInvalid):
            storage.verify(padded)


def test_a_link_cannot_be_repointed_or_extended(storage, clock) -> None:
    """Re-encoding the payload with another key or a later expiry breaks the signature."""
    put_bytes(storage, "jobs/other/source", b"someone else's", "audio/wav")
    token = token_of(storage.signed_url(KEY, 900))
    _, _, signature = token.partition(".")
    for key, expires in (
        ("jobs/other/source", int(clock.now) + 900),
        (KEY, int(clock.now) + 10**9),
    ):
        forged_body = local_module.sign(b"x" * 32, key, expires).partition(".")[0]
        with pytest.raises(LinkInvalid):
            storage.verify(f"{forged_body}.{signature}")
    # An expired link stays expired however it is re-labelled.
    clock.now += 901
    with pytest.raises(LinkExpired):
        storage.verify(token)


def test_a_link_signed_with_another_secret_is_rejected(storage, tmp_path, clock) -> None:
    other = LocalStorage(tmp_path / "store", b"another-secret-of-at-least-32-bytes!", clock=clock)
    with pytest.raises(LinkInvalid):
        storage.verify(token_of(other.signed_url(KEY, 900)))


@pytest.mark.parametrize(
    "token",
    [
        "",
        ".",
        "abc",
        "abc.def",
        "abc.def.ghi",
        "....",
        "é.é",
        "e30.e30",
        "x" * 5000,
        "!!!!.????",
    ],
)
def test_malformed_tokens_are_rejected(storage, token) -> None:
    with pytest.raises(LinkInvalid):
        storage.verify(token)


def test_a_correctly_signed_token_with_a_bad_payload_is_still_rejected(storage, clock) -> None:
    """Defence in depth: even with the secret, a token cannot name a path outside the root."""
    import base64
    import hashlib
    import hmac
    import json

    def forge(payload: object) -> str:
        raw = json.dumps(payload).encode()
        body = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
        mac = hmac.new(SECRET, body.encode(), hashlib.sha256).digest()
        return f"{body}.{base64.urlsafe_b64encode(mac).rstrip(b'=').decode()}"

    later = int(clock.now) + 900
    assert storage.verify(forge({"k": KEY, "e": later, "f": None})).key == KEY
    for payload in (
        {"k": "../../etc/passwd", "e": later, "f": None},
        {"k": "/etc/passwd", "e": later, "f": None},
        {"k": KEY, "e": "tomorrow", "f": None},
        {"k": KEY, "e": True, "f": None},
        {"k": KEY, "e": later + 0.5, "f": None},
        {"k": KEY, "e": later, "f": 7},
        {"k": KEY, "e": later},
        {"e": later, "f": None},
        [KEY, later],
        "text",
    ):
        with pytest.raises(LinkInvalid):
            storage.verify(forge(payload))


def test_download_names_are_cleaned(storage) -> None:
    cases = {
        '../../etc/pass"wd\r\nX-Injected: 1': "etcpasswdX-Injected: 1",
        "..hidden": "hidden",
        "  song.wav  ": "song.wav",
        "song\x00.wav": "song.wav",
        "C:\\Users\\me\\song.wav": "C:Usersmesong.wav",
        "": None,
        "///": None,
        "é" * 400: "é" * 150,
    }
    for given, expected in cases.items():
        link = storage.verify(token_of(storage.signed_url(KEY, 60, filename=given)))
        assert link.filename == expected


def test_base_url_is_configurable(tmp_path) -> None:
    store = LocalStorage(tmp_path / "s", SECRET, base_url="https://example.invalid/api/files/")
    assert store.signed_url(KEY, 60).startswith("https://example.invalid/api/files/")
    assert "//" not in store.signed_url(KEY, 60).removeprefix("https://")


# -- the storage root --------------------------------------------------------------------


def test_nothing_is_written_outside_the_root(tmp_path) -> None:
    root = tmp_path / "store"
    store = LocalStorage(root, SECRET)
    for key in ("../escape", "jobs/../../escape", "/tmp/escape", "jobs/..", ".."):
        with pytest.raises(InvalidKey):
            put_bytes(store, key, b"x", "text/plain")
    put_bytes(store, "jobs/abc/source", b"x", "audio/wav")
    outside = [p for p in tmp_path.rglob("*") if p.is_file() and root not in p.parents]
    assert outside == []


def test_a_symlink_inside_the_root_cannot_lead_out(tmp_path) -> None:
    root = tmp_path / "store"
    secret_dir = tmp_path / "elsewhere"
    secret_dir.mkdir()
    (secret_dir / "private.txt").write_text("not yours")
    store = LocalStorage(root, SECRET)
    (root / "objects" / "jobs").mkdir()
    (root / "objects" / "jobs" / "link").symlink_to(secret_dir, target_is_directory=True)
    with pytest.raises(InvalidKey):
        store.open("jobs/link/private.txt")
    with pytest.raises(InvalidKey):
        store.exists("jobs/link/private.txt")
    with pytest.raises(InvalidKey):
        put_bytes(store, "jobs/link/new.txt", b"x", "text/plain")
    with pytest.raises(InvalidKey):
        store.delete_prefix("jobs/link/private.txt")
    assert (secret_dir / "private.txt").read_text() == "not yours"
    assert not (secret_dir / "new.txt").exists()


def test_an_object_cannot_also_be_a_folder(storage) -> None:
    with pytest.raises(StorageError):
        put_bytes(storage, "jobs/abc/stems", b"x", "text/plain")
    with pytest.raises(StorageError):
        put_bytes(storage, f"{KEY}/inner", b"x", "text/plain")
    assert read_bytes(storage, KEY) == b"RIFF...."


def test_a_failed_put_leaves_the_old_object_and_no_temporary_file(storage) -> None:
    class Breaks(io.RawIOBase):
        def __init__(self) -> None:
            self.sent = False

        def readable(self) -> bool:
            return True

        def read(self, size: int = -1) -> bytes:
            if not self.sent:
                self.sent = True
                return b"partial"
            raise OSError("connection lost")

    with pytest.raises(OSError, match="connection lost"):
        storage.put(KEY, Breaks(), "audio/wav")
    assert read_bytes(storage, KEY) == b"RIFF...."
    leftovers = [p.name for p in storage.root.rglob("*") if p.name.startswith(".tmp-")]
    assert leftovers == []


def test_stat_gives_size_content_type_and_path(storage) -> None:
    info = storage.stat(KEY)
    assert (info.key, info.size, info.content_type) == (KEY, 8, "audio/wav")
    assert info.path.read_bytes() == b"RIFF...."
    with pytest.raises(ObjectNotFound):
        storage.stat("jobs/abc/stems/missing.wav")
    with pytest.raises(ObjectNotFound):
        storage.stat("jobs/abc/stems")


def test_delete_prefix_removes_metadata_and_empty_folders(storage) -> None:
    assert storage.delete_prefix("jobs/abc") == 1
    remaining = sorted(p.relative_to(storage.root).as_posix() for p in storage.root.rglob("*"))
    assert remaining == ["meta", "objects"]


def test_a_short_secret_is_refused_and_never_shown(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        LocalStorage(tmp_path / "s", b"too short")
    store = LocalStorage(tmp_path / "s", SECRET)
    assert SECRET.decode() not in repr(store)
    assert SECRET.decode() not in str(vars(store).keys())


# -- choosing the backend ----------------------------------------------------------------


def env(values: dict[str, str]):
    return lambda name: values.get(name) or None


def test_local_is_the_default_and_makes_its_secret_once(tmp_path) -> None:
    root = tmp_path / "storage"
    getenv = env({"STEMS_STORAGE_LOCAL_DIR": str(root)})
    first = storage_from_env(getenv)
    assert isinstance(first, LocalStorage)
    assert first.root == root
    secret_file = root / "signing-secret"
    assert stat.S_IMODE(os.stat(secret_file).st_mode) == 0o600
    made = secret_file.read_text()
    assert len(made) >= 48
    # A second process reads the same secret, so its links verify here and the reverse.
    second = storage_from_env(getenv)
    assert secret_file.read_text() == made
    put_bytes(first, KEY, b"x", "audio/wav")
    assert second.verify(token_of(first.signed_url(KEY, 60))).key == KEY
    # The secret file is not reachable as an object.
    assert not first.exists("signing-secret")


def test_the_secret_comes_from_the_environment_when_set(tmp_path) -> None:
    root = tmp_path / "storage"
    getenv = env({"STEMS_STORAGE_LOCAL_DIR": str(root), "STEMS_SIGNING_SECRET": SECRET.decode()})
    store = storage_from_env(getenv)
    assert not (root / "signing-secret").exists()
    same = LocalStorage(tmp_path / "other", SECRET)
    assert same.verify(token_of(store.signed_url(KEY, 60))).key == KEY
    with pytest.raises(StorageConfigError, match="STEMS_SIGNING_SECRET"):
        storage_from_env(
            env({"STEMS_STORAGE_LOCAL_DIR": str(root), "STEMS_SIGNING_SECRET": "short"})
        )


def test_a_damaged_secret_file_is_reported(tmp_path) -> None:
    root = tmp_path / "storage"
    root.mkdir()
    (root / "signing-secret").write_text("")
    with pytest.raises(StorageConfigError, match="damaged"):
        storage_from_env(env({"STEMS_STORAGE_LOCAL_DIR": str(root)}))


def test_unknown_backend_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(StorageConfigError, match="not a storage backend"):
        storage_from_env(env({"STEMS_STORAGE_BACKEND": "ftp"}))


def test_default_directory_is_under_runs() -> None:
    assert storage_factory.DEFAULT_LOCAL_DIR.parts[-3:] == ("runs", "dev", "storage")
