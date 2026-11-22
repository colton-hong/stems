"""The download manager resumes, skips, verifies and pins, tested against a local server."""

from __future__ import annotations

import hashlib
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stems_models import download
from stems_models.download import (
    AlreadyRunningError,
    ChecksumLock,
    ChecksumMismatchError,
    DownloadError,
    SizeMismatchError,
    ensure_model,
    fetch_file,
    read_status,
)
from stems_models.registry import ModelEntry, ModelFile, Registry

PAYLOAD = bytes(range(256)) * 2000  # 512,000 bytes, not a multiple of the read block


class _QuickServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # The stock server_bind looks up the host's fully qualified name, which can take
        # tens of seconds on a machine with slow reverse DNS. Tests do not need it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class _Server:
    """A small HTTP server with range support and switchable misbehaviour."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str | None]] = []  # (path, Range header)
        self.ignore_range = False
        self.drop_after: int | None = None  # send this many body bytes once, then hang up
        self.html = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep test output quiet
                pass

            def do_GET(self) -> None:
                outer.requests.append((self.path, self.headers.get("Range")))
                body = outer.files.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                start = 0
                range_header = self.headers.get("Range")
                if range_header and not outer.ignore_range:
                    start = int(range_header.removeprefix("bytes=").split("-")[0])
                    if start >= len(body):
                        self.send_error(416)
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                chunk = body[start:]
                content_type = "text/html" if outer.html else "application/octet-stream"
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                if outer.drop_after is not None:
                    cut, outer.drop_after = outer.drop_after, None
                    self.wfile.write(chunk[:cut])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(chunk)

        self.httpd = _QuickServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = _Server()
    srv.files["/weights.ckpt"] = PAYLOAD
    yield srv
    srv.close()


def _file(server: _Server, name: str = "weights.ckpt", size: int | None = len(PAYLOAD), **kw):
    # model_construct skips validation, which would reject a plain-http test URL.
    return ModelFile.model_construct(
        role="weights", filename=name, url=f"{server.base}/{name}", size_bytes=size, stem=None, **kw
    )


def _fetch(file: ModelFile, tmp_path, **kw):
    lock = ChecksumLock(tmp_path / "checksums.lock.json")
    return fetch_file(file, tmp_path / "weights" / "m", lock=lock, retry_wait=0, **kw), lock


# --- acceptance ----------------------------------------------------------------------------


def test_first_download_writes_the_file_and_pins_its_checksum(server, tmp_path) -> None:
    (path, outcome), lock = _fetch(_file(server), tmp_path)
    assert outcome == "downloaded"
    assert path.read_bytes() == PAYLOAD
    assert not path.with_name(path.name + ".part").exists()

    pinned = json.loads(lock.path.read_text())
    assert pinned[f"{server.base}/weights.ckpt"] == {
        "filename": "weights.ckpt",
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "size_bytes": len(PAYLOAD),
    }


def test_a_complete_file_is_skipped_without_touching_the_network(server, tmp_path) -> None:
    _fetch(_file(server), tmp_path)
    before = len(server.requests)
    (path, outcome), _ = _fetch(_file(server), tmp_path)
    assert outcome == "present"
    assert len(server.requests) == before
    assert path.read_bytes() == PAYLOAD


def test_a_partial_file_is_resumed_from_where_it_stopped(server, tmp_path) -> None:
    target = tmp_path / "weights" / "m"
    target.mkdir(parents=True)
    (target / "weights.ckpt.part").write_bytes(PAYLOAD[:100_000])

    (path, outcome), _ = _fetch(_file(server), tmp_path)
    assert outcome == "downloaded"
    assert path.read_bytes() == PAYLOAD
    assert server.requests == [("/weights.ckpt", "bytes=100000-")]


def test_a_dropped_connection_is_resumed_within_one_call(server, tmp_path) -> None:
    server.drop_after = 70_000
    (path, _), _ = _fetch(_file(server), tmp_path)
    assert path.read_bytes() == PAYLOAD
    assert len(server.requests) == 2
    resumed_from = int(server.requests[1][1].removeprefix("bytes=").rstrip("-"))
    assert 0 < resumed_from <= 70_000


def test_a_server_that_ignores_ranges_does_not_corrupt_the_file(server, tmp_path) -> None:
    server.ignore_range = True
    target = tmp_path / "weights" / "m"
    target.mkdir(parents=True)
    (target / "weights.ckpt.part").write_bytes(PAYLOAD[:100_000])
    (path, _), _ = _fetch(_file(server), tmp_path)
    assert path.read_bytes() == PAYLOAD


@pytest.mark.parametrize("claimed", [len(PAYLOAD) - 1, len(PAYLOAD) + 1])
def test_size_is_verified_against_the_registry(server, tmp_path, claimed) -> None:
    with pytest.raises(SizeMismatchError, match="registry records"):
        _fetch(_file(server, size=claimed), tmp_path)
    assert list((tmp_path / "weights" / "m").iterdir()) == []


def test_unknown_size_is_accepted_and_recorded(server, tmp_path) -> None:
    (path, _), lock = _fetch(_file(server, size=None), tmp_path)
    assert path.read_bytes() == PAYLOAD
    assert lock.get(f"{server.base}/weights.ckpt")["size_bytes"] == len(PAYLOAD)


def test_a_later_download_with_different_content_is_refused(server, tmp_path) -> None:
    (path, _), lock = _fetch(_file(server), tmp_path)
    path.unlink()
    server.files["/weights.ckpt"] = PAYLOAD[::-1]  # same size, different bytes

    with pytest.raises(ChecksumMismatchError, match="does not match the pinned"):
        fetch_file(_file(server), path.parent, lock=lock, retry_wait=0)
    assert list(path.parent.iterdir()) == []


def test_a_file_changed_on_disk_is_refused(server, tmp_path) -> None:
    (path, _), lock = _fetch(_file(server), tmp_path)
    path.write_bytes(PAYLOAD[:-1] + b"\x00")
    with pytest.raises(ChecksumMismatchError):
        fetch_file(_file(server), path.parent, lock=lock, retry_wait=0)


# --- failures ------------------------------------------------------------------------------


def test_a_missing_file_fails_at_once_without_retrying(server, tmp_path) -> None:
    with pytest.raises(DownloadError, match="HTTP 404"):
        _fetch(_file(server, name="absent.ckpt"), tmp_path)
    assert len(server.requests) == 1


def test_an_html_page_is_not_accepted_as_weights(server, tmp_path) -> None:
    server.html = True
    with pytest.raises(DownloadError, match="HTML page"):
        _fetch(_file(server), tmp_path)
    assert not (tmp_path / "weights" / "m" / "weights.ckpt").exists()


def test_an_unreachable_server_gives_up_after_the_attempt_limit(tmp_path) -> None:
    file = ModelFile.model_construct(
        role="weights", filename="w.ckpt", url="http://127.0.0.1:9/w.ckpt", size_bytes=10, stem=None
    )
    lock = ChecksumLock(tmp_path / "lock.json")
    with pytest.raises(DownloadError, match="gave up after 2 attempts"):
        fetch_file(file, tmp_path / "m", lock=lock, retry_wait=0, max_attempts=2, timeout=2)


# --- whole models, status files, locking ---------------------------------------------------


def _registry(server: _Server) -> Registry:
    server.files["/bass.th"] = b"bass" * 1000
    server.files["/drums.th"] = b"drums" * 1000
    server.files["/config.yaml"] = b"chunk_size: 1\n"

    def file(name: str, role: str, stem: str | None) -> ModelFile:
        return ModelFile.model_construct(
            role=role,
            filename=name,
            url=f"{server.base}/{name}",
            size_bytes=len(server.files[f"/{name}"]),
            stem=stem,
        )

    entry = ModelEntry.model_construct(
        id="fake_bag",
        name="Fake bag",
        kind="separator",
        architecture="fake",
        loader="demucs",
        stems=["bass", "drums"],
        candidate_for=["bass", "drums"],
        tier="candidate",
        origin="added",
        status="available",
        sample_rate_hz=44100,
        source_page="https://example.invalid",
        files=[
            file("bass.th", "weights", "bass"),
            file("drums.th", "weights", "drums"),
            file("config.yaml", "config", None),
        ],
        config={},
        licence_note="test",
        notes=None,
        unverified=[],
    )
    hosted = ModelEntry.model_construct(
        id="fake_hosted", status="hosted", files=[], tier="candidate", stems=[], candidate_for=[]
    )
    return Registry.model_construct(schema_version=1, models=[entry, hosted])


def _ensure(server, tmp_path, **kw):
    return ensure_model(
        "fake_bag",
        registry=_registry(server),
        directory=tmp_path / "weights",
        lock=ChecksumLock(tmp_path / "lock.json"),
        runs_dir=tmp_path / "runs",
        retry_wait=0,
        **kw,
    )


def test_ensure_model_downloads_every_file_and_reports_status(server, tmp_path) -> None:
    paths = _ensure(server, tmp_path)
    assert set(paths) == {"bass.th", "drums.th", "config.yaml"}
    assert paths["bass.th"] == tmp_path / "weights" / "fake_bag" / "bass.th"
    assert paths["drums.th"].read_bytes() == b"drums" * 1000

    status = read_status("fake_bag", tmp_path / "runs")
    assert status["state"] == "done"
    assert status["error"] is None
    assert {f["filename"]: f["state"] for f in status["files"]} == {
        "bass.th": "downloaded",
        "drums.th": "downloaded",
        "config.yaml": "downloaded",
    }
    assert all(f["bytes_done"] == f["size_bytes"] for f in status["files"])


def test_ensure_model_is_idempotent(server, tmp_path) -> None:
    _ensure(server, tmp_path)
    before = len(server.requests)
    _ensure(server, tmp_path)
    assert len(server.requests) == before
    states = {f["state"] for f in read_status("fake_bag", tmp_path / "runs")["files"]}
    assert states == {"present"}


def test_a_single_stem_request_fetches_only_that_stems_files(server, tmp_path) -> None:
    paths = _ensure(server, tmp_path, stem="bass")
    assert set(paths) == {"bass.th", "config.yaml"}
    assert not (tmp_path / "weights" / "fake_bag" / "drums.th").exists()


def test_a_failure_is_written_to_the_status_file(server, tmp_path) -> None:
    del server.files["/weights.ckpt"]
    registry = _registry(server)
    del server.files["/drums.th"]
    with pytest.raises(DownloadError):
        ensure_model(
            "fake_bag",
            registry=registry,
            directory=tmp_path / "weights",
            lock=ChecksumLock(tmp_path / "lock.json"),
            runs_dir=tmp_path / "runs",
            retry_wait=0,
        )
    status = read_status("fake_bag", tmp_path / "runs")
    assert status["state"] == "failed"
    assert "HTTP 404" in status["error"]
    assert status["files"][0]["state"] == "downloaded"  # bass finished before drums failed


def test_models_without_weights_cannot_be_downloaded(server, tmp_path) -> None:
    with pytest.raises(DownloadError, match="nothing to download"):
        ensure_model("fake_hosted", registry=_registry(server), directory=tmp_path / "w")


def test_two_processes_cannot_download_the_same_model_at_once(server, tmp_path) -> None:
    held = download._model_lock(tmp_path / "weights" / "fake_bag")
    with held, pytest.raises(AlreadyRunningError):
        _ensure(server, tmp_path)
    assert set(_ensure(server, tmp_path)) == {"bass.th", "drums.th", "config.yaml"}


# --- command line --------------------------------------------------------------------------


@pytest.fixture
def cli(server, tmp_path, monkeypatch):
    """Point the command line at the test server and at temporary directories."""
    registry = _registry(server)
    monkeypatch.setattr(download, "load_registry", lambda: registry)
    monkeypatch.setattr(download, "DOWNLOAD_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(download, "LOCK_PATH", tmp_path / "lock.json")
    return ["--dir", str(tmp_path / "weights")]


def test_cli_downloads_by_id(cli, tmp_path, capsys) -> None:
    assert download.main(["fake_bag", *cli]) == 0
    assert (tmp_path / "weights" / "fake_bag" / "bass.th").exists()
    assert "ok      fake_bag" in capsys.readouterr().out
    assert read_status("fake_bag", tmp_path / "runs")["state"] == "done"


def test_cli_reports_failure_with_a_nonzero_exit(cli, server, capsys) -> None:
    del server.files["/bass.th"]
    assert download.main(["fake_bag", *cli]) == 1
    assert "FAILED  fake_bag" in capsys.readouterr().out


def test_cli_rejects_an_unknown_id(cli, capsys) -> None:
    assert download.main(["nope", *cli]) == 2
    assert "unknown model id" in capsys.readouterr().err


def test_background_writes_status_files_and_detaches(cli, tmp_path, monkeypatch, capsys) -> None:
    spawned: list[dict] = []

    class FakeProcess:
        pid = 4242

    def fake_popen(command, **kwargs):
        spawned.append({"command": command, **kwargs})
        return FakeProcess()

    monkeypatch.setattr(download.subprocess, "Popen", fake_popen)
    assert download.main(["fake_bag", "--background", *cli]) == 0

    status = read_status("fake_bag", tmp_path / "runs")
    assert status["state"] == "queued"
    assert [f["state"] for f in status["files"]] == ["pending"] * 3

    (call,) = spawned
    assert call["command"][1:4] == ["-m", "stems_models.download", "fake_bag"]
    assert "--background" not in call["command"]
    assert call["start_new_session"] is True
    assert (tmp_path / "runs" / "background.log").exists()
    assert "started pid 4242" in capsys.readouterr().out


def test_status_command_lists_progress(cli, capsys) -> None:
    download.main(["--status", *cli])
    assert "fake_bag" in capsys.readouterr().out
    download.main(["fake_bag", *cli])
    capsys.readouterr()
    download.main(["--status", "fake_bag", *cli])
    assert "done" in capsys.readouterr().out
