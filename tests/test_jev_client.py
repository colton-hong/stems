"""The Jev client: pinned model, retries, timeouts, a complete log, and no key in it."""

from __future__ import annotations

import json
import logging
import socket
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stems_common.settings import load_defaults
from stems_worker.decision import jev
from stems_worker.decision.jev import (
    HttpTransport,
    JevClient,
    JevError,
    Reply,
    RequestRefused,
    ResponseError,
    build_request,
    choice,
    noul,
    parse_answers,
    request_key,
    retry_after_seconds,
    score,
)

SETTINGS = load_defaults().decision
MODEL = "jev-1.14.0"
KEY = "ts-live-0123456789abcdefSECRET"
QUESTIONS = {
    "wants.vocals": noul("Does the user want the vocals?", yes="They ask for it", no="They do not"),
    "format": choice("Which file format?", {"wav": "WAV", "flac": "FLAC", "unspecified": None}),
    "quality": score("How good is the stem?", ["unusable", "poor", "acceptable", "good"]),
}
ANSWERS = {
    "wants.vocals": {"type": "noul", "noul": 0.93},
    "format": {
        "type": "choice",
        "choice": "flac",
        "confidence": 0.9,
        "probabilities": {"wav": 0.06, "flac": 0.9, "unspecified": 0.04},
    },
    "quality": {
        "type": "score",
        "score": 2.6,
        "confidence": 0.7,
        "legend": {"0": "unusable", "1": "poor", "2": "acceptable", "3": "good"},
        "probabilities": {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.7},
    },
}


def ok(answers=None, model=MODEL, usage=None) -> Reply:
    payload = {
        "model": model,
        "answers": ANSWERS if answers is None else answers,
        "usage": {"input_tokens": 212, "output_tokens": 9} if usage is None else usage,
    }
    return Reply(200, payload)


class Scripted:
    """A transport that gives the next scripted reply, or raises it if it is an exception."""

    def __init__(self, *script, secrets=()) -> None:
        self.script = list(script)
        self.secrets = tuple(secrets)
        self.bodies: list[dict] = []
        self.timeouts: list[float] = []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        self.timeouts.append(timeout)
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def client(transport, tmp_path=None, **changes):
    sleeps: list[float] = []
    settings = SETTINGS.model_copy(update=changes)
    made = JevClient(
        transport,
        settings,
        log_path=None if tmp_path is None else tmp_path / "jev.jsonl",
        sleep=sleeps.append,
    )
    return made, sleeps


# --- the request --------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ["jev-latest", "jev-preview", "jev", "jev-1.14", "jev-1.14.0b"])
def test_an_alias_is_refused_before_anything_is_sent(alias: str) -> None:
    with pytest.raises(RequestRefused, match="not a pinned version"):
        build_request(alias, "state", QUESTIONS)
    transport = Scripted(ok())
    # Settings validation refuses an alias; a settings object built around it is refused too.
    with pytest.raises(RequestRefused, match="not a pinned version"):
        JevClient(
            transport, SETTINGS.model_construct(**{**SETTINGS.model_dump(), "jev_model": alias})
        )
    assert transport.bodies == []


def test_the_request_body_is_the_documented_shape() -> None:
    body = build_request(MODEL, {"text": "vocals please"}, QUESTIONS)
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == MODEL
    assert body["questions"]["wants.vocals"] == {
        "type": "noul",
        "instructions": "Does the user want the vocals?",
        "criteria": {"true": "They ask for it", "false": "They do not"},
    }
    assert body["questions"]["format"]["criteria"] == {
        "wav": "WAV",
        "flac": "FLAC",
        "unspecified": None,
    }
    assert body["questions"]["quality"]["criteria"] == ["unusable", "poor", "acceptable", "good"]
    assert noul("Is it?") == {"type": "noul", "instructions": "Is it?"}
    assert request_key(body) == request_key(json.loads(json.dumps(body)))
    assert request_key(body) != request_key(build_request(MODEL, "other", QUESTIONS))


@pytest.mark.parametrize(
    ("questions", "state", "message"),
    [
        ({}, "s", "at least one question"),
        (QUESTIONS, "", "needs a state"),
        ({"bad id": noul("x")}, "s", "question id"),
        ({"q": {"type": "noul"}}, "s", "no instructions"),
        ({"q": choice("x", {"only": None})}, "s", "2 to 255 options"),
        ({"q": score("x", ["one"])}, "s", "2 to 10 levels"),
        ({"q": {"type": "box", "instructions": "x"}}, "s", "unknown type"),
    ],
)
def test_a_malformed_request_is_refused(questions, state, message) -> None:
    with pytest.raises(RequestRefused, match=message):
        build_request(MODEL, state, questions)


# --- the answers --------------------------------------------------------------------------


def test_answers_are_checked_and_noul_confidence_is_derived() -> None:
    parsed = parse_answers(ok().payload, QUESTIONS, MODEL)
    assert parsed["model"] == MODEL
    assert parsed["usage"] == {"input_tokens": 212, "output_tokens": 9}
    # A noul answer has no confidence field: it is max(p, 1 - p).
    assert parsed["answers"]["wants.vocals"] == {
        "type": "noul",
        "probability": 0.93,
        "confidence": 0.93,
    }
    assert parsed["answers"]["format"]["choice"] == "flac"
    assert parsed["answers"]["format"]["confidence"] == 0.9
    quality = parsed["answers"]["quality"]
    assert quality["score"] == 2.6  # fractional: the probability-weighted average
    assert quality["top_level"] == 3
    assert quality["probabilities"] == {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.7}
    assert "legend" not in quality


@pytest.mark.parametrize(
    ("p", "confidence"), [(0.0, 1.0), (0.1, 0.9), (0.5, 0.5), (0.8, 0.8), (1.0, 1.0)]
)
def test_noul_confidence(p: float, confidence: float) -> None:
    assert jev.noul_confidence(p) == pytest.approx(confidence)
    answers = {**ANSWERS, "wants.vocals": {"type": "noul", "noul": p}}
    parsed = parse_answers(ok(answers).payload, QUESTIONS, MODEL)["answers"]["wants.vocals"]
    assert parsed["confidence"] == pytest.approx(confidence)


def _changed(qid: str, **fields) -> dict:
    return {**ANSWERS, qid: {**ANSWERS[qid], **fields}}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("<html>bad gateway</html>", "no answers object"),
        ({"model": MODEL, "answers": []}, "no answers object"),
        (ok(model="jev-latest").payload, "asked jev-1.14.0, but 'jev-latest' answered"),
        (ok(model="jev-1.13.0").payload, "asked jev-1.14.0, but 'jev-1.13.0' answered"),
        (ok({k: v for k, v in ANSWERS.items() if k != "format"}).payload, "was not answered"),
        (ok(_changed("wants.vocals", noul=1.4)).payload, "not a probability"),
        (ok(_changed("wants.vocals", noul=True)).payload, "not a probability"),
        (ok(_changed("format", probabilities={"wav": 1.0})).payload, "do not cover"),
        (ok(_changed("format", choice="ogg")).payload, "is not an option"),
        (ok(_changed("format", confidence=None)).payload, "not a probability"),
        (
            ok(
                _changed("format", probabilities={"wav": 0.5, "flac": 0.9, "unspecified": 0.2})
            ).payload,
            "sum to",
        ),
        (ok(_changed("quality", score=3.5)).payload, "outside 0 to 3"),
    ],
)
def test_an_unusable_reply_is_an_error(payload, message) -> None:
    with pytest.raises(ResponseError, match=message):
        parse_answers(payload, QUESTIONS, MODEL)
    made, sleeps = client(Scripted(Reply(200, payload)))
    with pytest.raises(ResponseError, match=message):
        made.ask("gate", "state", QUESTIONS)
    assert sleeps == []  # a bad 200 is not retried
    assert made.calls[0]["ok"] is False


# --- retries ------------------------------------------------------------------------------


def test_429_and_5xx_are_retried_with_backoff_and_retry_after_is_honoured() -> None:
    transport = Scripted(
        Reply(429, {"detail": "rate limited"}, {"retry-after": "2"}),
        Reply(503, "upstream unavailable"),
        ok(),
    )
    made, sleeps = client(transport, max_attempts=4, backoff_seconds=0.5)
    result = made.ask("plan", {"a": 1}, QUESTIONS)
    # First wait: what the reply asked for. Second: the backoff for a second failure.
    assert sleeps == [2.0, 1.0]
    assert result.attempts == 3
    assert result.model == MODEL
    assert [a["status"] for a in made.calls[0]["attempts"]] == [429, 503, 200]
    assert made.calls[0]["attempts"][0]["retry_after_seconds"] == 2.0
    assert len(transport.bodies) == 3
    assert transport.bodies[0] == transport.bodies[2]


def test_backoff_doubles_and_stops_at_max_attempts() -> None:
    transport = Scripted(*[Reply(500, {"detail": "boom"})] * 4)
    made, sleeps = client(transport, max_attempts=4, backoff_seconds=0.25)
    with pytest.raises(JevError, match="HTTP 500"):
        made.ask("gate", "s", QUESTIONS)
    assert sleeps == [0.25, 0.5, 1.0]
    assert len(transport.bodies) == 4
    assert made.calls[0]["ok"] is False
    assert made.calls[0]["answers"] is None


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529, 408, 429])
def test_retryable_statuses(status: int) -> None:
    made, sleeps = client(Scripted(Reply(status, "x"), ok()))
    assert made.ask("gate", "s", QUESTIONS).attempts == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_other_failures_are_not_retried(status: int) -> None:
    transport = Scripted(Reply(status, {"detail": "no"}), ok())
    made, sleeps = client(transport)
    with pytest.raises(JevError, match=f"HTTP {status}"):
        made.ask("parse", "s", QUESTIONS)
    assert sleeps == []
    assert len(transport.bodies) == 1


def test_a_retry_after_longer_than_allowed_ends_the_call_at_once() -> None:
    transport = Scripted(Reply(429, "slow down", {"retry-after": "120"}), ok())
    made, sleeps = client(transport, max_retry_wait_seconds=30)
    with pytest.raises(JevError, match="asked to wait 120 s, more than the 30 s allowed"):
        made.ask("plan", "s", QUESTIONS)
    assert sleeps == []
    assert len(transport.bodies) == 1


def test_retry_after_accepts_seconds_or_a_date() -> None:
    now = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    assert retry_after_seconds({}) is None
    assert retry_after_seconds({"retry-after": " 7 "}) == 7.0
    assert retry_after_seconds({"retry-after": "1.5"}) == 1.5
    assert retry_after_seconds({"retry-after": "-3"}) == 0.0
    assert retry_after_seconds({"retry-after": "Sun, 22 Nov 2026 12:00:09 GMT"}, now) == 9.0
    assert retry_after_seconds({"retry-after": "Sun, 22 Nov 2026 11:00:00 GMT"}, now) == 0.0
    assert retry_after_seconds({"retry-after": "soon"}) is None


# --- timeouts -----------------------------------------------------------------------------


def test_a_timeout_is_retried_and_then_reported() -> None:
    transport = Scripted(TimeoutError("timed out"), ok())
    made, sleeps = client(transport, request_timeout_seconds=3)
    assert made.ask("gate", "s", QUESTIONS).attempts == 2
    assert transport.timeouts == [3, 3]  # every attempt is given the configured limit
    assert sleeps == [SETTINGS.backoff_seconds]
    assert made.calls[0]["attempts"][0] == {
        "status": None,
        "error": "timeout",
        "waited_seconds": SETTINGS.backoff_seconds,
    }

    never = Scripted(*[TimeoutError("timed out")] * 3)
    made, sleeps = client(never, request_timeout_seconds=3, max_attempts=3)
    with pytest.raises(JevError, match="no reply within 3 s"):
        made.ask("gate", "s", QUESTIONS)
    assert len(never.bodies) == 3


def test_a_network_error_is_retried() -> None:
    made, _ = client(Scripted(ConnectionResetError("reset by peer"), ok()))
    assert made.ask("gate", "s", QUESTIONS).attempts == 2
    assert made.calls[0]["attempts"][0]["error"] == "reset by peer"


# --- the log ------------------------------------------------------------------------------


def test_every_call_logs_model_usage_and_probabilities(tmp_path, caplog) -> None:
    made, _ = client(Scripted(ok(), Reply(402, {"detail": "balance exhausted"})), tmp_path)
    with caplog.at_level(logging.INFO, logger="stems.decision.jev"):
        result = made.ask("parse", {"text": "vocals as flac"}, QUESTIONS)
        with pytest.raises(JevError, match="HTTP 402"):
            made.ask("gate", "stem metrics", QUESTIONS)
    lines = [json.loads(x) for x in (tmp_path / "jev.jsonl").read_text().splitlines()]
    assert lines == made.calls
    assert len(lines) == 2
    first, second = lines
    assert first["kind"] == "parse"
    assert first["provider"] == "jev"
    assert first["requested_model"] == MODEL
    assert first["model"] == MODEL  # the responding model field
    assert first["usage"] == {"input_tokens": 212, "output_tokens": 9}
    assert first["request_sha256"] == result.request_sha256
    assert first["state"] == {"text": "vocals as flac"}
    assert first["questions"]["format"]["criteria"]["flac"] == "FLAC"
    assert first["answers"]["wants.vocals"]["probability"] == 0.93
    assert first["answers"]["format"]["probabilities"] == {
        "wav": 0.06,
        "flac": 0.9,
        "unspecified": 0.04,
    }
    assert first["answers"]["quality"]["probabilities"]["3"] == 0.7
    assert first["attempts"] == [{"status": 200}]
    assert first["latency_ms"] >= 0
    assert first["ok"] is True
    assert first["error"] is None
    # A failed call is logged too, with what went wrong and nothing made up.
    assert second["ok"] is False
    assert "HTTP 402" in second["error"]
    assert second["model"] is None
    assert second["usage"] is None
    assert second["answers"] is None
    assert "model=jev-1.14.0" in caplog.text
    assert "'input_tokens': 212" in caplog.text
    assert "error=HTTP 402" in caplog.text


def test_the_key_never_reaches_a_log_or_an_error(tmp_path, caplog) -> None:
    # A transport that leaks its key every way it can: in a network error, in an error
    # body echoed by the server, and in a successful reply.
    echoed = {**ANSWERS, "format": {**ANSWERS["format"]}}
    leaky = Scripted(
        OSError(f"could not send header Authorization: Bearer {KEY}"),
        Reply(200, {**ok(echoed).payload, "debug": f"token {KEY}"}),
        Reply(401, {"detail": f"invalid token {KEY}"}),
        secrets=(KEY,),
    )
    made, _ = client(leaky, tmp_path)
    with caplog.at_level(logging.DEBUG):
        made.ask("parse", f"my key is {KEY}", QUESTIONS)
        with pytest.raises(JevError) as raised:
            made.ask("parse", "state", QUESTIONS)
    logged = (tmp_path / "jev.jsonl").read_text()
    assert KEY not in logged
    assert KEY not in json.dumps(made.calls)
    assert KEY not in caplog.text
    assert KEY not in str(raised.value)
    assert "invalid token [redacted]" in str(raised.value)
    assert "Bearer [redacted]" in logged
    assert "my key is [redacted]" in logged


def test_no_key_means_no_client(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(jev.KEY_ENV, raising=False)
    monkeypatch.setattr("stems_common.env.DOTENV_PATH", tmp_path / "absent.env")
    assert HttpTransport.from_env() is None
    assert JevClient.from_env(SETTINGS) is None
    with pytest.raises(JevError, match="TYPESAFE_API_KEY is not set"):
        HttpTransport("")
    monkeypatch.setenv(jev.KEY_ENV, KEY)
    made = JevClient.from_env(SETTINGS)
    assert made is not None
    assert made.provider == "jev"
    assert made.model == MODEL
    assert KEY not in repr(made._transport)
    assert made._transport.secrets == (KEY,)


# --- the real transport, against a server on this machine ----------------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.seen.append({"auth": self.headers["Authorization"], "body": body, "path": self.path})
        status, headers, payload = server.replies.pop(0)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.seen, httpd.replies = [], []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_http_transport_sends_the_key_and_reads_status_headers_and_body(server, tmp_path) -> None:
    server.replies += [
        (429, {"Retry-After": "0"}, {"detail": "rate limited"}),
        (502, {}, b"<html>bad gateway</html>"),
        (200, {}, ok().payload),
    ]
    url = f"http://127.0.0.1:{server.server_port}/v1/systemone"
    transport = HttpTransport(KEY, endpoint=url)
    made = JevClient(transport, SETTINGS, log_path=tmp_path / "jev.jsonl", sleep=lambda s: None)
    result = made.ask("parse", {"text": "vocals"}, QUESTIONS)
    assert result.attempts == 3
    assert result.answers["format"]["choice"] == "flac"
    assert [s["auth"] for s in server.seen] == [f"Bearer {KEY}"] * 3
    assert server.seen[0]["path"] == "/v1/systemone"
    assert server.seen[0]["body"] == build_request(MODEL, {"text": "vocals"}, QUESTIONS)
    attempts = made.calls[0]["attempts"]
    assert [a["status"] for a in attempts] == [429, 502, 200]
    assert attempts[0]["retry_after_seconds"] == 0.0
    assert KEY not in (tmp_path / "jev.jsonl").read_text()

    server.replies.append((422, {"X-Trace": "t1"}, {"detail": []}))
    reply = transport({"model": MODEL}, 5.0)
    assert reply.status == 422
    assert reply.payload == {"detail": []}
    assert reply.headers["x-trace"] == "t1"


def test_http_transport_times_out_on_a_server_that_never_answers() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)  # the connections are accepted by the kernel and never answered
    try:
        transport = HttpTransport(KEY, endpoint=f"http://127.0.0.1:{listener.getsockname()[1]}/")
        with pytest.raises(TimeoutError):
            transport({"model": MODEL}, 0.3)
        made = JevClient(
            transport,
            SETTINGS.model_copy(update={"request_timeout_seconds": 0.2, "max_attempts": 2}),
            sleep=lambda s: None,
        )
        with pytest.raises(JevError, match=r"no reply within 0\.2 s"):
            made.ask("gate", "s", QUESTIONS)
        assert [a["error"] for a in made.calls[0]["attempts"]] == ["timeout", "timeout"]
    finally:
        listener.close()
