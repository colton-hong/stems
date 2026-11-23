"""Recorded Jev replies, replayed by request content. No network and no key.

A recording is one JSON object per exchange: the request body, its SHA-256, the HTTP
status, the reply, and an ``origin`` saying where the reply came from:

- ``jev``: captured from the real service by ``RecordingTransport`` (P4.9);
- ``handwritten``: written by hand for a test. It shows what the code does with such an
  answer, and says nothing about what Jev would answer;
- ``scripted``: produced by a function standing in for Jev.

``ReplayTransport`` answers a request only if exactly that request was recorded: same
model, same state, same question wording. Anything else raises ``UnrecordedRequest``,
which is deliberately not a ``JevError``, so the decision layer does not quietly cover it
with its rule-based fallback and a stale recording cannot pass for a working one.

Files are ``.jsonl`` (one exchange per line) or ``.json`` (a list of exchanges).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from stems_common.settings import DecisionSettings
from stems_worker.decision.jev import JevClient, Reply, Transport, is_retryable, request_key

ORIGINS = ("jev", "handwritten", "scripted")


class UnrecordedRequest(LookupError):
    """A request was made that no recording answers."""


class RecordingError(ValueError):
    """A recording file is malformed, or its request no longer matches its hash."""


def exchange(
    body: Mapping[str, Any],
    payload: Any,
    *,
    origin: str,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One recorded exchange, ready to be written to a file."""
    if origin not in ORIGINS:
        raise RecordingError(f"origin must be one of {ORIGINS}, got {origin!r}")
    return {
        "request_sha256": request_key(body),
        "origin": origin,
        "request": dict(body),
        "status": status,
        "headers": dict(headers or {}),
        "response": payload,
    }


def handwritten(
    body: Mapping[str, Any], answers: Mapping[str, Any], *, input_tokens: int = 0
) -> dict[str, Any]:
    """A hand-written 200 reply to ``body``, in the service's wire format.

    ``answers`` are as the API documents them (``{"type": "noul", "noul": 0.9}`` and so
    on). Token usage is 0 unless given: nothing was billed for a reply nobody sent.
    """
    payload = {
        "model": body["model"],
        "answers": {qid: dict(answer) for qid, answer in answers.items()},
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
    return exchange(body, payload, origin="handwritten")


def save_exchange(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_exchanges(path: Path) -> list[dict[str, Any]]:
    """The exchanges in one ``.jsonl`` or ``.json`` file, each checked against its hash."""
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix == ".json":
            entries = json.loads(text)
        else:
            entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        raise RecordingError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(entries, list):
        raise RecordingError(f"{path}: expected a list of exchanges")
    for number, entry in enumerate(entries, 1):
        where = f"{path.name} #{number}"
        if not isinstance(entry, dict) or not {"request", "status", "response"} <= set(entry):
            raise RecordingError(f"{where}: an exchange needs request, status and response")
        if entry.get("origin") not in ORIGINS:
            raise RecordingError(f"{where}: origin must be one of {ORIGINS}")
        if entry.get("request_sha256") != request_key(entry["request"]):
            raise RecordingError(
                f"{where}: the request does not match its request_sha256 (edited by hand?)"
            )
    return entries


class ReplayTransport:
    """Answers from recordings, by request content."""

    def __init__(self, *sources: Path) -> None:
        self._saved: dict[str, dict[str, Any]] = {}
        self._used: set[str] = set()
        self.calls = 0
        for source in sources:
            files = (
                sorted(p for p in source.iterdir() if p.suffix in (".json", ".jsonl"))
                if source.is_dir()
                else [source]
            )
            for file in files:
                for entry in read_exchanges(file):
                    known = self._saved.setdefault(entry["request_sha256"], entry)
                    if (known["status"], known["response"]) != (entry["status"], entry["response"]):
                        raise RecordingError(
                            f"{file}: request {entry['request_sha256'][:12]} is recorded twice "
                            "with different replies"
                        )

    def __len__(self) -> int:
        return len(self._saved)

    def __call__(self, body: dict[str, Any], timeout: float) -> Reply:
        self.calls += 1
        key = request_key(body)
        entry = self._saved.get(key)
        if entry is None:
            state = json.dumps(body.get("state"), ensure_ascii=False)
            raise UnrecordedRequest(
                f"no recorded reply for request {key[:12]} "
                f"(model {body.get('model')}; questions {sorted(body.get('questions', {}))}; "
                f"state {state[:200]}). {len(self._saved)} recordings are loaded; record this "
                "request or add a fixture for it."
            )
        self._used.add(key)
        return Reply(entry["status"], entry["response"], dict(entry.get("headers") or {}))

    def origins(self) -> set[str]:
        """Where the replies given so far came from."""
        return {self._saved[key]["origin"] for key in self._used}

    def unused(self) -> list[str]:
        """Hashes of recordings no request has asked for."""
        return sorted(set(self._saved) - self._used)


class RecordingTransport:
    """Passes requests to ``inner`` and appends each settled exchange to ``path``.

    A reply the client will retry (429, 5xx) is passed on but not recorded.
    """

    def __init__(self, inner: Transport, path: Path, *, origin: str = "jev") -> None:
        if origin not in ORIGINS:
            raise RecordingError(f"origin must be one of {ORIGINS}, got {origin!r}")
        self._inner = inner
        self._path = path
        self._origin = origin
        self.secrets = tuple(getattr(inner, "secrets", ()))

    def __call__(self, body: dict[str, Any], timeout: float) -> Reply:
        reply = self._inner(body, timeout)
        if is_retryable(reply.status):
            # The client asks again; only the reply that settles a request is a recording.
            return reply
        save_exchange(
            self._path,
            exchange(
                body,
                reply.payload,
                origin=self._origin,
                status=reply.status,
                # Only what a replay needs. Other headers can identify the account.
                headers={k: v for k, v in reply.headers.items() if k == "retry-after"},
            ),
        )
        return reply


class ScriptedTransport:
    """A stand-in for the service: ``responder`` turns a request body into wire answers.

    For tests and offline demonstrations of the whole pipeline. Its answers are whatever
    the function says; they are not Jev's.
    """

    def __init__(self, responder: Callable[[dict[str, Any]], Mapping[str, Any]]) -> None:
        self._responder = responder
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, body: dict[str, Any], timeout: float) -> Reply:
        self.bodies.append(body)
        payload = {
            "model": body["model"],
            "answers": dict(self._responder(body)),
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return Reply(200, payload)


def mock_client(
    settings: DecisionSettings, *sources: Path, log_path: Path | None = None
) -> JevClient:
    """A client that answers from recordings; decision logs show its provider as ``mock``."""
    return JevClient(ReplayTransport(*sources), settings, log_path=log_path, provider="mock")
