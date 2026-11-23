"""A client for Clef-omni on Cloudflare Workers AI, built for a test that must be auditable.

The request and response shapes are the ones on the model page
(developers.cloudflare.com/workers-ai/models/clef-omni/, read 2026-11-22): a ``state``, a
map of typed ``questions`` (noul, choice, score) and up to four embedded ``audio`` clips
go in; ``answers`` keyed by question id and token ``usage`` come out. The request limits
are read from the registry entry ``clef_omni`` and checked here before anything is sent.

Every call, whether it succeeds or not, appends one line to a JSONL log: the question ids
and wording, each clip's id, size and SHA-256 (never its audio), the status, every
probability returned and the token usage. The log's first line is a header that names the
pre-registration commit. The token is never written anywhere.

Three transports: ``HttpTransport`` talks to Cloudflare; ``RecordingTransport`` wraps it
and saves each exchange; ``ReplayTransport`` answers from saved exchanges with no network,
which is what the tests use. Not yet confirmed against the live service (no token,
BLOCKED.md B5): whether the REST endpoint wraps the body in Cloudflare's usual
``{"result": ..., "success": ...}`` envelope. Both forms are accepted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from stems_common.env import get_env

ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"
QUESTION_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class ClefError(RuntimeError):
    pass


class RequestRefused(ClefError):
    """The request breaks a limit and was not sent."""


class ResponseError(ClefError):
    """The service answered, but not with a usable set of answers."""


@dataclass(frozen=True)
class Limits:
    max_questions: int
    max_clips: int
    max_clip_bytes: int
    max_clip_seconds: float
    model_id: str
    model_field: str

    @classmethod
    def from_registry(cls) -> Limits:
        from stems_models.registry import load_registry

        config = load_registry().get("clef_omni").config
        return cls(
            max_questions=int(config["max_questions_per_request"]),
            max_clips=int(config["max_audio_clips_per_request"]),
            max_clip_bytes=int(config["max_clip_bytes"]),
            max_clip_seconds=float(config["max_clip_seconds"]),
            model_id=str(config["model_id"]),
            model_field=str(config["request_model_field"]),
        )


@dataclass(frozen=True)
class Clip:
    id: str
    data: bytes
    seconds: float
    content_type: str = "audio/wav"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def noul(instructions: str, *, yes: str | None = None, no: str | None = None) -> dict[str, Any]:
    """A yes/no question. The answer is the probability of yes."""
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if yes is not None or no is not None:
        question["criteria"] = {"true": yes, "false": no}
    return question


def choice(instructions: str, options: Mapping[str, str | None]) -> dict[str, Any]:
    """Pick one of ``options`` (id to description)."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: Sequence[str]) -> dict[str, Any]:
    """Rate on ordered ``levels``, lowest first. The answer is a level index, maybe fractional."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def build_request(
    state: Any, questions: Mapping[str, Mapping[str, Any]], clips: Sequence[Clip], limits: Limits
) -> dict[str, Any]:
    """The request body, or ``RequestRefused`` naming the limit it would break."""
    if not 1 <= len(questions) <= limits.max_questions:
        raise RequestRefused(
            f"{len(questions)} questions; a request takes 1 to {limits.max_questions}"
        )
    for qid, question in questions.items():
        if not QUESTION_ID.match(qid):
            raise RequestRefused(
                f"question id {qid!r}: letters, digits, '_', '.', '-', at most 100"
            )
        kind, criteria = question.get("type"), question.get("criteria")
        if not question.get("instructions"):
            raise RequestRefused(f"question {qid!r} has no instructions")
        if kind == "choice" and not 2 <= len(criteria or {}) <= 255:
            raise RequestRefused(f"question {qid!r}: a choice takes 2 to 255 options")
        if kind == "score" and not 2 <= len(criteria or []) <= 10:
            raise RequestRefused(f"question {qid!r}: a score takes 2 to 10 levels")
        if kind not in ("noul", "choice", "score"):
            raise RequestRefused(f"question {qid!r} has unknown type {kind!r}")
    if len(clips) > limits.max_clips:
        raise RequestRefused(f"{len(clips)} clips; a request takes at most {limits.max_clips}")
    for clip in clips:
        if len(clip.data) > limits.max_clip_bytes:
            raise RequestRefused(
                f"clip {clip.id} is {len(clip.data)} bytes; the limit is {limits.max_clip_bytes}"
            )
        if clip.seconds > limits.max_clip_seconds:
            raise RequestRefused(
                f"clip {clip.id} is {clip.seconds:.1f} s; "
                f"the limit is {limits.max_clip_seconds:.0f} s"
            )
        if not clip.content_type.startswith("audio/"):
            raise RequestRefused(f"clip {clip.id} has content type {clip.content_type!r}")
    body: dict[str, Any] = {
        "model": limits.model_field,
        "state": state,
        "questions": {qid: dict(q) for qid, q in questions.items()},
    }
    if clips:
        body["audio"] = [
            {"content_type": clip.content_type, "base64": base64.b64encode(clip.data).decode()}
            for clip in clips
        ]
    return body


def request_key(body: Mapping[str, Any]) -> str:
    """Identifies a request by content: the same questions about the same audio."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Transport(Protocol):
    def __call__(self, body: dict[str, Any]) -> tuple[int, Any]:
        """Send one request. Returns the HTTP status and the decoded JSON body (or text)."""
        ...


class HttpTransport:
    """POSTs to Workers AI. Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN."""

    def __init__(self, account_id: str, token: str, model_id: str, timeout: float = 120.0) -> None:
        import certifi

        if not account_id or not token:
            raise ClefError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN are both needed")
        self._url = ENDPOINT.format(account_id=account_id, model_id=model_id)
        self._token = token
        self._timeout = timeout
        self._context = ssl.create_default_context(cafile=certifi.where())

    @classmethod
    def from_env(cls, limits: Limits) -> HttpTransport:
        return cls(
            get_env("CLOUDFLARE_ACCOUNT_ID") or "",
            get_env("CLOUDFLARE_AUTH_TOKEN") or "",
            limits.model_id,
        )

    def __call__(self, body: dict[str, Any]) -> tuple[int, Any]:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout, context=self._context
            ) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        text = raw.decode("utf-8", errors="replace")
        try:
            return status, json.loads(text)
        except json.JSONDecodeError:
            return status, text[:2000]


class ReplayTransport:
    """Answers from saved exchanges, by request content. No network."""

    def __init__(self, path: Path) -> None:
        self._saved: dict[str, tuple[int, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                self._saved[entry["request_sha256"]] = (entry["status"], entry["response"])
        self.calls = 0

    def __call__(self, body: dict[str, Any]) -> tuple[int, Any]:
        self.calls += 1
        key = request_key(body)
        if key not in self._saved:
            raise ClefError(f"no saved response for request {key[:12]}")
        return self._saved[key]


class RecordingTransport:
    """Passes requests to ``inner`` and appends each exchange to ``path`` for replay."""

    def __init__(self, inner: Transport, path: Path) -> None:
        self._inner = inner
        self._path = path

    def __call__(self, body: dict[str, Any]) -> tuple[int, Any]:
        status, response = self._inner(body)
        save_exchange(self._path, body, status, response)
        return status, response


def save_exchange(path: Path, body: Mapping[str, Any], status: int, response: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"request_sha256": request_key(body), "status": status, "response": response}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _probability(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ResponseError(f"{what} is not a probability: {value!r}")
    return float(value)


def parse_answers(payload: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Check a response against the questions asked and return its parts.

    Returns ``{"model", "answers", "usage"}`` where every answer has ``type`` and the
    fields the model page documents for it. Raises ``ResponseError`` if a question is
    unanswered or an answer is not what its type promises.
    """
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        if payload.get("success") is False:
            raise ResponseError(f"the service reported failure: {payload.get('errors')}")
        payload = payload["result"]
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ResponseError("the response has no answers object")
    answers: dict[str, Any] = {}
    for qid, question in questions.items():
        raw = payload["answers"].get(qid)
        kind = question["type"]
        if not isinstance(raw, dict):
            raise ResponseError(f"question {qid!r} was not answered")
        if kind == "noul":
            answers[qid] = {"type": kind, "probability": _probability(raw.get("noul"), qid)}
            continue
        options = (
            list(question["criteria"])
            if kind == "choice"
            else [str(i) for i in range(len(question["criteria"]))]
        )
        given = raw.get("probabilities")
        if not isinstance(given, dict) or set(given) != set(options):
            raise ResponseError(f"question {qid!r}: probabilities do not cover its options")
        probabilities = {o: _probability(given[o], f"{qid}/{o}") for o in options}
        if abs(sum(probabilities.values()) - 1.0) > 0.01:
            raise ResponseError(
                f"question {qid!r}: probabilities sum to {sum(probabilities.values()):.3f}"
            )
        answer = {
            "type": kind,
            "probabilities": probabilities,
            "confidence": _probability(raw.get("confidence"), f"{qid} confidence"),
        }
        if kind == "choice":
            if raw.get("choice") not in options:
                raise ResponseError(f"question {qid!r}: {raw.get('choice')!r} is not an option")
            answer["choice"] = raw["choice"]
        else:
            value = raw.get("score")
            top = len(options) - 1
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= top
            ):
                raise ResponseError(f"question {qid!r}: score {value!r} is outside 0 to {top}")
            answer["score"] = float(value)
        answers[qid] = answer
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "model": payload.get("model"),
        "answers": answers,
        "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens")},
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ClefClient:
    """Asks Clef-omni and writes down everything that was asked and answered."""

    def __init__(
        self,
        transport: Transport,
        log_path: Path,
        *,
        limits: Limits | None = None,
        header: Mapping[str, Any] | None = None,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._log_path = log_path
        self._limits = limits or Limits.from_registry()
        self._max_attempts = max_attempts
        self._sleep = sleep
        if not log_path.exists() or log_path.stat().st_size == 0:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._append(
                {
                    "kind": "header",
                    "started": _now(),
                    "model_id": self._limits.model_id,
                    **dict(header or {}),
                }
            )

    def _append(self, entry: Mapping[str, Any]) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def ask(
        self,
        request_id: str,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        clips: Sequence[Clip] = (),
    ) -> dict[str, Any]:
        """Send one request and return ``{"model", "answers", "usage"}``.

        Retries on timeouts, rate limits and server errors, waiting 1, 2, 4... seconds. The
        outcome is logged either way; on failure the exception is raised after logging.
        """
        body = build_request(state, questions, clips, self._limits)
        entry: dict[str, Any] = {
            "kind": "request",
            "time": _now(),
            "request_id": request_id,
            "request_sha256": request_key(body),
            "state": state,
            "questions": {qid: dict(q) for qid, q in questions.items()},
            "clips": [
                {"id": c.id, "sha256": c.sha256, "bytes": len(c.data), "seconds": c.seconds}
                for c in clips
            ],
        }
        started = time.monotonic()
        error: Exception | None = None
        result: dict[str, Any] | None = None
        attempts = 0
        status: int | None = None
        for attempt in range(1, self._max_attempts + 1):
            attempts = attempt
            try:
                status, payload = self._transport(body)
            except (OSError, TimeoutError) as e:
                status, error = None, ClefError(f"network error: {e}")
            else:
                if status == 200:
                    try:
                        result, error = parse_answers(payload, questions), None
                    except ResponseError as e:
                        error = e
                    break
                error = ClefError(f"HTTP {status}: {str(payload)[:300]}")
                if status not in RETRY_STATUSES:
                    break
            if attempt < self._max_attempts:
                self._sleep(2.0 ** (attempt - 1))
        entry.update(
            status=status,
            attempts=attempts,
            elapsed_seconds=round(time.monotonic() - started, 3),
            ok=result is not None,
            error=None if error is None else str(error),
            model=result and result["model"],
            answers=result and result["answers"],
            usage=result and result["usage"],
        )
        self._append(entry)
        if result is None:
            raise error or ClefError("no response")
        return result


def read_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A log's header and its request entries."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines or lines[0].get("kind") != "header":
        raise ClefError(f"{path} does not start with a header")
    return lines[0], lines[1:]
