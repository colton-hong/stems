"""A client for Jev, TypeSafe's "System One" decision model.

Jev never writes text. A request carries a ``state`` (text or JSON) and named, typed
``questions``; the reply carries one typed answer per question, with probabilities. The
shapes are the ones in the API reference (docs.typesafe.ai/api, read 2026-11-22):

- ``noul``: yes or no. The answer is ``{"noul": p}``, the probability of yes. It has **no**
  confidence field (docs/corrections.md, C1), so ``max(p, 1 - p)`` is used as one here.
- ``choice``: one of 1 to 255 named options. ``choice``, ``probabilities``, ``confidence``.
- ``score``: a rating on 1 to 10 ordered levels. ``score`` is the probability-weighted
  average and may fall between levels (C2); ``probabilities`` are keyed "0", "1", ...

The model is always a pinned version such as ``jev-1.14.0``. An alias (``jev-latest``,
``jev-preview``) is refused before anything is sent, and so is a reply that names a
different model than the one asked for.

Every call is written down: the question ids and wording, the HTTP status of each attempt,
the responding ``model`` field, the token usage, every probability, and the time taken. The
API key is never written anywhere, and is removed from any error text before that text is
logged or raised.

Plain HTTPS through the standard library, not the vendor SDK: the SDK defaults to an alias
and does its own retries, and both are things this module has to control and log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

from stems_common.env import get_env
from stems_common.settings import DecisionSettings

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
KEY_ENV = "TYPESAFE_API_KEY"
PINNED_MODEL = re.compile(r"^jev-\d+\.\d+\.\d+$")
QUESTION_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")
REDACTED = "[redacted]"

log = logging.getLogger("stems.decision.jev")


class JevError(RuntimeError):
    """Jev could not be asked, or did not give a usable answer. The caller falls back."""


class RequestRefused(JevError):
    """The request is not one this client sends (an alias, a malformed question)."""


class ResponseError(JevError):
    """The service answered, but not with a usable set of answers."""


# --- questions ----------------------------------------------------------------------------


def noul(instructions: str, *, yes: str | None = None, no: str | None = None) -> dict[str, Any]:
    """A yes/no question. The answer is the probability of yes."""
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if yes is not None or no is not None:
        question["criteria"] = {"true": yes, "false": no}
    return question


def choice(instructions: str, options: Mapping[str, str | None]) -> dict[str, Any]:
    """Pick one of ``options`` (name to description; the names come back as the answer)."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: Sequence[str]) -> dict[str, Any]:
    """Rate on ordered ``levels``, lowest first. The answer may fall between levels."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def check_model(model: str) -> str:
    if not PINNED_MODEL.match(model):
        raise RequestRefused(
            f"model {model!r} is not a pinned version; use an exact one such as jev-1.14.0"
        )
    return model


def build_request(
    model: str, state: Any, questions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """The request body, or ``RequestRefused`` saying what is wrong with it."""
    check_model(model)
    if not questions:
        raise RequestRefused("a request needs at least one question")
    if state is None or state == "":
        raise RequestRefused("a request needs a state")
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
    return {
        "model": model,
        "state": state,
        "questions": {qid: dict(q) for qid, q in questions.items()},
    }


def request_key(body: Mapping[str, Any]) -> str:
    """Identifies a request by its content: model, state and questions."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --- answers ------------------------------------------------------------------------------


def _probability(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ResponseError(f"{what} is not a probability: {value!r}")
    return float(value)


def noul_confidence(probability: float) -> float:
    """How far a yes/no probability is from a coin toss: ``max(p, 1 - p)``."""
    return max(probability, 1.0 - probability)


def parse_answers(
    payload: Any, questions: Mapping[str, Mapping[str, Any]], model: str
) -> dict[str, Any]:
    """Check a reply against the questions asked and return ``{"model", "answers", "usage"}``.

    Every answer has ``type`` and ``confidence``; a noul has ``probability``; a choice has
    ``choice`` and ``probabilities``; a score has ``score``, ``probabilities`` keyed by
    level index, and ``top_level``, the most probable level. Raises ``ResponseError`` if a
    question is unanswered, an answer is not what its type promises, or another model
    than ``model`` answered.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ResponseError("the response has no answers object")
    if payload.get("model") != model:
        raise ResponseError(f"asked {model}, but {payload.get('model')!r} answered")
    answers: dict[str, Any] = {}
    for qid, question in questions.items():
        raw = payload["answers"].get(qid)
        kind = question["type"]
        if not isinstance(raw, dict):
            raise ResponseError(f"question {qid!r} was not answered")
        if kind == "noul":
            p = _probability(raw.get("noul"), qid)
            answers[qid] = {"type": kind, "probability": p, "confidence": noul_confidence(p)}
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
        answer: dict[str, Any] = {
            "type": kind,
            "probabilities": probabilities,
            "confidence": _probability(raw.get("confidence"), f"{qid} confidence"),
        }
        if kind == "choice":
            if raw.get("choice") not in options:
                raise ResponseError(f"question {qid!r}: {raw.get('choice')!r} is not an option")
            answer["choice"] = raw["choice"]
        else:
            value, top = raw.get("score"), len(options) - 1
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= top
            ):
                raise ResponseError(f"question {qid!r}: score {value!r} is outside 0 to {top}")
            answer["score"] = float(value)
            answer["top_level"] = int(max(options, key=lambda o: (probabilities[o], -int(o))))
        answers[qid] = answer
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "model": payload.get("model"),
        "answers": answers,
        "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens")},
    }


# --- transport ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    status: int
    # The decoded JSON body, or the first 2000 characters of a body that is not JSON.
    payload: Any
    # Header names in lower case.
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def __call__(self, body: dict[str, Any], timeout: float) -> Reply:
        """Send one request and wait at most ``timeout`` seconds for the reply.

        Raises ``TimeoutError`` or ``OSError`` when no reply arrives.
        """
        ...


class HttpTransport:
    """POSTs to the TypeSafe API. The key is held here and nowhere else."""

    def __init__(self, key: str, *, endpoint: str = ENDPOINT) -> None:
        if not key:
            raise JevError(f"{KEY_ENV} is not set")
        self._key = key
        self._endpoint = endpoint
        self._context: ssl.SSLContext | None = None
        if endpoint.startswith("https://"):
            import certifi

            self._context = ssl.create_default_context(cafile=certifi.where())

    @classmethod
    def from_env(cls) -> HttpTransport | None:
        """A transport using ``TYPESAFE_API_KEY``, or None when no key is set."""
        key = get_env(KEY_ENV)
        return cls(key) if key else None

    @property
    def secrets(self) -> tuple[str, ...]:
        return (self._key,)

    def __repr__(self) -> str:
        return f"HttpTransport({self._endpoint})"

    def __call__(self, body: dict[str, Any], timeout: float) -> Reply:
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=self._context) as reply:
                status, raw, headers = reply.status, reply.read(), reply.headers
        except urllib.error.HTTPError as e:
            status, raw, headers = e.code, e.read(), e.headers
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise TimeoutError(str(e.reason)) from None
            raise
        text = raw.decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError:
            payload = text[:2000]
        return Reply(status, payload, {k.lower(): v for k, v in (headers or {}).items()})


def retry_after_seconds(headers: Mapping[str, str], now: datetime | None = None) -> float | None:
    """The wait a ``retry-after`` header asks for: seconds, or an HTTP date. None if absent."""
    value = (headers.get("retry-after") or "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


# --- client -------------------------------------------------------------------------------


@dataclass(frozen=True)
class JevResult:
    # The ``model`` field of the reply.
    model: str
    answers: dict[str, Any]
    # ``input_tokens`` and ``output_tokens``.
    usage: dict[str, Any]
    latency_ms: float
    attempts: int
    request_sha256: str


def is_retryable(status: int) -> bool:
    """Whether a reply with this status is asked again: 408, 429 and any 5xx."""
    return status in (408, 429) or 500 <= status <= 599


class JevClient:
    """Asks Jev and writes down everything that was asked and answered.

    ``provider`` names where the answers come from in decision logs: ``jev`` for the real
    service, ``mock`` when the transport replays recorded replies.
    """

    def __init__(
        self,
        transport: Transport,
        settings: DecisionSettings,
        *,
        log_path: Path | None = None,
        provider: str = "jev",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = check_model(settings.jev_model)
        self.provider = provider
        self._transport = transport
        self._timeout = settings.request_timeout_seconds
        self._max_attempts = settings.max_attempts
        self._backoff = settings.backoff_seconds
        self._max_wait = settings.max_retry_wait_seconds
        self._log_path = log_path
        self._sleep = sleep
        self._clock = clock
        self._secrets = tuple(s for s in getattr(transport, "secrets", ()) if s)
        # One entry per call, as written to the log file.
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_env(
        cls, settings: DecisionSettings, *, log_path: Path | None = None
    ) -> JevClient | None:
        """A client for the real service, or None when ``TYPESAFE_API_KEY`` is not set."""
        transport = HttpTransport.from_env()
        return None if transport is None else cls(transport, settings, log_path=log_path)

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text

    def _clean(self, value: Any) -> Any:
        """``value`` with every secret removed, whatever it is nested in."""
        if not self._secrets:
            return value
        return json.loads(self._scrub(json.dumps(value, ensure_ascii=False)))

    def _record(self, entry: dict[str, Any]) -> None:
        entry = self._clean(entry)
        self.calls.append(entry)
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info(
            "jev %s: ok=%s model=%s tokens=%s attempts=%d %.0f ms%s",
            entry["kind"],
            entry["ok"],
            entry["model"],
            entry["usage"],
            len(entry["attempts"]),
            entry["latency_ms"],
            "" if entry["ok"] else f" error={entry['error']}",
        )

    def ask(self, kind: str, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> JevResult:
        """Send one request. ``kind`` labels it in the log (parse, plan, gate).

        A 429 or a 5xx reply, a timeout and a network error are retried, up to
        ``max_attempts`` attempts in all. The wait before a retry is the reply's
        ``retry-after`` when it has one, otherwise ``backoff_seconds`` doubling each time. A
        ``retry-after`` longer than ``max_retry_wait_seconds`` ends the call at once: the
        caller's fallback is faster than the wait, and retrying sooner would ignore it.
        Anything else (400, 401, 402, 422) fails at once. The outcome is logged either
        way; on failure ``JevError`` is raised after logging.
        """
        body = build_request(self.model, state, questions)
        key = request_key(body)
        started = self._clock()
        attempts: list[dict[str, Any]] = []
        error: JevError | None = None
        result: dict[str, Any] | None = None
        for attempt in range(1, self._max_attempts + 1):
            wait: float | None = None
            try:
                reply = self._transport(body, self._timeout)
            except TimeoutError:
                error = JevError(f"no reply within {self._timeout:g} s")
                attempts.append({"status": None, "error": "timeout"})
            except OSError as e:
                error = JevError(f"network error: {e}")
                attempts.append({"status": None, "error": str(e)})
            else:
                attempts.append({"status": reply.status})
                if reply.status == 200:
                    try:
                        result, error = parse_answers(reply.payload, questions, self.model), None
                    except ResponseError as e:
                        error = e
                    break
                error = JevError(f"HTTP {reply.status}: {str(reply.payload)[:300]}")
                if not is_retryable(reply.status):
                    break
                wait = retry_after_seconds(reply.headers)
                if wait is not None:
                    attempts[-1]["retry_after_seconds"] = wait
                    if wait > self._max_wait:
                        error = JevError(
                            f"HTTP {reply.status}: asked to wait {wait:g} s, "
                            f"more than the {self._max_wait:g} s allowed"
                        )
                        break
            if attempt < self._max_attempts:
                wait = self._backoff * 2.0 ** (attempt - 1) if wait is None else wait
                attempts[-1]["waited_seconds"] = wait
                self._sleep(wait)
        latency_ms = round((self._clock() - started) * 1000.0, 1)
        self._record(
            {
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
                "kind": kind,
                "provider": self.provider,
                "requested_model": self.model,
                "request_sha256": key,
                "state": state,
                "questions": {qid: dict(q) for qid, q in questions.items()},
                "attempts": attempts,
                "latency_ms": latency_ms,
                "ok": result is not None,
                "error": None if error is None else str(error),
                "model": result and result["model"],
                "usage": result and result["usage"],
                "answers": result and result["answers"],
            }
        )
        if result is None:
            failure = error or JevError("no response")
            raise type(failure)(self._scrub(str(failure)))
        return JevResult(
            model=result["model"],
            answers=result["answers"],
            usage=result["usage"],
            latency_ms=latency_ms,
            attempts=len(attempts),
            request_sha256=key,
        )
