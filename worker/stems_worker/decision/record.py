"""What every decision returns: the schema object, and a log record of how it was reached.

A record has the columns of the ``decision_calls`` table (PLAN.md section 7): which kind
of decision, who decided (``jev``, ``mock`` or ``fallback``), the responding model, token
usage, latency, every answer with its probabilities, and why the rules decided instead of
Jev when they did.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from stems_worker.decision.jev import JevClient, JevError, JevResult

FALLBACK = "fallback"
# Why the rule-based fallback decided.
NO_KEY = "no_key"
JEV_FAILED = "jev_failed"
LOW_CONFIDENCE = "low_confidence"
NOT_NEEDED = "not_needed"


@dataclass(frozen=True)
class Decision:
    kind: str  # parse | plan | gate
    # The assembled object, valid against its schema.
    value: dict[str, Any]
    record: dict[str, Any]

    @property
    def provider(self) -> str:
        return self.record["provider"]


@dataclass(frozen=True)
class Asked:
    """The outcome of trying to ask Jev: an answer, or the reason there is none."""

    result: JevResult | None
    # None when Jev answered. Otherwise ``no_key`` or ``jev_failed: <what happened>``.
    failure: str | None
    provider: str

    @property
    def answers(self) -> dict[str, Any] | None:
        return None if self.result is None else self.result.answers

    @property
    def contacted(self) -> bool:
        """Whether a request went out (to Jev or to a mock of it), answered or not."""
        return self.result is not None or (self.failure or "").startswith(JEV_FAILED)


def ask(
    client: JevClient | None, kind: str, state: Any, questions: Mapping[str, Mapping[str, Any]]
) -> Asked:
    """Ask Jev if there is a client. A failure is returned, not raised: the caller falls back.

    An unrecorded request to a replaying mock is not a failure of Jev and is raised.
    """
    if client is None:
        return Asked(None, NO_KEY, FALLBACK)
    try:
        return Asked(client.ask(kind, state, questions), None, client.provider)
    except JevError as e:
        return Asked(None, f"{JEV_FAILED}: {e}", client.provider)


def record(
    kind: str, asked: Asked, *, fallback_reason: str | None, notes: list[str] | None = None
) -> dict[str, Any]:
    """The log record. ``fallback_reason`` is None exactly when Jev's answers decided."""
    result = asked.result
    usage = {} if result is None else result.usage
    return {
        "kind": kind,
        "provider": asked.provider if fallback_reason is None else FALLBACK,
        # What was asked, when anything was: jev or mock. None when no client existed.
        "asked": asked.provider if asked.contacted else None,
        "model": None if result is None else result.model,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "latency_ms": None if result is None else result.latency_ms,
        "attempts": None if result is None else result.attempts,
        "request_sha256": None if result is None else result.request_sha256,
        # Jev's answers are kept even when they were not trusted, so the log shows why.
        "answers": asked.answers,
        "fallback_reason": fallback_reason,
        "notes": list(notes or []),
    }
