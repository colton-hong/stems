"""The live-check tool, without the network; and the real recordings, once they exist."""

from __future__ import annotations

import pytest

from stems_common import schemas
from stems_common.settings import load_defaults
from stems_models.registry import load_registry
from stems_worker.decision import live
from stems_worker.decision.jev import JevClient, Reply, parse_answers
from stems_worker.decision.mock import (
    ReplayTransport,
    ScriptedTransport,
    UnrecordedRequest,
    read_exchanges,
)

DEFAULTS = load_defaults()
REGISTRY = load_registry()
SCHEMAS = {
    "parse": "parsed_stem_request",
    "plan": "ensemble_plan",
    "gate": "stem_quality_assessment",
}


def confident(body: dict) -> dict:
    """Wire-format answers a sure judge would give: yes, the first option, a good score."""
    answers = {}
    for qid, question in body["questions"].items():
        if question["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.97}
        elif question["type"] == "choice":
            options = list(question["criteria"])
            rest = 0.05 / max(1, len(options) - 1)
            answers[qid] = {
                "type": "choice",
                "choice": options[0],
                "confidence": 0.95,
                "probabilities": {o: (0.95 if o == options[0] else rest) for o in options},
            }
        else:
            answers[qid] = {
                "type": "score",
                "score": 3.4,
                "confidence": 0.9,
                "legend": {},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.5, "4": 0.4},
            }
    return answers


class Counting:
    """A scripted service that counts the requests it is sent."""

    def __init__(self) -> None:
        self.inner, self.calls = ScriptedTransport(confident), 0

    def __call__(self, body: dict, timeout: float) -> Reply:
        self.calls += 1
        return self.inner(body, timeout)


def run(tmp_path, transport, **options) -> tuple[dict, list[str]]:
    said: list[str] = []
    records = live.run_live(
        transport,
        defaults=DEFAULTS,
        registry=REGISTRY,
        directory=tmp_path,
        say=said.append,
        **options,
    )
    return records, said


def test_each_kind_is_asked_once_and_recorded(tmp_path) -> None:
    transport = Counting()
    records, said = run(tmp_path, transport)
    assert list(records) == ["parse", "plan", "gate"]
    for kind, record in records.items():
        assert live.answered(record), kind
        assert record["model"] == DEFAULTS.decision.jev_model
        (exchange,) = read_exchanges(live.recording_path(kind, tmp_path))
        assert exchange["origin"] == "jev"
        assert exchange["request"]["model"] == "jev-1.14.0"
        assert exchange["request_sha256"] == record["request_sha256"]
    assert len(said) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "gate.recorded.jsonl",
        "parse.recorded.jsonl",
        "plan.recorded.jsonl",
    ]
    # A second run asks nothing: the recordings are there.
    calls = transport.calls
    again, said = run(tmp_path, transport)
    assert again == {}
    assert transport.calls == calls == 3
    assert all("already recorded" in line for line in said)
    # --force asks again and replaces them.
    forced, _ = run(tmp_path, transport, force=True)
    assert len(forced) == 3
    assert transport.calls == 6
    assert len(read_exchanges(live.recording_path("parse", tmp_path))) == 1


def test_a_call_that_fails_leaves_no_fixture(tmp_path) -> None:
    def refuses(body: dict, timeout: float) -> Reply:
        return Reply(401, {"detail": "invalid api key"}, {})

    records, said = run(tmp_path, refuses)
    assert len(records) == 3
    for record in records.values():
        assert not live.answered(record)
        assert record["provider"] == "fallback"
        assert record["fallback_reason"].startswith("jev_failed")
    assert list(tmp_path.iterdir()) == []
    assert all("jev_failed" in line for line in said)


def test_without_a_key_nothing_is_sent(monkeypatch, capsys) -> None:
    monkeypatch.setattr(live.HttpTransport, "from_env", classmethod(lambda cls: None))
    assert live.main([]) == 2
    assert "TYPESAFE_API_KEY is not set; nothing was sent" in capsys.readouterr().err


def test_the_fixed_inputs_are_valid_on_their_own() -> None:
    """With no client at all the same three decisions come out of the fallbacks."""
    for kind in live.KINDS:
        decision = live.decide(kind, None, DEFAULTS, REGISTRY)
        assert decision.provider in ("fallback", "selection"), kind
        assert schemas.errors(SCHEMAS[kind], decision.value) == []


# -- the real recordings -----------------------------------------------------------------


@pytest.mark.parametrize("kind", live.KINDS)
def test_a_real_recording_replays_into_a_valid_decision(kind) -> None:
    path = live.recording_path(kind)
    if not path.exists():
        pytest.skip(f"no live recording yet: run python -m stems_worker.decision.live ({kind})")
    (exchange,) = read_exchanges(path)
    assert exchange["origin"] == "jev"
    assert exchange["status"] == 200
    assert exchange["request"]["model"] == "jev-1.14.0"
    # The reply is in the documented wire format: our parser reads every answer.
    read = parse_answers(
        exchange["response"], exchange["request"]["questions"], exchange["request"]["model"]
    )
    assert set(read["answers"]) == set(exchange["request"]["questions"])
    assert read["model"] == "jev-1.14.0"
    usage = exchange["response"]["usage"]
    assert usage["input_tokens"] > 0
    # Nothing that identifies the account was kept.
    assert set(exchange["headers"]) <= {"retry-after"}
    # Replayed through the decision code it gives a valid document, decided by the mock.
    client = JevClient(ReplayTransport(path), DEFAULTS.decision, provider="mock")
    try:
        decision = live.decide(kind, client, DEFAULTS, REGISTRY)
    except UnrecordedRequest:
        pytest.skip(f"the {kind} request has changed since it was recorded: record it again")
    assert decision.record["asked"] == "mock"  # the recording was what answered
    assert schemas.errors(SCHEMAS[kind], decision.value) == []
