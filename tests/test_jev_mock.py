"""Recorded replies: replayed by request content, and loud about anything unrecorded."""

from __future__ import annotations

import json

import pytest

from stems_common.settings import load_defaults
from stems_worker.decision.jev import JevClient, JevError, Reply, build_request, choice, noul
from stems_worker.decision.mock import (
    RecordingError,
    RecordingTransport,
    ReplayTransport,
    ScriptedTransport,
    UnrecordedRequest,
    exchange,
    handwritten,
    mock_client,
    read_exchanges,
    save_exchange,
)

SETTINGS = load_defaults().decision
MODEL = SETTINGS.jev_model
QUESTIONS = {
    "wants.drums": noul("Does the user want the drums?"),
    "format": choice("Which file format?", {"wav": None, "flac": None}),
}
ANSWERS = {
    "wants.drums": {"type": "noul", "noul": 0.96},
    "format": {
        "type": "choice",
        "choice": "wav",
        "confidence": 0.88,
        "probabilities": {"wav": 0.88, "flac": 0.12},
    },
}


def body(state="drums please"):
    return build_request(MODEL, state, QUESTIONS)


def test_a_recorded_request_is_replayed_by_its_content(tmp_path) -> None:
    path = tmp_path / "parse.jsonl"
    save_exchange(path, handwritten(body(), ANSWERS))
    save_exchange(
        path,
        handwritten(
            body("only the drums"), {**ANSWERS, "wants.drums": {"type": "noul", "noul": 0.99}}
        ),
    )
    client = mock_client(SETTINGS, path, log_path=tmp_path / "calls.jsonl")
    assert client.provider == "mock"

    result = client.ask("parse", "drums please", QUESTIONS)
    assert result.answers["wants.drums"]["probability"] == 0.96
    assert result.answers["format"]["choice"] == "wav"
    assert result.model == MODEL
    assert result.usage == {"input_tokens": 0, "output_tokens": 0}  # nothing was billed
    # The same content again, built separately, finds the same recording.
    assert client.ask("parse", "drums please", dict(QUESTIONS)).answers == result.answers
    assert (
        client.ask("parse", "only the drums", QUESTIONS).answers["wants.drums"]["probability"]
        == 0.99
    )
    logged = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [entry["provider"] for entry in logged] == ["mock"] * 3


@pytest.mark.parametrize(
    "change",
    [
        lambda b: {**b, "state": "drums please!"},
        lambda b: {**b, "model": "jev-1.13.0"},
        lambda b: {**b, "questions": {**b["questions"], "wants.drums": noul("Drums?")}},
        lambda b: {**b, "questions": {"format": b["questions"]["format"]}},
    ],
)
def test_an_unrecorded_request_fails_loudly(tmp_path, change) -> None:
    path = tmp_path / "parse.jsonl"
    save_exchange(path, handwritten(body(), ANSWERS))
    replay = ReplayTransport(path)
    assert len(replay) == 1
    with pytest.raises(UnrecordedRequest, match="no recorded reply for request") as raised:
        replay(change(body()), 1.0)
    assert "1 recordings are loaded" in str(raised.value)
    assert replay.unused() == [handwritten(body(), ANSWERS)["request_sha256"]]


def test_the_client_does_not_turn_an_unrecorded_request_into_a_jev_failure(tmp_path) -> None:
    path = tmp_path / "parse.jsonl"
    save_exchange(path, handwritten(body(), ANSWERS))
    sleeps: list[float] = []
    client = JevClient(ReplayTransport(path), SETTINGS, provider="mock", sleep=sleeps.append)
    # Not a JevError: the decision layer's fallback must not hide it. And not retried.
    assert not issubclass(UnrecordedRequest, (JevError, OSError))
    with pytest.raises(UnrecordedRequest, match='state "something else"'):
        client.ask("parse", "something else", QUESTIONS)
    assert sleeps == []


def test_recordings_load_from_files_and_directories(tmp_path) -> None:
    folder = tmp_path / "fixtures"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps([handwritten(body("a"), ANSWERS)]))
    save_exchange(folder / "b.jsonl", handwritten(body("b"), ANSWERS))
    (folder / "notes.txt").write_text("ignored")
    single = tmp_path / "c.jsonl"
    save_exchange(single, handwritten(body("c"), ANSWERS))
    save_exchange(single, handwritten(body("a"), ANSWERS))  # the same exchange twice is fine
    replay = ReplayTransport(folder, single)
    assert len(replay) == 3
    for state in "abc":
        assert replay(body(state), 1.0).status == 200
    assert replay.calls == 3
    assert replay.unused() == []
    assert replay.origins() == {"handwritten"}


def test_a_recording_that_no_longer_matches_its_hash_is_refused(tmp_path) -> None:
    entry = handwritten(body(), ANSWERS)
    edited = {**entry, "request": {**entry["request"], "state": "edited afterwards"}}
    path = tmp_path / "x.jsonl"
    save_exchange(path, edited)
    with pytest.raises(RecordingError, match="does not match its request_sha256"):
        ReplayTransport(path)

    for bad, message in [
        ("not json", "not valid JSON"),
        (json.dumps({"request": {}}), "needs request, status and response"),
        (json.dumps({**entry, "origin": "guess"}), "origin must be one of"),
    ]:
        path.write_text(bad + "\n")
        with pytest.raises(RecordingError, match=message):
            read_exchanges(path)
    (tmp_path / "y.json").write_text(json.dumps({"request": {}}))
    with pytest.raises(RecordingError, match="expected a list"):
        read_exchanges(tmp_path / "y.json")
    with pytest.raises(RecordingError, match="origin must be one of"):
        exchange(body(), {}, origin="invented")


def test_one_request_recorded_with_two_different_replies_is_refused(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    save_exchange(path, handwritten(body(), ANSWERS))
    save_exchange(
        path, handwritten(body(), {**ANSWERS, "wants.drums": {"type": "noul", "noul": 0.1}})
    )
    with pytest.raises(RecordingError, match="recorded twice with different replies"):
        ReplayTransport(path)


def test_recording_then_replaying_gives_the_same_answers(tmp_path) -> None:
    def responder(request):
        wants = "drum" in str(request["state"])
        return {**ANSWERS, "wants.drums": {"type": "noul", "noul": 0.95 if wants else 0.03}}

    inner = ScriptedTransport(responder)
    path = tmp_path / "recorded" / "session.jsonl"
    recording = JevClient(RecordingTransport(inner, path, origin="scripted"), SETTINGS)
    first = [recording.ask("parse", s, QUESTIONS).answers for s in ("drums", "just vocals")]
    assert len(inner.bodies) == 2
    assert [e["origin"] for e in read_exchanges(path)] == ["scripted", "scripted"]

    replay = ReplayTransport(path)
    client = JevClient(replay, SETTINGS, provider="mock")
    again = [client.ask("parse", s, QUESTIONS).answers for s in ("drums", "just vocals")]
    assert again == first
    assert again[0]["wants.drums"]["probability"] == 0.95
    assert again[1]["wants.drums"]["probability"] == 0.03
    assert replay.origins() == {"scripted"}
    assert len(inner.bodies) == 2  # the replay asked nobody


def test_only_settled_replies_are_recorded_and_secrets_are_not(tmp_path) -> None:
    class Flaky:
        secrets = ("ts-secret-key",)

        def __init__(self) -> None:
            self.replies = [
                Reply(429, {"detail": "slow down"}, {"retry-after": "0", "x-account": "acct-1"}),
                Reply(200, handwritten(body(), ANSWERS)["response"], {"x-account": "acct-1"}),
                Reply(400, {"detail": "max_tokens_exceeded"}),
            ]

        def __call__(self, request, timeout):
            return self.replies.pop(0)

    path = tmp_path / "flaky.jsonl"
    recorder = RecordingTransport(Flaky(), path)
    assert recorder.secrets == ("ts-secret-key",)
    client = JevClient(recorder, SETTINGS, sleep=lambda s: None)
    assert client.ask("parse", "drums please", QUESTIONS).attempts == 2
    with pytest.raises(JevError, match="HTTP 400"):
        client.ask("parse", "far too long", QUESTIONS)
    text = path.read_text()
    assert "ts-secret-key" not in text
    assert "acct-1" not in text  # only retry-after is kept from the headers
    # The 429 was retried and is not a recording; the 200 and the 400 are.
    assert [e["status"] for e in read_exchanges(path)] == [200, 400]
    replayed = JevClient(ReplayTransport(path), SETTINGS, provider="mock", sleep=lambda s: None)
    assert replayed.ask("parse", "drums please", QUESTIONS).attempts == 1
    with pytest.raises(JevError, match="HTTP 400"):
        replayed.ask("parse", "far too long", QUESTIONS)
    with pytest.raises(RecordingError, match="origin must be one of"):
        RecordingTransport(Flaky(), path, origin="made up")
