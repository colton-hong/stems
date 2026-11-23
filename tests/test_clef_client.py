"""The Clef-omni client: limits, parsing, logging and replay. No network anywhere.

The saved responses used here are written by hand to the format on Cloudflare's model
page. They are not captures of the live service; none exists yet (BLOCKED.md B5).
"""

from __future__ import annotations

import base64
import json

import pytest

from stems_bench.clef import client as cl

LIMITS = cl.Limits(
    max_questions=64,
    max_clips=4,
    max_clip_bytes=8 * 1024 * 1024,
    max_clip_seconds=300.0,
    model_id="@cf/cloudflare/clef-omni",
    model_field="clef-omni",
)
STATE = "A clip of music."
QUESTIONS = {
    "present.guitar": cl.noul("Is a guitar audible in this clip?"),
    "lead_backing": cl.choice(
        "Which vocals are audible?", {"lead only": None, "backing only": None, "both": None}
    ),
    "bleed.score": cl.score("How much bleed?", ["Clean", "Faint", "Audible", "Heavy"]),
}
ANSWERS = {
    "present.guitar": {"noul": 0.91},
    "lead_backing": {
        "choice": "both",
        "probabilities": {"lead only": 0.2, "backing only": 0.1, "both": 0.7},
        "confidence": 0.6,
    },
    "bleed.score": {
        "score": 0.4,
        "legend": {"0": "Clean", "1": "Faint", "2": "Audible", "3": "Heavy"},
        "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1, "3": 0.0},
        "confidence": 0.55,
    },
}
RESPONSE = {
    "model": "clef-omni",
    "answers": ANSWERS,
    "usage": {"input_tokens": 431, "output_tokens": 0},
}


def clip(name: str = "a01", size: int = 1000, seconds: float = 30.0) -> cl.Clip:
    return cl.Clip(name, bytes(size), seconds)


class Scripted:
    """A transport that returns the given outcomes in order."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.bodies: list[dict] = []

    def __call__(self, body):
        self.bodies.append(body)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make(tmp_path, *outcomes, **options):
    transport = Scripted(*outcomes)
    waits: list[float] = []
    client = cl.ClefClient(
        transport,
        tmp_path / "log.jsonl",
        limits=LIMITS,
        header={"preregistration_commit": "abc123"},
        sleep=waits.append,
        **options,
    )
    return client, transport, waits


def test_limits_come_from_the_registry():
    assert cl.Limits.from_registry() == LIMITS


def test_the_request_body_is_the_documented_shape():
    body = cl.build_request(STATE, QUESTIONS, [clip()], LIMITS)
    assert set(body) == {"model", "state", "questions", "audio"}
    assert body["model"] == "clef-omni"
    assert body["questions"]["present.guitar"] == {
        "type": "noul",
        "instructions": "Is a guitar audible in this clip?",
    }
    assert body["questions"]["bleed.score"]["criteria"] == ["Clean", "Faint", "Audible", "Heavy"]
    assert body["audio"] == [
        {"content_type": "audio/wav", "base64": base64.b64encode(bytes(1000)).decode()}
    ]
    assert "audio" not in cl.build_request(STATE, QUESTIONS, [], LIMITS)
    described = cl.noul("Is it loud?", yes="clearly loud", no="quiet")
    assert described["criteria"] == {"true": "clearly loud", "false": "quiet"}


@pytest.mark.parametrize(
    ("questions", "clips", "message"),
    [
        ({f"q{i}": cl.noul("?") for i in range(65)}, [], "65 questions"),
        ({}, [], "0 questions"),
        ({"q": cl.noul("?")}, [clip(str(i)) for i in range(5)], "5 clips"),
        ({"q": cl.noul("?")}, [clip(size=8 * 1024 * 1024 + 1)], "8388609 bytes"),
        ({"q": cl.noul("?")}, [clip(seconds=300.5)], "300.5 s"),
        ({"q": cl.noul("?")}, [cl.Clip("v", b"x", 1.0, "video/mp4")], "content type"),
        ({"bad id": cl.noul("?")}, [], "question id"),
        ({"q" * 101: cl.noul("?")}, [], "question id"),
        ({"q": cl.noul("")}, [], "no instructions"),
        ({"q": cl.choice("?", {"only": None})}, [], "2 to 255 options"),
        ({"q": cl.score("?", ["one"])}, [], "2 to 10 levels"),
        ({"q": cl.score("?", [str(i) for i in range(11)])}, [], "2 to 10 levels"),
        ({"q": {"type": "box", "instructions": "?"}}, [], "unknown type"),
    ],
)
def test_a_request_over_a_limit_is_refused_before_sending(tmp_path, questions, clips, message):
    with pytest.raises(cl.RequestRefused, match=message):
        cl.build_request(STATE, questions, clips, LIMITS)
    client, transport, _ = make(tmp_path)
    with pytest.raises(cl.RequestRefused):
        client.ask("r1", STATE, questions, clips)
    assert transport.bodies == []


def test_requests_at_the_limits_are_accepted():
    questions = {f"q{i}": cl.noul("?") for i in range(64)}
    clips = [clip(str(i), size=8 * 1024 * 1024, seconds=300.0) for i in range(4)]
    body = cl.build_request(STATE, questions, clips, LIMITS)
    assert len(body["questions"]) == 64
    assert len(body["audio"]) == 4


def test_answers_are_parsed_by_question_type():
    parsed = cl.parse_answers(RESPONSE, QUESTIONS)
    assert parsed["model"] == "clef-omni"
    assert parsed["usage"] == {"input_tokens": 431, "output_tokens": 0}
    assert parsed["answers"]["present.guitar"] == {"type": "noul", "probability": 0.91}
    choice = parsed["answers"]["lead_backing"]
    assert (choice["choice"], choice["confidence"]) == ("both", 0.6)
    assert choice["probabilities"]["both"] == 0.7
    score = parsed["answers"]["bleed.score"]
    assert score["score"] == 0.4
    assert score["probabilities"] == {"0": 0.7, "1": 0.2, "2": 0.1, "3": 0.0}
    # Cloudflare's REST envelope, if it turns out to be used, gives the same result.
    wrapped = {"result": RESPONSE, "success": True, "errors": [], "messages": []}
    assert cl.parse_answers(wrapped, QUESTIONS) == parsed


def broken(question: str, **changes):
    answers = json.loads(json.dumps(ANSWERS))
    for key, value in changes.items():
        if value is None:
            del answers[question][key]
        else:
            answers[question][key] = value
    return {**RESPONSE, "answers": answers}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("<html>bad gateway</html>", "no answers object"),
        ({"answers": {}}, "was not answered"),
        (broken("present.guitar", noul=1.2), "not a probability"),
        (broken("present.guitar", noul=None), "not a probability"),
        (broken("present.guitar", noul=True), "not a probability"),
        (broken("lead_backing", choice="neither"), "is not an option"),
        (broken("lead_backing", probabilities={"both": 1.0}), "do not cover"),
        (broken("lead_backing", confidence=None), "not a probability"),
        (broken("bleed.score", score=3.5), "outside 0 to 3"),
        (
            broken("bleed.score", probabilities={"0": 0.9, "1": 0.5, "2": 0.1, "3": 0.0}),
            "sum to 1.500",
        ),
        ({"result": RESPONSE, "success": False, "errors": [{"message": "no"}]}, "reported failure"),
    ],
)
def test_a_response_that_does_not_answer_the_questions_is_an_error(payload, message):
    with pytest.raises(cl.ResponseError, match=message):
        cl.parse_answers(payload, QUESTIONS)


def test_every_call_is_logged_with_questions_probabilities_and_usage(tmp_path):
    client, transport, _ = make(tmp_path, (200, RESPONSE))
    result = client.ask("a01", STATE, QUESTIONS, [clip()])
    assert result["answers"]["present.guitar"]["probability"] == 0.91
    header, entries = cl.read_log(tmp_path / "log.jsonl")
    assert header["kind"] == "header"
    assert header["preregistration_commit"] == "abc123"
    assert header["model_id"] == "@cf/cloudflare/clef-omni"
    (entry,) = entries
    assert entry["request_id"] == "a01"
    assert entry["ok"] is True
    assert (entry["status"], entry["attempts"], entry["error"]) == (200, 1, None)
    assert list(entry["questions"]) == list(QUESTIONS)
    assert entry["questions"]["present.guitar"]["instructions"].startswith("Is a guitar")
    assert entry["answers"]["lead_backing"]["probabilities"]["both"] == 0.7
    assert entry["answers"]["bleed.score"]["probabilities"]["0"] == 0.7
    assert entry["usage"] == {"input_tokens": 431, "output_tokens": 0}
    assert entry["clips"] == [
        {"id": "a01", "sha256": clip().sha256, "bytes": 1000, "seconds": 30.0}
    ]
    assert entry["request_sha256"] == cl.request_key(transport.bodies[0])
    # The audio itself is never written to the log.
    raw = (tmp_path / "log.jsonl").read_text()
    assert base64.b64encode(bytes(1000)).decode() not in raw


def test_a_second_client_appends_to_the_log_without_a_second_header(tmp_path):
    first, _, _ = make(tmp_path, (200, RESPONSE))
    first.ask("a01", STATE, QUESTIONS)
    second, _, _ = make(tmp_path, (200, RESPONSE))
    second.ask("a02", STATE, QUESTIONS)
    header, entries = cl.read_log(tmp_path / "log.jsonl")
    assert header["kind"] == "header"
    assert [e["request_id"] for e in entries] == ["a01", "a02"]


def test_rate_limits_and_server_errors_are_retried_with_backoff(tmp_path):
    client, transport, waits = make(
        tmp_path, (429, {"errors": ["slow down"]}), TimeoutError("timed out"), (200, RESPONSE)
    )
    assert client.ask("a01", STATE, QUESTIONS)["usage"]["input_tokens"] == 431
    assert len(transport.bodies) == 3
    assert waits == [1.0, 2.0]
    _, (entry,) = cl.read_log(tmp_path / "log.jsonl")
    assert (entry["attempts"], entry["ok"]) == (3, True)


def test_failures_are_logged_and_raised(tmp_path):
    client, transport, waits = make(tmp_path, (401, {"errors": [{"message": "bad token"}]}))
    with pytest.raises(cl.ClefError, match="HTTP 401"):
        client.ask("a01", STATE, QUESTIONS)
    assert len(transport.bodies) == 1  # an authentication failure is not retried
    assert waits == []
    client, transport, _ = make(tmp_path, *[(503, "unavailable")] * 3)
    with pytest.raises(cl.ClefError, match="HTTP 503"):
        client.ask("a02", STATE, QUESTIONS)
    assert len(transport.bodies) == 3
    client, _, _ = make(tmp_path, (200, {"answers": {}}))
    with pytest.raises(cl.ResponseError, match="was not answered"):
        client.ask("a03", STATE, QUESTIONS)
    _, entries = cl.read_log(tmp_path / "log.jsonl")
    assert [(e["request_id"], e["ok"], e["status"], e["attempts"]) for e in entries] == [
        ("a01", False, 401, 1),
        ("a02", False, 503, 3),
        ("a03", False, 200, 1),
    ]
    assert all(e["answers"] is None and e["error"] for e in entries)


def test_saved_exchanges_replay_without_a_network(tmp_path):
    saved = tmp_path / "saved.jsonl"
    live = Scripted((200, RESPONSE))
    recording = cl.ClefClient(
        cl.RecordingTransport(live, saved), tmp_path / "live.jsonl", limits=LIMITS
    )
    first = recording.ask("a01", STATE, QUESTIONS, [clip()])
    replay = cl.ReplayTransport(saved)
    client = cl.ClefClient(replay, tmp_path / "replay.jsonl", limits=LIMITS)
    assert client.ask("a01", STATE, QUESTIONS, [clip()]) == first
    assert replay.calls == 1
    # A different clip is a different request: nothing saved answers it.
    with pytest.raises(cl.ClefError, match="no saved response"):
        client.ask("a02", STATE, QUESTIONS, [clip(size=999)])


def test_the_token_never_reaches_the_log_or_an_error(tmp_path, monkeypatch):
    secret = "cf-token-that-must-not-leak"
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", secret)
    transport = cl.HttpTransport.from_env(LIMITS)
    assert transport._url.endswith("/accounts/acct/ai/run/@cf/cloudflare/clef-omni")
    assert secret not in repr(vars(transport)["_url"])

    def offline(*args, **kwargs):
        raise OSError("network is unreachable")

    monkeypatch.setattr(cl.urllib.request, "urlopen", offline)
    client = cl.ClefClient(
        transport, tmp_path / "log.jsonl", limits=LIMITS, max_attempts=1, sleep=lambda s: None
    )
    with pytest.raises(cl.ClefError, match="network error") as caught:
        client.ask("a01", STATE, QUESTIONS, [clip()])
    assert secret not in str(caught.value)
    assert secret not in (tmp_path / "log.jsonl").read_text()


def test_missing_credentials_are_reported_plainly(monkeypatch):
    monkeypatch.setattr(cl, "get_env", lambda name: None)
    with pytest.raises(cl.ClefError, match="CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN"):
        cl.HttpTransport.from_env(LIMITS)
