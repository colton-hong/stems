"""Request parsing: Jev's answers mapped to a ParsedStemRequest, and the keyword rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stems_common import schemas
from stems_common.settings import load_defaults
from stems_common.stems import PRODUCT_STEMS
from stems_worker.decision import parse
from stems_worker.decision.jev import JevClient, Reply, build_request
from stems_worker.decision.mock import (
    ReplayTransport,
    UnrecordedRequest,
    handwritten,
    save_exchange,
)
from stems_worker.decision.parse import from_selection, parse_request, read_keywords, wanted_stems

SETTINGS = load_defaults().decision
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "jev" / "parse.handwritten.json").read_text()
)
CASES = FIXTURE["cases"]
ID = "7b0c9a52-13f6-4c0e-8a55-0c1c5f1f8f10"
ALL = PRODUCT_STEMS
NO_VOCALS = tuple(s for s in ALL if s != "vocals")


@pytest.fixture
def client(tmp_path):
    """A client that replays the hand-written answers for the fixture's request texts."""
    path = tmp_path / "parse.jsonl"
    for case in CASES.values():
        body = build_request(SETTINGS.jev_model, parse.state(case["text"]), parse.questions())
        save_exchange(path, handwritten(body, case["answers"]))
    return JevClient(ReplayTransport(path), SETTINGS, provider="mock", sleep=lambda s: None)


def ask(client, name: str):
    return parse_request(CASES[name]["text"], client=client, settings=SETTINGS, request_id=ID)


# --- Jev's answers ------------------------------------------------------------------------


def test_the_fixture_says_what_it_is() -> None:
    assert FIXTURE["origin"] == "handwritten"
    assert "not recordings of Jev" in FIXTURE["about"]


def test_answers_become_a_valid_parsed_request(client) -> None:
    decision = ask(client, "vocals_drums_flac")
    parsed = decision.value
    assert schemas.errors("parsed_stem_request", parsed) == []
    assert wanted_stems(parsed) == ("vocals", "drums")
    assert parsed["request_id"] == ID
    assert parsed["raw_text"] == CASES["vocals_drums_flac"]["text"]
    assert parsed["stems"]["vocals"] == {"wanted": True, "probability": 0.97, "confidence": 0.97}
    # A confident no: probability 0.02, confidence max(p, 1 - p) = 0.98.
    assert parsed["stems"]["bass"] == {"wanted": False, "probability": 0.02, "confidence": 0.98}
    assert parsed["output_format"] == "flac"
    assert parsed["quality_mode"] == "unspecified"
    assert parsed["vocal_split"] == "single_vocals"  # a confident no, and vocals are wanted
    assert parsed["wanted_threshold"] == SETTINGS.wanted_threshold == 0.7
    # The lowest confidence among the answers used: the quality choice, at 0.9.
    assert parsed["confidence"] == pytest.approx(0.9)
    assert parsed["needs_review"] is False
    assert "review_reason" not in parsed

    log = decision.record
    assert decision.kind == log["kind"] == "parse"
    assert decision.provider == log["provider"] == "mock"
    assert log["asked"] == "mock"
    assert log["model"] == SETTINGS.jev_model
    assert log["fallback_reason"] is None
    assert log["input_tokens"] == 0
    assert log["answers"]["stem.vocals"]["probability"] == 0.97
    assert log["answers"]["format"]["probabilities"]["flac"] == 0.93


def test_everything_and_quality(client) -> None:
    parsed = ask(client, "everything_best").value
    assert wanted_stems(parsed) == ALL
    assert parsed["quality_mode"] == "best_quality"
    assert parsed["output_format"] == "unspecified"
    assert parsed["vocal_split"] == "single_vocals"
    assert parsed["confidence"] == pytest.approx(0.9)
    assert parsed["needs_review"] is False


def test_lead_and_backing_split(client) -> None:
    parsed = ask(client, "lead_backing").value
    assert wanted_stems(parsed) == ("vocals",)
    assert parsed["vocal_split"] == "lead_and_backing"
    assert parsed["needs_review"] is False


def test_a_stem_is_wanted_from_the_threshold_up(client) -> None:
    parsed = ask(client, "threshold_edge").value
    assert wanted_stems(parsed) == ("bass", "guitar")
    assert parsed["stems"]["guitar"]["confidence"] == pytest.approx(0.81)
    assert parsed["vocal_split"] == "unspecified"  # no vocals, so nothing to split
    stricter = SETTINGS.model_copy(update={"wanted_threshold": 0.9})
    again = parse_request(CASES["threshold_edge"]["text"], client=client, settings=stricter)
    assert wanted_stems(again.value) == ("bass",)
    assert again.value["wanted_threshold"] == 0.9


def test_no_stem_detected_needs_review_and_assumes_nothing(client) -> None:
    decision = ask(client, "no_stem")
    parsed = decision.value
    assert wanted_stems(parsed) == ()
    assert parsed["needs_review"] is True
    assert parsed["review_reason"] == "no_stem_detected"
    assert decision.provider == "mock"  # Jev was sure; there is simply nothing asked for
    # The unsure quality answer (0.6) is not acted on, and does not lower the confidence.
    assert parsed["quality_mode"] == "unspecified"
    assert parsed["confidence"] == pytest.approx(0.94)
    assert schemas.errors("parsed_stem_request", parsed) == []


def test_an_unsure_secondary_answer_falls_back_to_unspecified(client) -> None:
    decision = ask(client, "unsure_format")
    assert decision.provider == "mock"
    assert wanted_stems(decision.value) == ("drums",)
    assert decision.value["output_format"] == "unspecified"
    assert decision.value["needs_review"] is False


def test_unsure_stem_answers_hand_the_request_to_the_rules(client) -> None:
    decision = ask(client, "unsure_rhythm")
    parsed, log = decision.value, decision.record
    assert decision.provider == "fallback"
    assert log["asked"] == "mock"
    assert log["fallback_reason"] == "low_confidence: stem.drums 0.55, stem.bass 0.50 (minimum 0.8)"
    # Jev's answers stay in the log even though they were not used.
    assert log["answers"]["stem.drums"]["probability"] == 0.55
    assert log["model"] == SETTINGS.jev_model
    # The rules find "vocals" in the text; the user is asked to confirm.
    assert wanted_stems(parsed) == ("vocals",)
    assert parsed["stems"]["vocals"] == {"wanted": True, "probability": 1.0, "confidence": 1.0}
    assert parsed["needs_review"] is True
    assert "Jev was unsure" in parsed["review_reason"]
    assert schemas.errors("parsed_stem_request", parsed) == []


def test_without_a_client_the_rules_answer(client) -> None:
    decision = parse_request("drums and bass as wav", client=None, settings=SETTINGS)
    assert decision.provider == "fallback"
    assert decision.record["fallback_reason"] == "no_key"
    assert decision.record["asked"] is None
    assert decision.record["model"] is None
    assert decision.record["answers"] is None
    assert wanted_stems(decision.value) == ("drums", "bass")
    assert decision.value["output_format"] == "wav"
    assert decision.value["needs_review"] is False
    assert decision.value["confidence"] == 1.0
    assert schemas.errors("parsed_stem_request", decision.value) == []


def test_a_failing_jev_hands_over_to_the_rules() -> None:
    class Down:
        def __call__(self, body, timeout):
            return Reply(503, "unavailable")

    failing = JevClient(Down(), SETTINGS, sleep=lambda s: None)
    decision = parse_request("just the piano", client=failing, settings=SETTINGS)
    assert decision.provider == "fallback"
    assert decision.record["asked"] == "jev"
    assert decision.record["fallback_reason"].startswith("jev_failed: HTTP 503")
    assert wanted_stems(decision.value) == ("piano",)
    assert len(failing.calls) == 1
    assert len(failing.calls[0]["attempts"]) == SETTINGS.max_attempts


def test_an_unrecorded_request_is_not_covered_by_the_rules(client) -> None:
    with pytest.raises(UnrecordedRequest):
        parse_request("a text nobody recorded", client=client, settings=SETTINGS)


def test_the_request_sent_to_jev_has_one_question_per_decision() -> None:
    asked = parse.questions()
    assert list(asked) == [*(f"stem.{s}" for s in ALL), "format", "quality", "vocal_split"]
    assert all(asked[f"stem.{s}"]["type"] == "noul" for s in ALL)
    # Criteria keys are the schema's enum values, so mapping is a direct lookup.
    schema = schemas.load("parsed_stem_request")["properties"]
    assert list(asked["format"]["criteria"]) == schema["output_format"]["enum"]
    assert list(asked["quality"]["criteria"]) == schema["quality_mode"]["enum"]
    assert parse.state("hi") == {"request_text": "hi", "available_stems": list(ALL)}


@pytest.mark.parametrize("text", ["", "   ", "x" * 2001])
def test_empty_or_oversized_text_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="request text"):
        parse_request(text, client=None, settings=SETTINGS)


# --- the keyword rules, no network ----------------------------------------------------------

PHRASES = [
    # (text, stems, format, quality, vocal split)
    ("vocals", ("vocals",), "unspecified", "unspecified", "unspecified"),
    (
        "Just the vocals and drums please",
        ("vocals", "drums"),
        "unspecified",
        "unspecified",
        "unspecified",
    ),
    ("all stems", ALL, "unspecified", "unspecified", "unspecified"),
    ("Everything, please!", ALL, "unspecified", "unspecified", "unspecified"),
    ("split it into all six", ALL, "unspecified", "unspecified", "unspecified"),
    ("every instrument on its own track", ALL, "unspecified", "unspecified", "unspecified"),
    (
        "everything except the drums",
        ("vocals", "bass", "guitar", "piano", "other"),
        "unspecified",
        "unspecified",
        "unspecified",
    ),
    (
        "all stems but no piano, as FLAC",
        ("vocals", "drums", "bass", "guitar", "other"),
        "flac",
        "unspecified",
        "unspecified",
    ),
    (
        "everything except drums, bass and piano",
        ("vocals", "guitar", "other"),
        "unspecified",
        "unspecified",
        "unspecified",
    ),
    (
        "no vocals, just drums and bass",
        ("drums", "bass"),
        "unspecified",
        "unspecified",
        "unspecified",
    ),
    ("acapella", ("vocals",), "unspecified", "unspecified", "unspecified"),
    ("an a cappella version in wav", ("vocals",), "wav", "unspecified", "unspecified"),
    ("instrumental", NO_VOCALS, "unspecified", "unspecified", "unspecified"),
    ("a karaoke version", NO_VOCALS, "unspecified", "unspecified", "unspecified"),
    ("remove the vocals", NO_VOCALS, "unspecified", "unspecified", "unspecified"),
    (
        "without the drums",
        ("vocals", "bass", "guitar", "piano", "other"),
        "unspecified",
        "unspecified",
        "unspecified",
    ),
    ("bass and guitar, mp3", ("bass", "guitar"), "mp3_320", "unspecified", "unspecified"),
    ("drums in mp3 192", ("drums",), "mp3_192", "unspecified", "unspecified"),
    ("isolate the piano, best quality", ("piano",), "unspecified", "best_quality", "unspecified"),
    ("a quick drum stem", ("drums",), "unspecified", "fastest", "unspecified"),
    ("balanced quality, guitars only", ("guitar",), "unspecified", "balanced", "unspecified"),
    (
        "lead and backing vocals separately",
        ("vocals",),
        "unspecified",
        "unspecified",
        "lead_and_backing",
    ),
    (
        "the vocals with the harmonies split off",
        ("vocals",),
        "unspecified",
        "unspecified",
        "lead_and_backing",
    ),
    ("keys and synths", ("piano", "other"), "unspecified", "unspecified", "unspecified"),
    ("the bass guitar", ("bass",), "unspecified", "unspecified", "unspecified"),
    ("bass drum and snare", ("drums",), "unspecified", "unspecified", "unspecified"),
    ("VOCALS + DRUMS", ("vocals", "drums"), "unspecified", "unspecified", "unspecified"),
    (
        "the singer and everything else",
        ("vocals", "other"),
        "unspecified",
        "unspecified",
        "unspecified",
    ),
]


@pytest.mark.parametrize(("text", "stems", "fmt", "quality", "split"), PHRASES)
def test_the_rules_read_the_test_phrases(text, stems, fmt, quality, split) -> None:
    reading = read_keywords(text)
    assert reading.wanted == stems
    assert (reading.output_format, reading.quality_mode, reading.vocal_split) == (
        fmt,
        quality,
        split,
    )
    assert reading.problems == ()
    decision = parse_request(text, client=None, settings=SETTINGS, request_id=ID)
    parsed = decision.value
    assert schemas.errors("parsed_stem_request", parsed) == []
    assert wanted_stems(parsed) == stems
    assert parsed["needs_review"] is False
    assert all(s["probability"] in (0.0, 1.0) for s in parsed["stems"].values())
    assert decision.provider == "fallback"


@pytest.mark.parametrize("text", ["Make it sound better", "hello?", "can you master this", "42"])
def test_the_rules_never_guess_when_no_stem_is_named(text: str) -> None:
    parsed = parse_request(text, client=None, settings=SETTINGS).value
    assert wanted_stems(parsed) == ()
    assert parsed["needs_review"] is True
    assert parsed["review_reason"] == "no_stem_detected"
    assert parsed["confidence"] == 0.0


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("vocals, but no vocals", "contradictory: vocals both asked for and excluded"),
        ("drums as wav or flac", "more than one file format"),
        ("bass, fast, best quality", "both best quality and speed"),
    ],
)
def test_an_unclear_request_is_flagged_for_review(text: str, problem: str) -> None:
    reading = read_keywords(text)
    assert any(problem in p for p in reading.problems)
    parsed = parse_request(text, client=None, settings=SETTINGS).value
    assert parsed["needs_review"] is True
    assert problem in parsed["review_reason"]


def test_what_the_rules_get_wrong() -> None:
    # Known limit, pinned so that it is visible: an exclusion that comes after what it
    # excludes is not understood. "the rest" is read as the "other" stem. Jev is asked
    # first whenever there is a key; with the rules alone the user sees this parse in the
    # confirmation step only if something else flags it.
    assert read_keywords("percussion; the rest I don't need").wanted == ("drums", "other")


def test_the_rules_are_deterministic() -> None:
    text = "everything except the drums, as flac, as fast as you can"
    assert read_keywords(text) == read_keywords(text)
    first = parse_request(text, client=None, settings=SETTINGS, request_id=ID)
    assert first.value == parse_request(text, client=None, settings=SETTINGS, request_id=ID).value


# --- checkboxes ---------------------------------------------------------------------------


def test_checkboxes_are_not_parsed() -> None:
    decision = from_selection(
        ["drums", "vocals"], settings=SETTINGS, split_vocals=True, output_format="flac"
    )
    parsed = decision.value
    assert schemas.errors("parsed_stem_request", parsed) == []
    assert wanted_stems(parsed) == ("vocals", "drums")
    assert parsed["vocal_split"] == "lead_and_backing"
    assert parsed["output_format"] == "flac"
    assert parsed["needs_review"] is False
    assert decision.provider == "selection"
    assert decision.record["answers"] is None
    assert from_selection(["bass"], settings=SETTINGS).value["vocal_split"] == "unspecified"
    for bad in ([], ["strings"]):
        with pytest.raises(ValueError, match="stems must be among"):
            from_selection(bad, settings=SETTINGS)
