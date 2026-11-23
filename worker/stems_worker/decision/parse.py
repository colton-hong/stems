"""Decision 1 of 3: turn a free-text request into a ``ParsedStemRequest`` (schema (a)).

With Jev: one yes/no question per stem, a choice for the file format, a choice for
quality against speed, and a yes/no for splitting lead from backing vocals. The answers
are probabilities; this module thresholds them and assembles the object.

Without Jev, or when Jev fails or is not confident enough, keyword rules read the text
instead. The rules know the stem names and their common synonyms, "all" and "everything",
exclusions ("everything except the drums", "no vocals"), "instrumental" and "acapella",
the file formats and a few words for quality and speed. A rule either matches or it does
not, so its probabilities are 1 and 0: that is the rule's certainty, not a calibrated one.
They read left to right: an exclusion placed after the thing it excludes ("the drums I
don't need") is not understood.

Either way, a request in which no stem is recognised is never guessed at: it comes back
with ``needs_review`` set and ``review_reason`` ``no_stem_detected``, for the user to
confirm (the report's rule: never silently assume all six).

A user who ticked checkboxes is not parsed at all; ``from_selection`` builds the same
object from the ticks.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from stems_common import schemas
from stems_common.settings import DecisionSettings
from stems_common.stems import PRODUCT_STEMS
from stems_worker.decision import jev
from stems_worker.decision.jev import JevClient
from stems_worker.decision.record import (
    LOW_CONFIDENCE,
    NOT_NEEDED,
    Asked,
    Decision,
    ask,
    record,
)

KIND = "parse"
MAX_TEXT = 2000
FORMATS = ("wav", "flac", "mp3_320", "mp3_192", "unspecified")
QUALITY_MODES = ("best_quality", "balanced", "fastest", "unspecified")
NO_STEM = "no_stem_detected"

STEM_DESCRIPTIONS = {
    "vocals": "the vocals (singing, voice, an acapella)",
    "drums": "the drums and percussion",
    "bass": "the bass (bass guitar or synth bass)",
    "guitar": "the guitars",
    "piano": "the piano or keyboards",
    "other": "the remaining instruments, anything that is not vocals, drums, bass, guitar "
    "or piano (synths, strings, brass, effects), often called 'other' or 'the rest'",
}


def questions() -> dict[str, dict[str, Any]]:
    """The questions sent with a request text. Their ids are part of the recordings."""
    asked: dict[str, dict[str, Any]] = {
        f"stem.{stem}": jev.noul(
            f"Does the user want {STEM_DESCRIPTIONS[stem]} as a separate stem?",
            yes="The request asks for it by name, or asks for all stems or everything.",
            no="The request does not ask for it, or asks for it to be left out.",
        )
        for stem in PRODUCT_STEMS
    }
    asked["format"] = jev.choice(
        "Which audio file format does the user ask for?",
        {
            "wav": "WAV, or uncompressed",
            "flac": "FLAC",
            "mp3_320": "MP3 at 320 kbps, or MP3 with no bit rate given",
            "mp3_192": "MP3 at 192 kbps, or a small MP3",
            "unspecified": "No format is mentioned",
        },
    )
    asked["quality"] = jev.choice(
        "Does the user ask for the best quality, for speed, or for a balance?",
        {
            "best_quality": "Quality matters most, even if it takes longer",
            "balanced": "A balance of quality and speed is asked for",
            "fastest": "Speed matters most",
            "unspecified": "Neither quality nor speed is mentioned",
        },
    )
    asked["vocal_split"] = jev.noul(
        "Does the user want the lead vocal and the backing vocals as two separate stems?",
        yes="The request mentions lead and backing vocals, harmonies or background vocals "
        "as separate things.",
        no="The request wants vocals as one stem, or does not mention vocals.",
    )
    return asked


def state(text: str) -> dict[str, Any]:
    return {"request_text": text, "available_stems": list(PRODUCT_STEMS)}


# --- keyword rules ------------------------------------------------------------------------

_STEM_WORDS = {
    "vocals": r"vocals?|vox|voices?|sing(?:er|ers|ing)|acapp?ella|a capp?ella",
    "drums": r"drums?|drum ?kit|percussion|kick|snare|hi-?hats?|cymbals?",
    "bass": r"bass(?:line)?",
    "guitar": r"guitars?",
    "piano": r"pianos?|keys|keyboards?",
    "other": r"others?|synths?|synthesi[sz]ers?|strings|brass|horns|pads",
}
# Checked before the single words, so "bass drum" is a drum and "bass guitar" is a bass.
_PHRASES: tuple[tuple[str, str, str], ...] = (
    ("stem", "other", r"everything else|all the rest|the rest|what(?:'s| is) left|remaining"),
    ("stem", "drums", r"bass drums?"),
    ("stem", "bass", r"bass guitars?"),
    ("only_vocals", "", r"acapp?ella|a capp?ella"),
    ("no_vocals", "", r"instrumentals?|karaoke|backing tracks?|minus[- ]one"),
    (
        "all",
        "",
        r"everything|all (?:six |6 |the |of the )?(?:stems|tracks|instruments|parts|layers)"
        r"|all six|all 6|all of (?:them|it)|every (?:single )?(?:stem|instrument|track|part)"
        r"|each (?:stem|instrument|track|part)|full (?:split|separation)|\ball\b",
    ),
    (
        "neg",
        "",
        r"except(?: for)?|apart from|other than|but not|not|no|without|minus|remove|removing"
        r"|exclud(?:e|ing)|drop|skip|leave out|leaving out|don'?t (?:want|need|include)|get rid of",
    ),
    ("pos", "", r"just|only|keep|give me|i want|i need|i'd like|with|include|including|plus"),
    ("list", "", r"and|or|&|\+"),
    ("break", "", r"[.;!?]|\bbut\b"),
    ("comma", "", r","),
)
_TOKENS = re.compile(
    "|".join(
        [
            f"(?P<p{i}>\\b(?:{pattern})\\b)"
            if kind not in ("break", "comma", "list")
            else f"(?P<p{i}>{pattern})"
            for i, (kind, _, pattern) in enumerate(_PHRASES)
        ]
        + [f"(?P<s_{stem}>\\b(?:{pattern})\\b)" for stem, pattern in _STEM_WORDS.items()]
    ),
    re.IGNORECASE,
)
_SPLIT = re.compile(
    r"\b(?:lead (?:and|&|\+|from) (?:the )?(?:backing|background)|backing vocals?"
    r"|background vocals?|backing singers?|harmon(?:y|ies)|lead vocals?|lead singer"
    r"|split (?:up )?the vocals?|vocal split)\b",
    re.IGNORECASE,
)
_BEST = re.compile(
    r"\b(?:(?:best|highest|high|max(?:imum)?|top|studio) (?:possible )?quality|best possible"
    r"|as clean as possible|cleanest|quality over speed)\b",
    re.IGNORECASE,
)
_FAST = re.compile(
    r"\b(?:fast(?:est)?|quick(?:ly|est)?|asap|in a hurry|rough (?:cut|version|split)|draft"
    r"|speed over quality)\b",
    re.IGNORECASE,
)
_BALANCED = re.compile(r"\bbalanced?\b", re.IGNORECASE)


@dataclass(frozen=True)
class RuleReading:
    """What the keyword rules found in a request."""

    wanted: tuple[str, ...]
    excluded: tuple[str, ...]
    output_format: str
    quality_mode: str
    vocal_split: str
    # Why the reading should be confirmed by the user; empty when it is clear.
    problems: tuple[str, ...] = field(default=())


def _tokens(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for match in _TOKENS.finditer(text):
        name = match.lastgroup or ""
        if name.startswith("s_"):
            found.append(("stem", name[2:]))
        else:
            kind, stem, _ = _PHRASES[int(name[1:])]
            found.append((kind, stem))
    return found


def _format(text: str) -> tuple[str, str | None]:
    lowered = text.lower()
    found: list[str] = []
    if re.search(r"\bflac\b", lowered):
        found.append("flac")
    if re.search(r"\bwave?\b|\buncompressed\b", lowered):
        found.append("wav")
    if re.search(r"\bmp3\b", lowered):
        found.append("mp3_192" if re.search(r"\b192\b", lowered) else "mp3_320")
    if len(found) > 1:
        return "unspecified", f"more than one file format is named ({', '.join(found)})"
    return (found[0] if found else "unspecified"), None


def read_keywords(text: str) -> RuleReading:
    """Read a request with the keyword rules. Deterministic, and needs nothing but the text."""
    positive: list[str] = []
    negative: list[str] = []
    everything = False
    negated = False
    tokens = _tokens(text)
    for index, (kind, stem) in enumerate(tokens):
        if kind == "neg":
            negated = True
        elif kind in ("pos", "break"):
            negated = False
        elif kind == "comma":
            # A comma ends an exclusion unless the list simply carries on after it.
            following = tokens[index + 1][0] if index + 1 < len(tokens) else None
            if following not in ("stem", "list"):
                negated = False
        elif kind == "all":
            everything = everything or not negated
        elif kind == "only_vocals":
            (negative if negated else positive).append("vocals")
        elif kind == "no_vocals":
            if not negated:
                negative.append("vocals")
                positive += [s for s in PRODUCT_STEMS if s != "vocals"]
        elif kind == "stem":
            (negative if negated else positive).append(stem)

    problems: list[str] = []
    clash = sorted(set(positive) & set(negative), key=PRODUCT_STEMS.index)
    if clash:
        problems.append(f"contradictory: {', '.join(clash)} both asked for and excluded")
    if everything or (negative and not positive):
        # "everything except the drums", and also a bare "no vocals".
        chosen = [s for s in PRODUCT_STEMS if s not in negative]
    else:
        chosen = [s for s in PRODUCT_STEMS if s in positive and s not in negative]
    output_format, format_problem = _format(text)
    if format_problem:
        problems.append(format_problem)
    best, fast = bool(_BEST.search(text)), bool(_FAST.search(text))
    if best and fast:
        quality = "unspecified"
        problems.append("both best quality and speed are asked for")
    else:
        quality = (
            "best_quality"
            if best
            else "fastest"
            if fast
            else "balanced"
            if _BALANCED.search(text)
            else "unspecified"
        )
    split = "lead_and_backing" if _SPLIT.search(text) and "vocals" in chosen else "unspecified"
    return RuleReading(
        wanted=tuple(chosen),
        excluded=tuple(s for s in PRODUCT_STEMS if s in negative),
        output_format=output_format,
        quality_mode=quality,
        vocal_split=split,
        problems=tuple(problems),
    )


# --- assembling the object ----------------------------------------------------------------


def _finish(parsed: dict[str, Any]) -> dict[str, Any]:
    schemas.validate("parsed_stem_request", parsed)
    return parsed


def _check_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("the request text is empty")
    if len(text) > MAX_TEXT:
        raise ValueError(f"the request text is longer than {MAX_TEXT} characters")
    return text


def _from_rules(
    text: str, request_id: str, settings: DecisionSettings, extra_reason: str | None = None
) -> dict[str, Any]:
    reading = read_keywords(text)
    reasons = [NO_STEM] if not reading.wanted else []
    reasons += reading.problems
    if extra_reason:
        reasons.append(extra_reason)
    parsed: dict[str, Any] = {
        "request_id": request_id,
        "raw_text": text,
        "stems": {
            stem: {
                "wanted": stem in reading.wanted,
                "probability": 1.0 if stem in reading.wanted else 0.0,
                "confidence": 1.0,
            }
            for stem in PRODUCT_STEMS
        },
        "vocal_split": reading.vocal_split,
        "output_format": reading.output_format,
        "quality_mode": reading.quality_mode,
        "wanted_threshold": settings.wanted_threshold,
        # A rule matched or it did not; a reading with a problem is not a confident one.
        "confidence": 0.0 if reasons else 1.0,
        "needs_review": bool(reasons),
    }
    if reasons:
        parsed["review_reason"] = "; ".join(reasons)[:500]
    return _finish(parsed)


def _from_answers(
    text: str, request_id: str, answers: dict[str, Any], settings: DecisionSettings
) -> tuple[dict[str, Any], list[str]]:
    """The object as Jev's answers give it, and the answers too unsure to act on."""
    floor = settings.min_confidence
    unsure: list[str] = []
    used: list[float] = []
    stems: dict[str, Any] = {}
    for stem in PRODUCT_STEMS:
        answer = answers[f"stem.{stem}"]
        stems[stem] = {
            "wanted": answer["probability"] >= settings.wanted_threshold,
            "probability": answer["probability"],
            "confidence": answer["confidence"],
        }
        used.append(answer["confidence"])
        if answer["confidence"] < floor:
            unsure.append(f"stem.{stem} {answer['confidence']:.2f}")
    chosen: dict[str, str] = {}
    for name in ("format", "quality"):
        answer = answers[name]
        # An unsure answer here costs nothing: the configured default is used instead.
        if answer["confidence"] >= floor:
            chosen[name] = answer["choice"]
            used.append(answer["confidence"])
        else:
            chosen[name] = "unspecified"
    split = answers["vocal_split"]
    vocal_split = "unspecified"
    if split["confidence"] >= floor and stems["vocals"]["wanted"]:
        vocal_split = "lead_and_backing" if split["probability"] >= 0.5 else "single_vocals"
        used.append(split["confidence"])
    nothing = not any(s["wanted"] for s in stems.values())
    parsed: dict[str, Any] = {
        "request_id": request_id,
        "raw_text": text,
        "stems": stems,
        "vocal_split": vocal_split,
        "output_format": chosen["format"],
        "quality_mode": chosen["quality"],
        "wanted_threshold": settings.wanted_threshold,
        "confidence": min(used),
        "needs_review": nothing or bool(unsure),
    }
    if nothing:
        parsed["review_reason"] = NO_STEM
    return parsed, unsure


def parse_request(
    text: str,
    *,
    client: JevClient | None,
    settings: DecisionSettings,
    request_id: str | None = None,
) -> Decision:
    """Parse a free-text request. Jev decides when it is there and sure; otherwise the rules."""
    text = _check_text(text)
    request_id = request_id or str(uuid.uuid4())
    asked = ask(client, KIND, state(text), questions())
    if asked.answers is None:
        value = _from_rules(text, request_id, settings)
        return Decision(KIND, value, record(KIND, asked, fallback_reason=asked.failure))
    parsed, unsure = _from_answers(text, request_id, asked.answers, settings)
    if unsure:
        reason = f"{LOW_CONFIDENCE}: {', '.join(unsure)} (minimum {settings.min_confidence:g})"
        value = _from_rules(
            text, request_id, settings, extra_reason="Jev was unsure of the request"
        )
        return Decision(KIND, value, record(KIND, asked, fallback_reason=reason))
    return Decision(KIND, _finish(parsed), record(KIND, asked, fallback_reason=None))


def from_selection(
    stems: tuple[str, ...] | list[str],
    *,
    settings: DecisionSettings,
    output_format: str = "unspecified",
    quality_mode: str = "unspecified",
    split_vocals: bool = False,
    request_id: str | None = None,
) -> Decision:
    """The same object for a user who ticked boxes. Nothing is asked and nothing is guessed."""
    unknown = [s for s in stems if s not in PRODUCT_STEMS]
    if unknown or not stems:
        raise ValueError(f"stems must be among {PRODUCT_STEMS}, got {list(stems)}")
    wanted = [s for s in PRODUCT_STEMS if s in stems]
    split = "unspecified"
    if "vocals" in wanted:
        split = "lead_and_backing" if split_vocals else "single_vocals"
    value = {
        "request_id": request_id or str(uuid.uuid4()),
        "raw_text": "selected: " + ", ".join(wanted),
        "stems": {
            stem: {
                "wanted": stem in wanted,
                "probability": float(stem in wanted),
                "confidence": 1.0,
            }
            for stem in PRODUCT_STEMS
        },
        "vocal_split": split,
        "output_format": output_format,
        "quality_mode": quality_mode,
        "wanted_threshold": settings.wanted_threshold,
        "confidence": 1.0,
        "needs_review": False,
    }
    asked = Asked(None, NOT_NEEDED, "selection")
    notes = ["checkboxes: nothing was parsed"]
    return Decision(KIND, _finish(value), record(KIND, asked, fallback_reason=None, notes=notes))


def wanted_stems(parsed: dict[str, Any]) -> tuple[str, ...]:
    return tuple(stem for stem in PRODUCT_STEMS if parsed["stems"][stem]["wanted"])
