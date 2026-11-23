"""Writes tests/fixtures/jev/parse.handwritten.json from a compact description."""

import json
import pathlib
import sys

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
FORMATS = ("wav", "flac", "mp3_320", "mp3_192", "unspecified")
QUALITY = ("best_quality", "balanced", "fastest", "unspecified")


def choice(options, picked, confidence, p=None):
    p = confidence if p is None else p
    rest = round((1 - p) / (len(options) - 1), 4)
    probabilities = {o: (p if o == picked else rest) for o in options}
    return {
        "type": "choice",
        "choice": picked,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def case(text, stems, fmt, quality, split):
    answers = {f"stem.{s}": {"type": "noul", "noul": stems.get(s, 0.03)} for s in STEMS}
    answers["format"] = choice(FORMATS, *fmt)
    answers["quality"] = choice(QUALITY, *quality)
    answers["vocal_split"] = {"type": "noul", "noul": split}
    return {"text": text, "answers": answers}


CASES = {
    "vocals_drums_flac": case(
        "Can I get just the vocals and the drums, as FLAC?",
        {"vocals": 0.97, "drums": 0.96, "bass": 0.02, "guitar": 0.04},
        ("flac", 0.93),
        ("unspecified", 0.9),
        0.03,
    ),
    "everything_best": case(
        "Split everything out at the best quality you can",
        dict.fromkeys(STEMS, 0.95),
        ("unspecified", 0.97),
        ("best_quality", 0.92),
        0.1,
    ),
    "lead_backing": case(
        "I need the lead vocal and the backing vocals separately",
        {"vocals": 0.98},
        ("unspecified", 0.95),
        ("unspecified", 0.91),
        0.94,
    ),
    "unsure_rhythm": case(
        "vocals and maybe some of the rhythm section?",
        {"vocals": 0.95, "drums": 0.55, "bass": 0.5},
        ("unspecified", 0.96),
        ("unspecified", 0.9),
        0.05,
    ),
    "no_stem": case("Make it sound better", {}, ("unspecified", 0.94), ("best_quality", 0.6), 0.02),
    "unsure_format": case(
        "drums, compressed is fine", {"drums": 0.97}, ("mp3_320", 0.55), ("unspecified", 0.88), 0.01
    ),
    "threshold_edge": case(
        "the bass, and I guess the guitar too",
        {"bass": 0.99, "guitar": 0.81},
        ("unspecified", 0.9),
        ("unspecified", 0.9),
        0.02,
    ),
}
out = {
    "about": "Hand-written answers in Jev's wire format, for tests of how answers are mapped "
    "to a ParsedStemRequest. They are not recordings of Jev and say nothing about what Jev "
    "would answer; real recordings come with P4.9. The request body is built by the test "
    "from the current question wording.",
    "origin": "handwritten",
    "cases": CASES,
}
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(path, len(CASES))
