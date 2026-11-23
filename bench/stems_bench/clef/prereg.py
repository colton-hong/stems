"""The Clef-omni test's pre-registration: everything decided before the first request.

Research report section 5 asks for the thresholds, the 50 track ids, the 30-second
windows and the class-to-stem mapping to be fixed before Clef-omni is run. This module
writes them, with the questions, the clip plan and the known answers, into one JSON file.
The file is committed; the test's log header carries the hash of that commit, and the
test refuses to run from a file that is not committed as it stands.

    python -m stems_bench.clef.prereg --freeze     # needs MOISESDB_PATH; written once

A frozen file is never rewritten. A change to the design is a new file with a new version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from stems_bench.clef import clips as c
from stems_bench.clef.clipset import make_plan
from stems_bench.datasets import Dataset, DatasetError, get_dataset, is_committed
from stems_common.levels import SILENCE_DBFS, SILENCE_WINDOW_SECONDS
from stems_common.paths import REPO_ROOT
from stems_common.stems import PRODUCT_STEMS
from stems_worker.listening.audioset import STEM_MAP_PATH, load_stem_map

VERSION = 1
SEED = 2307
DATASET = "moisesdb_subset"
PREREG_PATH = REPO_ROOT / "bench" / "preregistration" / "clef-omni-v1.json"
MODEL = "@cf/cloudflare/clef-omni"

# Report section 5, "Pass thresholds". Accuracies and rates are fractions.
THRESHOLDS: dict[str, Any] = {
    "A1": {
        "what": "presence accuracy over every mix decision (tracks x 6 stems)",
        "min_accuracy": 0.85,
    },
    "A2": {
        "what": "presence accuracy per stem",
        "min_accuracy": {
            "vocals": 0.80,
            "drums": 0.80,
            "bass": 0.80,
            "guitar": 0.80,
            "piano": 0.80,
            "other": 0.70,
        },
    },
    "A3": {"what": "vocals present or absent", "min_accuracy": 0.90},
    "B1": {
        "what": "clean against bleed, by the probability of the 'any other instrument' "
        "question, as area under the ROC curve",
        "min_auc_pooled": 0.85,
        "min_auc_hardest": 0.80,
        "hardest_sir_db": 30.0,
    },
    "B2": {
        "what": "Spearman correlation between the bleed score and the true bleed level "
        "(clean 0, +30 dB 1, +20 dB 2, +12 dB 3, +6 dB 4)",
        "min_spearman": 0.70,
    },
    "B3": {
        "what": "share of clean clips called bleeding: score at or above 1, or the "
        "'any other instrument' probability at or above 0.5",
        "max_false_positive_rate": 0.10,
    },
    "C": {
        "what": "Clef-omni's margin over BEATs on A1 (accuracy) and B1 (pooled AUC), "
        "in percentage points, on the same clips",
        "min_margin_points": 5.0,
    },
}

DECISION_RULE = (
    "Clef-omni passes for quality gating only if A1, A2, A3, B1, B2 and B3 all hold, and "
    "is preferred over BEATs only if C also holds on both A1 and B1. If it fails any of "
    "B1 to B3 but passes A1 to A3 it may be used for presence and routing only, never for "
    "gating. Otherwise BEATs is used, with CLAP as a second opinion."
)

PRESENCE_QUESTIONS = {
    "vocals": "Are vocals (singing, rapping or speech) audible in this clip?",
    "drums": "Are drums audible in this clip?",
    "bass": "Is a bass instrument (bass guitar, double bass or synth bass) audible in this clip?",
    "guitar": "Is a guitar audible in this clip?",
    "piano": "Is a piano audible in this clip?",
    "other": "Is any instrument other than vocals, drums, bass, guitar and piano audible "
    "in this clip?",
}
TARGET_NAMES = {
    "vocals": "vocals",
    "drums": "drums",
    "bass": "the bass",
    "guitar": "the guitar",
    "piano": "the piano",
    "other": "the named instrument",
}
BLEED_QUESTION = "Is any instrument other than {target} audible?"
BLEED_LEGEND = (
    "Clean — only the named instrument is audible",
    "Faint bleed — another instrument is just audible",
    "Audible bleed — another instrument is clearly audible",
    "Heavy bleed — another instrument is prominent",
)
BLEED_SCORE_QUESTION = (
    "This clip should contain only {target}. How much of another instrument is audible?"
)
LEAD_BACKING_QUESTION = "Which vocals are audible in this clip?"
REVERB_QUESTION = "Is there audible reverberation on {target} in this clip?"

# How the questions are put to Clef-omni. One clip per request, so no question has to say
# which clip it is about.
REQUEST: dict[str, Any] = {
    "state": "The attached audio clip is an excerpt of a music recording. "
    "Answer from what is audible in the clip.",
    "clips_per_request": 1,
    "A": {
        "clip": "the mix excerpt",
        "question_ids": {stem: f"present.{stem}" for stem in PRODUCT_STEMS},
    },
    "lead_backing": {"clip": "the isolated vocals excerpt", "question_ids": ["lead_backing"]},
    "B": {
        "clip": "each clean and each bleed clip, on its own",
        "question_ids": ["bleed.any", "bleed.score"],
    },
    "reverb": {
        "clip": "each reverb clip, and separately the clean clip it was made from",
        "question_ids": ["reverb"],
    },
}
AMENDMENTS = [
    {
        "date": "2026-11-23",
        "what": "Added clef_omni.request: the state text, one clip per request, and the "
        "question ids. Nothing else changed.",
        "before_any_request": True,
    }
]


class PreregError(RuntimeError):
    pass


def _counts(plan: dict[str, Any]) -> dict[str, Any]:
    tracks = plan["tracks"].values()
    present = {s: sum(1 for t in tracks if t["present"][s]) for s in PRODUCT_STEMS}
    answers = Counter(plan["tracks"][t]["lead_backing"] for t in plan["lead_backing"])
    return {
        "tracks": len(plan["tracks"]),
        "mix_decisions": len(plan["tracks"]) * len(PRODUCT_STEMS),
        "present": present,
        "absent": {s: len(plan["tracks"]) - n for s, n in present.items()},
        "lead_backing_clips": len(plan["lead_backing"]),
        "lead_backing_answers": {a: answers.get(a, 0) for a in c.LEAD_BACKING_ANSWERS},
        "part_b_clean": len(plan["part_b"]),
        "part_b_bleed": len(plan["part_b"]) * len(c.SIR_LEVELS_DB),
        "part_b_targets": dict(Counter(i["target"] for i in plan["part_b"])),
        "reverb_clips": len(plan["reverb"]),
    }


def build(
    dataset: Dataset, *, seed: int = SEED, created: str, **plan_options: Any
) -> dict[str, Any]:
    """The whole pre-registration for ``dataset``, as plain data."""
    stem_map = load_stem_map()
    plan = make_plan(dataset, seed, **plan_options)
    counts = _counts(plan)
    notes = [
        "The report asks for at least 8 instrumental tracks among the 50. MoisesDB has one "
        "track without vocals in 240, so the frozen subset has one (docs/corrections.md, "
        "C15).",
        f"As a result A3 rests on few negatives: vocals are absent from "
        f"{counts['absent']['vocals']} of {counts['tracks']} excerpts by the presence rule. "
        "A3 is still scored against its threshold as written; the report also gives the "
        "counts, and the accuracy of always answering 'present', so the reader can see how "
        "much A3 can show.",
        "Part B's five guitar-or-piano stems are taken as guitar, piano, guitar, piano, "
        "guitar. The reverb probe uses Part B's clean vocal stems.",
        "The lead-or-backing sub-test and the reverb probe are exploratory and are not "
        "part of the pass decision.",
    ]
    return {
        "version": VERSION,
        "title": "Clef-omni against BEATs on MoisesDB: pre-registration",
        "created": created,
        "source": "docs/research-report.pdf, section 5, with docs/corrections.md",
        "dataset": {
            "name": dataset.name,
            "frozen_list": dataset.frozen.path.name,
            "frozen_sha256": dataset.frozen.sha256,
        },
        "seed": seed,
        "excerpt": {
            "rule": "the loudest window of the mix by RMS; the same window for the mix "
            "and every stem of the track",
            "seconds": plan_options.get("window_seconds", c.WINDOW_SECONDS),
            "grid_seconds": c.GRID_SECONDS,
        },
        "presence_rule": {
            "rule": "a stem is present if its RMS over the excerpt is at least "
            "rms_dbfs_min and it is non-silent for at least sounding_seconds_min",
            "rms_dbfs_min": c.PRESENT_RMS_DBFS,
            "sounding_seconds_min": c.PRESENT_SECONDS,
            "silence_dbfs": SILENCE_DBFS,
            "silence_window_seconds": SILENCE_WINDOW_SECONDS,
        },
        "clip_format": {
            "sample_rate_hz": c.CLIP_SAMPLE_RATE,
            "channels": 1,
            "encoding": "WAV, 16-bit PCM; turned down only if it would clip",
            "max_bytes": c.MAX_CLIP_BYTES,
        },
        "class_to_stem": {
            "file": STEM_MAP_PATH.relative_to(REPO_ROOT).as_posix(),
            "version": stem_map.version,
            "sha256": stem_map.sha256,
            "labels_sha256": stem_map.labels.sha256,
        },
        "clef_omni": {
            "model": MODEL,
            "present_threshold": 0.5,
            "presence_questions": {"type": "noul", "per_stem": PRESENCE_QUESTIONS},
            "bleed_question": {"type": "noul", "text": BLEED_QUESTION},
            "bleed_score": {
                "type": "score",
                "text": BLEED_SCORE_QUESTION,
                "legend": list(BLEED_LEGEND),
            },
            "lead_backing": {
                "type": "choice",
                "text": LEAD_BACKING_QUESTION,
                "options": list(c.LEAD_BACKING_ANSWERS),
                "asked_on": "the isolated vocals stem of each lead_backing track",
            },
            "reverb": {"type": "noul", "text": REVERB_QUESTION},
            "target_names": TARGET_NAMES,
            "request": REQUEST,
        },
        "beats": {
            "windows": "three consecutive 10-second windows of the excerpt; class "
            "probabilities are averaged over them",
            "stem_score": "the largest probability among the stem's classes",
            "present_threshold": 0.5,
            "secondary": "also reported at the threshold with the best F1, labelled as "
            "chosen after the fact",
            "bleed_score": "the largest probability among the classes of every stem "
            "other than the target",
        },
        "bleed_levels": {
            "clean": 0,
            **{f"+{s:.0f} dB SIR": n + 1 for n, s in enumerate(c.SIR_LEVELS_DB)},
        },
        "reverb_wet_db": c.REVERB_WET_DB,
        "thresholds": THRESHOLDS,
        "decision_rule": DECISION_RULE,
        "logging": "every request and response is appended to a JSONL log with question "
        "ids, probabilities and token usage; the log header carries this file's commit",
        "counts": counts,
        "notes": notes,
        "cannot_show": [
            "anything above 8 kHz: both models hear 16 kHz audio",
            "stereo bleed: the clips are mono",
            "differences of a point or two between the models: 300 decisions give about "
            "4 points of precision on A1, hence the 5-point margin in C",
        ],
        "amendments": AMENDMENTS,
        "plan": plan,
    }


def write(data: dict[str, Any], path: Path) -> str:
    """Write a new pre-registration and return its SHA-256. Never overwrites."""
    if path.exists():
        raise PreregError(f"{path} exists; a pre-registration is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load(path: Path = PREREG_PATH) -> tuple[dict[str, Any], str]:
    """The committed pre-registration and the commit that last changed it.

    Raises if the file is missing, untracked or differs from what is committed: the test
    must not run from a design that is not on record.
    """
    if not path.is_file():
        raise PreregError(f"{path} does not exist; run python -m stems_bench.clef.prereg --freeze")
    if not is_committed(path):
        raise PreregError(f"{path} is not committed as it stands; commit it before any request")
    commit = subprocess.run(
        ["git", "-C", str(path.parent), "log", "-n", "1", "--format=%H", "--", path.name],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not commit:
        raise PreregError(f"no commit found for {path}")
    return json.loads(path.read_text(encoding="utf-8")), commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--freeze", action="store_true", help="write the pre-registration file")
    parser.add_argument("--created", help="date to record (default: today, UTC)")
    args = parser.parse_args(argv)
    if not args.freeze:
        try:
            data, commit = load()
        except PreregError as e:
            print(e, file=sys.stderr)
            return 1
        counts = data["counts"]
        print(f"pre-registration v{data['version']} ({data['created']}), commit {commit[:12]}")
        print(f"  {counts['tracks']} tracks, {counts['mix_decisions']} mix decisions")
        print(f"  present: {counts['present']}")
        print(f"  Part B: {counts['part_b_clean']} clean, {counts['part_b_bleed']} bleed clips")
        return 0
    from datetime import UTC, datetime

    try:
        dataset = get_dataset(DATASET)
        dataset.check_ready()
        data = build(dataset, created=args.created or datetime.now(UTC).date().isoformat())
        digest = write(data, PREREG_PATH)
    except (DatasetError, PreregError, c.ClipError) as e:
        print(e, file=sys.stderr)
        return 2
    print(f"wrote {PREREG_PATH.relative_to(REPO_ROOT)} (sha256 {digest[:12]})")
    print(json.dumps(data["counts"], indent=2))
    print("Commit it before any Clef-omni request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
