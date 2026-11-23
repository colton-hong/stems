"""The Clef-omni test end to end on the synthetic set, with stand-ins for both models.

The stand-ins answer from the manifest's known answers, so a correct pipeline scores them
perfectly; any slip in matching clips, questions and answers shows up as a wrong score.
"""

from __future__ import annotations

import base64
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from stems_bench.clef import client as cl
from stems_bench.clef import prereg as pr
from stems_bench.clef import run as clef_run
from stems_bench.datasets.synthetic import SyntheticDataset
from stems_common.stems import PRODUCT_STEMS
from stems_worker.listening.audioset import default_stem_map

OPTIONS = {
    "window_seconds": 8.0,
    "part_b_targets": ("vocals", "drums", "bass", "guitar", "piano"),
    "reverb_clips": 1,
}
BLEED_P = [0.05, 0.6, 0.7, 0.8, 0.9]
PART_OF = {"present": "A", "lead_backing": "lead_backing", "bleed": "B", "reverb": "reverb"}


class OracleClef:
    """Answers each request correctly, finding the clip by the audio it was sent."""

    def __init__(self, manifest, clips_dir, fail=()) -> None:
        # The same audio can be more than one clip (a clean vocal stem is also a
        # lead-or-backing clip), so clips are kept per audio and told apart by the question.
        self.by_audio: dict[str, list[dict]] = {}
        for c in manifest["clips"]:
            digest = hashlib.sha256((clips_dir / c["file"]).read_bytes()).hexdigest()
            self.by_audio.setdefault(digest, []).append(c)
        self.fail = set(fail)
        self.sent: list[str] = []

    def __call__(self, body):
        (audio,) = body["audio"]
        same = self.by_audio[hashlib.sha256(base64.b64decode(audio["base64"])).hexdigest()]
        part = PART_OF[next(iter(body["questions"])).split(".")[0]]
        clip = next((c for c in same if c["part"] == part), None) or next(
            c
            for c in same
            if c["part"] == "B"  # the reverb question on a clean clip
        )
        self.sent.append(clip["id"])
        if clip["id"] in self.fail:
            return 400, {"errors": ["refused"]}
        answers = {}
        for qid, question in body["questions"].items():
            if qid.startswith("present."):
                answers[qid] = {"noul": 0.95 if clip["present"][qid.split(".")[1]] else 0.05}
            elif qid == "lead_backing":
                options = list(question["criteria"])
                answers[qid] = {
                    "choice": clip["answer"],
                    "probabilities": {o: 1.0 if o == clip["answer"] else 0.0 for o in options},
                    "confidence": 1.0,
                }
            elif qid == "bleed.any":
                answers[qid] = {"noul": BLEED_P[clip["bleed_level"]]}
            elif qid == "bleed.score":
                level = min(clip["bleed_level"], 3)
                answers[qid] = {
                    "score": clip["bleed_level"] * 0.7,
                    "legend": {str(i): text for i, text in enumerate(question["criteria"])},
                    "probabilities": {str(i): 1.0 if i == level else 0.0 for i in range(4)},
                    "confidence": 1.0,
                }
            elif qid == "reverb":
                answers[qid] = {"noul": 0.8 if clip["part"] == "reverb" else 0.2}
        return 200, {"model": "clef-omni", "answers": answers,
                     "usage": {"input_tokens": 100, "output_tokens": 0}}  # fmt: skip


class OracleTagger:
    """Three windows of class probabilities that name exactly what the clip holds."""

    name = "oracle"
    window_seconds = 10.0

    def __init__(self, manifest, clips_dir) -> None:
        self.by_audio = {}
        for c in manifest["clips"]:
            if c["part"] not in ("A", "B"):
                continue
            audio = sf.read(clips_dir / c["file"], dtype="float32", always_2d=True)[0].T
            self.by_audio[hashlib.sha256(audio.tobytes()).hexdigest()] = c
        stem_map = default_stem_map()
        self.first_class = {s: stem_map.stems[s][0] for s in PRODUCT_STEMS}

    def class_probabilities(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        assert sample_rate == 16000
        clip = self.by_audio[hashlib.sha256(audio.tobytes()).hexdigest()]
        row = np.zeros(527)
        if clip["part"] == "A":
            for stem, present in clip["present"].items():
                row[self.first_class[stem]] = 0.9 if present else 0.1
        else:
            row[self.first_class[clip["target"]]] = 0.9
            row[self.first_class[clip["contaminant"]]] = BLEED_P[clip["bleed_level"]]
        # Uneven windows with the right mean: the mean over windows is what counts.
        return np.stack([row * 0.5, row, row * 1.5])


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    dataset = SyntheticDataset()
    prereg = pr.build(dataset, created="2026-11-22", **OPTIONS)
    clips_dir = tmp_path_factory.mktemp("clips")
    manifest = clef_run.ensure_clips(dataset, prereg, clips_dir)
    return dataset, prereg, clips_dir, manifest


def run(setup, run_dir, **kwargs):
    dataset, prereg, clips_dir, _ = setup
    return clef_run.run_test(dataset, prereg, "c0ffee" * 6, run_dir, clips_dir=clips_dir, **kwargs)


def test_both_models_answering_correctly_pass_everything_they_can(setup, tmp_path):
    _, prereg, clips_dir, manifest = setup
    transport = OracleClef(manifest, clips_dir)
    scores = run(setup, tmp_path / "r1", transport=transport,
                 tagger=OracleTagger(manifest, clips_dir))  # fmt: skip
    clef, beats = scores["models"]["clef_omni"], scores["models"]["beats"]
    assert scores["expected"] == {"part_a": 48, "part_b": 25}
    for model in (clef, beats):
        assert model["complete"] is True
        assert model["A1"]["value"] == 1.0
        assert model["A1"]["n"] == 48
        assert model["A2"]["pass"] is True
        assert model["B1"]["auc_pooled"] == 1.0
        assert model["B1"]["auc_hardest"] == 1.0
        assert model["B3"]["value"] == 0.0
        assert model["B2"]["value"] > 0.9
        assert model["passes_b"] is True
    assert clef["B2"]["rated_by"] == "bleed_score"
    assert beats["B2"]["rated_by"] == "bleed_probability"
    assert clef["exploratory"]["lead_backing"]["accuracy"] == 1.0
    assert clef["exploratory"]["lead_backing"]["n"] == len(prereg["plan"]["lead_backing"])
    assert clef["exploratory"]["reverb"] == {"wet": [0.8], "clean": [0.2], "auc": 1.0}
    # One request per clip, plus the clean counterpart of the reverb clip.
    assert len(transport.sent) == len(manifest["clips"]) + 1
    sent = len(transport.sent)
    assert clef["usage"] == {
        "requests": sent,
        "answered": sent,
        "input_tokens": 100 * sent,
        "output_tokens": 0,
    }
    # Equal scores: no margin over BEATs, so BEATs is kept.
    assert scores["C"]["pass"] is False
    assert scores["verdict"]["listening_layer"].startswith("BEATs")
    header, entries = cl.read_log(tmp_path / "r1" / "clef-log.jsonl")
    assert header["preregistration_commit"] == "c0ffee" * 6
    assert all(e["state"] == prereg["clef_omni"]["request"]["state"] for e in entries)
    saved = json.loads((tmp_path / "r1" / "scores.json").read_text())
    assert saved["models"]["beats"]["tagger"] == "oracle"
    info = json.loads((tmp_path / "r1" / "run.json").read_text())
    assert info["ran"] == ["beats", "clef_omni"]
    assert info["preregistration"]["plan_sha256"] == clef_run.plan_sha(prereg)
    rows = (tmp_path / "r1" / "tagger-scores.jsonl").read_text().splitlines()
    assert len(rows) == 8 + 25
    assert json.loads(rows[0])["windows"] == 3


def test_questions_are_the_preregistered_wording(setup):
    _, prereg, _, manifest = setup
    clips = {c["part"]: c for c in reversed(manifest["clips"])}
    ((_, a),) = clef_run.requests_for(clips["A"], prereg)
    assert list(a) == [f"present.{s}" for s in PRODUCT_STEMS]
    assert a["present.piano"]["instructions"] == pr.PRESENCE_QUESTIONS["piano"]
    ((_, b),) = clef_run.requests_for(clips["B"], prereg)
    target = pr.TARGET_NAMES[clips["B"]["target"]]
    assert b["bleed.any"]["instructions"] == pr.BLEED_QUESTION.format(target=target)
    assert b["bleed.score"]["criteria"] == list(pr.BLEED_LEGEND)
    ((_, r),) = clef_run.requests_for(clips["reverb"], prereg)
    assert r["reverb"]["instructions"] == pr.REVERB_QUESTION.format(target="vocals")


def test_a_run_is_resumed_without_asking_again(setup, tmp_path):
    _, _, clips_dir, manifest = setup
    first = OracleClef(manifest, clips_dir, fail={"a03", "b02-sir12"})
    partial = run(setup, tmp_path / "r", transport=first)
    assert partial["models"]["clef_omni"]["complete"] is False
    assert partial["models"]["clef_omni"]["A1"]["n"] == 42
    assert "verdict" not in partial
    assert "C" not in partial
    second = OracleClef(manifest, clips_dir)
    complete = run(setup, tmp_path / "r", transport=second)
    assert sorted(second.sent) == ["a03", "b02-sir12"]
    assert complete["models"]["clef_omni"]["complete"] is True
    assert complete["models"]["clef_omni"]["A1"]["n"] == 48
    third = OracleClef(manifest, clips_dir)
    run(setup, tmp_path / "r", transport=third)
    assert third.sent == []


def test_the_report_states_each_threshold_its_value_and_the_verdict(setup, tmp_path):
    _, _, clips_dir, manifest = setup
    run(setup, tmp_path / "full", transport=OracleClef(manifest, clips_dir),
        tagger=OracleTagger(manifest, clips_dir))  # fmt: skip
    text = clef_run.render(tmp_path / "full")
    assert "## Clef-omni" in text
    assert "## BEATs" in text
    assert "| A1 presence accuracy, 48 decisions | 100.0% | 85.0% | pass |" in text
    assert "| B3 clean clips called bleeding (0 of 5) | 0.0% | at most 10.0% | pass |" in text
    assert "Margin on A1: 0.0 points." in text
    assert "Clef-omni: **passes, but BEATs preferred**" in text
    assert "commit `c0ffeec0ffee`" in text
    run(setup, tmp_path / "half", tagger=OracleTagger(manifest, clips_dir),
        skipped={"clef_omni": "no token"})  # fmt: skip
    half = clef_run.render(tmp_path / "half")
    assert "**Clef-omni did not run:** no token" in half
    assert "No verdict: it needs complete results for both" in half
    assert "## Clef-omni\n" not in half


def test_a_run_refuses_inputs_that_are_not_the_preregistered_ones(setup, tmp_path):
    dataset, prereg, clips_dir, _ = setup
    changed = {**prereg, "class_to_stem": {**prereg["class_to_stem"], "sha256": "0" * 64}}
    with pytest.raises(pr.PreregError, match="class-to-stem mapping"):
        clef_run.run_test(dataset, changed, "abc", tmp_path / "x", clips_dir=clips_dir)
    changed = {**prereg, "dataset": {**prereg["dataset"], "frozen_sha256": "0" * 64}}
    with pytest.raises(pr.PreregError, match="frozen track list"):
        clef_run.run_test(dataset, changed, "abc", tmp_path / "y", clips_dir=clips_dir)


def test_a_damaged_clip_set_is_rebuilt(setup, tmp_path):
    dataset, prereg, _, manifest = setup
    clips_dir = tmp_path / "clips"
    built = clef_run.ensure_clips(dataset, prereg, clips_dir)
    assert [c["sha256"] for c in built["clips"]] == [c["sha256"] for c in manifest["clips"]]
    victim = clips_dir / built["clips"][0]["file"]
    victim.write_bytes(b"not audio")
    clef_run.ensure_clips(dataset, prereg, clips_dir)
    assert hashlib.sha256(victim.read_bytes()).hexdigest() == built["clips"][0]["sha256"]


def test_the_command_reports_what_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(clef_run, "CLEF_DIR", tmp_path)
    assert clef_run.main(["--report", str(tmp_path / "r.md")]) == 2
    assert "has not been run" in capsys.readouterr().err
