"""Run the pre-registered Clef-omni test, for Clef-omni, for BEATs, or for both.

    python -m stems_bench.clef.run                 # both, as far as keys and weights allow
    python -m stems_bench.clef.run --only beats
    python -m stems_bench.clef.run --report docs/clef-omni-test.md

Everything is driven by the committed pre-registration file: the clips are cut from its
plan, the questions are its wording, the thresholds are its numbers. A run folder under
``runs/clef_test/<run-id>/`` holds ``run.json`` (what ran, from which commit), the request
log, the per-clip tagger scores and ``scores.json``. Requests already answered in the log
are not sent again, so an interrupted run is resumed by running it again.

Clef-omni needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN; the tagger needs the BEATs
checkpoint. Whichever is missing is skipped and named in ``run.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stems_bench.clef import client as cl
from stems_bench.clef import prereg as pr
from stems_bench.clef import scoring
from stems_bench.clef.clips import ClipError
from stems_bench.clef.clipset import build_clips
from stems_bench.datasets import Dataset, DatasetError, get_dataset
from stems_bench.run import git_state, write_json
from stems_common.paths import RUNS_DIR
from stems_common.stems import PRODUCT_STEMS
from stems_worker.listening.audioset import StemMap, load_stem_map

CLEF_DIR = RUNS_DIR / "clef_test"


def plan_sha(prereg: Mapping[str, Any]) -> str:
    raw = json.dumps(prereg["plan"], sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def ensure_clips(dataset: Dataset, prereg: Mapping[str, Any], clips_dir: Path) -> dict[str, Any]:
    """The clip set for this plan: built once, then reused if every file is intact."""
    manifest_path = clips_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        intact = all(
            (clips_dir / c["file"]).is_file()
            and hashlib.sha256((clips_dir / c["file"]).read_bytes()).hexdigest() == c["sha256"]
            for c in manifest["clips"]
        )
        if intact:
            return manifest
    return build_clips(dataset, dict(prereg["plan"]), clips_dir)


def requests_for(clip: Mapping[str, Any], prereg: Mapping[str, Any]) -> list[tuple[str, dict]]:
    """The requests the pre-registration asks about one clip: (request id, questions)."""
    spec = prereg["clef_omni"]
    names = spec["target_names"]
    part = clip["part"]
    if part == "A":
        ids = spec["request"]["A"]["question_ids"]
        asked = spec["presence_questions"]["per_stem"]
        return [(clip["id"], {ids[s]: cl.noul(asked[s]) for s in PRODUCT_STEMS})]
    if part == "lead_backing":
        options = dict.fromkeys(spec["lead_backing"]["options"])
        return [(clip["id"], {"lead_backing": cl.choice(spec["lead_backing"]["text"], options)})]
    target = names[clip["target"]]
    reverb = {"reverb": cl.noul(spec["reverb"]["text"].format(target=target))}
    if part == "reverb":
        return [(clip["id"], reverb)]
    bleed = {
        "bleed.any": cl.noul(spec["bleed_question"]["text"].format(target=target)),
        "bleed.score": cl.score(
            spec["bleed_score"]["text"].format(target=target), spec["bleed_score"]["legend"]
        ),
    }
    return [(clip["id"], bleed)]


def reverb_pairs(manifest: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, str]:
    """Each reverb clip's id mapped to the id of the clean clip it was made from."""
    clean = {
        (c["track_id"], c["target"]): c["id"]
        for c in manifest["clips"]
        if c["part"] == "B" and c["bleed_level"] == 0
    }
    return {
        c["id"]: clean[(c["track_id"], c["target"])]
        for c in manifest["clips"]
        if c["part"] == "reverb"
    }


def ask_clef(
    client: cl.ClefClient,
    log_path: Path,
    manifest: Mapping[str, Any],
    prereg: Mapping[str, Any],
    clips_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Every pre-registered request, skipping those the log already answers.

    Returns answers by request id. A request that fails is logged by the client and left
    out; the caller decides what an incomplete set means.
    """
    by_id = {c["id"]: c for c in manifest["clips"]}
    wanted: list[tuple[str, dict, Mapping[str, Any]]] = []
    for clip in manifest["clips"]:
        wanted += [(rid, q, clip) for rid, q in requests_for(clip, prereg)]
    spec = prereg["clef_omni"]
    for wet_id, clean_id in reverb_pairs(manifest, prereg).items():
        clean = by_id[clean_id]
        question = cl.noul(
            spec["reverb"]["text"].format(target=spec["target_names"][clean["target"]])
        )
        wanted.append((f"{wet_id}-clean", {"reverb": question}, clean))
    done: dict[str, dict[str, Any]] = {}
    if log_path.is_file():
        for entry in cl.read_log(log_path)[1]:
            if entry.get("ok"):
                done[entry["request_id"]] = entry["answers"]
    state = spec["request"]["state"]
    for request_id, questions, clip in wanted:
        if request_id in done:
            continue
        audio = cl.Clip(clip["id"], (clips_dir / clip["file"]).read_bytes(), clip["seconds"])
        try:
            done[request_id] = client.ask(request_id, state, questions, [audio])["answers"]
        except cl.ClefError:
            continue
    return {rid: done[rid] for rid, _, _ in wanted if rid in done}


def clef_items(
    manifest: Mapping[str, Any], answers: Mapping[str, Mapping[str, Any]], prereg: Mapping[str, Any]
) -> dict[str, Any]:
    """Clef-omni's answers arranged for scoring, plus the two exploratory results."""
    ids = prereg["clef_omni"]["request"]["A"]["question_ids"]
    part_a, part_b, lead_backing = [], [], []
    for clip in manifest["clips"]:
        given = answers.get(clip["id"])
        if given is None:
            continue
        if clip["part"] == "A":
            part_a += [
                {"track_id": clip["track_id"], "stem": s, "truth": clip["present"][s],
                 "probability": given[ids[s]]["probability"]}
                for s in PRODUCT_STEMS
            ]  # fmt: skip
        elif clip["part"] == "B":
            part_b.append(
                {"clip_id": clip["id"], "level": clip["bleed_level"],
                 "bleed_probability": given["bleed.any"]["probability"],
                 "bleed_score": given["bleed.score"]["score"]}
            )  # fmt: skip
        elif clip["part"] == "lead_backing":
            lead_backing.append(given["lead_backing"]["choice"] == clip["answer"])
    pairs = reverb_pairs(manifest, prereg)
    wet = [answers[w]["reverb"]["probability"] for w in pairs if w in answers]
    dry = [answers[f"{w}-clean"]["reverb"]["probability"] for w in pairs if f"{w}-clean" in answers]
    return {
        "part_a": part_a,
        "part_b": part_b,
        "exploratory": {
            "lead_backing": {
                "n": len(lead_backing),
                "accuracy": scoring.accuracy([True] * len(lead_backing), lead_backing),
            },
            "reverb": {"wet": wet, "clean": dry, "auc": scoring.auc(dry, wet)},
        },
    }


def tagger_items(
    tagger: Any, manifest: Mapping[str, Any], clips_dir: Path, stem_map: StemMap
) -> dict[str, Any]:
    """The tagger on the same clips: class probabilities averaged over its windows, then
    a stem's score is its largest class, and bleed is the largest score of any other stem."""
    import soundfile as sf

    part_a, part_b, per_clip = [], [], []
    for clip in manifest["clips"]:
        if clip["part"] not in ("A", "B"):
            continue
        audio, rate = sf.read(clips_dir / clip["file"], dtype="float32", always_2d=True)
        probabilities = np.asarray(tagger.class_probabilities(audio.T, rate), dtype=np.float64)
        scores = stem_map.stem_scores(probabilities.mean(axis=0))
        per_clip.append({"clip_id": clip["id"], "windows": len(probabilities), "stems": scores})
        if clip["part"] == "A":
            part_a += [
                {"track_id": clip["track_id"], "stem": s, "truth": clip["present"][s],
                 "probability": scores[s]}
                for s in PRODUCT_STEMS
            ]  # fmt: skip
        else:
            bleed = max(v for s, v in scores.items() if s != clip["target"])
            part_b.append({"clip_id": clip["id"], "level": clip["bleed_level"],
                           "bleed_probability": bleed, "bleed_score": None})  # fmt: skip
    return {"part_a": part_a, "part_b": part_b, "per_clip": per_clip}


def usage_total(log_path: Path) -> dict[str, int]:
    entries = cl.read_log(log_path)[1] if log_path.is_file() else []
    used = [e["usage"] for e in entries if e.get("ok") and e.get("usage")]
    return {
        "requests": len(entries),
        "answered": len(used),
        "input_tokens": sum(int(u.get("input_tokens") or 0) for u in used),
        "output_tokens": sum(int(u.get("output_tokens") or 0) for u in used),
    }


def run_test(
    dataset: Dataset,
    prereg: Mapping[str, Any],
    commit: str,
    run_dir: Path,
    *,
    transport: cl.Transport | None = None,
    tagger: Any | None = None,
    clips_dir: Path | None = None,
    skipped: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run whichever of the two models is available and write the run folder."""
    if prereg["dataset"]["frozen_sha256"] != dataset.frozen.sha256:
        raise pr.PreregError("the frozen track list is not the one that was pre-registered")
    stem_map = load_stem_map()
    if prereg["class_to_stem"]["sha256"] != stem_map.sha256:
        raise pr.PreregError("the class-to-stem mapping is not the one that was pre-registered")
    run_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = clips_dir or CLEF_DIR / "clips" / plan_sha(prereg)[:16]
    manifest = ensure_clips(dataset, prereg, clips_dir)
    thresholds = prereg["thresholds"]
    expected = {
        "part_a": len(prereg["plan"]["tracks"]) * len(PRODUCT_STEMS),
        "part_b": len(prereg["plan"]["part_b"]) * 5,
    }
    scores: dict[str, Any] = {"expected": expected, "models": {}}
    log_path = run_dir / "clef-log.jsonl"
    if transport is not None:
        header = {"preregistration_commit": commit, "preregistration_version": prereg["version"]}
        client = cl.ClefClient(transport, log_path, header=header)
        answers = ask_clef(client, log_path, manifest, prereg, clips_dir)
        items = clef_items(manifest, answers, prereg)
        complete = (len(items["part_a"]), len(items["part_b"])) == tuple(expected.values())
        scores["models"]["clef_omni"] = {
            "complete": complete,
            **scoring.evaluate(items["part_a"], items["part_b"], thresholds),
            "exploratory": items["exploratory"],
            "usage": usage_total(log_path),
        }
    if tagger is not None:
        items = tagger_items(tagger, manifest, clips_dir, stem_map)
        with (run_dir / "tagger-scores.jsonl").open("w", encoding="utf-8") as f:
            for row in items["per_clip"]:
                f.write(json.dumps(row) + "\n")
        scores["models"]["beats"] = {
            "tagger": tagger.name,
            "complete": True,
            **scoring.evaluate(items["part_a"], items["part_b"], thresholds),
        }
    models = scores["models"]
    if "clef_omni" in models and "beats" in models:
        scores["C"] = scoring.compare(models["clef_omni"], models["beats"], thresholds)
        if models["clef_omni"]["complete"]:
            scores["verdict"] = scoring.verdict(models["clef_omni"], scores["C"])
    write_json(run_dir / "scores.json", scores)
    write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": dataset.name,
            "preregistration": {"commit": commit, "version": prereg["version"],
                                "plan_sha256": plan_sha(prereg)},
            "clips": {"dir": str(clips_dir), "count": len(manifest["clips"])},
            "ran": sorted(models),
            "skipped": dict(skipped or {}),
            "git": git_state(),
        },
    )  # fmt: skip
    return scores


def _percent(value: float | None) -> str:
    return "not measured" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f}"


def _verdict(passed: bool) -> str:
    return "pass" if passed else "fail"


def render(run_dir: Path) -> str:
    """The pass/fail report, from a run folder's files alone."""
    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    models = scores["models"]
    names = {"clef_omni": "Clef-omni", "beats": "BEATs"}
    out = [
        "# Clef-omni against BEATs: test report",
        "",
        f"Generated from `runs/clef_test/{run['run_id']}/` (finished {run['finished']}), "
        f"pre-registration version {run['preregistration']['version']} at commit "
        f"`{run['preregistration']['commit'][:12]}`. Reproduce with "
        "`python -m stems_bench.clef.run`.",
        "",
    ]
    for key, why in run["skipped"].items():
        out += [f"**{names.get(key, key)} did not run:** {why}", ""]
    for key in ("clef_omni", "beats"):
        if key not in models:
            continue
        m = models[key]
        out += [f"## {names[key]}", ""]
        if not m["complete"]:
            out += ["**Incomplete:** some requests failed, so the figures below cover only "
                    "the clips that were answered and no verdict is drawn.", ""]  # fmt: skip
        out += ["| Test | Measured | Needed | Result |", "|---|---|---|---|"]
        a1, a3, b1, b2, b3 = m["A1"], m["A3"], m["B1"], m["B2"], m["B3"]
        out.append(f"| A1 presence accuracy, {a1['n']} decisions | {_percent(a1['value'])} | "
                   f"{_percent(a1['threshold'])} | {_verdict(a1['pass'])} |")  # fmt: skip
        for stem, s in m["A2"]["per_stem"].items():
            out.append(f"| A2 {stem} ({s['present']} of {s['n']} present) | {_percent(s['value'])} "
                       f"| {_percent(s['threshold'])} | {_verdict(s['pass'])} |")  # fmt: skip
        out.append(f"| A3 vocals ({a3['absent']} of {a3['n']} absent; always answering "
                   f"'present' scores {_percent(a3['always_present_accuracy'])}) | "
                   f"{_percent(a3['value'])} | {_percent(a3['threshold'])} | "
                   f"{_verdict(a3['pass'])} |")  # fmt: skip
        out.append(f"| B1 AUC, {b1['clean']} clean against {b1['bleed']} bleed | "
                   f"{_number(b1['auc_pooled'])} | {_number(b1['threshold_pooled'])} | "
                   f"{_verdict(b1['pass'])} |")  # fmt: skip
        out.append(f"| B1 AUC, clean against +30 dB SIR | {_number(b1['auc_hardest'])} | "
                   f"{_number(b1['threshold_hardest'])} | |")  # fmt: skip
        out.append(f"| B2 Spearman, {b2['rated_by']} against bleed level | "
                   f"{_number(b2['value'])} | {_number(b2['threshold'])} | "
                   f"{_verdict(b2['pass'])} |")  # fmt: skip
        out.append(f"| B3 clean clips called bleeding ({b3['flagged']} of {b3['clean']}) | "
                   f"{_percent(b3['value'])} | at most {_percent(b3['threshold'])} | "
                   f"{_verdict(b3['pass'])} |")  # fmt: skip
        out.append("")
        if "usage" in m:
            u = m["usage"]
            out += [f"{u['answered']} of {u['requests']} requests answered, "
                    f"{u['input_tokens']} input tokens.", ""]  # fmt: skip
        if "exploratory" in m:
            e = m["exploratory"]
            out += [
                "Exploratory, not part of the decision: lead or backing answered correctly "
                f"on {_percent(e['lead_backing']['accuracy'])} of {e['lead_backing']['n']} "
                f"clips; reverb clips against their clean versions, AUC "
                f"{_number(e['reverb']['auc'])} ({len(e['reverb']['wet'])} pairs).",
                "",
            ]
    if "C" in scores:
        c = scores["C"]
        out += [
            "## Clef-omni against BEATs (threshold C)",
            "",
            f"Margin on A1: {c['A1_margin_points']} points. Margin on B1: "
            f"{c['B1_margin_points']} points. Needed on both: {c['threshold']}. "
            f"Result: {_verdict(c['pass'])}.",
            "",
        ]
    out += ["## Verdict", ""]
    if "verdict" in scores:
        v = scores["verdict"]
        out += [f"Clef-omni: **{v['clef_omni']}**. Listening layer: **{v['listening_layer']}**."]
    else:
        out += ["No verdict: it needs complete results for both Clef-omni and BEATs."]
    out += [""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default="clef-omni-v1")
    parser.add_argument("--only", choices=("clef", "beats"))
    parser.add_argument("--report", type=Path, help="write the report for the run and stop")
    args = parser.parse_args(argv)
    run_dir = CLEF_DIR / args.run_id
    if args.report:
        if not (run_dir / "scores.json").is_file():
            print(f"{run_dir} has no scores.json; the test has not been run", file=sys.stderr)
            return 2
        args.report.write_text(render(run_dir), encoding="utf-8")
        print(f"wrote {args.report}")
        return 0
    try:
        prereg, commit = pr.load()
        dataset = get_dataset(prereg["dataset"]["name"])
        dataset.check_ready()
        skipped: dict[str, str] = {}
        transport = tagger = None
        if args.only != "beats":
            try:
                transport = cl.HttpTransport.from_env(cl.Limits.from_registry())
            except cl.ClefError as e:
                skipped["clef_omni"] = str(e)
        if args.only != "clef":
            skipped["beats"] = (
                "the BEATs wrapper is not built yet (PROGRESS.md P3.1, BLOCKED.md B10)"
            )
        if transport is None and tagger is None:
            for name, why in skipped.items():
                print(f"{name}: {why}", file=sys.stderr)
            return 2
        scores = run_test(dataset, prereg, commit, run_dir,
                          transport=transport, tagger=tagger, skipped=skipped)  # fmt: skip
    except (pr.PreregError, DatasetError, ClipError, cl.ClefError) as e:
        print(e, file=sys.stderr)
        return 2
    print(f"ran {sorted(scores['models'])}; results in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
