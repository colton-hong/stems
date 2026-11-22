"""Comparing runs: run-to-run noise against the spread between devices."""

from __future__ import annotations

import json

import pytest

from stems_bench import cli
from stems_bench.compare import compare_runs, write_comparison
from stems_bench.report import ReportError, load_run

FROZEN = "f" * 64


def _make_run(root, run_id, device, scores, *, trial=0, overlap=0.25, frozen=FROZEN):
    """``scores``: {(model, output): {track: sdr or None}}."""
    run_dir = root / "bench" / run_id
    run_dir.mkdir(parents=True)
    by_job: dict[tuple[str, str], dict] = {}
    for (model, output), per_track in scores.items():
        for track, sdr in per_track.items():
            result = by_job.setdefault(
                (model, track),
                {
                    "dataset": "fake_set",
                    "frozen_sha256": frozen,
                    "protocol_version": 1,
                    "model_id": model,
                    "track_id": track,
                    "outputs": {},
                },
            )
            result["outputs"][output] = {"target": output, "bss": {"sdr": sdr}}
    jobs = []
    for i, ((model, track), result) in enumerate(by_job.items()):
        path = root / "bench" / "results" / run_id / f"{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        jobs.append(
            {
                "model_id": model,
                "track_id": track,
                "outcome": "computed",
                "result": f"bench/results/{run_id}/{i}.json",
            }
        )
    jobs.append({"model_id": "broken", "track_id": "t1", "outcome": "failed", "result": None})
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    run = {
        "run_id": run_id,
        "dataset": "fake_set",
        "frozen_sha256": frozen,
        "protocol_version": 1,
        "device": device,
        "overlap": overlap,
        "command": f"bench run --dataset fake_set --run-id {run_id}",
        "git": {"commit": "abc123", "dirty": False},
    }
    if trial:
        run["trial"] = trial
    (run_dir / "run.json").write_text(json.dumps(run))
    (run_dir / "status.json").write_text(json.dumps({"state": "done"}))
    return run_dir


BASE = {
    ("alpha", "vocals"): {"t1": 8.0, "t2": 10.0, "t3": None},
    ("alpha", "drums"): {"t1": 5.0, "t2": 6.0, "t3": 7.0},
    ("beta", "vocals"): {"t1": 9.0, "t2": 9.5},
}
AGAIN = {
    ("alpha", "vocals"): {"t1": 8.001, "t2": 9.998, "t3": None},
    ("alpha", "drums"): {"t1": 5.0, "t2": 6.002, "t3": 7.0},
    ("beta", "vocals"): {"t1": 9.0, "t2": 9.5},
}
OTHER_DEVICE = {
    ("alpha", "vocals"): {"t1": 8.25, "t2": 9.5, "t3": 4.0},
    ("alpha", "drums"): {"t1": 5.0, "t2": 6.125},  # t3 not done there
    ("gamma", "vocals"): {"t1": 1.0},  # a model the first run did not have
}


@pytest.fixture
def runs(tmp_path):
    return {
        "base": _make_run(tmp_path, "base", "mps", BASE),
        "again": _make_run(tmp_path, "again", "mps", AGAIN, trial=1),
        "cpu": _make_run(tmp_path, "cpu", "cpu", OTHER_DEVICE),
    }


def _load(run_dir, tmp_path):
    return load_run(run_dir, repo_root=tmp_path)


def test_differences_are_per_track_over_what_both_runs_scored(runs, tmp_path) -> None:
    comparison = compare_runs(_load(runs["base"], tmp_path), _load(runs["cpu"], tmp_path))
    assert comparison.same_device is False
    by_key = {(d.system, d.output): d for d in comparison.differences}
    assert set(by_key) == {("alpha", "vocals"), ("alpha", "drums")}  # beta and gamma: one run each
    vocals = by_key["alpha", "vocals"]
    assert vocals.deltas == {
        "t1": pytest.approx(0.25),
        "t2": pytest.approx(-0.5),
    }  # t3 unscored in base
    assert (vocals.tracks, vocals.sdr_a, vocals.sdr_b) == (2, 9.0, 8.875)
    assert vocals.largest == pytest.approx(0.5)
    assert vocals.worst_track == "t2"
    drums = by_key["alpha", "drums"]
    assert drums.tracks == 2
    assert drums.largest == pytest.approx(0.125)
    assert sorted(comparison.all_deltas) == pytest.approx([0.0, 0.125, 0.25, 0.5])
    assert comparison.largest == pytest.approx(0.5)
    assert comparison.median == pytest.approx(0.1875)


def test_document_states_noise_spread_and_the_consequence(runs, tmp_path) -> None:
    out = tmp_path / "docs" / "spread.md"
    pairs = [(runs["base"], runs["again"]), (runs["base"], runs["cpu"])]
    text = write_comparison(pairs, out, repo_root=tmp_path).read_text()
    assert "# Score differences between runs: fake_set" in text
    assert f"- Frozen track list: sha256 `{FROZEN}`" in text
    assert (
        "- `again` (mps, trial 1): `bench run --dataset fake_set --run-id again`, commit `abc123`"
        in text
    )
    assert (
        "| `again` (mps, trial 1) | `base` (mps) | same device, run again | 7 | 0.000 | 0.002 |"
        in text
    )
    assert "| `cpu` (cpu) | `base` (mps) | another device | 4 | 0.188 | 0.500 |" in text
    assert "Moving to another device changes a score by up to 0.500 dB" in text
    assert "running again on the same device changes it by up to 0.002 dB" in text
    assert "must not be compared with each other" in text
    assert "| `alpha` | vocals | 2 | 9.000 | 8.875 | -0.125 | 0.500 | `t2` |" in text
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert [c["second"] for c in sidecar["comparisons"]] == ["again", "cpu"]
    assert sidecar["comparisons"][1]["largest"] == pytest.approx(0.5)
    assert set(sidecar["result_files"]) == {"base", "again", "cpu"}
    assert all(len(f["sha256"]) == 64 for f in sidecar["result_files"]["cpu"])


def test_when_devices_agree_within_noise_the_document_says_so(tmp_path) -> None:
    base = _make_run(tmp_path, "base", "mps", {("m", "vocals"): {"t1": 8.0, "t2": 9.0}})
    again = _make_run(tmp_path, "again", "mps", {("m", "vocals"): {"t1": 8.01, "t2": 9.0}}, trial=1)
    cpu = _make_run(tmp_path, "cpu", "cpu", {("m", "vocals"): {"t1": 8.005, "t2": 9.0}})
    out = tmp_path / "spread.md"
    text = write_comparison([(base, again), (base, cpu)], out, repo_root=tmp_path).read_text()
    assert "no more than running again on the same device (0.010 dB)" in text
    assert "must not be compared" not in text
    # With only one kind of comparison there is nothing to conclude, and nothing is.
    text = write_comparison([(base, cpu)], out, repo_root=tmp_path).read_text()
    assert "## What follows from it" not in text


def test_runs_that_did_different_jobs_are_not_compared(tmp_path) -> None:
    base = _make_run(tmp_path, "base", "mps", BASE)
    other_list = _make_run(tmp_path, "other", "cpu", BASE, frozen="0" * 64)
    other_overlap = _make_run(tmp_path, "overlap", "cpu", BASE, overlap=0.5)
    for other, field in ((other_list, "frozen_sha256"), (other_overlap, "overlap")):
        with pytest.raises(ReportError, match=f"differ in {field}"):
            write_comparison([(base, other)], tmp_path / "x.md", repo_root=tmp_path)
    absent = tmp_path / "bench" / "absent"
    with pytest.raises(ReportError, match="is not a benchmark run"):
        write_comparison([(base, absent)], tmp_path / "x.md", repo_root=tmp_path)


def test_command_compares_pairs(runs, tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(write_comparison, "__kwdefaults__", {"repo_root": tmp_path})
    out = tmp_path / "spread.md"
    argv = ["compare", "--pair", "base", "again", "--pair", "base", "cpu", "--out", str(out)]
    assert cli.main(argv, bench_dir=tmp_path / "bench") == 0
    assert "another device" in out.read_text()
    assert f"wrote {out}" in capsys.readouterr().out
    bad = ["compare", "--pair", "base", "absent", "--out", str(out)]
    assert cli.main(bad, bench_dir=tmp_path / "bench") == 2
