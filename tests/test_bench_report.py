"""Reports are built only from a run's result files, and say where every figure is from."""

from __future__ import annotations

import hashlib
import json
import re
import statistics

import pytest

from stems_bench import cli
from stems_bench.report import ReportError, aggregate, load_run, render, write_report

FROZEN = "f" * 64
COMMIT = "c0ffee" * 6 + "c0ff"


def _output(
    sdr,
    *,
    target,
    sir=None,
    sar=None,
    present=True,
    derived=False,
    high=0.0,
    silence=0.0,
    same_length=True,
):
    return {
        "target": target,
        "derived": derived,
        "reference_present": present,
        "bss": {
            "sdr": sdr,
            "sir": sir,
            "sar": sar,
            "isr": None,
            "windows": 10,
            "scored_windows": 0 if sdr is None else 10,
        },
        "signal": {
            "estimate": {"silence_fraction": silence},
            "reference": {"silence_fraction": 0.0},
            "duration_matches": same_length,
            "high_band_ratio_db": high,
        },
    }


def _result(model, track, outputs, wall, *, audio=200.0, device="mps", commit=COMMIT):
    return {
        "dataset": "fake_set",
        "frozen_sha256": FROZEN,
        "protocol_version": 1,
        "model_id": model,
        "track_id": track,
        "outputs": outputs,
        "wall_seconds": wall,
        "audio_seconds": audio,
        "seconds_per_audio_second": wall / audio,
        "device": device,
        "git": {"commit": commit, "dirty": False},
    }


# Hand-built results: three tracks, two models, one of which also failed once.
RESULTS = [
    _result("alpha", "t1", {"vocals": _output(8.0, target="vocals", sir=15.0, sar=9.0, high=-1.0),
                            "other": _output(4.0, target="other_4stem", sir=7.0, sar=5.0)}, 20.0),
    _result("alpha", "t2", {"vocals": _output(10.0, target="vocals", sir=17.0, sar=11.0, high=-3.0),
                            "other": _output(6.0, target="other_4stem", sir=9.0, sar=7.0)}, 30.0),
    _result("alpha", "t3", {"vocals": _output(None, target="vocals", present=False, high=None),
                            "other": _output(5.0, target="other_4stem", sir=8.0, sar=6.0)}, 40.0),
    _result("beta", "t1", {"vocals": _output(9.5, target="vocals", sir=12.0, sar=13.0, high=-20.0, same_length=False)}, 2.0, device="cpu"),
    _result("beta", "t2", {"vocals": _output(11.5, target="vocals", sir=14.0, sar=15.0, high=-22.0, silence=0.9)}, 4.0, device="cpu"),
]  # fmt: skip


def _make_run(
    root, results=RESULTS, *, state="done", reliable=("sdr", "sir", "sar"), extra_jobs=()
):
    run_dir = root / "bench" / "r1"
    run_dir.mkdir(parents=True)
    jobs = []
    for i, result in enumerate(results):
        path = root / "bench" / "results" / f"{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        jobs.append(
            {
                "model_id": result["model_id"],
                "track_id": result["track_id"],
                "outcome": "computed" if i % 2 == 0 else "cached",
                "result": f"bench/results/{i}.json",
                "detail": None,
            }
        )
    jobs += list(extra_jobs)
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    run = {
        "run_id": "r1",
        "dataset": "fake_set",
        "frozen_sha256": FROZEN,
        "protocol_version": 1,
        "device": "mps",
        "overlap": 0.25,
        "batch_size": 1,
        "models": ["alpha", "beta"],
        "tracks": ["t1", "t2", "t3"],
        "reliable_metrics": list(reliable),
        "command": "bench run --dataset fake_set --models alpha,beta",
        "git": {"commit": COMMIT, "dirty": False},
        "packages": {"torch": "2.15.0", "museval": "0.4.1", "demucs": None},
        "started_at": "2026-11-22T10:00:00+00:00",
    }
    (run_dir / "run.json").write_text(json.dumps(run))
    (run_dir / "status.json").write_text(json.dumps({"state": state, "run_id": "r1"}))
    return run_dir


EXTRA = (
    {"model_id": "beta", "track_id": "t3", "outcome": "failed", "result": None, "detail": "RuntimeError: out of memory"},
    {"model_id": "gamma", "track_id": "t1", "outcome": "not_applicable", "result": None, "detail": "needs the 'vocals_wet' input, which fake_set does not have"},
    {"model_id": "gamma", "track_id": "t2", "outcome": "not_applicable", "result": None, "detail": "needs the 'vocals_wet' input, which fake_set does not have"},
)  # fmt: skip


def test_aggregate_takes_the_median_over_tracks_and_counts_what_was_scored() -> None:
    rows = {(r.system, r.output): r for r in aggregate(RESULTS)}
    assert set(rows) == {("alpha", "vocals"), ("alpha", "other"), ("beta", "vocals")}
    vocals = rows["alpha", "vocals"]
    assert vocals.medians == {"sdr": 9.0, "sir": 16.0, "sar": 10.0, "isr": None}
    assert (vocals.tracks, vocals.tracks_scored) == (3, 2)  # t3 has no vocals
    assert vocals.seconds_per_track == 30.0
    assert vocals.seconds_per_audio_minute == pytest.approx(60 * 30.0 / 200.0)
    assert vocals.devices == ["mps"]
    assert vocals.high_band_ratio_db == -2.0
    other = rows["alpha", "other"]
    assert (other.target, other.medians["sdr"], other.tracks_scored) == ("other_4stem", 5.0, 3)
    beta = rows["beta", "vocals"]
    assert beta.medians["sdr"] == 10.5
    assert beta.seconds_per_track == 3.0
    assert beta.duration_mismatches == 1
    assert beta.silence_fraction == pytest.approx(0.45)
    assert beta.devices == ["cpu"]


def test_report_states_command_commit_list_hash_and_puts_time_and_device_by_each_score(
    tmp_path,
) -> None:
    run_dir = _make_run(tmp_path, extra_jobs=EXTRA)
    path = write_report(run_dir, repo_root=tmp_path)
    text = path.read_text()
    assert path == run_dir / "report.md"
    assert "# Benchmark: fake_set" in text
    assert "- Command: `bench run --dataset fake_set --models alpha,beta`" in text
    assert f"- Code: commit `{COMMIT}`" in text
    assert f"- Frozen track list: sha256 `{FROZEN}`, 3 tracks in this run" in text
    assert "- Protocol version: 1" in text
    assert "- Packages: torch 2.15.0, museval 0.4.1" in text
    assert "3 computed in this run, 2 taken from earlier runs, 1 failed, 2 not applicable" in text
    # One table per target; the best SDR first; time and device on every row.
    vocals = text[text.index("## vocals") : text.index("## other_4stem")]
    assert (
        "| model | SDR | SIR | SAR | tracks | s per track | s per min of audio | device |" in vocals
    )
    rows = [line for line in vocals.splitlines() if line.startswith("| `")]
    assert rows[0].startswith(
        "| `beta` | 10.50 | 13.00 | 14.00 | 2 of 2 | 3.00 | 0.90 | cpu | -21.0 |"
    )
    assert rows[1].startswith(
        "| `alpha` | 9.00 | 16.00 | 10.00 | 2 of 3 | 30.00 | 9.00 | mps | -2.0 |"
    )
    assert "wrong length on 1 track(s)" in rows[0]
    assert "| `alpha` | 5.00 | 8.00 | 6.00 | 3 of 3 | 30.00 | 9.00 | mps | 0.0 |" in text
    assert "Four-stem other: guitar, piano and everything else" in text
    assert "## Failures" in text
    assert "- `beta` on `t3`: RuntimeError: out of memory" in text
    assert text.count("- `gamma`: needs the 'vocals_wet' input") == 1


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"-?\d+\.\d+", text))


def test_every_figure_in_the_tables_comes_from_the_result_files(tmp_path) -> None:
    run_dir = _make_run(tmp_path, extra_jobs=EXTRA)
    text = write_report(run_dir, repo_root=tmp_path).read_text()
    table = "\n".join(line for line in text.splitlines() if line.startswith("| `"))

    # Recompute, independently of the report code, every figure a table may show.
    allowed = set()
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in RESULTS:
        for output in result["outputs"]:
            groups.setdefault((result["model_id"], output), []).append(result)
    for (_, output), items in groups.items():
        for metric in ("sdr", "sir", "sar"):
            values = [r["outputs"][output]["bss"][metric] for r in items]
            values = [v for v in values if v is not None]
            if values:
                allowed.add(f"{statistics.median(values):.2f}")
        allowed.add(f"{statistics.median(r['wall_seconds'] for r in items):.2f}")
        per_minute = 60 * statistics.median(r["wall_seconds"] / r["audio_seconds"] for r in items)
        allowed.add(f"{per_minute:.2f}")
        highs = [r["outputs"][output]["signal"]["high_band_ratio_db"] for r in items]
        highs = [h for h in highs if h is not None]
        if highs:
            allowed.add(f"{statistics.median(highs):.1f}")
    assert _numbers(table) <= allowed
    assert len(_numbers(table)) >= 12  # and the tables are not empty

    # The sidecar names every result file with its hash, and holds the same rows.
    sidecar = json.loads((run_dir / "report.json").read_text())
    assert [f["path"] for f in sidecar["result_files"]] == [
        f"bench/results/{i}.json" for i in range(5)
    ]
    for entry in sidecar["result_files"]:
        assert (
            entry["sha256"] == hashlib.sha256((tmp_path / entry["path"]).read_bytes()).hexdigest()
        )
    assert {(r["system"], r["output"]) for r in sidecar["rows"]} == set(groups)
    assert sidecar["frozen_sha256"] == FROZEN


def test_changing_a_result_file_changes_the_report_and_nothing_else_does(tmp_path) -> None:
    run_dir = _make_run(tmp_path)
    before = write_report(run_dir, repo_root=tmp_path).read_text()
    assert write_report(run_dir, repo_root=tmp_path).read_text() == before
    path = tmp_path / "bench" / "results" / "3.json"
    result = json.loads(path.read_text())
    result["outputs"]["vocals"]["bss"]["sdr"] = 3.25
    path.write_text(json.dumps(result))
    after = write_report(run_dir, repo_root=tmp_path).read_text()
    assert "| `beta` | 7.38 |" in after  # median of 3.25 and 11.5
    assert "10.50" not in after


def test_unreliable_metrics_are_left_out_with_the_reason(tmp_path) -> None:
    run_dir = _make_run(tmp_path, reliable=("sdr",))
    text = write_report(run_dir, repo_root=tmp_path).read_text()
    assert "| model | SDR | tracks | s per track |" in text
    assert "SIR |" not in text
    assert "**Only SDR is reported for this dataset.**" in text
    assert "| `beta` | 10.50 | 2 of 2 | 3.00 | 0.90 | cpu |" in text
    assert "13.00" not in text  # beta's SIR is in the files but not in the report


def test_derived_outputs_unfinished_runs_and_mixed_commits_are_flagged(tmp_path) -> None:
    results = [
        _result("six", "t1", {"other_4stem": _output(7.0, target="other_4stem", derived=True)}, 5.0),
        _result("six", "t2", {"other_4stem": _output(9.0, target="other_4stem", derived=True)}, 7.0, commit="b" * 40),
    ]  # fmt: skip
    run_dir = _make_run(tmp_path, results, state="interrupted")
    text = write_report(run_dir, repo_root=tmp_path).read_text()
    assert "sum of its guitar, piano and other" in text
    assert "**This run did not finish (state `interrupted`).**" in text
    assert f"Results computed on commit(s): `{'b' * 40}`, `{COMMIT}`" in text


def test_a_result_from_another_list_or_protocol_is_refused(tmp_path) -> None:
    run_dir = _make_run(tmp_path)
    path = tmp_path / "bench" / "results" / "0.json"
    result = json.loads(path.read_text())
    path.write_text(json.dumps({**result, "frozen_sha256": "0" * 64}))
    with pytest.raises(ReportError, match="its frozen_sha256 is not the run's"):
        load_run(run_dir, repo_root=tmp_path)
    with pytest.raises(ReportError, match="is not a benchmark run"):
        load_run(tmp_path / "nowhere")


def test_a_tiny_negative_is_not_printed_as_minus_zero(tmp_path) -> None:
    results = [_result("m", "t1", {"vocals": _output(-0.001, target="vocals", high=-0.04)}, 1.0)]
    text = write_report(_make_run(tmp_path, results), repo_root=tmp_path).read_text()
    assert "| `m` | 0.00 | n/a | n/a | 1 of 1 | 1.00 | 0.30 | mps | 0.0 |" in text


def _system(mode, sdr, wall, members, weights, *, track="t1", minus=()):
    result = _result(f"vocals:{mode}", track, {"vocals": _output(sdr, target="vocals")}, wall)
    extra = {"kind": "ensemble", "slot": "vocals", "mode": mode, "members": members}
    return {**result, **extra, "weights": weights, "method": "waveform", "minus": list(minus)}


def test_blends_get_their_own_section_with_gain_and_cost(tmp_path) -> None:
    results = [
        _result("alpha", "t1", {"vocals": _output(8.0, target="vocals")}, 20.0),
        _result("alpha", "t2", {"vocals": _output(10.0, target="vocals")}, 30.0),
        _system("full_ensemble", 9.75, 70.0, ["alpha", "beta", "gamma"], [0.5, 0.25, 0.25]),
        _system("single_best", 8.0, 20.0, ["alpha"], [1.0]),
        _system("two_model_blend", 8.5, 45.0, ["alpha", "beta"], [0.5, 0.5]),
        _system("two_model_blend", 7.5, 55.0, ["alpha", "beta"], [0.5, 0.5], track="t2"),
    ]
    text = write_report(_make_run(tmp_path, results), repo_root=tmp_path).read_text()
    section = text[text.index("## Single best, two-model blend, full ensemble") :]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "vocals" in line]
    assert rows == [
        "| single best | vocals | vocals | 8.00 | n/a | n/a |  | 1 of 1 | 20.00 | 6.00 | mps | `alpha` (1.00) |  |",
        "| two-model blend | vocals | vocals | 8.00 | n/a | n/a | +0.00 | 2 of 2 | 50.00 | 15.00 | mps | `alpha` (0.50), `beta` (0.50) | waveform |",
        "| full ensemble | vocals | vocals | 9.75 | n/a | n/a | +1.75 | 1 of 1 | 70.00 | 21.00 | mps | `alpha` (0.50), `beta` (0.25), `gamma` (0.25) | waveform |",
    ]
    # The per-model table has the model only.
    table = text[text.index("## vocals") : text.index("## Single best")]
    assert [
        line.split("|")[1].strip() for line in table.splitlines() if line.startswith("| `")
    ] == ["`alpha`"]


def test_a_composed_system_names_what_is_taken_out(tmp_path) -> None:
    minus = [{"slot": "guitar", "members": ["g"], "weights": [1.0]}]
    system = _system("single_best", 3.0, 9.0, ["o"], [1.0], minus=minus)
    system |= {"slot": "other_composed", "model_id": "other_composed:single_best"}
    text = write_report(_make_run(tmp_path, [system]), repo_root=tmp_path).read_text()
    assert "### other_composed" in text
    assert "The four-stem blend with the same mode's guitar taken out" in text
    assert "`o` (1.00), minus guitar: `g`" in text


def test_render_is_a_function_of_the_loaded_run_only(tmp_path) -> None:
    data = load_run(_make_run(tmp_path), repo_root=tmp_path)
    rows = aggregate([r for _, _, r in data.results])
    assert render(data, rows) == render(data, rows)
    assert len(data.results) == 5


def test_command_writes_a_report_where_asked(tmp_path, capsys, monkeypatch) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr("stems_bench.report.REPO_ROOT", tmp_path)
    monkeypatch.setattr(write_report, "__kwdefaults__", {"repo_root": tmp_path})
    out = tmp_path / "docs" / "fake_set.md"
    assert cli.main(["report", "r1", "--out", str(out)], bench_dir=run_dir.parent) == 0
    assert "# Benchmark: fake_set" in out.read_text()
    assert out.with_suffix(".json").exists()
    assert f"wrote {out}" in capsys.readouterr().out
    assert cli.main(["report"], bench_dir=run_dir.parent) == 0  # the latest run
    assert (run_dir / "report.md").exists()
    assert cli.main(["report", "absent"], bench_dir=run_dir.parent) == 2
