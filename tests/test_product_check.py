"""Tracks through the product's path, scored beside the benchmark: the bookkeeping.

The models here are the weightless stand-ins and the tracks the synthetic set, so the
scores mean nothing; what is tested is that a track goes down the worker's road, is
scored, is compared with the right benchmark figure, and leaves no audio behind.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
if importlib.util.find_spec("museval") is None:
    pytest.skip("museval is not installed", allow_module_level=True)

from stems_bench import product_check as pc  # noqa: E402
from stems_bench.datasets import get_dataset  # noqa: E402
from stems_common.settings import load_defaults  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_worker.fakes import fake_pipeline  # noqa: E402


def stem(name: str, models: list[str]) -> dict:
    return {"name": name, "model_ids": models}


def test_a_stem_is_compared_with_the_same_model_scored_against_the_same_stem() -> None:
    measured = {
        "six": {
            "piano": {"target": "piano", "sdr_db": 5.0, "device": "mps"},
            "other": {"target": "other", "sdr_db": None, "device": "mps"},
        },
        "four": {"other": {"target": "other_4stem", "sdr_db": 7.0, "device": "mps"}},
    }
    lookup = lambda model_id: measured.get(model_id, {})  # noqa: E731
    assert pc.compare_stem(stem("piano", ["six"]), 5.3, lookup) == {
        "model_id": "six",
        "sdr_db": 5.0,
        "device": "mps",
        "difference_db": 0.3,
        "agrees": True,
    }
    assert pc.compare_stem(stem("piano", ["six"]), 4.4, lookup)["agrees"] is False
    assert pc.compare_stem(stem("piano", ["six"]), 5.5, lookup)["agrees"] is True  # inclusive
    # No score for the product's stem: the benchmark's figure is shown, nothing is claimed.
    assert pc.compare_stem(stem("piano", ["six"]), None, lookup)["agrees"] is None
    # A blend, a four-stem "other", an output the benchmark could not score, an unknown model.
    assert pc.compare_stem(stem("piano", ["six", "four"]), 5.0, lookup) is None
    assert pc.compare_stem(stem("other", ["four"]), 5.0, lookup) is None
    assert pc.compare_stem(stem("other", ["six"]), 5.0, lookup) is None
    assert pc.compare_stem(stem("piano", ["nine"]), 5.0, lookup) is None


def test_benchmark_figures_are_read_for_the_model_and_the_track(tmp_path) -> None:
    folder = tmp_path / "results" / "set" / "six"
    folder.mkdir(parents=True)
    for key, track, sdr in (("a", "t1", 4.0), ("b", "t2", 6.0), ("c", "t2", 9.0)):
        result = {
            "kind": "model",
            "track_id": track,
            "device": "mps",
            "outputs": {"piano": {"target": "piano", "bss": {"sdr": sdr}}},
        }
        (folder / f"{key}.json").write_text(json.dumps(result))
    (folder / "0.json").write_text(json.dumps({"kind": "blend", "track_id": "t2"}))
    assert pc.benchmark_scores(tmp_path, "set", "t2", "six") == {
        "piano": {"target": "piano", "sdr_db": 6.0, "device": "mps"}  # the first of two runs
    }
    assert pc.benchmark_scores(tmp_path, "set", "t3", "six") == {}
    assert pc.benchmark_scores(tmp_path, "set", "t1", "seven") == {}


def test_the_same_track_of_the_same_run_is_the_same_job() -> None:
    assert pc.job_id_for("run", "track") == pc.job_id_for("run", "track")
    assert pc.job_id_for("run", "track") != pc.job_id_for("run2", "track")


def test_tracks_go_down_the_workers_road_are_scored_and_leave_no_audio(tmp_path) -> None:
    options, defaults = fake_pipeline(load_defaults())
    dataset = get_dataset("synthetic")
    tracks = dataset.track_ids()[:2]
    bench = tmp_path / "bench"
    folder = bench / "results" / dataset.name / "fake_bands"
    folder.mkdir(parents=True)
    result = {
        "kind": "model",
        "track_id": tracks[0],
        "device": "mps",
        "outputs": {
            "bass": {"target": "bass", "bss": {"sdr": 1.0}},
            "other": {"target": "other_4stem", "bss": {"sdr": 2.0}},
        },
    }
    (folder / "k.json").write_text(json.dumps(result))
    run_dir = tmp_path / "run"
    said: list[str] = []
    arguments = {
        "run_id": "test",
        "defaults": defaults,
        "device": "cpu",
        "bench_dir": bench,
        "pipeline": options,
        "progress": said.append,
    }
    assert pc.run(dataset, tracks, run_dir, **arguments) == 2
    assert said == [f"track 1 of 2: {tracks[0]}", f"track 2 of 2: {tracks[1]}"]
    records = pc.read_results(run_dir)
    assert [record["track_id"] for record in records] == sorted(tracks)
    first = next(record for record in records if record["track_id"] == tracks[0])
    assert (first["state"], first["device"], first["audio_seconds"]) == ("ready", "cpu", 12.0)
    assert first["wall_seconds"] > 0
    assert list(first["stems"]) == list(PRODUCT_STEMS)
    for name, entry in first["stems"].items():
        assert entry["length_matches"] is True, name
        assert entry["in_song"] is (entry["reference_sounding_seconds"] >= 5.0)
        assert isinstance(entry["flagged_empty"], bool)
        assert entry["attempts"] >= 1
    # Scored by the benchmark's scorer against the dataset's reference.
    assert isinstance(first["stems"]["bass"]["sdr_db"], float)
    # Set beside the benchmark only where the benchmark scored the same thing.
    bass = first["stems"]["bass"]
    if bass["model_ids"] == ["fake_bands"]:
        assert bass["benchmark"]["sdr_db"] == 1.0
        assert bass["benchmark"]["difference_db"] == round(bass["sdr_db"] - 1.0, 3)
    assert first["stems"]["other"]["benchmark"] is None
    assert first["stems"]["piano"]["benchmark"] is None
    second = next(record for record in records if record["track_id"] == tracks[1])
    assert all(entry["benchmark"] is None for entry in second["stems"].values())
    # Nothing of the audio is left: the storage folder holds no object of any job.
    left = [p for p in (run_dir / "storage").rglob("*") if p.is_file()]
    assert [p.name for p in left if p.suffix in (".flac", ".wav", ".zip")] == []
    assert not any("jobs" in p.parts and p.name != "signing-secret" for p in left), left
    # Run again: nothing is done twice.
    assert pc.run(dataset, tracks, run_dir, **arguments) == 0

    summary = pc.summarise(records)
    assert (summary["tracks"], summary["ready"], summary["failed"]) == (2, 2, [])
    assert summary["stems"] == 12
    text = pc.render(
        records,
        {
            "run_id": "test",
            "finished": "2026-11-24T13:00:00+00:00",
            "tracks": 2,
            "device": "cpu",
            "git": {"commit": "0123456789abcdef", "dirty": False},
        },
    )
    assert "the first 2 of the frozen `moisesdb_subset` list" in text
    assert f"| `{tracks[0][:8]}` | bass | `" in text
    assert "## Did the gate flag the right stems?" in text
    assert "## Time" in text


def record(track: str, stems: dict, *, state: str = "ready", wall: float = 100.0) -> dict:
    return {
        "track_id": track,
        "state": state,
        "error": None if state == "ready" else "The separation failed inside the server.",
        "device": "cpu",
        "audio_seconds": 200.0,
        "wall_seconds": wall,
        "timings": [
            {"model_id": "m", "stage": "separation", "wall_seconds": 60.0, "cached": False},
            {"model_id": "m", "stage": "separation", "wall_seconds": 60.0, "cached": True},
            {"model_id": "tagger", "stage": "listening", "wall_seconds": 12.5, "cached": False},
        ],
        "stems": stems,
    }


def made(sdr, *, bench=None, in_song=True, flagged=False, models=("m",), warning="w") -> dict:
    benchmark = None
    if bench is not None:
        difference = round(sdr - bench, 3)
        benchmark = {
            "model_id": models[0],
            "sdr_db": bench,
            "device": "mps",
            "difference_db": difference,
            "agrees": abs(difference) <= pc.AGREE_DB,
        }
    return {
        "model_ids": list(models),
        "attempts": 3 if flagged else 1,
        "warning": warning,
        "flagged_empty": flagged,
        "in_song": in_song,
        "sdr_db": sdr,
        "benchmark": benchmark,
    }


def test_the_summary_counts_agreement_flags_and_time() -> None:
    records = [
        record(
            "aaaaaaaa-1",
            {
                "vocals": made(10.0, bench=10.2),
                "drums": made(9.0, bench=10.0),
                "piano": made(None, in_song=False, flagged=True),
                "other": made(None, in_song=False),
                "guitar": made(0.1, flagged=True),
                "bass": made(8.0, models=("m", "n"), warning=None),
            },
        ),
        record("bbbbbbbb-2", {}, state="failed"),
    ]
    summary = pc.summarise(records)
    assert (summary["tracks"], summary["ready"]) == (2, 1)
    assert summary["failed"] == ["bbbbbbbb-2"]
    assert (summary["stems"], summary["compared"], summary["agree"]) == (6, 2, 1)
    assert summary["median_difference_db"] == -0.6
    assert summary["largest_difference_db"] == -1.0
    assert (summary["not_in_song"], summary["not_in_song_flagged"]) == (2, 1)
    assert summary["not_in_song_unflagged"] == [
        {"track_id": "aaaaaaaa-1", "name": "other", "attempts": 1, "model_ids": ["m"]}
    ]
    assert summary["in_song"] == 4
    assert summary["in_song_flagged"] == [
        {"track_id": "aaaaaaaa-1", "name": "guitar", "sdr_db": 0.1}
    ]
    assert pc.seconds_by_stage(records[0]) == {"separation": 60.0, "listening": 12.5}

    text = pc.render(
        records,
        {"run_id": "r", "finished": "now", "tracks": 2, "device": "cpu", "git": {"commit": "c"}},
    )
    assert "**1 of 2 jobs did not finish.**" in text
    assert "| `aaaaaaaa` | vocals | `m` | 1 | yes | 10.00 | 10.20 | -0.20 | not checked |" in text
    assert "| `aaaaaaaa` | piano | `m` | 3 | no | - | - | - | probably empty |" in text
    assert "| `aaaaaaaa` | bass | `m` + `n` | 1 | yes | 8.00 | - | - | passed |" in text
    assert "| `bbbbbbbb` | - | - | - | - | - | - | - | job failed:" in text
    assert "2 delivered stems could be set beside a benchmark figure. 1 of them are within" in text
    assert "Of 2 stems whose instrument is not in the song, the gate flagged 1" in text
    assert "- `aaaaaaaa` guitar: in the song, flagged; its SDR is 0.10 dB." in text
    assert (
        "- `aaaaaaaa` other: not in the song, not flagged: the first attempt, by `m`, was not "
        "found empty." in text
    )
    retried = pc.render(
        [record("cccccccc-3", {"guitar": {**made(None, in_song=False), "attempts": 2}})],
        {"run_id": "r", "finished": "now", "tracks": 1, "device": "cpu", "git": {"commit": "c"}},
    )
    assert "an earlier attempt was found empty, the retry with `m` was not" in retried
    assert "| `aaaaaaaa` | 200 s | 100 s | 60 s | 12 s | 28 s |" in text


def test_a_difference_that_rounds_to_zero_is_not_shown_as_minus_zero() -> None:
    assert pc._num(-0.001) == "0.00"
    assert pc._num(-0.006) == "-0.01"
    assert pc._num(None) == "-"


def test_report_refuses_a_run_that_has_not_finished(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(pc, "PRODUCT_DIR", tmp_path)
    assert pc.main(["report", "--run-id", "nothing"]) == 2
    assert "has not finished" in capsys.readouterr().err
