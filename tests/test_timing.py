"""Every real model run is logged with its device and speed; cache hits are not."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from test_cli import OVERRIDES, Song  # noqa: E402

from stems_common.synthetic import generate_track  # noqa: E402
from stems_worker import cli, timing  # noqa: E402
from stems_worker.cache import OutputCache  # noqa: E402
from stems_worker.compose import ModelRun, run_model  # noqa: E402
from stems_worker.fakes import OracleSeparator  # noqa: E402
from stems_worker.timing import TimingLog, read_timings, summarise  # noqa: E402

SR = 44100


@pytest.fixture(scope="module")
def track():
    return generate_track(seed=51, duration_seconds=2.0)


def _factory(track):
    def factory(model_id, stems):
        return [OracleSeparator({s: track.stems[s] for s in stems}, model_id=model_id)]

    return factory


# --- acceptance ----------------------------------------------------------------------------


def test_a_model_run_appends_one_line_with_the_required_fields(track, tmp_path) -> None:
    log = TimingLog(tmp_path / "runs" / "timings.jsonl", {"audio_sha256": "abc", "overlap": 0.25})
    _, records = run_model(
        ModelRun("fake_model", ("drums", "bass")),
        track.mix,
        SR,
        _factory(track),
        device="cpu",
        timings=log,
    )

    (line,) = [json.loads(raw) for raw in log.path.read_text().splitlines()]
    assert line["model_id"] == "fake_model"
    assert line["device"] == "cpu"
    assert line["audio_seconds"] == pytest.approx(2.0)
    assert line["wall_seconds"] == pytest.approx(records[0].wall_seconds, abs=1e-3)
    assert line["realtime_factor"] == pytest.approx(line["wall_seconds"] / 2.0, abs=1e-4)
    assert line["outputs"] == ["drums", "bass"]
    assert line["stage"] == "separation"
    assert line["num_chunks"] == records[0].num_chunks
    assert (line["audio_sha256"], line["overlap"]) == ("abc", 0.25)
    assert line["at"].endswith("+00:00")


def test_runs_append_and_cache_hits_are_not_logged(track, tmp_path) -> None:
    log = TimingLog(tmp_path / "timings.jsonl")
    cache = OutputCache(
        tmp_path / "cache", audio_hash="a", overlap=0.25, device="cpu", fingerprint=lambda m: m
    )
    run = ModelRun("fake_model", ("drums",))
    for _ in range(3):
        run_model(run, track.mix, SR, _factory(track), device="cpu", timings=log, cache=cache)
    assert len(read_timings(log.path)) == 1  # the second and third came from the cache

    run_model(
        ModelRun("other_model", ("bass",)),
        track.mix,
        SR,
        _factory(track),
        device="cpu",
        timings=log,
    )
    assert [line["model_id"] for line in read_timings(log.path)] == ["fake_model", "other_model"]


def test_a_bag_logs_one_line_per_model_it_ran(track, tmp_path) -> None:
    def bag(model_id, stems):
        return [OracleSeparator({s: track.stems[s]}, model_id=model_id) for s in stems]

    log = TimingLog(tmp_path / "timings.jsonl")
    run_model(
        ModelRun("fake_bag", ("drums", "bass")), track.mix, SR, bag, device="cpu", timings=log
    )
    assert [line["outputs"] for line in read_timings(log.path)] == [["drums"], ["bass"]]


# --- through the pipeline and the command --------------------------------------------------


def test_separating_a_file_logs_every_model_and_stage(tmp_path) -> None:
    song = Song(tmp_path)
    manifest = song.separate(tmp_path / "out", stems=["vocals", "guitar"], split_vocals=True)
    lines = read_timings(song.timings_path)

    assert [(line["model_id"], line["stage"]) for line in lines] == [
        ("fake4", "separation"),
        ("fake6", "separation"),
        ("fake_kara", "vocal_split"),
    ]
    for line in lines:
        assert line["device"] == "cpu"
        assert line["audio_sha256"] == manifest["input"]["audio_sha256"]
        assert line["overlap"] == manifest["request"]["overlap"]
        assert "torch" in line
        assert "platform" in line
        assert "code_version" in line
    # The log identifies audio by hash only: no file name, no path.
    assert song.path.name not in song.timings_path.read_text()
    assert str(tmp_path) not in song.timings_path.read_text()

    song.separate(tmp_path / "again", stems=["vocals", "guitar"], split_vocals=True)
    assert len(read_timings(song.timings_path)) == 3  # all cached: nothing new


def test_the_log_can_be_switched_off(tmp_path) -> None:
    song = Song(tmp_path)
    song.separate(tmp_path / "out", stems=["drums"], timings_path=None)
    assert not song.timings_path.exists()


def test_command_line_options(tmp_path) -> None:
    song = Song(tmp_path)
    chosen = tmp_path / "elsewhere" / "t.jsonl"
    base = [
        str(song.path),
        *OVERRIDES,
        "--device",
        "cpu",
        "--quiet",
        "--no-cache",
        "--stems",
        "bass",
    ]
    assert cli.main([*base, "--timings-log", str(chosen)], **song.inject()) == 0
    assert [line["model_id"] for line in read_timings(chosen)] == ["fake4"]
    assert cli.main([*base, "--timings-log", str(chosen), "--no-timings"], **song.inject()) == 0
    assert len(read_timings(chosen)) == 1


def test_default_log_location_is_under_runs() -> None:
    assert timing.DEFAULT_TIMINGS_PATH.name == "timings.jsonl"
    assert timing.DEFAULT_TIMINGS_PATH.parent.name == "runs"


# --- reading and summarising ---------------------------------------------------------------


def test_damaged_lines_are_skipped(tmp_path) -> None:
    path = tmp_path / "timings.jsonl"
    good = {"model_id": "m", "device": "cpu", "wall_seconds": 1.0, "audio_seconds": 2.0}
    path.write_text(json.dumps(good) + "\n" + '{"model_id": "m", "wall_sec' + "\n[1, 2]\n\n")
    assert read_timings(path) == [good]
    assert read_timings(tmp_path / "absent.jsonl") == []


def test_summary_groups_by_model_and_device() -> None:
    def line(model, device, factor, audio=10.0):
        return {
            "model_id": model,
            "device": device,
            "audio_seconds": audio,
            "wall_seconds": factor * audio,
            "realtime_factor": factor,
        }

    rows = summarise(
        [
            line("a", "cpu", 0.9),
            line("a", "cpu", 0.5),
            line("a", "cpu", 0.7),
            line("a", "mps", 0.2),
            line("b", "cpu", 2.0),
        ]
    )
    assert [(r["model_id"], r["device"], r["runs"]) for r in rows] == [
        ("a", "cpu", 3),
        ("a", "mps", 1),
        ("b", "cpu", 1),
    ]
    assert rows[0]["median_realtime_factor"] == 0.7
    assert (rows[0]["min_realtime_factor"], rows[0]["max_realtime_factor"]) == (0.5, 0.9)
    assert rows[0]["audio_seconds"] == 30.0


def test_summary_command(tmp_path, capsys) -> None:
    log = TimingLog(tmp_path / "t.jsonl")
    log.record(
        model_id="m", device="cpu", outputs=["x"], audio_seconds=10, wall_seconds=5, num_chunks=2
    )
    assert timing.main([str(log.path)]) == 0
    printed = capsys.readouterr().out
    assert "m" in printed
    assert "0.500" in printed
    assert timing.main([str(tmp_path / "none.jsonl")]) == 0
    assert "no timings" in capsys.readouterr().out
