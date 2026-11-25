"""The two ways of making `other`, joined: the gaps, the level, the join and the report.

Synthetic songs and a made-up benchmark folder; no model runs and nothing is scored by
museval (a stand-in scorer takes its place).
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from stems_bench import other_joined as oj  # noqa: E402
from stems_bench.datasets import get_dataset  # noqa: E402

RULES = {"silent_fraction_min": 0.98, "empty_below_mix_db": -21.68}


class OneSongWithoutOther:
    """The synthetic songs, the first of them with its `other` taken out of everything."""

    name = "synthetic"

    def __init__(self) -> None:
        self._songs = get_dataset("synthetic")

    def track_ids(self) -> list[str]:
        return self._songs.track_ids()

    def load(self, track_id: str):
        track = self._songs.load(track_id)
        if track_id != self.track_ids()[0]:
            return track
        other = track.references["other"]
        references = {
            **track.references,
            "other": np.zeros_like(other),
            "other_4stem": track.references["other_4stem"] - other,
        }
        return dataclasses.replace(
            track,
            references=references,
            inputs={**track.inputs, "mix": track.inputs["mix"] - other},
        )


def fake_bench(bench_dir, dataset, *, skip: tuple = ()) -> None:
    """Kept outputs as a benchmark run leaves them. ``six`` has the true guitar and piano,
    and an `other` that is missing twice in every song (0.5 s to 6 s, and 6.5 s to the
    end); ``four`` has the true four-stem other.
    """
    for number, track_id in enumerate(dataset.track_ids()):
        track = dataset.load(track_id)
        refs, rate = track.references, track.sample_rate
        own = refs["other"].copy()
        own[:, rate // 2 : 6 * rate] = 0.0
        own[:, 13 * rate // 2 :] = 0.0
        kept = {
            "six": {"other": (own, "other"), "guitar": (refs["guitar"], "guitar"),
                    "piano": (refs["piano"], "piano")},
            "four": {"other": (refs["other_4stem"], "other_4stem")},
        }  # fmt: skip
        key = f"{number:064x}"
        for model, stems in kept.items():
            folder = bench_dir / "estimates" / dataset.name / model / key
            folder.mkdir(parents=True)
            outputs = {}
            for stem, (audio, target) in stems.items():
                if (model, stem, track_id) in skip:
                    continue
                np.save(folder / f"{stem}.npy", audio.astype(np.float32))
                outputs[stem] = {"target": target, "derived": False, "bss": {"sdr": 6.0}}
            result = {"kind": "model", "input": "mix", "track_id": track_id, "outputs": outputs}
            path = bench_dir / "results" / dataset.name / model / f"{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))


def scorer(true, estimate, mix, rate):
    """Stands in for the benchmark's protocol: a window is scored unless the estimate is
    silent in it, and the score is 9 dB for the true stem and 1 dB for anything else."""
    windows = true.shape[-1] // rate
    scored = sum(
        1 for w in range(windows) if np.any(estimate[..., w * rate : (w + 1) * rate])
    )  # fmt: skip
    exact = np.allclose(estimate, true, atol=1e-6)
    return SimpleNamespace(sdr=9.0 if exact else 1.0, scored_windows=scored, windows=windows)


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    root = tmp_path_factory.mktemp("other-joined")
    dataset = OneSongWithoutOther()
    fake_bench(root / "bench", dataset)
    rows_path, joined_path = root / "run" / "rows.jsonl", root / "run" / "joined.jsonl"
    options = {"rules": RULES, "bench_dir": root / "bench"}
    said: list[str] = []
    for name in oj.SETS:
        added = oj.collect(name, dataset, "six", "four", rows_path, progress=said.append, **options)
        assert added == len(dataset.track_ids())
    rows = oj.read_rows(rows_path)
    level = oj.choose_level(rows)
    for name in oj.SETS:
        oj.join_songs(
            name, dataset, "six", "four", rows, level, joined_path, score=scorer, **options
        )
    return root, dataset, rows_path, joined_path, said


def test_the_gaps_of_the_defaults_own_and_what_the_subtracted_one_has_there(made) -> None:
    root, dataset, rows_path, _, said = made
    tracks = dataset.track_ids()
    assert said[-1] == f"test: gaps of {len(tracks)} of {len(tracks)} tracks"
    rows = [row for row in oj.read_rows(rows_path) if row["set"] == "tuning"]
    without, song = rows[0], rows[1]
    # Two gaps a song, each a dropout: the true `other` plays through both.
    assert song["sounding"] == 12.0
    assert [(gap["start"], gap["seconds"], gap["true"]) for gap in song["gaps"]] == [
        (0.5, 5.5, 5.5),
        (6.5, 5.5, 5.5),
    ]
    assert song["own"]["dropped"] == 11.0
    assert song["subtracted"]["dropped"] == 0.0
    # The song with no `other`: own is silent throughout, one long rest, and noted.
    assert without["sounding"] == 0.0
    assert [(gap["seconds"], gap["true"]) for gap in without["gaps"]] == [(12.0, 0.0)]
    assert without["own"]["noted"]
    assert without["subtracted"]["noted"]
    # The subtracted stem is far louder over a dropout than over the rest.
    assert song["gaps"][0]["second_below_song_db"] > -30.0
    assert without["gaps"][0]["second_below_song_db"] < -60.0
    again = oj.collect(
        "tuning", dataset, "six", "four", rows_path, rules=RULES, bench_dir=root / "bench"
    )
    assert again == 0


def test_the_join_is_own_with_the_subtracted_stem_in_the_gaps_it_fills() -> None:
    rate = 1000
    own = np.ones((2, 3 * rate), dtype=np.float32)
    own[:, 1000:2000] = 0.0
    subtracted = np.full((2, 3 * rate), 5.0, dtype=np.float32)
    joined = oj.join(own, subtracted, [{"start": 1.0, "seconds": 1.0}], rate)
    assert np.all(joined[:, :1000] == 1.0)
    assert np.all(joined[:, 1000:2000] == 5.0)
    assert np.all(joined[:, 2000:] == 1.0)
    assert np.all(oj.join(own, subtracted, [], rate) == own)
    # A gap that runs to the end of the song stops there.
    to_the_end = oj.join(own, subtracted, [{"start": 2.5, "seconds": 5.0}], rate)
    assert to_the_end.shape == own.shape
    assert np.all(to_the_end[:, 2500:] == 5.0)


def test_joining_gives_back_what_the_defaults_own_dropped(made) -> None:
    _, dataset, rows_path, joined_path, _ = made
    tracks = dataset.track_ids()
    rows, joined = oj.read_rows(rows_path), oj.read_rows(joined_path)
    analysis = oj.analyse(rows, joined)
    # The level is the lowest that is reliable: here even the one rest may be filled, since
    # fourteen of fifteen gaps are dropouts.
    n = len(tracks) - 1
    assert analysis["outcomes"]["tuning"]["gaps"] == 2 * n + 1
    assert analysis["outcomes"]["tuning"]["dropouts"] == 2 * n
    assert analysis["level"] is not None
    assert analysis["outcomes"]["tuning"]["reliable"]
    assert analysis["verdict"] == "stands"
    assert analysis["joined_tracks"] == {"tuning": len(tracks), "test": len(tracks)}
    test = analysis["versions"]["test"]
    assert test["own"]["present"]["share"] == pytest.approx(11.0 / 12.0)
    assert test["own"]["present"]["long"] == n
    assert test["subtracted"]["present"]["share"] == 0.0
    assert test["joined"]["present"]["share"] == 0.0
    assert test["joined"]["present"]["long"] == 0
    # The stand-in scorer: own is scored in the two seconds of twelve that it sounds in.
    assert (test["own"]["scored_windows"], test["own"]["windows"]) == (2 * n, 12 * n)
    assert test["joined"]["scored_windows"] == 12 * n
    assert test["joined"]["present"]["sdr"] == 9.0
    # The song with no `other` is still empty after the join, and noted.
    assert test["joined"]["absent"] == {
        "songs": 1,
        "below_song_db": pytest.approx(-100.0, abs=25.0),
        "noted": 1,
    }
    one = next(row for row in joined if row["set"] == "test" and row["track_id"] == tracks[1])
    assert (one["filled"], one["joined"]["dropped"]) == (2, 0.0)


def gap(true: float, level: float) -> dict:
    return {"start": 1.0, "seconds": 9.0, "song": 9.0, "true": true, "second": 9.0,
            "second_below_song_db": level}  # fmt: skip


def rows_of(tuning: list, test: list, songs: int = 12) -> list[dict]:
    stem = {"dropped": 9.0, "longest": 9.0, "below_song_db": -10.0, "noted": False}
    return [
        {"id": f"{name}/t{i}", "set": name, "track_id": f"t{i}", "sounding": 100.0,
         "own": stem, "subtracted": {**stem, "dropped": 0.0, "longest": 0.0}, "gaps": list(found)}
        for name, found in (("tuning", tuning), ("test", test))
        for i in range(songs)
    ]  # fmt: skip


RUN = {
    "run_id": "r",
    "default_model": "six",
    "sub_model": "four",
    "git": {"commit": "abcdef0123", "dirty": False},
}


def test_the_three_outcomes_and_what_the_report_says() -> None:
    dropout, rest, loud_rest = gap(9.0, -8.0), gap(0.0, -40.0), gap(0.0, -8.0)
    stands = oj.analyse(rows_of([dropout, rest], [dropout, rest]))
    assert (stands["level"], stands["verdict"]) == (-8.0, "stands")
    assert stands["outcomes"]["test"]["flagged"] == 12
    text = oj.render(stands, RUN)
    assert "from the run `r`, made at commit `abcdef0`." in text
    assert "**Own** is `six`'s `other`. **Subtracted** is `four`'s four-stem `other`" in text
    assert "**The join stands:" in text
    assert "On the tuning tracks: T = -8.00 dB." in text
    assert "| tuning | 12 | 24 | 12 | 12 | 12 | 100% | 12 of 12 | yes |" in text
    # No song was joined in these made-up rows, so the joined stem has no row of its own.
    assert "| test | the default's own | 12 | 9.0% | 12 | - | - | 0 | - | 0 of 0 |" in text
    assert "the two joined" not in text

    fails = oj.analyse(rows_of([dropout, rest], [dropout, loud_rest]))
    assert fails["verdict"] == "fails"
    assert fails["outcomes"]["test"]["precision"] == pytest.approx(0.5)
    assert "the join does not stand.**" in oj.render(fails, RUN)

    none = oj.analyse(rows_of([dropout, loud_rest], [dropout, rest]))
    assert (none["level"], none["verdict"]) == (None, "none")
    assert none["nearest"] == {"level": -8.0, "flagged": 24, "right": 12, "share": 0.5}
    text = oj.render(none, RUN)
    assert "**No level is reliable on the tuning tracks, so nothing was joined" in text
    assert (
        "no level is. The nearest: at -8.00 dB, 24 gaps filled, 12 of them dropouts (50%)." in text
    )
    assert "| no: no level found |" in text
    assert "No stem was scored in this run: scoring is part of joining" in text
    assert "No stem was scored in this run" not in oj.render(stands, RUN)
    assert oj.RULES_COMMIT.startswith("5c539a1d25f1")
    # Joined rows made at another level than the one now chosen are not used.
    stale = [{"id": "test/t0", "set": "test", "level": -99.0, "joined": {}, "scores": {}}]
    assert oj.analyse(rows_of([dropout, rest], [dropout, rest]), stale)["joined_tracks"] == {
        "tuning": 0,
        "test": 0,
    }


def test_nothing_is_counted_until_everything_it_needs_is_kept(tmp_path) -> None:
    dataset = OneSongWithoutOther()
    tracks = dataset.track_ids()
    fake_bench(tmp_path / "bench", dataset, skip=(("six", "other", tracks[2]),))
    with pytest.raises(oj.JoinError, match=f"not kept: six's other of {tracks[2]}"):
        oj.collect(
            "tuning", dataset, "six", "four", tmp_path / "rows.jsonl", rules=RULES,
            bench_dir=tmp_path / "bench",
        )  # fmt: skip
    assert not (tmp_path / "rows.jsonl").exists()


def test_the_plan_names_a_four_stem_model_for_the_subtraction() -> None:
    pytest.importorskip("yaml")
    default_model, sub_model, rules = oj.plan()
    assert default_model != sub_model
    assert set(rules) == {"silent_fraction_min", "empty_below_mix_db"}


def test_the_commands_wait_for_one_another(made, tmp_path, monkeypatch, capsys) -> None:
    _, _, rows_path, joined_path, _ = made
    monkeypatch.setattr(oj, "OUT_DIR", tmp_path)
    assert oj.main(["report", "--run-id", "nothing"]) == 2
    assert "has not finished" in capsys.readouterr().err
    assert oj.main(["join", "--run-id", "nothing"]) == 2
    assert "run `collect` first" in capsys.readouterr().err
    run_dir = tmp_path / "done"
    run_dir.mkdir()
    (run_dir / "rows.jsonl").write_text(rows_path.read_text())
    (run_dir / "joined.jsonl").write_text(joined_path.read_text())
    (run_dir / "run.json").write_text(json.dumps(RUN))
    (run_dir / "status.json").write_text(json.dumps({"state": "done"}))
    assert oj.main(["report", "--run-id", "done"]) == 2  # joined, but not said to be finished
    capsys.readouterr()
    (run_dir / "join-status.json").write_text(json.dumps({"state": "done"}))
    report = tmp_path / "report.md"
    assert oj.main(["report", "--run-id", "done", "--out", str(report)]) == 0
    text = report.read_text()
    assert "**The join stands:" in text
    assert "| test | the two joined | " in text
