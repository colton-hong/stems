"""A dropout or a rest: the gaps of a stem, the three rules and the verdict.

Tones and made-up rows only; no model runs and no kept benchmark output is read.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from stems_bench import gaps  # noqa: E402
from stems_bench.datasets import get_dataset  # noqa: E402

SR = 8000
LIMIT = -21.68


def tone(freq: float, seconds: float, gain: float = 0.1):
    t = np.arange(round(seconds * SR)) / SR
    return (gain * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :].repeat(2, axis=0)


def span(start: float, stop: float) -> slice:
    return slice(round(start * SR), round(stop * SR))


@pytest.fixture
def song():
    """Forty seconds. The true bass rests from 30 s to 36 s and the song ends at 36 s.

    The default model's bass is also missing from 5 s to 15 s (a dropout) and from 20 s to
    23 s (too short to count). The second model has half of the bass from 5 s to 12 s.
    """
    true = tone(60.0, 40.0)
    true[:, span(30, 40)] = 0.0
    guitar = tone(440.0, 40.0)
    guitar[:, span(36, 40)] = 0.0
    mix = true + guitar
    made = true.copy()
    made[:, span(5, 15)] = 0.0
    made[:, span(20, 23)] = 0.0
    second = true.copy()
    second[:, span(5, 12)] *= 0.5
    second[:, span(12, 15)] = 0.0
    return mix, true, made, second


def test_runs_of_true() -> None:
    assert gaps.runs_of(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert gaps.runs_of(np.ones(3, dtype=bool)) == [(0, 3)]
    assert gaps.runs_of(np.zeros(3, dtype=bool)) == []
    assert gaps.runs_of(np.zeros(0, dtype=bool)) == []


def test_a_gap_is_a_long_silence_of_the_stem_while_the_song_goes_on(song) -> None:
    mix, true, made, second = song
    first, last = gaps.measure_stem(mix, true, made, second, SR)
    # The dropout: ten seconds of song and of true bass; the second model sounds for seven.
    assert (first["start"], first["seconds"], first["song"]) == (5.0, 10.0, 10.0)
    assert (first["true"], first["second"]) == (10.0, 7.0)
    assert gaps.is_dropout(first)
    # Half of the bass for seven of the ten seconds.
    assert first["second_share"] == pytest.approx(0.35, abs=0.005)
    # Its level against the song over the gap: 0.05 for 7 of 10 s against two tones of 0.1.
    assert first["second_below_song_db"] == pytest.approx(-10.58, abs=0.05)
    assert first["true_below_song_db"] == pytest.approx(-3.01, abs=0.05)
    # The rest runs on into the silence after the song, which does not count towards it.
    assert (last["start"], last["seconds"], last["song"]) == (30.0, 10.0, 6.0)
    assert (last["true"], last["second"]) == (0.0, 0.0)
    assert not gaps.is_dropout(last)
    assert "second_share" not in last  # no true stem there to have a share of
    assert last["second_below_song_db"] < -60.0


def test_silence_while_the_song_is_silent_is_not_a_gap(song) -> None:
    mix, true, made, second = song
    quiet = slice(0, round(36 * SR))
    for audio in (mix, true, made, second):
        audio[:, span(24, 36)] = 0.0  # the song stops for twelve seconds
    found = gaps.measure_stem(mix[:, quiet], true[:, quiet], made[:, quiet], second[:, quiet], SR)
    assert [gap["start"] for gap in found] == [5.0]
    with pytest.raises(gaps.GapsError, match="fallback's audio has 8000 samples"):
        gaps.measure_stem(mix, true, made, second[:, :SR], SR)


def test_a_stem_the_gate_notes_already(song) -> None:
    mix, true, _, _ = song
    rules = {"silent_fraction_min": 0.98, "empty_below_mix_db": LIMIT}
    assert not gaps.is_noted(true, mix, SR, **rules)
    assert gaps.is_noted(np.zeros_like(true), mix, SR, **rules)
    faint = 0.05 * true  # 26 dB under the bass, and more under the song
    assert gaps.is_noted(faint, mix, SR, **rules)
    assert not gaps.is_noted(faint, mix, SR, silent_fraction_min=0.98, empty_below_mix_db=None)


def gap(true: float, second: float = 0.0, level: float = -80.0, **more) -> dict:
    return {"true": true, "second": second, "second_below_song_db": level, **more}


def test_the_three_rules() -> None:
    assert gaps.is_dropout(gap(5.0))
    assert not gaps.is_dropout(gap(4.9))
    assert gaps.flags("silence", gap(0.0))
    assert gaps.flags("second", gap(0.0, second=5.0))
    assert not gaps.flags("second", gap(9.0, second=4.9))
    assert gaps.flags("loud_second", gap(0.0, level=-20.0), -20.0)
    assert not gaps.flags("loud_second", gap(0.0, level=-20.1), -20.0)
    assert not gaps.flags("loud_second", gap(0.0, level=0.0), None)
    with pytest.raises(gaps.GapsError, match="unknown rule"):
        gaps.flags("anything", gap(0.0))


def test_the_level_is_the_lowest_that_is_reliable() -> None:
    dropouts = [gap(20.0, level=-10.0 - i) for i in range(12)]  # -10 down to -21
    rests = [gap(0.0, level=-30.0 - i) for i in range(8)]  # -30 down to -37
    loud_rest = gap(0.0, level=-15.0)
    # At -30 it would flag 14 with 12 right (86%); at -21, 13 with 12 right (92%).
    assert gaps.choose_threshold([*dropouts, *rests, loud_rest]) == -21.0
    # Eight dropouts cannot make ten flagged gaps nine tenths right.
    assert gaps.choose_threshold([*dropouts[:8], *rests]) is None
    # Rests louder than every dropout: no level works.
    assert gaps.choose_threshold([*dropouts, *[gap(0.0, level=-5.0)] * 3]) is None
    assert gaps.choose_threshold([]) is None


def test_a_rule_is_judged_by_what_it_flags() -> None:
    some = [*[gap(20.0, second=20.0)] * 9, gap(0.0, second=20.0), *[gap(0.0)] * 5, gap(20.0)]
    silence = gaps.outcome(some, "silence")
    assert (silence["gaps"], silence["dropouts"], silence["flagged"], silence["right"]) == (
        16,
        10,
        16,
        10,
    )
    assert silence["precision"] == pytest.approx(10 / 16)
    assert not silence["reliable"]
    second = gaps.outcome(some, "second")
    assert (second["flagged"], second["right"]) == (10, 9)
    assert second["precision"] == pytest.approx(0.9)
    assert second["recall"] == pytest.approx(0.9)
    assert second["reliable"]
    # Nine of nine is not enough gaps to go by.
    few = gaps.outcome([gap(20.0, second=20.0)] * 9, "second")
    assert few["precision"] == 1.0
    assert not few["reliable"]
    assert (gaps.why(second), gaps.why(silence), gaps.why(few)) == (
        "yes",
        "no: wrong too often",
        "no: too few flagged",
    )
    nothing = gaps.outcome([], "silence")
    assert (nothing["precision"], nothing["recall"], nothing["reliable"]) == (None, None, False)


def row(set_name: str, track: str, stem: str, found: list, *, noted: bool = False) -> dict:
    return {
        "id": f"{set_name}/{track}/{stem}",
        "set": set_name,
        "track_id": track,
        "stem": stem,
        "seconds": 200.0,
        "noted": noted,
        "gaps": found,
    }


DROPOUT = gap(20.0, second=20.0, level=-10.0, second_share=0.6)
REST = gap(0.0, second=0.0, level=-70.0)
LOUD_REST = gap(0.0, second=20.0, level=-10.0)
# A dropout the second model has for three seconds only, but loudly.
BRIEF = gap(20.0, second=3.0, level=-12.0, second_share=0.2)


def world(tuning: list, test: list, *, tracks: int = 12) -> list[dict]:
    rows = []
    for name, found in (("tuning", tuning), ("test", test)):
        for number in range(tracks):
            rows.append(row(name, f"t{number:02d}", "bass", list(found)))
            rows.append(row(name, f"t{number:02d}", "piano", []))
    return rows


def test_a_rule_that_holds_on_both_lists_stands() -> None:
    rows = world([DROPOUT, REST], [DROPOUT, REST])
    # A stem that carries the gate's note has no gaps counted, whatever is in it.
    rows.append(row("tuning", "t99", "bass", [LOUD_REST] * 5, noted=True))
    analysis = gaps.analyse(rows)
    assert analysis["tracks"] == {"tuning": 13, "test": 12}
    bass = analysis["stems"]["tuning"]["bass"]
    assert (bass["stems"], bass["noted"], bass["with_a_gap"], bass["with_a_dropout"]) == (
        13,
        1,
        12,
        12,
    )
    assert (bass["gaps"], bass["dropouts"]) == (24, 12)
    assert analysis["stems"]["tuning"]["piano"]["gaps"] == 0
    assert "guitar" not in analysis["stems"]["tuning"]
    # Silence alone is right half the time. The other two are right every time and find as
    # many, so the earlier in the list is the one tried.
    assert not analysis["outcomes"]["tuning"]["silence"]["reliable"]
    assert analysis["threshold"] == -10.0
    assert analysis["reliable_on_tuning"] == ["second", "loud_second"]
    assert analysis["tried"] == "second"
    assert analysis["verdict"] == "stands"
    second = analysis["second"]["test"]
    assert (second["dropouts"], second["speaks_in_dropouts"], second["share"]) == (12, 12, 0.6)
    assert (second["rests"], second["speaks_in_rests"]) == (12, 0)
    assert (second["level_in_dropouts"], second["level_in_rests"]) == (-10.0, -70.0)


def test_the_rule_that_finds_the_most_dropouts_is_the_one_tried() -> None:
    analysis = gaps.analyse(world([DROPOUT, BRIEF, REST], [DROPOUT, BRIEF, REST]))
    assert analysis["threshold"] == -12.0
    assert analysis["outcomes"]["tuning"]["second"]["right"] == 12
    assert analysis["outcomes"]["tuning"]["loud_second"]["right"] == 24
    assert analysis["tried"] == "loud_second"
    assert analysis["verdict"] == "stands"


def test_a_rule_that_does_not_hold_on_the_test_tracks_fails() -> None:
    analysis = gaps.analyse(world([DROPOUT, REST], [DROPOUT, LOUD_REST]))
    assert analysis["tried"] == "second"
    assert analysis["outcomes"]["test"]["second"]["precision"] == pytest.approx(0.5)
    assert analysis["verdict"] == "fails"
    # Too few gaps on the test tracks is not reliable either.
    few = gaps.analyse(world([DROPOUT, REST], [DROPOUT, REST])[: 24 + 9 * 2])
    assert few["outcomes"]["test"]["second"]["flagged"] == 9
    assert few["verdict"] == "fails"


def test_when_nothing_tells_them_apart_no_rule_is_tried() -> None:
    analysis = gaps.analyse(world([DROPOUT, LOUD_REST], [DROPOUT, REST]))
    assert analysis["threshold"] is None
    assert analysis["reliable_on_tuning"] == []
    assert analysis["tried"] is None
    assert analysis["verdict"] == "none"
    # The third rule flags nothing without a level, on either list.
    assert analysis["outcomes"]["test"]["loud_second"]["flagged"] == 0


RUN = {
    "run_id": "r",
    "models": {"bass": ["main", "backup"], "piano": ["main", "keys"]},
    "left_out": ["vocals", "drums", "guitar", "other"],
    "git": {"commit": "abcdef0123", "dirty": False},
}


def test_the_report_says_what_was_counted_and_what_follows() -> None:
    stands = gaps.render(gaps.analyse(world([DROPOUT, REST], [DROPOUT, REST])), RUN)
    assert "from the run `r`, made at commit `abcdef0`. No model was run" in stands
    assert "the 12 tuning tracks and the 12 test tracks" in stands
    assert "Counted: bass (`main`, then `backup`), piano (`main`, then `keys`)" in stands
    assert "for every track: vocals, drums, guitar, other." in stands
    assert "**The rule stands:" in stands
    assert "| tuning | bass | 12 | 0 | 12 | 12 | 24 | 12 | 12 |" in stands
    assert "| test | piano | 12 | 0 | 0 | 0 | 0 | 0 | 0 |" in stands
    assert "flags 10 gaps or more and at least 90% of them are dropouts" in stands
    assert "on the tuning tracks: -10.00 dB against the song." in stands
    assert (
        "| `silence` | every gap | tuning | 24 | 12 | 50% | 12 of 12 | no: wrong too often |"
        in stands
    )
    assert "| test | 12 | 12 | 100% | 12 of 12 | yes |" in stands
    assert "Tried on the test tracks: `second`." in stands
    assert "| test | 12 | 12 | -10.0 dB | 60% | 12 | 0 | -70.0 dB |" in stands
    assert "2 of the six stems" in stands
    fails = gaps.render(gaps.analyse(world([DROPOUT, REST], [DROPOUT, LOUD_REST])), RUN)
    assert "is not reliable on the test tracks" in fails
    none = gaps.render(gaps.analyse(world([DROPOUT, LOUD_REST], [DROPOUT, REST])), RUN)
    assert "**No rule is reliable on the tuning tracks" in none
    assert "nothing here shows that the product can tell a dropout from a rest.**" in none
    assert "| tuning | 0 | 0 | - | 0 of 12 | no: too few flagged |" in none
    assert "no level is reliable there, so the rule flags nothing." in none
    assert "No rule was tried on the test tracks" in none
    dirty = gaps.render(gaps.analyse([]), {**RUN, "git": {"commit": "abcdef0123", "dirty": True}})
    assert "`abcdef0`, with uncommitted changes." in dirty


# -- collecting, from a made-up benchmark folder --------------------------------------------

MODELS = {"bass": ("main", "backup"), "guitar": ("main", "strings"), "piano": ("main", "keys")}


def fake_bench(bench_dir, dataset, *, skip: tuple = ()) -> None:
    """Kept outputs as a benchmark run leaves them. The main model's bass is missing from
    2 s to 9 s, its guitar is silent throughout, and its piano is the true one."""
    for number, track_id in enumerate(dataset.track_ids()):
        track = dataset.load(track_id)
        rate = track.sample_rate
        made = {
            "bass": track.references["bass"].copy(),
            "guitar": np.zeros_like(track.references["guitar"]),
            "piano": track.references["piano"].copy(),
        }
        made["bass"][:, 2 * rate : 9 * rate] = 0.0
        kept = {"main": made}
        for stem, (_, fallback) in MODELS.items():
            kept[fallback] = {stem: 0.5 * track.references[stem]}
        for model, stems in kept.items():
            if (model, track_id) in skip:
                continue
            key = f"{number:064x}"
            folder = bench_dir / "estimates" / dataset.name / model / key
            folder.mkdir(parents=True)
            for stem, audio in stems.items():
                np.save(folder / f"{stem}.npy", audio)
            result = {
                "kind": "model",
                "input": "mix",
                "track_id": track_id,
                "outputs": {stem: {"target": stem, "derived": False, "bss": {}} for stem in stems},
            }
            path = bench_dir / "results" / dataset.name / model / f"{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))


def test_a_stem_is_counted_only_when_both_models_are_kept_for_every_track(tmp_path) -> None:
    dataset = get_dataset("synthetic")
    track_ids = dataset.track_ids()
    fake_bench(tmp_path, dataset, skip=(("keys", track_ids[3]),))
    found = gaps.index(tmp_path, dataset.name)
    assert ("main", track_ids[0], "bass") in found
    assert ("keys", track_ids[3], "piano") not in found
    lists = {"tuning": track_ids, "test": track_ids[:2]}
    indexes = {"tuning": found, "test": found}
    assert gaps.covered_stems(indexes, lists, MODELS) == ["bass", "guitar"]
    assert gaps.covered_stems(indexes, {"tuning": track_ids[:3], "test": []}, MODELS) == [
        "bass",
        "guitar",
        "piano",
    ]


def test_collect_makes_a_row_for_each_stem_once_and_resumes(tmp_path) -> None:
    dataset = get_dataset("synthetic")
    fake_bench(tmp_path / "bench", dataset)
    out = tmp_path / "run" / "rows.jsonl"
    said: list[str] = []
    options = {
        "silent_fraction_min": 0.98,
        "empty_below_mix_db": LIMIT,
        "bench_dir": tmp_path / "bench",
        "progress": said.append,
    }
    tracks = len(dataset.track_ids())
    assert gaps.collect("tuning", dataset, MODELS, out, **options) == 3 * tracks
    assert said[-1] == f"tuning: {tracks} of {tracks} tracks"
    rows = gaps.read_rows(out)
    assert {r["id"] for r in rows} == {
        f"tuning/{track_id}/{stem}" for track_id in dataset.track_ids() for stem in MODELS
    }
    by_stem = {stem: [r for r in rows if r["stem"] == stem] for stem in MODELS}
    # The bass: one gap of seven seconds, a dropout, and the second model has half of it.
    for found in by_stem["bass"]:
        assert not found["noted"]
        (only,) = found["gaps"]
        assert (only["start"], only["seconds"], only["true"], only["second"]) == (
            2.0,
            7.0,
            7.0,
            7.0,
        )
        assert only["second_share"] == pytest.approx(0.5, abs=0.001)
    assert all(found["noted"] for found in by_stem["guitar"])  # silent: the gate's note
    # The piano is the true one: no gaps, and the gate's note only where the song has none.
    without = {t for t in dataset.track_ids() if not dataset.load(t).references["piano"].any()}
    assert {found["track_id"] for found in by_stem["piano"] if found["noted"]} == without
    assert all(found["gaps"] == [] for found in by_stem["piano"] if not found["noted"])
    # A second pass makes nothing; a list of another name makes its own rows.
    assert gaps.collect("tuning", dataset, MODELS, out, **options) == 0
    assert gaps.collect("test", dataset, {"bass": MODELS["bass"]}, out, **options) == tracks
    analysis = gaps.analyse(gaps.read_rows(out))
    assert analysis["stems"]["tuning"]["guitar"]["noted"] == tracks
    assert analysis["outcomes"]["tuning"]["silence"]["flagged"] == tracks


def test_the_plan_names_two_different_models_for_every_stem() -> None:
    pytest.importorskip("yaml")
    pairs = gaps.plan_models()
    assert pairs
    assert all(default != fallback for default, fallback in pairs.values())


def test_report_refuses_a_run_that_has_not_finished(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(gaps, "GAPS_DIR", tmp_path)
    assert gaps.main(["report", "--run-id", "nothing"]) == 2
    assert "has not finished (state: None)" in capsys.readouterr().err
    run_dir = tmp_path / "done"
    run_dir.mkdir()
    rows = world([DROPOUT, REST], [DROPOUT, REST])
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "run.json").write_text(json.dumps(RUN))
    (run_dir / "status.json").write_text(json.dumps({"state": "running"}))
    assert gaps.main(["report", "--run-id", "done"]) == 2
    (run_dir / "status.json").write_text(json.dumps({"state": "done"}))
    out = tmp_path / "report.md"
    assert gaps.main(["report", "--run-id", "done", "--out", str(out)]) == 0
    assert "**The rule stands:" in out.read_text()
