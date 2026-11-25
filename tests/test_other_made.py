"""`other` made by subtraction: putting it together from kept outputs, and the report.

Synthetic songs and a made-up benchmark folder; no model runs.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

np = pytest.importorskip("numpy")

from stems_bench import other_made as om  # noqa: E402
from stems_bench import product_check as pc  # noqa: E402
from stems_bench.datasets import get_dataset  # noqa: E402

LIMIT = -21.68
RULES = {"silent_fraction_min": 0.98, "empty_below_mix_db": LIMIT}
FOUR, LEAKY, HOLLOW = (om.way_name(model) for model in ("four", "leaky", "hollow"))


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
    """Kept outputs as a benchmark run leaves them. Three four-stem models: ``four`` has the
    true four-stem other; ``leaky`` has some of the vocals in it too; ``hollow`` has the
    guitar and the piano and no `other` at all. ``six`` has the true guitar and piano.
    """
    for number, track_id in enumerate(dataset.track_ids()):
        track = dataset.load(track_id)
        refs = track.references
        kept = {
            "four": {"other": (refs["other_4stem"], "other_4stem")},
            "leaky": {"other": (refs["other_4stem"] + 0.3 * refs["vocals"], "other_4stem")},
            "hollow": {"other": (refs["guitar"] + refs["piano"], "other_4stem")},
            "six": {"guitar": (refs["guitar"], "guitar"), "piano": (refs["piano"], "piano")},
        }
        key = f"{number:064x}"
        for model, stems in kept.items():
            if (model, track_id) in skip:
                continue
            folder = bench_dir / "estimates" / dataset.name / model / key
            folder.mkdir(parents=True)
            outputs = {}
            for stem, (audio, target) in stems.items():
                np.save(folder / f"{stem}.npy", audio.astype(np.float32))
                outputs[stem] = {"target": target, "derived": False, "bss": {"sdr": 6.0}}
            result = {"kind": "model", "input": "mix", "track_id": track_id, "outputs": outputs}
            path = bench_dir / "results" / dataset.name / model / f"{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        # The benchmark scored one subtraction itself: `four` minus the guitar and piano of `six`.
        scored = {
            "kind": "ensemble",
            "track_id": track_id,
            "members": ["four"],
            "minus": [
                {"slot": "guitar", "members": ["six"]},
                {"slot": "piano", "members": ["six"]},
            ],
            "outputs": {"other": {"target": "other", "bss": {"sdr": 4.0}}},
        }
        path = bench_dir / "results" / dataset.name / "_ensembles" / om.SCORED_SLOT / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scored))


@pytest.fixture(scope="module")
def counted(tmp_path_factory):
    root = tmp_path_factory.mktemp("other-made")
    dataset = OneSongWithoutOther()
    fake_bench(root / "bench", dataset)
    out = root / "run" / "rows.jsonl"
    said: list[str] = []
    for name in om.LISTS:
        added = om.collect(
            name, dataset, "six", out, bench_dir=root / "bench", progress=said.append, **RULES
        )
        assert added == len(dataset.track_ids())
    return root, dataset, out, said


def test_what_is_found_among_the_kept_outputs(tmp_path) -> None:
    dataset = OneSongWithoutOther()
    tracks = dataset.track_ids()
    fake_bench(tmp_path, dataset, skip=(("leaky", tracks[2]),))
    others = om.four_stem_others(tmp_path, dataset.name)
    assert {model for model, _ in others} == {"four", "leaky", "hollow"}  # not `six`
    assert ("leaky", tracks[2]) not in others
    assert om.four_stem_others(tmp_path, "nothing") == {}
    scores = om.scored_subtraction(tmp_path, dataset.name, "six")
    assert scores == {("four", track): 4.0 for track in tracks}
    # Scored with another model's guitar and piano taken out: not this way of making it.
    assert om.scored_subtraction(tmp_path, dataset.name, "another") == {}
    assert om.way_name("four") == "four minus guitar and piano"


def test_collect_puts_other_together_and_counts_it_once(counted) -> None:
    root, dataset, out, said = counted
    tracks = dataset.track_ids()
    assert said[-1] == f"MoisesDB tuning: {len(tracks)} of {len(tracks)} tracks"
    rows = [row for row in om.read_rows(out) if row["list"] == "MoisesDB test"]
    assert [row["track_id"] for row in rows] == tracks
    without, with_other = rows[0], rows[1]
    # The true four-stem other, less the guitar and the piano, is the true `other`.
    assert with_other["sounding"] == 12.0
    assert with_other["ways"][FOUR]["dropped"] == 0.0
    assert with_other["ways"][FOUR]["sdr"] == 4.0
    assert not with_other["ways"][FOUR]["noted"]
    # A four-stem other with no `other` in it leaves nothing: all of it is dropped.
    assert with_other["ways"][HOLLOW] == {
        "dropped": 12.0,
        "longest": 12.0,
        "below_song_db": pytest.approx(-100.0, abs=25.0),
        "noted": True,
        "sdr": None,
    }
    # In the song with no `other`, what a leaky model leaves is loud enough to pass for one.
    assert without["sounding"] == 0.0
    assert without["ways"][FOUR]["noted"]
    assert not without["ways"][LEAKY]["noted"]
    assert LIMIT < without["ways"][LEAKY]["below_song_db"] < -5.0
    again = om.collect("MoisesDB test", dataset, "six", out, bench_dir=root / "bench", **RULES)
    assert again == 0


def test_without_the_defaults_guitar_and_piano_nothing_is_put_together(tmp_path) -> None:
    dataset = OneSongWithoutOther()
    fake_bench(tmp_path / "bench", dataset, skip=(("six", dataset.track_ids()[4]),))
    out = tmp_path / "rows.jsonl"
    om.collect("MoisesDB test", dataset, "six", out, bench_dir=tmp_path / "bench", **RULES)
    assert all(row["ways"] == {} for row in om.read_rows(out))
    assert om.by_way(om.read_rows(out), "MoisesDB test") == {}


def test_an_output_of_the_wrong_length_stops_the_count(tmp_path) -> None:
    dataset = OneSongWithoutOther()
    fake_bench(tmp_path / "bench", dataset)
    first = sorted((tmp_path / "bench" / "estimates" / dataset.name / "four").iterdir())[1]
    np.save(first / "other.npy", np.zeros((2, 100), dtype=np.float32))
    with pytest.raises((om.OtherError, ValueError)):
        om.collect(
            "MoisesDB test", dataset, "six", tmp_path / "r.jsonl", bench_dir=tmp_path / "bench",
            **RULES,
        )  # fmt: skip


def product_records(tracks: list[str]) -> list[dict]:
    """The product's own run on three songs: one has no `other` and carries the note."""

    def record(track: str, sounding: float, flagged: bool) -> dict:
        lost = {"other": {"sounding": sounding, "dropped": sounding / 4, "longest": 3.0}}
        return {
            "track_id": track,
            "lost": {"stems": lost},
            "stems": {"other": {"sdr_db": 2.5 if sounding else None, "flagged_empty": flagged}},
        }

    return [
        record(tracks[0], 0.0, True),
        record(tracks[1], 12.0, False),
        record(tracks[2], 12.0, False),
        {"track_id": "failed", "state": "failed"},
    ]


RUN = {"run_id": "r", "default_model": "six", "git": {"commit": "abcdef0123", "dirty": False}}


def test_the_report_sets_the_ways_beside_the_defaults_own(counted) -> None:
    _, dataset, out, _ = counted
    rows = om.read_rows(out)
    tracks = dataset.track_ids()
    n = len(tracks)
    ways = om.by_way(rows, "MoisesDB test")
    assert list(ways) == [FOUR, HOLLOW, LEAKY]
    assert (ways[FOUR]["present"]["stems"], ways[FOUR]["present"]["share"]) == (n - 1, 0.0)
    assert ways[FOUR]["present"]["sdr"] == 4.0
    assert (ways[HOLLOW]["present"]["share"], ways[HOLLOW]["present"]["long"]) == (1.0, n - 1)
    assert ways[FOUR]["absent"] == {
        "songs": 1,
        "below_song_db": pytest.approx(-100.0, abs=25.0),
        "noted": 1,
    }
    assert (ways[LEAKY]["absent"]["songs"], ways[LEAKY]["absent"]["noted"]) == (1, 0)
    own, own_tracks = om.product_other(product_records(tracks))
    assert own_tracks == set(tracks[:3])
    assert (own["present"]["stems"], own["present"]["share"], own["present"]["sdr"]) == (
        2,
        0.25,
        2.5,
    )
    assert own["absent"] == {"songs": 1, "below_song_db": None, "noted": 1}
    assert om.product_other([]) == (None, set())

    analysis = om.analyse(rows, product_records(tracks))
    assert analysis["own_tracks"] == 3
    assert analysis["beside_own"][FOUR]["present"]["stems"] == 2
    text = om.render(analysis, RUN)
    assert om.RULES_COMMIT.startswith("3b37fcd51f69")
    assert "and committed (`3b37fcd`) before any kept output was read for this." in text
    assert "from the run `r`, made at commit `abcdef0`. No model was run" in text
    assert "## Beside the default's own, on the product's 3 tracks" in text
    assert (
        "| `six`'s own (the default) | 2 | 0.4 | 0.1 | 25.0% | 0 | 2.50 | 1 | - | 1 of 1 |" in text
    )
    assert f"| `{FOUR}` | 2 | 0.4 | 0.0 | 0.0% | 0 | 4.00 | 1 | " in text
    assert f"## MoisesDB test: {n} tracks" in text
    minutes = f"{(n - 1) * 0.2:.1f}"
    assert f"| `{HOLLOW}` | {n - 1} | {minutes} | {minutes} | 100.0% | {n - 1} | - | 1 | " in text
    assert "| 0 of 1 |" in text  # the leaky way, where the song has no `other`
    bare = om.render(om.analyse(rows), RUN)
    assert "The product's own run was not found" in bare
    empty = om.render(om.analyse([]), RUN)
    assert "Nothing could be put together for this list from what was kept." in empty


def test_report_refuses_a_run_that_has_not_finished(counted, tmp_path, monkeypatch, capsys) -> None:
    out = counted[2]
    monkeypatch.setattr(om, "OUT_DIR", tmp_path)
    monkeypatch.setattr(pc, "PRODUCT_DIR", tmp_path / "product")
    assert om.main(["report", "--run-id", "nothing"]) == 2
    assert "has not finished (state: None)" in capsys.readouterr().err
    run_dir = tmp_path / "done"
    run_dir.mkdir()
    (run_dir / "rows.jsonl").write_text(out.read_text())
    (run_dir / "run.json").write_text(json.dumps(RUN))
    (run_dir / "status.json").write_text(json.dumps({"state": "done"}))
    report = tmp_path / "report.md"
    assert om.main(["report", "--run-id", "done", "--out", str(report)]) == 0
    assert "# `other`, made the other way" in report.read_text()
