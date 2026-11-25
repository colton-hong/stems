"""Dropouts by model: the count over kept outputs, the tallies and the report.

Synthetic songs and a made-up benchmark folder; no model runs.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from stems_bench import dropouts_by_model as dbm  # noqa: E402
from stems_bench import product_check as pc  # noqa: E402
from stems_bench.datasets import get_dataset  # noqa: E402

BLEND = "blend of second and main"
BLENDS = {
    "bass": {"members": ["second", "main"], "weights": None, "method": "waveform"},
    # One member of this blend was never kept, so the blend cannot be counted.
    "piano": {"members": ["main", "absent"], "weights": None, "method": "waveform"},
}


def fake_bench(bench_dir, dataset, *, skip: tuple = ()) -> None:
    """Kept outputs as a benchmark run leaves them. ``main`` has the bass, missing from 2 s
    to 9 s, and the true piano; ``second`` has the bass at half level throughout.
    """
    for number, track_id in enumerate(dataset.track_ids()):
        track = dataset.load(track_id)
        rate = track.sample_rate
        holed = track.references["bass"].copy()
        holed[:, 2 * rate : 9 * rate] = 0.0
        kept = {
            "main": {"bass": (holed, 5.0), "piano": (track.references["piano"].copy(), 9.0)},
            "second": {"bass": (0.5 * track.references["bass"], 3.0)},
        }
        for model, stems in kept.items():
            if (model, track_id) in skip:
                continue
            key = f"{number:064x}"
            folder = bench_dir / "estimates" / dataset.name / model / key
            folder.mkdir(parents=True)
            outputs = {}
            for stem, (audio, sdr) in stems.items():
                np.save(folder / f"{stem}.npy", audio)
                outputs[stem] = {"target": stem, "derived": False, "bss": {"sdr": sdr}}
            result = {"kind": "model", "input": "mix", "track_id": track_id, "outputs": outputs}
            path = bench_dir / "results" / dataset.name / model / f"{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))


@pytest.fixture(scope="module")
def counted(tmp_path_factory):
    """The synthetic songs counted twice, once as each of the two MoisesDB lists."""
    root = tmp_path_factory.mktemp("by-model")
    dataset = get_dataset("synthetic")
    fake_bench(root / "bench", dataset)
    out = root / "run" / "rows.jsonl"
    said: list[str] = []
    for name in ("MoisesDB tuning", "MoisesDB test"):
        added = dbm.collect(
            name, dataset, BLENDS, out, bench_dir=root / "bench", progress=said.append
        )
        assert added == len(dataset.track_ids())
    return root, dataset, out, said


def test_the_choices_are_the_models_kept_for_every_track_and_the_blends_of_them(tmp_path) -> None:
    dataset = get_dataset("synthetic")
    track_ids = dataset.track_ids()
    fake_bench(tmp_path, dataset, skip=(("second", track_ids[3]),))
    found = dbm.index(tmp_path, dataset.name)
    assert found[("main", track_ids[0], "bass")]["sdr"] == 5.0
    some = dbm.choices_for(found, track_ids[:3], BLENDS)
    assert some == {
        "bass": {"main": ["main"], "second": ["second"], BLEND: ["second", "main"]},
        "piano": {"main": ["main"]},
    }
    # With one of its tracks missing, a model is not a choice, and neither is its blend.
    assert dbm.choices_for(found, track_ids, BLENDS)["bass"] == {"main": ["main"]}
    assert dbm.blend_name(["a", "b"]) == "blend of a and b"


def test_collect_counts_every_choice_of_every_stem_once_and_resumes(counted) -> None:
    root, dataset, out, said = counted
    tracks = dataset.track_ids()
    assert said[-1] == f"MoisesDB test: {len(tracks)} of {len(tracks)} tracks"
    rows = dbm.read_rows(out)
    assert {row["id"] for row in rows} == {
        f"{name}/{track}" for name in ("MoisesDB tuning", "MoisesDB test") for track in tracks
    }
    bass = rows[0]["stems"]["bass"]
    assert bass["sounding"] == 12.0
    # The model with the hole drops seven seconds; the other, and the blend of the two, none.
    assert bass["choices"]["main"] == {"dropped": 7.0, "longest": 7.0, "sdr": 5.0}
    assert bass["choices"]["second"] == {"dropped": 0.0, "longest": 0.0, "sdr": 3.0}
    assert bass["choices"][BLEND] == {"dropped": 0.0, "longest": 0.0, "sdr": None}
    assert set(rows[0]["stems"]["piano"]["choices"]) == {"main"}
    again = dbm.collect("MoisesDB test", dataset, BLENDS, out, bench_dir=root / "bench")
    assert again == 0


def test_an_output_of_the_wrong_length_stops_the_count(tmp_path) -> None:
    dataset = get_dataset("synthetic")
    fake_bench(tmp_path / "bench", dataset)
    first = sorted((tmp_path / "bench" / "estimates" / dataset.name / "main").iterdir())[0]
    np.save(first / "bass.npy", np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(dbm.ByModelError, match="has 100 samples"):
        dbm.collect(
            "MoisesDB test", dataset, {}, tmp_path / "rows.jsonl", bench_dir=tmp_path / "bench"
        )


def test_a_tally_is_over_the_songs_that_have_the_instrument(counted) -> None:
    _, dataset, out, _ = counted
    rows = dbm.read_rows(out)
    tracks = dataset.track_ids()
    there = dbm.by_choice(rows, "MoisesDB tuning")
    main = there["bass"]["main"]
    assert (main["stems"], main["long"], main["sdr"]) == (len(tracks), len(tracks), 5.0)
    assert main["sounding_minutes"] == pytest.approx(len(tracks) * 12.0 / 60.0)
    assert main["share"] == pytest.approx(7.0 / 12.0)
    assert there["bass"][BLEND]["share"] == 0.0
    assert there["bass"][BLEND]["sdr"] is None
    # One synthetic song has no piano: it is not counted for the piano.
    without = sum(1 for t in tracks if not dataset.load(t).references["piano"].any())
    assert without > 0
    assert there["piano"]["main"]["stems"] == len(tracks) - without
    # Over some of the tracks only.
    few = dbm.by_choice(rows, "MoisesDB tuning", only=set(tracks[:3]))
    assert few["bass"]["main"]["stems"] == 3
    assert dbm.by_choice(rows, "MUSDB18-HQ test") == {}
    # The least is by share; among equals, the first by name.
    assert dbm.least(there["bass"]) == BLEND
    assert dbm.least({"b": {"share": 0.2}, "a": {"share": 0.2}, "c": {"share": None}}) == "a"
    assert dbm.least({}) is None
    nothing = dbm.tally([])
    assert (nothing["stems"], nothing["share"], nothing["sdr"]) == (0, None, None)


def product_records(tracks: list[str]) -> list[dict]:
    """The product's own run on three songs: its piano is silent for a quarter of the time."""
    stems = {stem: {"sounding": 12.0, "dropped": 3.0, "longest": 3.0} for stem in dbm.PRODUCT_STEMS}
    records = [{"track_id": track, "lost": {"stems": stems}} for track in tracks[:3]]
    return [*records, {"track_id": "failed", "state": "failed"}]


RUN = {"run_id": "r", "git": {"commit": "abcdef0123", "dirty": False}}


def test_the_report_puts_the_least_beside_the_default(counted) -> None:
    _, dataset, out, _ = counted
    rows = dbm.read_rows(out)
    tracks = dataset.track_ids()
    # The bass's default was kept; the piano's was not, so the product's own run stands in.
    defaults = {"bass": "main", "piano": "the_default"}
    analysis = dbm.analyse(rows, defaults, product_records(tracks))
    assert analysis["stems"] == ["bass", "piano"]
    assert analysis["tracks"] == {
        "MoisesDB test": len(tracks),
        "MoisesDB tuning": len(tracks),
        "MUSDB18-HQ test": 0,
    }
    assert analysis["least_on_tuning"] == {"bass": BLEND, "piano": "main"}
    assert list(analysis["default_from_product"]) == ["piano"]
    beside = analysis["default_from_product"]["piano"]
    assert beside["tracks"] == 3
    assert (beside["default"]["stems"], beside["default"]["share"]) == (3, 0.25)
    assert beside["others"]["main"]["stems"] == 3
    text = dbm.render(analysis, RUN)
    assert "from the run `r`, made at commit `abcdef0`. No model was run" in text
    n = len(tracks)
    assert f"| bass | `{BLEND}` | MoisesDB tuning | 0.0% | - | 58.3% | 5.00 |" in text
    assert f"| bass | `{BLEND}` | MUSDB18-HQ test | - | - | not kept | - |" in text
    assert "| piano | `main` | MoisesDB test | 0.0% | 9.00 | not kept | - |" in text
    assert f"## MoisesDB tuning: {n} tracks" in text
    assert f"| bass | `main` (the default) | {n} | {n * 0.2:.1f} | " in text
    assert f"| bass | `second` | {n} | {n * 0.2:.1f} | 0.0 | 0.0% | 0 | 3.00 |" in text
    assert "## MUSDB18-HQ test: 0 tracks" in text
    assert "No stem has a kept output for every track of this list." in text
    assert "## The default where the benchmark did not keep it" in text
    assert "| piano | `the_default` (the default) | 3 | 0.6 | 0.1 | 25.0% | 0 | - |" in text
    assert "| piano | `main` | 3 | 0.6 | 0.0 | 0.0% | 0 | 9.00 |" in text
    assert "No output was kept for: vocals, drums, guitar, other." in text
    # Without the product's run nothing stands in, and the report says so.
    bare = dbm.render(dbm.analyse(rows, defaults), RUN)
    assert "The product's run was not found, so nothing stands for piano." in bare
    # A default that was kept needs no stand-in.
    kept = dbm.analyse(rows, {"bass": "main", "piano": "main"}, product_records(tracks))
    assert kept["default_from_product"] == {}
    assert "The default where the benchmark did not keep it" not in dbm.render(kept, RUN)


def test_the_plan_comes_from_the_config() -> None:
    pytest.importorskip("yaml")
    defaults, blends = dbm.plan()
    assert set(defaults) == set(dbm.PRODUCT_STEMS) == set(blends)
    assert all(len(blend["members"]) == 2 and blend["method"] for blend in blends.values())


def test_report_refuses_a_run_that_has_not_finished(counted, tmp_path, monkeypatch, capsys) -> None:
    out = counted[2]
    monkeypatch.setattr(dbm, "OUT_DIR", tmp_path)
    monkeypatch.setattr(pc, "PRODUCT_DIR", tmp_path / "product")
    assert dbm.main(["report", "--run-id", "nothing"]) == 2
    assert "has not finished (state: None)" in capsys.readouterr().err
    run_dir = tmp_path / "done"
    run_dir.mkdir()
    (run_dir / "rows.jsonl").write_text(out.read_text())
    (run_dir / "run.json").write_text(json.dumps({**RUN, "defaults": {"bass": "main"}}))
    (run_dir / "status.json").write_text(json.dumps({"state": "running"}))
    assert dbm.main(["report", "--run-id", "done"]) == 2
    (run_dir / "status.json").write_text(json.dumps({"state": "done"}))
    report = tmp_path / "report.md"
    assert dbm.main(["report", "--run-id", "done", "--out", str(report)]) == 0
    assert "# Dropouts by model" in report.read_text()
