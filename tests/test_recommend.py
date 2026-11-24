"""Choosing default models from benchmark rows: the rule, on made-up scores."""

from __future__ import annotations

import pytest

from stems_bench import recommend as rec

STEMS = ("vocals", "drums", "bass")


def row(system: str, stem: str, sdr, *, tracks: int = 30, rate: float = 10.0, **more) -> dict:
    return {
        "system": system,
        "output": stem,
        "target": stem,
        "derived": False,
        "tracks": 30,
        "tracks_scored": tracks,
        "medians": {"sdr": sdr},
        "seconds_per_audio_minute": rate,
        **more,
    }


def test_the_rule_is_what_the_report_says_it_is() -> None:
    assert (rec.MARGIN_DB, rec.MIN_TRACKS) == (0.3, 10)


def test_only_a_models_own_output_scored_against_the_stem_is_a_candidate() -> None:
    rows = [
        row("a", "vocals", 10.0),
        row("vocals:two_model_blend", "vocals", 12.0),  # a blend
        row("b", "vocals", 11.0, target="instrumental"),  # scored against something else
        row("c", "vocals", 11.0, derived=True),
        row("d", "vocals", 11.0, tracks=9),  # too few tracks
        row("e", "vocals", None),  # never scored
        row("f", "drums", 11.0),
    ]
    assert rec.candidates(rows, "vocals") == {
        "a": {"sdr": 10.0, "tracks": 30, "seconds_per_audio_minute": 10.0}
    }


def test_the_best_model_wins_when_nothing_else_is_close() -> None:
    rows = [
        row("a", "vocals", 10.0),
        row("b", "vocals", 9.0),
        row("b", "drums", 12.0),
        row("a", "drums", 8.0),
    ]
    chosen = rec.choose(rows, STEMS)
    assert (chosen["vocals"]["model"], chosen["drums"]["model"]) == ("a", "b")
    assert chosen["vocals"]["gives_up_db"] == 0.0
    assert chosen["vocals"]["best_model"] == "a"
    assert chosen["bass"] == {"model": None, "best_model": None, "candidates": {}}


def test_a_model_within_the_margin_that_serves_more_stems_replaces_the_best() -> None:
    rows = [
        row("solo", "vocals", 10.2),
        row("many", "vocals", 10.0),  # 0.2 dB behind: acceptable
        row("many", "drums", 12.0),
        row("many", "bass", 9.0),
        row("deep", "bass", 9.4),  # 0.4 dB ahead of "many": outside the margin
    ]
    chosen = rec.choose(rows, STEMS)
    assert {stem: chosen[stem]["model"] for stem in STEMS} == {
        "vocals": "many",
        "drums": "many",
        "bass": "deep",
    }
    assert chosen["vocals"]["gives_up_db"] == pytest.approx(0.2)
    assert chosen["vocals"]["best_model"] == "solo"
    # Exactly at the margin still counts as acceptable.
    edge = rec.choose(
        [row("x", "vocals", 10.0), row("y", "vocals", 9.7), row("y", "drums", 5.0)], STEMS
    )
    assert edge["vocals"]["model"] == "y"


def test_ties_go_to_the_least_given_up_then_the_faster_then_the_name() -> None:
    # Each serves one stem; "a" gives up nothing, "b" gives up 0.1.
    rows = [row("b", "vocals", 9.9), row("a", "vocals", 10.0)]
    assert rec.choose(rows, STEMS)["vocals"]["model"] == "a"
    # Equal scores: the faster one.
    rows = [row("slow", "vocals", 10.0, rate=30.0), row("fast", "vocals", 10.0, rate=5.0)]
    assert rec.choose(rows, STEMS)["vocals"]["model"] == "fast"
    # Equal in everything: the name, so that the result never depends on file order.
    rows = [row("m2", "vocals", 10.0), row("m1", "vocals", 10.0)]
    assert rec.choose(rows, STEMS)["vocals"]["model"] == "m1"
    assert rec.choose(list(reversed(rows)), STEMS)["vocals"]["model"] == "m1"


def data(run: str, rows: list) -> dict:
    return {"run_id": run, "dataset": "set", "rows": rows}


def test_the_report_shows_the_choice_and_then_the_test_lists() -> None:
    tuning = data(
        "tune-run",
        [
            row("new", "vocals", 11.0, rate=12.0),
            row("old", "vocals", 9.0, rate=8.0),
            row("new", "drums", 12.0, rate=12.0),
            row("old4", "drums", 10.0, rate=20.0),
        ],
    )
    tests = {
        "list_a": data(
            "run-a", [row("new", "vocals", 10.5, tracks=50), row("old", "vocals", 9.5, tracks=50)]
        ),
        "list_b": data("run-b", [row("new", "drums", 13.0, tracks=50)]),
    }
    in_use = {"vocals": "old", "drums": "old4", "bass": "old4"}
    report = rec.build(tuning, tests, in_use)
    assert {s: e["model"] for s, e in report["chosen"].items() if e["model"]} == {
        "vocals": "new",
        "drums": "new",
    }
    assert report["checks"]["vocals"]["list_a"]["difference_db"] == pytest.approx(1.0)
    assert report["checks"]["drums"]["list_b"] == {
        "chosen": {"sdr": 13.0, "tracks": 50},
        "in_use": None,
        "difference_db": None,
    }
    # One model for both stems runs once; the two in use each run.
    assert report["time"]["chosen_models"] == ["new"]
    assert report["time"]["chosen"] == 12.0
    assert report["time"]["in_use_models"] == ["old", "old4"]
    assert report["time"]["in_use"] == 28.0
    text = rec.render(report)
    assert "| vocals | `new` | 11.00 | `new` | 11.00 | 0.00 | `old` | 9.00 |" in text
    assert "| vocals | `list_a` | 10.50 | 9.50 | +1.00 | 50 |" in text
    assert "| drums | `list_b` | 13.00 | - | - | 50 |" in text
    assert "- chosen: `new`: 12.0 s" in text
    assert "- in use until now: `old`, `old4`: 28.0 s" in text
    assert "`list_a` from run `run-a`" in text
    assert rec.seconds_per_minute(tuning["rows"], ["new", "nobody"]) is None


def test_the_command_refuses_without_the_tuning_report(tmp_path, capsys) -> None:
    assert rec.main(["--tuning", str(tmp_path / "none.json")]) == 2
    assert "run `bench report` on the tuning run" in capsys.readouterr().err
