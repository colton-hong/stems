"""Where a dropped instrument is, and what is in no stem. Synthetic tones only."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from stems_bench import lost  # noqa: E402
from stems_bench import product_check as pc  # noqa: E402
from stems_bench.datasets import get_dataset  # noqa: E402
from stems_common.levels import sounding_windows, window_power  # noqa: E402
from stems_common.settings import load_defaults  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_worker.fakes import fake_pipeline  # noqa: E402

SR = 44100
SECONDS = 20.0
# One tone per stem, far enough apart that no two resemble one another.
TONES = dict(zip(PRODUCT_STEMS, (440.0, 150.0, 60.0, 700.0, 1100.0, 1700.0), strict=True))
LIMIT = -21.68


def tone(freq: float, gain: float = 0.1):
    t = np.arange(round(SECONDS * SR)) / SR
    return (gain * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :].repeat(2, axis=0)


def span(start: float, stop: float) -> slice:
    return slice(round(start * SR), round(stop * SR))


@pytest.fixture
def song():
    """Six true stems, their mix, and a separation whose bass is missing from 5 s to 15 s:
    from 5 s to 10 s the bass is in ``other``; from 10 s to 15 s it is in no stem."""
    true = {stem: tone(freq) for stem, freq in TONES.items()}
    mix = sum(true.values())
    made = {stem: audio.copy() for stem, audio in true.items()}
    made["bass"][:, span(5, 15)] = 0.0
    made["other"][:, span(5, 10)] += true["bass"][:, span(5, 10)]
    return mix, true, made


def test_window_power_is_what_sounding_windows_compares() -> None:
    audio = tone(440.0)[:, : 2 * SR]
    audio[:, SR:] = 0.0
    power = window_power(audio, SR)
    assert power.shape == (20,)
    assert power[:10] == pytest.approx(0.1**2 / 2, rel=1e-3)
    assert np.all(power[10:] == 0.0)
    assert np.array_equal(sounding_windows(audio, SR), power >= 1e-6)
    assert window_power(audio[:, :0], SR).size == 0


def test_a_share_is_how_much_of_the_true_stem_a_place_holds() -> None:
    true = tone(60.0)
    everywhere = np.ones(true.shape[-1], dtype=bool)
    places = {"all": true, "half": true * 0.5, "none": tone(440.0), "silent": np.zeros_like(true)}
    found = lost.shares(true, places, everywhere)
    assert found["all"] == pytest.approx(1.0)
    assert found["half"] == pytest.approx(0.5)
    assert found["none"] == pytest.approx(0.0, abs=1e-3)
    assert found["silent"] == 0.0
    assert lost.shares(np.zeros_like(true), places, everywhere) is None
    assert lost.shares(true, places, np.zeros_like(everywhere)) is None


def test_a_dropped_stem_is_looked_for_in_the_six_and_in_what_is_left(song) -> None:
    mix, true, made = song
    result = lost.measure_track(mix, true, made, SR, empty_below_mix_db=LIMIT)
    bass = result["stems"]["bass"]
    assert (bass["sounding"], bass["dropped"], bass["longest"]) == (20.0, 10.0, 10.0)
    # Half of the dropped time it is in `other`, half of it in none of the six.
    assert bass["found_in"]["other"] == pytest.approx(0.5, abs=0.01)
    assert bass["found_in"][lost.NONE] == pytest.approx(0.5, abs=0.01)
    assert bass["found_in"]["bass"] == 0.0
    assert bass["found_in"]["drums"] == pytest.approx(0.0, abs=0.01)
    assert sum(bass["found_in"].values()) == pytest.approx(1.0, abs=0.01)
    # Where it is not dropped, its own stem holds all of it.
    assert bass["own_share_kept"] == pytest.approx(1.0, abs=0.01)
    # A stem with no dropped time has no "found in"; it is all its own.
    drums = result["stems"]["drums"]
    assert drums["dropped"] == 0.0
    assert "found_in" not in drums
    assert drums["own_share_kept"] == pytest.approx(1.0, abs=0.01)
    # In no stem: the bass, for 5 of the song's 20 seconds. One of six equal tones is
    # 7.8 dB under the song, which is not "very quiet".
    assert result["in_no_stem"]["share_of_time"] == pytest.approx(0.25, abs=0.01)
    assert result["in_no_stem"]["below_song_db"] == pytest.approx(-13.8, abs=0.2)
    assert result["true_stems_minus_song_db"] <= -100.0
    # With no level to call "very quiet" by, the share of time is not given.
    bare = lost.measure_track(mix, true, made, SR, empty_below_mix_db=None)
    assert "share_of_time" not in bare["in_no_stem"]
    with pytest.raises(lost.LostError, match="missing: other"):
        lost.measure_track(mix, true, {k: v for k, v in made.items() if k != "other"}, SR,
                           empty_below_mix_db=LIMIT)  # fmt: skip


def test_the_report_pools_by_dropped_time_and_lists_the_long_ones(song) -> None:
    mix, true, made = song
    first = lost.measure_track(mix, true, made, SR, empty_below_mix_db=LIMIT)
    clean = lost.measure_track(mix, true, true, SR, empty_below_mix_db=LIMIT)
    # A second song whose bass is missing for 2 s, all of it in no stem.
    short = {stem: audio.copy() for stem, audio in true.items()}
    short["bass"][:, span(1, 3)] = 0.0
    second = lost.measure_track(mix, true, short, SR, empty_below_mix_db=LIMIT)
    records = [
        {"track_id": "aaaaaaaa-1", "lost": first},
        {"track_id": "bbbbbbbb-2", "lost": second},
        {"track_id": "cccccccc-3", "lost": clean},
        {"track_id": "dddddddd-4", "state": "failed"},  # no measurement: not a song here
    ]
    analysis = lost.analyse(records)
    assert analysis["songs"] == 3
    bass = analysis["by_stem"]["bass"]
    assert (bass["stems"], bass["long"]) == (3, 1)
    assert bass["share"] == pytest.approx(12.0 / 60.0)
    # 10 s at one half in none and 2 s wholly in none: (10 * 0.5 + 2 * 1) / 12.
    assert bass["found_in"][lost.NONE] == pytest.approx(7.0 / 12.0, abs=0.01)
    assert bass["found_in"]["other"] == pytest.approx(5.0 / 12.0, abs=0.01)
    assert analysis["by_stem"]["drums"]["found_in"] is None
    assert [(e["track_id"], e["stem"]) for e in analysis["listed"]] == [("aaaaaaaa-1", "bass")]
    assert analysis["share"] == pytest.approx(12.0 / 360.0)
    run = {"run_id": "r", "dataset": "synthetic", "device": "cpu", "empty_below_mix_db": LIMIT}
    text = lost.render(analysis, run)
    assert "| bass | 3 | 1.0 | 0.2 | 20.0% | 1 |" in text
    assert "| all | | 6.0 | 0.2 | 3.3% | 1 |" in text
    assert "| bass | 0% | 0% | 0% | 0% | 0% | 42% | 58% | 100% |" in text
    assert "| drums | - | - | - | - | - | - | - | - |" in text
    assert "| `aaaaaaaa` | bass | 10.0 | 0% | 0% | 0% | 0% | 0% | 50% | 50% | 100% |" in text
    assert "bass 100%" in text
    # How far the long table can be trusted, and the figure the owner's choice turns on.
    assert "1 of the 1 stems listed are within 10 points of 100%" in text
    assert "they hold 0.2 of the 0.2 minutes dropped; 0 have a share below zero" in text
    assert "**In none of the six: 58% of what is dropped**" in text
    # (10 s at one half in `other`) / 12 s; nothing of it is in the bass stem itself.
    assert "; 42% is in a stem of another name and 0% in its own." in text
    assert (analysis["own"], analysis["elsewhere"]) == pytest.approx((0.0, 5.0 / 12.0), abs=0.01)
    assert "is 50% (`aaaaaaaa` bass, a row that adds up to 100%)" in text
    assert "more than 21.68 dB under the song" in text
    assert "(median; the most in one song is 25.0%)" in text
    nothing = lost.render(lost.analyse([{"track_id": "c", "lost": clean}]), run)
    assert "None." in nothing
    assert "In none of the six" not in nothing


def test_a_row_that_does_not_add_up_is_counted_as_one_that_cannot_be_read() -> None:
    def entry(found: dict[str, float]) -> dict:
        shares = dict.fromkeys(lost.PLACES, 0.0) | found
        return {"sounding": 60.0, "dropped": 30.0, "longest": 6.0, "found_in": shares}

    quiet = {"sounding": 0.0, "dropped": 0.0, "longest": 0.0}
    records = [
        {
            "track_id": f"{name}-0000",
            "lost": {
                "true_stems_minus_song_db": -95.0,
                "in_no_stem": {"below_song_db": -30.0, "share_of_time": 0.0},
                "stems": dict.fromkeys(PRODUCT_STEMS, quiet) | {"drums": entry(found)},
            },
        }
        for name, found in (
            ("adds", {"other": 0.9, "drums": 0.04, lost.NONE: 0.02}),
            ("over", {"guitar": 0.9, "piano": 0.5, lost.NONE: 0.04}),
            ("under", {"bass": -0.79, "vocals": 0.83, lost.NONE: 0.19}),
        )
    ]
    run = {"run_id": "r", "empty_below_mix_db": LIMIT}
    text = lost.render(lost.analyse(records), run)
    assert "1 of the 3 stems listed are within 10 points of 100%" in text
    assert "they hold 0.5 of the 1.5 minutes dropped; 1 have a share below zero" in text
    assert "is 19% (`under-00` drums, a row that adds up to 23%)" in text
    # Own: 0.04 in one of three equal rows. Elsewhere: (0.9 + 1.4 + 0.04) / 3.
    assert "**In none of the six: 8% of what is dropped**, over all stems; 78% is in a stem" in text
    assert "of another name and 1% in its own." in text
    assert "it is 2% (`adds-000` drums)" in text
    assert "or lower in every song (median 95.0 dB)" in text


def test_the_report_names_the_commit_that_fixed_the_measures(song) -> None:
    mix, true, made = song
    records = [{"track_id": "a", "lost": lost.measure_track(mix, true, made, SR,
                                                            empty_below_mix_db=LIMIT)}]  # fmt: skip
    assert lost.RULES_COMMIT.startswith("17908793fbe7")
    run = {"run_id": "r", "empty_below_mix_db": LIMIT}
    at = {"git": {"commit": lost.RULES_COMMIT, "dirty": False}}
    plain = lost.render(lost.analyse(records), run)
    assert "committed (`1790879`) before any real song was measured." in plain
    assert "the run's own record" not in plain
    made_there = lost.render(lost.analyse(records), run | at)
    assert "was made at that commit, with nothing uncommitted." in made_there
    dirty = {"git": {"commit": lost.RULES_COMMIT, "dirty": True}}
    assert "the run's own record" not in lost.render(lost.analyse(records), run | dirty)


def test_the_product_check_records_it_when_asked(tmp_path) -> None:
    options, defaults = fake_pipeline(load_defaults())
    dataset = get_dataset("synthetic")
    track = dataset.track_ids()[0]
    arguments = {
        "run_id": "lost",
        "defaults": defaults,
        "device": "cpu",
        "bench_dir": tmp_path / "bench",
        "pipeline": options,
    }
    assert pc.run(dataset, [track], tmp_path / "with", lost=True, **arguments) == 1
    (record,) = pc.read_results(tmp_path / "with")
    assert set(record["lost"]["stems"]) == set(PRODUCT_STEMS)
    # The stand-in models split the mix into bands that add back to it: nothing is in no
    # stem, up to the rounding of the FLAC files.
    assert record["lost"]["in_no_stem"]["below_song_db"] < -60.0
    assert record["lost"]["in_no_stem"]["share_of_time"] == 0.0
    assert pc.run(dataset, [track], tmp_path / "without", **arguments) == 1
    assert "lost" not in pc.read_results(tmp_path / "without")[0]


def test_the_command_says_when_there_is_nothing_to_report(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(pc, "PRODUCT_DIR", tmp_path)
    assert lost.main(["--run-id", "nothing"]) == 2
    assert "run `python -m stems_bench.product_check run --lost` first" in capsys.readouterr().err
