"""Counting where the true stem plays and the made stem is silent. Synthetic tones only."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")

from stems_bench import dropouts  # noqa: E402
from stems_common.levels import silence_fraction, sounding_windows  # noqa: E402
from stems_worker.audio import save_audio  # noqa: E402

SR = 44100


def tone(seconds: float, gain: float = 0.2, freq: float = 220.0):
    t = np.arange(round(seconds * SR)) / SR
    return (gain * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :].repeat(2, axis=0)


def gaps(audio, *spans: tuple[float, float]):
    """The audio with silence in each (from, to) span of seconds."""
    out = audio.copy()
    for start, stop in spans:
        out[:, round(start * SR) : round(stop * SR)] = 0.0
    return out


def test_windows_sound_or_not_and_the_silence_fraction_is_their_share() -> None:
    audio = gaps(tone(10.0), (2.0, 5.0))
    sounding = sounding_windows(audio, SR)
    assert sounding.shape == (100,)
    assert sounding[:20].all()
    assert not sounding[20:50].any()
    assert sounding[50:].all()
    assert silence_fraction(audio, SR) == pytest.approx(0.3)
    assert sounding_windows(audio[:, :0], SR).size == 0
    assert silence_fraction(audio[:, :0], SR) == 1.0
    # Quiet is not silent: 40 dB down from 0.2 is still above -60 dBFS.
    assert sounding_windows(audio * 0.01, SR)[:20].all()
    assert not sounding_windows(audio * 0.001, SR).any()


def test_a_dropout_is_the_true_stems_time_inside_one_stretch_of_silence() -> None:
    true = gaps(tone(20.0), (10.0, 12.0))  # sounds for 18 s, with a 2 s rest
    made = gaps(tone(20.0), (4.0, 14.0))  # silent from 4 s to 14 s
    # Dropped: 4-10 and 12-14. One stretch of silence, so the rest does not split it.
    assert dropouts.measure(true, made, SR) == {"sounding": 18.0, "dropped": 8.0, "longest": 8.0}
    # Two stretches of silence: the longest is the larger, not the sum.
    made = gaps(tone(20.0), (1.0, 4.0), (9.0, 16.0))
    assert dropouts.measure(true, made, SR) == {"sounding": 18.0, "dropped": 8.0, "longest": 5.0}
    # Quieter is not dropped, and silence where the true stem is silent is not either.
    assert dropouts.measure(true, tone(20.0) * 0.05, SR)["dropped"] == 0.0
    assert dropouts.measure(true, true, SR) == {"sounding": 18.0, "dropped": 0.0, "longest": 0.0}
    nothing = np.zeros_like(true)
    assert dropouts.measure(nothing, nothing, SR) == {
        "sounding": 0.0,
        "dropped": 0.0,
        "longest": 0.0,
    }
    assert dropouts.measure(true, nothing, SR) == {
        "sounding": 18.0,
        "dropped": 18.0,
        "longest": 18.0,
    }
    # The shorter of the two decides how much is compared.
    assert dropouts.measure(true, nothing[:, : 5 * SR], SR)["sounding"] == 5.0


@pytest.fixture
def run_dir(tmp_path):
    """A listening run's folder with one 20-second song of two stems."""
    run = tmp_path / "run"
    stems = run / "stems"
    bass, drums = tone(20.0, 0.2, 80.0), tone(20.0, 0.2, 400.0)
    save_audio(stems / "mix" / "song.wav", bass + drums, SR, "wav")
    made = {
        "reference": {"bass": bass, "drums": drums},
        # Now: the bass is missing from 6 s to 18 s; before: from 2 s to 5 s.
        "new": {"bass": gaps(bass, (6.0, 18.0)), "drums": drums},
        "old": {"bass": gaps(bass, (2.0, 5.0)), "drums": drums * 0.5},
    }
    for version, files in made.items():
        for stem, audio in files.items():
            save_audio(stems / version / "song" / f"{stem}.flac", audio, SR, "flac")
    (stems / "done").mkdir()
    (stems / "done" / "song.json").write_text(
        json.dumps({"track_id": "song", "stems": ["drums", "bass"], "seconds": 20.0})
    )
    (run / "key.json").write_text(json.dumps({"tracks": {"T01": {"id": "song"}}}))
    return run


def test_a_run_is_counted_stem_by_stem_and_the_report_lists_the_long_dropouts(run_dir) -> None:
    records = dropouts.collect(run_dir, empty_below_mix_db=-21.68)
    by = {(r["stem"], r["version"]): r for r in records}
    assert len(records) == 4
    assert (by["bass", "new"]["dropped"], by["bass", "new"]["longest"]) == (12.0, 12.0)
    assert (by["bass", "old"]["dropped"], by["bass", "old"]["longest"]) == (3.0, 3.0)
    assert by["drums", "new"]["dropped"] == by["drums", "old"]["dropped"] == 0.0
    assert not any(r["noted"] for r in records)  # each stem is loud enough over the whole song
    total = dropouts.summarise([r for r in records if r["version"] == "new"])
    assert (total["stems"], total["long"], total["long_noted"]) == (2, 1, 0)
    assert total["share"] == pytest.approx(12.0 / 40.0)
    run = {
        "run_id": "r",
        "tracks": 1,
        "empty_below_mix_db": -21.68,
        "new_models": {"bass": "model_b", "drums": "model_b"},
        "old_models": {"bass": "model_a", "drums": "model_b"},
    }
    text = dropouts.render(records, run)
    assert "| The product now | 2 | 0.7 | 0.2 | 30.0% | 1 |" in text
    assert "| The product before | 2 | 0.7 | 0.1 | 7.5% | 0 |" in text
    assert "| bass, now (`model_b`) | 1 | 0.3 | 0.2 | 60.0% | 1 |" in text
    assert "| bass, before (`model_a`) | 1 | 0.3 | 0.1 | 15.0% | 0 |" in text
    # Drums have the same model in both: one row, and a sentence saying so.
    assert "| drums, now (`model_b`)" in text
    assert "drums, before" not in text
    assert "For drums the product uses the same model now as before" in text
    assert "| T01 | bass | now | 20.0 | 12.0 | 12.0 |" in text
    assert "0 of these 1 stems would carry it" in text
    assert "the true stem is itself" not in text
    # A true stem that is itself next to nothing against the song is pointed out.
    faint = [{**r, "true_below_song_db": -30.0} for r in records]
    assert "In 1 of them (T01's bass) the true stem is itself more than 21.68 dB" in (
        dropouts.render(faint, run)
    )
    # A stem that is very quiet over the whole song would carry the product's note.
    noted = dropouts.collect(run_dir, empty_below_mix_db=-1.0)
    assert all(r["noted"] for r in noted)
    assert "None." in dropouts.render([r for r in records if r["stem"] == "drums"], run)


def test_the_command_says_when_there_is_no_run(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(dropouts, "LISTEN_DIR", tmp_path)
    assert dropouts.main(["--run-id", "nothing"]) == 2
    assert "run `listen make` first" in capsys.readouterr().err


def test_the_command_writes_the_report_and_the_records(
    run_dir, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(dropouts, "LISTEN_DIR", run_dir.parent)
    out = tmp_path / "dropouts.md"
    assert dropouts.main(["--run-id", "run", "--out", str(out)]) == 0
    assert "# Where the instrument plays" in out.read_text()
    saved = json.loads((run_dir / "dropouts.json").read_text())
    assert saved["long_seconds"] == dropouts.LONG_SECONDS == 5.0
    assert len(saved["records"]) == 4
