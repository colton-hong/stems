"""The blind listening-test kit, on synthetic audio. No listening test has been run."""

from __future__ import annotations

import csv
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")

import yaml  # noqa: E402

from stems_bench import listening_kit as kit_tool  # noqa: E402
from stems_bench.listening_kit import KitError  # noqa: E402
from stems_common.levels import peak_dbfs, rms_dbfs  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_worker.audio import load_audio, save_audio  # noqa: E402

SR = 44100
SYSTEMS = ("v1", "uvr", "lalal", "moises")
STEMS = ("vocals", "drums")
# How each stand-in "system" differs from the true stem: a gain, and added noise.
FLAVOUR = {"v1": (1.0, 0.0), "uvr": (0.5, 0.001), "lalal": (2.0, 0.002), "moises": (0.25, 0.001)}


@pytest.fixture
def material(tmp_path):
    """Two 40-second tracks, each 'separated' by four systems, and the manifest for them."""
    rng = np.random.default_rng(11)
    tracks = []
    for number in (1, 2):
        track = generate_track(seed=40 + number, duration_seconds=40.0)
        # Louder in the second half, so the loudest 30 seconds are not the first 30.
        ramp = np.linspace(0.3, 1.0, track.mix.shape[1], dtype=np.float32)
        mix_path = tmp_path / f"song-{number}.wav"
        save_audio(mix_path, track.mix * ramp * 0.5, SR, "wav")
        folders = {}
        for system in SYSTEMS:
            gain, noise = FLAVOUR[system]
            folder = tmp_path / system / f"song-{number}"
            for stem in STEMS:
                audio = track.stems[stem] * ramp * 0.5 * gain
                audio = audio + rng.standard_normal(audio.shape).astype(np.float32) * noise
                save_audio(folder / f"{stem}.wav", np.clip(audio, -1, 1), SR, "wav")
            folders[system] = str(folder)
        tracks.append({"id": f"song-{number}", "mix": str(mix_path), "systems": folders})
    manifest = {
        "ours": "v1",
        "free_reference": "uvr",
        "paid_references": ["lalal", "moises"],
        "stems": list(STEMS),
        "tracks": tracks,
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def build(material, tmp_path, **options):
    out, key_path = tmp_path / "kit", tmp_path / "private" / "key.json"
    key = kit_tool.build_kit(kit_tool.load_manifest(material), out, key_path, **options)
    return out, key_path, key


def fill(sheet, key, scores, cleanest="v1", skip=None) -> None:
    """Fill a sheet as a listener who gives each system the scores in ``scores``."""
    with sheet.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        labels = key["items"][f"{row['track']}/{row['stem']}"]["labels"]
        system = labels[row["sample"]]
        value = scores[system] if not callable(scores) else scores(system, row)
        row["overall"], row["bleed"], row["artifacts"] = value, value, value
        row["would_use"] = "y" if int(value) >= 3 else "n"
        if row["cleanest"] != "-":
            row["cleanest"] = next(lb for lb, name in labels.items() if name == cleanest)
    if skip:
        rows = [row for row in rows if not skip(row)]
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(out, key, sheets_dir=None):
    kit = json.loads((out / "kit.json").read_text())
    sheets, problems = {}, []
    for path in sorted((sheets_dir or out / "sheets").glob("*.csv")):
        rows, found = kit_tool.read_sheet(path, kit)
        sheets[path.stem] = rows
        problems += found
    return kit_tool.score_kit(kit, key, sheets), problems


# -- building ----------------------------------------------------------------------------


def test_the_kit_holds_the_same_excerpt_from_every_system_under_letters(material, tmp_path):
    out, _key_path, key = build(material, tmp_path, seed=3)
    kit = json.loads((out / "kit.json").read_text())
    assert [(i["track"], i["stem"]) for i in kit["items"]] == [
        ("T01", "vocals"),
        ("T01", "drums"),
        ("T02", "vocals"),
        ("T02", "drums"),
    ]
    assert all(item["samples"] == ["A", "B", "C", "D"] for item in kit["items"])
    assert sorted(p.name for p in (out / "audio" / "T01" / "vocals").iterdir()) == [
        "A.wav",
        "B.wav",
        "C.wav",
        "D.wav",
    ]
    for item in kit["items"]:
        clips = {
            label: load_audio(out / "audio" / item["track"] / item["stem"] / f"{label}.wav").samples
            for label in item["samples"]
        }
        # 30 seconds each, all the same length.
        assert {clip.shape for clip in clips.values()} == {(2, 30 * SR)}
        labels = key["items"][f"{item['track']}/{item['stem']}"]["labels"]
        by_system = {labels[label]: clip for label, clip in clips.items()}
        # The same stretch of the song: the systems' clips line up sample for sample.
        ours = by_system["v1"].mean(axis=0)
        for system in ("uvr", "lalal", "moises"):
            other = by_system[system].mean(axis=0)
            assert np.corrcoef(ours, other)[0, 1] > 0.9, (item, system)
    # The excerpt is the loud end of the track, not its start.
    track = generate_track(seed=41, duration_seconds=40.0)
    ramp = np.linspace(0.3, 1.0, track.mix.shape[1], dtype=np.float32)
    labels = key["items"]["T01/vocals"]["labels"]
    ours_label = next(lb for lb, name in labels.items() if name == "v1")
    clip = load_audio(out / "audio" / "T01" / "vocals" / f"{ours_label}.wav").samples
    last = (track.stems["vocals"] * ramp)[:, -30 * SR :]
    first = (track.stems["vocals"] * ramp)[:, : 30 * SR]
    assert np.corrcoef(clip.mean(axis=0), last.mean(axis=0))[0, 1] > 0.99
    assert abs(np.corrcoef(clip.mean(axis=0), first.mean(axis=0))[0, 1]) < 0.9


@pytest.fixture
def late_song(tmp_path):
    """A minute-long song that is loud from 25 s on; its piano plays in the first 20 s only."""
    samples = 60 * SR
    t = np.arange(samples) / SR

    def tone(freq: float, gain: float):
        return (gain * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :].repeat(2, axis=0)

    drums = tone(110.0, 0.4)
    drums[:, : 25 * SR] *= 0.05
    piano = tone(440.0, 0.1)
    piano[:, 20 * SR :] = 0.0
    mix_path = tmp_path / "late.wav"
    save_audio(mix_path, drums + piano, SR, "wav")
    rng = np.random.default_rng(5)
    folders = {}
    for system, noise in (("true", 0.0), ("a", 0.002), ("b", 0.004)):
        folder = tmp_path / system / "late"
        for stem, audio in (("drums", drums), ("piano", piano)):
            noisy = audio + rng.standard_normal(audio.shape).astype(np.float32) * noise
            save_audio(folder / f"{stem}.wav", noisy, SR, "wav")
        folders[system] = str(folder)
    return {
        "systems": ["true", "a", "b"],
        "ours": "a",
        "free_reference": "b",
        "paid_references": [],
        "stems": ["drums", "piano"],
        "tracks": [{"id": "late", "mix": str(mix_path), "systems": folders}],
    }


def clip_levels(out, key, item_id: str) -> dict[str, float]:
    labels = key["items"][item_id]["labels"]
    return {
        system: rms_dbfs(load_audio(out / "audio" / item_id / f"{label}.wav").samples)
        for label, system in labels.items()
    }


def test_each_stem_can_be_cut_where_one_system_has_it_loudest(late_song, tmp_path):
    # Cut where the song is loudest, the piano's true stem is silence: nothing to judge.
    out = tmp_path / "by-song"
    key = kit_tool.build_kit(late_song, out, tmp_path / "k1.json", listeners=1)
    assert key["items"]["T01/piano"]["start_seconds"] == key["items"]["T01/drums"]["start_seconds"]
    assert key["items"]["T01/drums"]["start_seconds"] >= 24.0
    assert clip_levels(out, key, "T01/piano")["true"] < kit_tool.SILENT_DBFS
    # Cut where the true stem is loudest, every step has its instrument playing.
    out = tmp_path / "by-stem"
    key = kit_tool.build_kit(late_song, out, tmp_path / "k2.json", listeners=1, window_by="true")
    assert key["items"]["T01/piano"]["start_seconds"] == 0.0
    assert key["items"]["T01/drums"]["start_seconds"] >= 24.0
    for item_id in ("T01/piano", "T01/drums"):
        levels = clip_levels(out, key, item_id)
        assert min(levels.values()) > -40.0, (item_id, levels)
        assert max(levels.values()) - min(levels.values()) < 0.2
    with pytest.raises(KitError, match="not one of the systems"):
        kit_tool.build_kit(late_song, tmp_path / "x", tmp_path / "kx.json", window_by="nobody")


def test_a_version_that_is_another_ones_audio_is_played_once(late_song, tmp_path):
    out = tmp_path / "kit"
    key = kit_tool.build_kit(
        late_song,
        out,
        tmp_path / "key.json",
        listeners=1,
        window_by="true",
        same_as={"piano": {"b": "a"}, "drums": {}},
    )
    kit = json.loads((out / "kit.json").read_text())
    samples = {item["stem"]: item["samples"] for item in kit["items"]}
    assert samples == {"drums": ["A", "B", "C"], "piano": ["A", "B"]}
    assert sorted(key["items"]["T01/piano"]["labels"].values()) == ["a", "true"]
    assert key["items"]["T01/piano"]["same_as"] == {"b": "a"}
    assert "same_as" not in key["items"]["T01/drums"]
    assert sorted(p.name for p in (out / "audio" / "T01" / "piano").iterdir()) == ["A.wav", "B.wav"]
    # The sheet asks for two rows there, and nothing in the kit says why.
    with (out / "sheets" / "listener-01.csv").open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["stem"] == "piano"]
    assert [row["sample"] for row in rows] == ["A", "B"]
    assert "same" not in (out / "kit.json").read_text()
    for number, (bad, options) in enumerate(
        [
            ({"piano": {"b": "nobody"}}, {}),
            ({"piano": {"b": "b"}}, {}),
            ({"piano": {"a": "b", "b": "a"}}, {}),
            ({"piano": {"true": "a"}}, {"window_by": "true"}),
        ]
    ):
        with pytest.raises(KitError):
            kit_tool.build_kit(
                late_song, tmp_path / f"bad{number}", tmp_path / f"bad{number}.json",
                same_as=bad, **options,
            )  # fmt: skip


def test_levels_are_matched_so_loudness_cannot_give_a_system_away(material, tmp_path):
    out, _, key = build(material, tmp_path)
    kit = json.loads((out / "kit.json").read_text())
    for item in kit["items"]:
        levels = [
            rms_dbfs(load_audio(out / "audio" / item["track"] / item["stem"] / f"{s}.wav").samples)
            for s in item["samples"]
        ]
        # The systems differed by up to 18 dB going in; they are within 0.2 dB coming out.
        assert max(levels) - min(levels) < 0.2, item
        peaks = [
            peak_dbfs(load_audio(out / "audio" / item["track"] / item["stem"] / f"{s}.wav").samples)
            for s in item["samples"]
        ]
        assert max(peaks) <= kit_tool.PEAK_CEILING_DBFS + 0.05
        gains = key["items"][f"{item['track']}/{item['stem']}"]["gains_db"]
        assert max(gains.values()) - min(gains.values()) > 10  # and the key says what was done


def test_level_matching_leaves_silence_alone_and_respects_the_ceiling() -> None:
    tone = (0.5 * np.sin(np.linspace(0, 2000, SR))).astype(np.float32)[None, :].repeat(2, axis=0)
    excerpts = {"a": tone, "b": tone * 0.1, "c": np.zeros_like(tone)}
    matched, gains = kit_tool.match_levels(excerpts)
    assert rms_dbfs(matched["a"]) == pytest.approx(rms_dbfs(matched["b"]), abs=0.01)
    assert np.all(matched["c"] == 0)
    assert gains["c"] == 0.0
    assert gains["b"] - gains["a"] == pytest.approx(20.0, abs=0.01)
    hot = {"a": tone * 1.9, "b": tone * 1.9}
    matched, gains = kit_tool.match_levels(hot)
    assert peak_dbfs(matched["a"]) == pytest.approx(kit_tool.PEAK_CEILING_DBFS, abs=0.01)
    assert gains["a"] == gains["b"] < 0
    _silent, gains = kit_tool.match_levels({"a": np.zeros_like(tone), "b": np.zeros_like(tone)})
    assert gains == {"a": 0.0, "b": 0.0}


def test_nothing_in_the_kit_says_which_system_is_which(material, tmp_path):
    out, key_path, key = build(material, tmp_path, seed=3)
    assert not key_path.is_relative_to(out)
    everything = "".join(
        p.read_text()
        for p in out.rglob("*")
        if p.is_file() and p.suffix in (".json", ".csv", ".md")
    )
    names = " ".join(str(p.relative_to(out)) for p in out.rglob("*"))
    for system in SYSTEMS:
        assert system not in everything
        assert system not in names
    assert "song-1" not in everything  # nor the tracks' own names
    assert "seed" not in json.loads((out / "kit.json").read_text())
    # The letters are not always in the same order: the mapping changes from stem to stem.
    orders = {tuple(item["labels"].values()) for item in key["items"].values()}
    assert len(orders) > 1
    # The same seed gives the same kit; another seed, another assignment.
    _, _, again = (
        tmp_path / "kit2",
        tmp_path / "private" / "key2.json",
        kit_tool.build_kit(
            kit_tool.load_manifest(material),
            tmp_path / "kit2",
            tmp_path / "private" / "key2.json",
            seed=3,
        ),
    )
    assert again["items"] == key["items"]
    other = kit_tool.build_kit(
        kit_tool.load_manifest(material),
        tmp_path / "kit3",
        tmp_path / "private" / "key3.json",
        seed=4,
    )
    assert other["items"] != key["items"]


def test_each_listener_gets_the_same_rows_in_an_order_of_their_own(material, tmp_path):
    out, _, _ = build(material, tmp_path, listeners=6)
    sheets = sorted((out / "sheets").glob("*.csv"))
    assert [p.name for p in sheets] == [f"listener-{n:02d}.csv" for n in range(1, 7)]
    contents = []
    for sheet in sheets:
        with sheet.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == [
            "track",
            "stem",
            "sample",
            "overall",
            "bleed",
            "artifacts",
            "would_use",
            "cleanest",
        ]
        assert len(rows) == 4 * 4  # four stems, four systems
        # "cleanest" is asked once per stem.
        assert sum(1 for row in rows if row["cleanest"] != "-") == 4
        contents.append([(row["track"], row["stem"]) for row in rows])
    assert all(sorted(order) == sorted(contents[0]) for order in contents)
    assert len({tuple(order) for order in contents}) > 1
    readme = (out / "README.md").read_text()
    assert "1 (heavy bleed) to 5 (clean)" in readme
    assert "letters mean nothing and change from stem to stem" in readme


def test_the_key_cannot_be_put_in_the_kit_and_a_kit_is_not_overwritten(material, tmp_path):
    manifest = kit_tool.load_manifest(material)
    with pytest.raises(KitError, match="must not be inside the kit"):
        kit_tool.build_kit(manifest, tmp_path / "kit", tmp_path / "kit" / "key.json")
    build(material, tmp_path)
    with pytest.raises(KitError, match="not empty"):
        kit_tool.build_kit(manifest, tmp_path / "kit", tmp_path / "private" / "other.json")


def test_a_manifest_that_cannot_give_a_fair_comparison_is_refused(material, tmp_path):
    data = yaml.safe_load(material.read_text())

    def refuse(change, says):
        changed = json.loads(json.dumps(data))
        change(changed)
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(KitError, match=says):
            kit_tool.build_kit(
                kit_tool.load_manifest(path),
                tmp_path / "bad-kit",
                tmp_path / "private" / "bad.json",
            )

    refuse(lambda d: d.update(ours="nobody"), "'ours' must name one of the systems")
    refuse(lambda d: d.update(free_reference="nobody"), "is not one of the systems")
    refuse(lambda d: d["tracks"][1]["systems"].pop("uvr"), "every track must have the same systems")
    refuse(lambda d: d["tracks"].append(dict(d["tracks"][0])), "listed twice")
    refuse(lambda d: d.update(tracks=[]), "needs a list of tracks")

    def one_stem_missing(d):
        import os

        os.remove(os.path.join(d["tracks"][0]["systems"]["lalal"], "drums.wav"))

    refuse(one_stem_missing, "drums is missing for lalal")


# -- scoring -----------------------------------------------------------------------------


def test_a_clear_winner_passes_the_bar(material, tmp_path):
    out, _, key = build(material, tmp_path, listeners=5)
    for sheet in (out / "sheets").glob("*.csv"):
        fill(sheet, key, {"v1": 5, "uvr": 4, "lalal": 3, "moises": 3})
    result, problems = score(out, key)
    assert problems == []
    assert (result["listeners"], result["ratings"], result["cleanest_votes"]) == (5, 80, 20)
    assert result["per_system"]["v1"]["overall"] == 5.0
    assert result["per_system"]["uvr"]["overall"] == 4.0
    assert result["per_system"]["v1"]["cleanest_votes"] == 20
    assert result["per_system"]["moises"]["would_use"] == 1.0
    assert result["checks"]["beats_free_reference"] == {
        "tracks_won": 2,
        "tracks": 2,
        "needed": 2,
        "met": True,
    }
    assert result["checks"]["beats_paid_references"]["gaps"] == {"lalal": 2.0, "moises": 2.0}
    assert result["checks"]["no_stem_lost_badly"]["worst_loss"] == 0.0
    assert result["passes"] is True
    text = kit_tool.render(result, key, problems)
    assert "`v1` clears the bar on this test." in text
    assert "| `v1` | 5.00 | 5.00 | 5.00 | 100% | 20 of 20 |" in text
    assert "The report planned 10 tracks; this test had 2." in text
    assert "| song-1 |" in text  # unblinded: the tracks' own names are back


@pytest.mark.parametrize(
    ("scores", "failed"),
    [
        # Ahead of the free tool, but by less than 0.3.
        ({"v1": 4, "uvr": 4, "lalal": 3, "moises": 3}, "beats_free_reference"),
        # Behind one paid tool.
        ({"v1": 4, "uvr": 3, "lalal": 5, "moises": 3}, "beats_paid_references"),
    ],
)
def test_missing_the_bar_is_reported_as_missing_it(material, tmp_path, scores, failed):
    out, _, key = build(material, tmp_path, listeners=5)
    for sheet in (out / "sheets").glob("*.csv"):
        fill(sheet, key, scores)
    result, _ = score(out, key)
    assert result["checks"][failed]["met"] is False
    assert result["passes"] is False
    text = kit_tool.render(result, key, [])
    assert "does **not** clear the bar" in text
    assert failed.replace("_", " ") in text
    assert "not supported by this test" in text


def test_losing_one_stem_badly_fails_even_when_the_averages_win(material, tmp_path):
    out, _, key = build(material, tmp_path, listeners=5)

    def scores(system, row):
        if row["stem"] == "drums":
            return {"v1": 4, "uvr": 3, "lalal": 5, "moises": 3}[system]  # lalal's drums are better
        return {"v1": 5, "uvr": 3, "lalal": 3, "moises": 3}[system]

    for sheet in (out / "sheets").glob("*.csv"):
        fill(sheet, key, scores)
    result, _ = score(out, key)
    assert result["checks"]["beats_free_reference"]["met"]
    assert result["checks"]["beats_paid_references"]["met"]
    worst = result["checks"]["no_stem_lost_badly"]
    assert (worst["stem"], worst["against"], worst["worst_loss"], worst["met"]) == (
        "drums",
        "lalal",
        1.0,
        False,
    )
    assert result["passes"] is False


def test_a_small_panel_or_a_missing_reference_cannot_pass(material, tmp_path):
    out, _, key = build(material, tmp_path, listeners=4)
    for sheet in (out / "sheets").glob("*.csv"):
        fill(sheet, key, {"v1": 5, "uvr": 3, "lalal": 3, "moises": 3})
    result, _ = score(out, key)
    assert all(check["met"] for check in result["checks"].values())
    assert result["panel_big_enough"] is False
    assert result["passes"] is False
    assert "fewer than 5 listeners" in kit_tool.render(result, key, [])
    no_paid = {**key, "paid_references": []}
    result, _ = score(out, no_paid)
    assert "beats_paid_references" not in result["checks"]
    assert result["passes"] is False


def test_bad_cells_and_missing_rows_are_reported_and_left_out(material, tmp_path):
    out, _, key = build(material, tmp_path, listeners=1)
    sheet = out / "sheets" / "listener-01.csv"
    fill(sheet, key, {"v1": 5, "uvr": 4, "lalal": 3, "moises": 3})
    with sheet.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["overall"] = "6"
    rows[1]["would_use"] = "maybe"
    rows[2]["bleed"] = ""
    dropped = rows.pop(5)
    rows.append({**rows[3]})  # a row twice
    rows.append({**rows[3], "track": "T99"})
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result, problems = score(out, key)
    text = "\n".join(problems)
    assert "overall must be 1 to 5, not '6'" in text
    assert "would_use must be y or n, not 'maybe'" in text
    assert "bleed must be 1 to 5, not ''" in text
    assert "appears twice" in text
    assert "T99" in text
    assert f"no row for {dropped['track']}/{dropped['stem']}/{dropped['sample']}" in text
    assert len(problems) == 6
    assert result["ratings"] == 16 - 4  # three bad cells and one missing row
    assert "Rows with a problem were left out." in kit_tool.render(result, key, problems)


# -- the command -------------------------------------------------------------------------


def test_build_and_score_from_the_command_line(material, tmp_path, capsys):
    out, key_path = tmp_path / "kit", tmp_path / "private" / "key.json"
    arguments = ["build", str(material), "--out", str(out), "--key", str(key_path)]
    assert kit_tool.main([*arguments, "--listeners", "5", "--seed", "9"]) == 0
    printed = capsys.readouterr().out
    assert "4 stems, 5 sheets" in printed
    assert "keep it away from the listeners" in printed
    key = json.loads(key_path.read_text())
    for sheet in (out / "sheets").glob("*.csv"):
        fill(sheet, key, {"v1": 5, "uvr": 4, "lalal": 4, "moises": 3})
    report = tmp_path / "results.md"
    assert (
        kit_tool.main(
            [
                "score",
                str(out),
                "--key",
                str(key_path),
                "--sheets",
                str(out / "sheets"),
                "--out",
                str(report),
            ]
        )
        == 0
    )
    assert "`v1` clears the bar on this test." in report.read_text()
    assert kit_tool.main(arguments) == 1  # the folder is no longer empty
    assert "not empty" in capsys.readouterr().err
    empty = tmp_path / "no-sheets"
    empty.mkdir()
    assert kit_tool.main(["score", str(out), "--key", str(key_path), "--sheets", str(empty)]) == 1
