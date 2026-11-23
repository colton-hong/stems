"""The Clef-omni test's clips: window choice, presence, bleed mixes, encoding, the clip set."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from stems_bench.clef import clips as c
from stems_bench.clef.clipset import build_clips, make_plan
from stems_bench.datasets.synthetic import SyntheticDataset
from stems_common.stems import PRODUCT_STEMS

SR = 44100


def tone(seconds: float, level: float = 0.1, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * freq * t))[None].astype(np.float32)


def test_the_loudest_window_is_found_on_the_grid():
    mix = np.concatenate([tone(3.4, 0.01), tone(4.0, 0.5), tone(5.0, 0.01)], axis=1)
    window = c.loudest_window(mix, SR, seconds=4.0)
    # The loud part starts at 3.4 s; whole-second starts 3 and 4 hold equal shares of it
    # only if it were centred, and 3 holds more.
    assert (window.start, window.length) == (3 * SR, 4 * SR)
    assert c.loudest_window(mix, SR, seconds=4.0, grid_seconds=0.1).start == round(3.4 * SR)
    assert window.cut(mix).shape == (1, 4 * SR)


def test_ties_go_to_the_earliest_window_and_short_tracks_are_used_whole():
    flat = tone(10.0, 0.2, freq=441.0)  # a whole number of cycles per second
    assert c.loudest_window(flat, SR, seconds=4.0).start == 0
    short = tone(2.5)
    assert c.loudest_window(short, SR, seconds=30.0) == c.Window(0, short.shape[1])
    with pytest.raises(c.ClipError, match="empty"):
        c.loudest_window(np.zeros((2, 0)), SR)


def test_presence_needs_both_level_and_five_sounding_seconds():
    silence = np.zeros((1, 10 * SR), dtype=np.float32)
    assert c.is_present(tone(10.0, 0.1), SR)
    assert not c.is_present(silence, SR)
    # Loud enough overall, but sounding for only 4 of 30 seconds.
    burst = np.concatenate([tone(4.0, 0.5), np.zeros((1, 26 * SR), dtype=np.float32)], axis=1)
    assert c.rms_dbfs(burst) > c.PRESENT_RMS_DBFS
    assert not c.is_present(burst, SR)
    # Sounding throughout, but 50 dB down.
    assert not c.is_present(tone(10.0, 10 ** (-50 / 20)), SR)
    # Just either side of -40 dBFS RMS (a sine's RMS is 3 dB under its peak).
    assert c.is_present(tone(10.0, 10 ** (-36.5 / 20)), SR)
    assert not c.is_present(tone(10.0, 10 ** (-37.5 / 20)), SR)


def test_lead_and_backing_answers():
    on, off = tone(10.0), np.zeros((1, 10 * SR), dtype=np.float32)
    window = c.Window(0, 10 * SR)
    cases = {(True, True): "both", (True, False): "lead only",
             (False, True): "backing only", (False, False): "neither"}  # fmt: skip
    for (lead, backing), answer in cases.items():
        assert (
            c.lead_backing_answer(on if lead else off, on if backing else off, window, SR) == answer
        )
        assert answer in c.LEAD_BACKING_ANSWERS


@pytest.mark.parametrize("sir", c.SIR_LEVELS_DB)
def test_bleed_mixes_hit_the_requested_sir(sir):
    rng = np.random.default_rng(3)
    target = (0.2 * rng.standard_normal((2, SR))).astype(np.float32)
    contaminant = (0.03 * rng.standard_normal((2, SR))).astype(np.float32)
    mixture, scaled = c.mix_at_sir(target, contaminant, sir)
    assert c.measured_sir_db(target, scaled) == pytest.approx(sir, abs=1e-6)
    np.testing.assert_allclose(mixture, target.astype(np.float64) + scaled)


def test_bleed_mixes_refuse_silence_and_mismatched_shapes():
    signal, silence = tone(1.0), np.zeros((1, SR), dtype=np.float32)
    with pytest.raises(c.ClipError, match="silent"):
        c.mix_at_sir(signal, silence, 20.0)
    with pytest.raises(c.ClipError, match="shapes differ"):
        c.mix_at_sir(signal, tone(2.0), 20.0)


def test_part_b_plan_is_reproducible_and_uses_each_track_once():
    present = {
        f"t{n}": {stem: (stem != "piano" or n % 2 == 0) for stem in PRODUCT_STEMS}
        for n in range(25)
    }
    plan = c.plan_part_b(present, seed=1)
    assert plan == c.plan_part_b(present, seed=1)
    assert plan != c.plan_part_b(present, seed=2)
    assert [item.target for item in plan] == list(c.PART_B_TARGETS)
    assert len({item.track_id for item in plan}) == 20
    for item in plan:
        assert item.contaminant != item.target
        assert present[item.track_id][item.target]
        assert present[item.track_id][item.contaminant]
    assert c.PART_B_TARGETS.count("piano") == 2


def test_part_b_plan_fails_loudly_when_tracks_run_out():
    present = {"only": dict.fromkeys(PRODUCT_STEMS, True)}
    with pytest.raises(c.ClipError, match="no unused track has vocals"):
        c.plan_part_b(present, seed=1, targets=("vocals", "vocals"))
    lonely = {"t": {**dict.fromkeys(PRODUCT_STEMS, False), "drums": True}}
    with pytest.raises(c.ClipError, match="no unused track has drums"):
        c.plan_part_b(lonely, seed=1, targets=("drums",))


def test_clips_are_16k_mono_pcm_and_only_ever_turned_down():
    stereo = np.concatenate([tone(2.0, 0.3), tone(2.0, 0.1)])
    clip = c.encode_clip(stereo, SR)
    audio, rate = sf.read(io.BytesIO(clip.data), dtype="float32")
    info = sf.info(io.BytesIO(clip.data))
    assert (rate, audio.ndim, info.subtype) == (16000, 1, "PCM_16")
    assert clip.seconds == pytest.approx(2.0)
    assert clip.gain == 1.0
    assert np.abs(audio).max() == pytest.approx(0.2, abs=0.01)  # the mean of the two channels
    assert len(clip.data) < c.MAX_CLIP_BYTES
    hot = c.encode_clip(tone(1.0, 1.8), SR)
    assert hot.gain == pytest.approx(0.999 / 1.8, rel=0.01)
    assert np.abs(sf.read(io.BytesIO(hot.data), dtype="float32")[0]).max() <= 1.0
    # 30 seconds, the test's clip length, is well under the request limit.
    assert len(c.encode_clip(tone(30.0), SR).data) < 1_000_000


@pytest.fixture(scope="module")
def clip_set(tmp_path_factory):
    dataset = SyntheticDataset()
    targets = ("vocals", "drums", "bass", "guitar", "piano")
    plan = make_plan(dataset, seed=5, window_seconds=8.0, part_b_targets=targets, reverb_clips=1)
    out = tmp_path_factory.mktemp("clips")
    return dataset, plan, out, build_clips(dataset, plan, out)


def test_the_plan_on_the_synthetic_set(clip_set):
    dataset, plan, _, _ = clip_set
    assert list(plan["tracks"]) == dataset.track_ids()
    assert plan == make_plan(dataset, seed=5, window_seconds=8.0,
                             part_b_targets=("vocals", "drums", "bass", "guitar", "piano"),
                             reverb_clips=1)  # fmt: skip
    json.dumps(plan)  # plain data
    for track_id, entry in plan["tracks"].items():
        window = entry["window"]
        assert window["seconds"] == 8.0
        assert window["start_seconds"] == int(window["start_seconds"])
        track = dataset.load(track_id)
        assert window["start_sample"] + window["num_samples"] <= track.inputs["mix"].shape[1]
        # Presence agrees with which stems the fixture actually generated.
        silent = {s for s in PRODUCT_STEMS if not np.any(track.references[s])}
        assert all(not entry["present"][s] for s in silent)
    assert plan["lead_backing"], "the fixture has tracks with lead and backing vocals"
    assert len(plan["part_b"]) == 5
    assert len(plan["reverb"]) == 1
    assert plan["reverb"][0]["target"] == "vocals"


def test_every_clip_is_written_with_its_answer_and_measured_sir(clip_set):
    dataset, plan, out, manifest = clip_set
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    assert json.loads((out / "manifest.json").read_text()) == manifest
    by_part = {part: [x for x in clips.values() if x["part"] == part]
               for part in ("A", "lead_backing", "B", "reverb")}  # fmt: skip
    assert len(by_part["A"]) == len(dataset.track_ids())
    assert len(by_part["lead_backing"]) == len(plan["lead_backing"])
    assert len(by_part["B"]) == 5 * 5
    assert len(by_part["reverb"]) == 1
    for clip in clips.values():
        data = (out / clip["file"]).read_bytes()
        info = sf.info(io.BytesIO(data))
        assert (info.samplerate, info.channels) == (16000, 1)
        assert info.duration == pytest.approx(8.0) == clip["seconds"]
        assert len(data) == clip["bytes"] < c.MAX_CLIP_BYTES
        assert 0 < clip["gain"] <= 1.0
    for clip in by_part["A"]:
        assert clip["present"] == plan["tracks"][clip["track_id"]]["present"]
    for clip in by_part["lead_backing"]:
        assert clip["answer"] in c.LEAD_BACKING_ANSWERS
    levels = sorted(clip["bleed_level"] for clip in by_part["B"])
    assert levels == sorted([0, 1, 2, 3, 4] * 5)
    for clip in by_part["B"]:
        if clip["sir_db"] is None:
            assert clip["measured_sir_db"] is None
            assert clip["id"].endswith("-clean")
        else:
            assert clip["measured_sir_db"] == pytest.approx(clip["sir_db"], abs=1e-6)
            assert clip["id"].endswith(f"-sir{clip['sir_db']:02.0f}")


def test_bleed_is_audible_in_the_written_clips_in_the_right_order(clip_set):
    _, _, out, manifest = clip_set
    read = lambda name: sf.read(out / name, dtype="float64")[0]  # noqa: E731
    for n in range(1, 6):
        clean = read(f"b{n:02d}-clean.wav")
        added = []
        for sir in c.SIR_LEVELS_DB:
            clip = next(x for x in manifest["clips"] if x["id"] == f"b{n:02d}-sir{sir:02.0f}")
            # Undo the anti-clipping gain, then what is left after removing the clean stem
            # is the contaminant as encoded.
            residue = read(clip["file"]) / clip["gain"] - clean
            added.append(float(np.mean(residue**2)))
        assert added == sorted(added), "more bleed at lower SIR"
        # 30 to 6 dB is a 24 dB step in the contaminant; mono and 16 kHz change both
        # parts, so allow a few dB.
        assert 10 * np.log10(added[-1] / added[0]) == pytest.approx(24.0, abs=3.0)


def test_a_plan_for_another_dataset_is_refused(clip_set, tmp_path):
    dataset, plan, _, _ = clip_set
    with pytest.raises(c.ClipError, match="the plan is for"):
        build_clips(dataset, {**plan, "dataset": "elsewhere"}, tmp_path)
