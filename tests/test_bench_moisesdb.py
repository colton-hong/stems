"""The MoisesDB loader and subset selection, against a fake tree in the dataset's layout."""

from __future__ import annotations

import json
import random
import shutil

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("soxr")

from stems_bench.datasets import (  # noqa: E402
    DatasetError,
    DatasetUnavailableError,
    get_dataset,
    moisesdb,
)
from stems_bench.datasets.moisesdb import (  # noqa: E402
    QUOTAS,
    MoisesDbDataset,
    TrackInfo,
    build_frozen,
    read_meta,
    scan,
    select_subset,
    strata_of,
)
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402

RATE = 44100
SECONDS = 1.5


def _write(path, audio, rate=RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, rate, subtype="PCM_24")


def _make_track(root, track_id, seed, genre, parts, provider="moisesdb_v0.1"):
    """``parts``: list of (stem name as in data.json, track type, audio, sample rate)."""
    folder = root / provider / track_id
    stems: dict[str, list] = {}
    for i, (stem, track_type, audio, rate) in enumerate(parts):
        source_id = f"src-{seed}-{i}"
        _write(folder / stem / f"{source_id}.wav", audio, rate)
        stems.setdefault(stem, []).append(
            {"id": source_id, "trackType": track_type, "extension": "wav", "has_bleed": False}
        )
    data = {
        "song": f"Song {seed}",
        "artist": f"Artist {seed}",
        "genre": genre,
        "stems": [{"id": f"stem-{n}", "stemName": n, "tracks": t} for n, t in stems.items()],
    }
    (folder / "data.json").write_text(json.dumps(data))


def _parts(seed, *, vocals="both", guitar=True, piano=True, loud_rhythm=True):
    t = generate_track(seed=seed, duration_seconds=SECONDS)
    gain = 1.0 if loud_rhythm else 0.01
    parts = [
        ("drums", "full acoustic drumkit", gain * t.stems["drums"], RATE),
        ("bass", "bass guitar", gain * t.stems["bass"], RATE),
        ("other_keys", "synth pad", t.stems["other"], RATE),
    ]
    if vocals in ("both", "lead"):
        parts.append(("vocals", "lead female singer", t.lead_vocals, RATE))
    if vocals == "both":
        parts.append(("vocals", "background vocals", t.backing_vocals, RATE))
    if guitar:
        parts.append(("guitar", "acoustic guitar", t.stems["guitar"], RATE))
    if piano:
        parts.append(("piano", "grand piano", t.stems["piano"], RATE))
    return parts


GENRES = ["rock", "pop", "jazz", "blues", "country", "rap", "reggae"]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("moisesdb")
    kinds = [
        {"vocals": "both"},
        {"vocals": "both", "guitar": False},
        {"vocals": "lead", "piano": False},
        {"vocals": "lead", "loud_rhythm": False},
        {"vocals": "none"},
        {"vocals": "none", "guitar": False},
        {"vocals": "none", "piano": False},
        {"vocals": "both", "piano": False, "guitar": False},
    ]
    for i, kind in enumerate(kinds):
        _make_track(root, f"track-{i:02d}", 30 + i, GENRES[i % len(GENRES)], _parts(30 + i, **kind))
    return root


SMALL = {"size": 6, "quotas": dict.fromkeys(QUOTAS, 2), "min_genres": 4, "expected_tracks": 8}


def _frozen(tree, tmp_path):
    path = tmp_path / "moisesdb_subset.json"
    moisesdb.write_frozen(path, build_frozen(tree, **SMALL))
    return MoisesDbDataset(root=tree, frozen_path=path)


# --- reading the layout -----------------------------------------------------------------


def test_scan_reads_genre_and_sources_from_data_json(tree) -> None:
    metas = scan(tree)
    assert [m.track_id for m in metas] == [f"track-{i:02d}" for i in range(8)]
    first = metas[0]
    assert (first.provider, first.genre) == ("moisesdb_v0.1", "rock")
    assert {s.stem for s in first.sources} == {
        "drums", "bass", "other_keys", "vocals", "guitar", "piano"
    }  # fmt: skip
    assert sorted(s.track_type for s in first.sources if s.stem == "vocals") == [
        "background vocals",
        "lead female singer",
    ]
    # Pointing straight at the provider folder works too.
    assert [m.track_id for m in scan(tree / "moisesdb_v0.1")] == [m.track_id for m in metas]


def test_stems_are_grouped_into_six_plus_lead_and_backing(tmp_path) -> None:
    t = generate_track(seed=5, duration_seconds=SECONDS)
    zeros = np.zeros_like(t.stems["drums"])
    shaker = 0.2 * t.stems["drums"]
    parts = [
        ("vocals", "lead male singer", t.lead_vocals, RATE),
        ("vocals", "human choir", t.backing_vocals, RATE),
        ("vocals", "other (vocoder, beatboxing etc)", 0.1 * t.stems["other"], RATE),
        ("drums", "kick drum", 0.5 * t.stems["drums"], RATE),
        ("drums", "overheads", 0.5 * t.stems["drums"], RATE),
        ("bass", "bass guitar", t.stems["bass"], RATE),
        ("guitar", "clean electric guitar", t.stems["guitar"], RATE),
        ("piano", "electric piano (rhodes, wurlitzer, piano sound alike)", t.stems["piano"], RATE),
        # Everything below belongs to "other". The data.json name has a space in it.
        ("percussion", "pitched percussion (mallets, glockenspiel, ...)", shaker, RATE),
        ("other keys", "synth lead", t.stems["other"], RATE),
        ("bowed_strings", "string section", 0.3 * t.stems["guitar"], RATE),
        ("wind", "other wind", zeros, RATE),
    ]  # fmt: skip
    _make_track(tmp_path, "grouped", 5, "pop", parts, provider="p")
    audio = moisesdb.decode(read_meta(tmp_path / "p" / "grouped", "p"), tmp_path)
    tolerance = 2.0**-22  # 24-bit files
    np.testing.assert_allclose(audio["drums"], t.stems["drums"], atol=tolerance)
    np.testing.assert_allclose(audio["piano"], t.stems["piano"], atol=tolerance)
    np.testing.assert_allclose(audio["lead_vocals"], t.lead_vocals, atol=tolerance)
    np.testing.assert_allclose(audio["backing_vocals"], t.backing_vocals, atol=tolerance)
    # The vocoder is vocals but neither lead nor backing.
    np.testing.assert_allclose(
        audio["vocals"], t.stems["vocals"] + 0.1 * t.stems["other"], atol=2 * tolerance
    )
    expected_other = 0.2 * t.stems["drums"] + t.stems["other"] + 0.3 * t.stems["guitar"]
    np.testing.assert_allclose(audio["other"], expected_other, atol=4 * tolerance)
    total = sum(audio[s].astype(np.float64) for s in PRODUCT_STEMS)
    np.testing.assert_allclose(audio["mix"], total, atol=1e-6)


def test_mono_resampled_and_short_sources_are_brought_to_one_shape(tmp_path) -> None:
    import soxr

    t = generate_track(seed=6, duration_seconds=SECONDS)
    drums_48k = soxr.resample(t.stems["drums"].T, RATE, 48000, quality="VHQ").T
    parts = [
        ("drums", "drum machine", drums_48k, 48000),
        ("bass", "bass guitar", t.stems["bass"][:1], RATE),  # a mono file
        ("guitar", "acoustic guitar", t.stems["guitar"][:, : RATE // 2], RATE),  # a short file
    ]
    _make_track(tmp_path, "odd", 6, "rock", parts, provider="p")
    audio = moisesdb.decode(read_meta(tmp_path / "p" / "odd", "p"), tmp_path)
    samples = round(SECONDS * RATE)
    assert {x.shape for x in audio.values()} == {(2, samples)}
    assert {x.dtype for x in audio.values()} == {np.dtype("float32")}
    np.testing.assert_allclose(audio["bass"][1], t.stems["bass"][0], atol=1e-6)
    half = RATE // 2
    np.testing.assert_allclose(audio["guitar"][:, :half], t.stems["guitar"][:, :half], atol=1e-6)
    assert not audio["guitar"][:, half:].any()  # padded with silence, not cut
    middle = slice(2000, samples - 2000)
    assert np.abs(audio["drums"] - t.stems["drums"])[:, middle].max() < 0.02
    assert not audio["vocals"].any()


def test_source_types_are_read_as_the_release_writes_them(tmp_path) -> None:
    # The real data.json files say "lead_male_singer" and "background_vocals".
    t = generate_track(seed=8, duration_seconds=SECONDS)
    parts = [
        ("vocals", "lead_male_singer", t.lead_vocals, RATE),
        ("vocals", "background_vocals", t.backing_vocals, RATE),
        ("vocals", "Human_Choir", 0.5 * t.backing_vocals, RATE),
        ("drums", "hi_hat", t.stems["drums"], RATE),
    ]
    _make_track(tmp_path, "underscored", 8, "rock", parts, provider="p")
    meta = read_meta(tmp_path / "p" / "underscored", "p")
    assert sorted({s.track_type for s in meta.sources}) == [
        "background vocals",
        "hi hat",
        "human choir",
        "lead male singer",
    ]
    audio = moisesdb.decode(meta, tmp_path)
    np.testing.assert_allclose(audio["lead_vocals"], t.lead_vocals, atol=2.0**-21)
    np.testing.assert_allclose(audio["backing_vocals"], 1.5 * t.backing_vocals, atol=2.0**-20)
    assert "lead_and_backing" in moisesdb.analyse(meta, tmp_path).strata


def test_unknown_stems_and_empty_tracks_are_errors(tmp_path) -> None:
    t = generate_track(seed=7, duration_seconds=0.5)
    _make_track(tmp_path, "weird", 7, "pop", [("theremin", "x", t.stems["bass"], RATE)], "p")
    with pytest.raises(DatasetError, match="unknown MoisesDB stem 'theremin'"):
        scan(tmp_path)
    shutil.rmtree(tmp_path / "p")
    (tmp_path / "p" / "empty").mkdir(parents=True)
    (tmp_path / "p" / "empty" / "data.json").write_text('{"genre": "pop", "stems": []}')
    with pytest.raises(DatasetError, match="lists no audio"):
        scan(tmp_path)


# --- strata and selection ----------------------------------------------------------------


def _levels(**given):
    levels = dict.fromkeys((*PRODUCT_STEMS, "lead_vocals", "backing_vocals"), -120.0)
    return {**levels, "mix": -18.0, **given}


def test_strata_follow_from_stem_levels() -> None:
    assert strata_of(_levels()) == ("instrumental",)
    voices = {"vocals": -24, "lead_vocals": -25, "backing_vocals": -35}
    full = _levels(**voices, guitar=-30, piano=-59.9, drums=-22, bass=-30)
    assert set(strata_of(full)) == set(QUOTAS) - {"instrumental"}
    # Bass more than 12 dB under the mix: present, not prominent.
    assert "prominent_drums_bass" not in strata_of(_levels(drums=-20, bass=-30.1))
    # A stem under -60 dBFS is not there.
    assert "guitar" not in strata_of(_levels(guitar=-60.1))
    # Backing only: vocal, but no lead.
    assert strata_of(_levels(vocals=-30, backing_vocals=-30)) == ()


def _candidates(count=240, seed=0):
    rng = random.Random(seed)
    genres = [f"genre-{i}" for i in range(12)]
    out = []
    for i in range(count):
        odds = (("guitar", 0.6), ("piano", 0.3), ("prominent_drums_bass", 0.5))
        strata = [s for s, p in odds if rng.random() < p]
        if rng.random() < 0.1:
            strata.append("instrumental")
        else:
            strata.append("lead_vocals")
            if rng.random() < 0.4:
                strata.append("lead_and_backing")
        out.append(TrackInfo(f"id-{i:03d}", rng.choice(genres), tuple(strata)))
    return out


def test_selection_meets_every_quota_and_is_reproducible() -> None:
    candidates = _candidates()
    chosen = select_subset(candidates)
    assert len(chosen) == 50
    assert len({t.track_id for t in chosen}) == 50
    for stratum, least in QUOTAS.items():
        assert sum(stratum in t.strata for t in chosen) >= least, stratum
    assert len({t.genre for t in chosen}) >= 6
    assert QUOTAS == {
        "lead_vocals": 8,
        "instrumental": 8,
        "guitar": 10,
        "piano": 10,
        "prominent_drums_bass": 10,
        "lead_and_backing": 8,
    }
    # The same candidates in any order give the same subset; another seed gives another.
    shuffled = candidates[::-1]
    assert select_subset(shuffled) == chosen
    assert select_subset(candidates, seed=1) != chosen
    assert [t.track_id for t in chosen] == sorted(t.track_id for t in chosen)


def test_a_stratum_the_dataset_lacks_takes_all_there_are_when_that_is_allowed() -> None:
    candidates = _candidates()
    few = [t for t in candidates if "instrumental" not in t.strata]
    few += [t for t in candidates if "instrumental" in t.strata][:3]
    with pytest.raises(DatasetError, match="only 3 tracks are 'instrumental'"):
        select_subset(few)
    chosen = select_subset(few, shortfall_ok=("instrumental",))
    assert len(chosen) == 50
    assert sum("instrumental" in t.strata for t in chosen) == 3  # every one that exists
    for stratum, least in QUOTAS.items():
        if stratum != "instrumental":
            assert sum(stratum in t.strata for t in chosen) >= least
    # Other strata are still enforced.
    no_piano = [
        TrackInfo(t.track_id, t.genre, tuple(s for s in t.strata if s != "piano")) for t in few
    ]
    with pytest.raises(DatasetError, match="only 0 tracks are 'piano'"):
        select_subset(no_piano, shortfall_ok=("instrumental",))
    assert moisesdb.SHORTFALL_OK == ("instrumental",)


def test_selection_is_random_not_the_first_tracks() -> None:
    chosen = {t.track_id for t in select_subset(_candidates())}
    assert chosen != {f"id-{i:03d}" for i in range(50)}
    assert max(chosen) > "id-200"


def test_selection_says_which_requirement_cannot_be_met() -> None:
    candidates = _candidates()
    no_instrumentals = [t for t in candidates if "instrumental" not in t.strata]
    with pytest.raises(DatasetError, match="only 0 tracks are 'instrumental'; the subset needs 8"):
        select_subset(no_instrumentals)
    with pytest.raises(DatasetError, match="fewer than 6 genres"):
        select_subset([TrackInfo(t.track_id, "rock", t.strata) for t in candidates])
    with pytest.raises(DatasetError, match="only 30 candidates, the subset needs 50"):
        select_subset(candidates[:30], quotas={})
    disjoint = [TrackInfo(f"a{i}", f"g{i}", ("guitar",)) for i in range(5)]
    disjoint += [TrackInfo(f"b{i}", f"g{i}", ("piano",)) for i in range(5)]
    with pytest.raises(DatasetError, match="takes 8 tracks, more than 6"):
        select_subset(disjoint, size=6, quotas={"guitar": 4, "piano": 4}, min_genres=1)
    with pytest.raises(DatasetError, match="duplicate track ids"):
        select_subset([candidates[0], candidates[0]])


# --- freezing and loading ----------------------------------------------------------------


def test_build_frozen_records_selection_rules_strata_and_file_hashes(tree) -> None:
    data = build_frozen(tree, **SMALL)
    assert data["dataset"] == "moisesdb_subset"
    assert data["protocol_version"] == 1
    assert data["selection"]["seed"] == moisesdb.SELECTION_SEED
    assert data["selection"]["candidates"] == 8
    assert len(data["selection"]["candidate_ids_sha256"]) == 64
    assert len(data["tracks"]) == 6
    assert build_frozen(tree, **SMALL) == data  # deterministic
    assert data["selection"]["available"]["instrumental"] == 3
    assert data["selection"]["shortfalls"] == {}
    scarce = build_frozen(tree, **{**SMALL, "quotas": {**SMALL["quotas"], "instrumental": 4}})
    assert scarce["selection"]["shortfalls"] == {"instrumental": 3}
    assert sum("instrumental" in t["strata"] for t in scarce["tracks"]) == 3
    by_id = {t["id"]: t for t in data["tracks"]}
    for stratum in QUOTAS:
        assert sum(stratum in t["strata"] for t in data["tracks"]) >= 2, stratum
    entry = next(iter(by_id.values()))
    assert "data.json" in entry["files"]
    assert all(len(v) == 64 for v in entry["files"].values())
    assert set(entry["levels_dbfs"]) == {*PRODUCT_STEMS, "lead_vocals", "backing_vocals", "mix"}
    assert "song" not in json.dumps(data).lower()  # titles and artists are not copied
    with pytest.raises(DatasetError, match="found 8 tracks, expected 240"):
        build_frozen(tree)


def test_fake_tree_strata_match_how_the_tracks_were_built(tree) -> None:
    infos = {m.track_id: moisesdb.analyse(m, tree) for m in scan(tree)}
    assert set(infos["track-00"].strata) == set(QUOTAS) - {"instrumental"}
    assert "lead_and_backing" not in infos["track-02"].strata
    assert "piano" not in infos["track-02"].strata
    assert "prominent_drums_bass" not in infos["track-03"].strata  # rhythm section 40 dB down
    assert "instrumental" in infos["track-04"].strata
    assert "lead_vocals" not in infos["track-04"].strata


def test_a_frozen_track_loads_with_all_targets(tree, tmp_path) -> None:
    dataset = _frozen(tree, tmp_path)
    assert len(dataset.track_ids()) == 6
    track = dataset.load(dataset.track_ids()[0])
    assert set(track.inputs) == {"mix", "vocals"}
    assert set(track.references) == set(dataset.targets)
    ref = track.references
    np.testing.assert_allclose(ref["instrumental"] + ref["vocals"], track.mix, atol=1e-6)
    np.testing.assert_allclose(
        ref["other_4stem"], ref["guitar"] + ref["piano"] + ref["other"], atol=1e-6
    )
    assert track.meta["genre"] in GENRES
    assert isinstance(track.meta["strata"], list)
    assert track.sample_rate == RATE


def test_loading_refuses_audio_that_changed_since_the_freeze(tree, tmp_path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(tree, copy)
    dataset = _frozen(copy, tmp_path)
    first, second, third = dataset.track_ids()[:3]
    entry = dataset.frozen.track(first)
    wav = next(name for name in entry["files"] if name.endswith(".wav"))
    path = copy / entry["provider"] / first / wav
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    sf.write(path, audio * 0.5, RATE, subtype="PCM_24")
    with pytest.raises(DatasetError, match=r"does not match moisesdb_subset\.json"):
        dataset.load(first)
    entry = dataset.frozen.track(second)
    wav = next(name for name in entry["files"] if name.endswith(".wav"))
    (copy / entry["provider"] / second / wav).unlink()
    with pytest.raises(DatasetError, match="is missing"):
        dataset.load(second)
    dataset.load(third)


def test_unavailable_and_unfrozen_are_reported_separately(tree, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MOISESDB_PATH", raising=False)
    monkeypatch.setattr(moisesdb, "get_env", lambda name: None)
    with pytest.raises(DatasetUnavailableError, match="MOISESDB_PATH is not set"):
        MoisesDbDataset(frozen_path=tmp_path / "none.json").track_ids()
    with pytest.raises(DatasetError, match="has not been frozen"):
        MoisesDbDataset(root=tree, frozen_path=tmp_path / "none.json").track_ids()
    monkeypatch.setattr(moisesdb, "get_env", lambda name: str(tmp_path / "nowhere"))
    with pytest.raises(DatasetUnavailableError, match="is not a folder"):
        MoisesDbDataset(frozen_path=tmp_path / "none.json").track_ids()
    assert isinstance(get_dataset("moisesdb_subset"), MoisesDbDataset)
    monkeypatch.setattr(moisesdb, "get_env", lambda name: None)
    monkeypatch.setattr(moisesdb, "FROZEN_PATH", tmp_path / "none.json")
    assert moisesdb.main(["--freeze"]) == 1
