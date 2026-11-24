"""The owner's listening test: making the kit, serving the page, saving and unblinding marks.

Synthetic audio and the weightless stand-in models throughout.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from stems_bench import listen  # noqa: E402
from stems_bench.datasets import get_dataset  # noqa: E402
from stems_common.settings import load_defaults  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_worker.fakes import fake_pipeline  # noqa: E402

# Before, another model for four of the stems; guitar and piano had the model they have now.
SAME = ("guitar", "piano")
OLD = {stem: "fake_bands" if stem in SAME else "fake_bands_b" for stem in PRODUCT_STEMS}


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """A kit of two synthetic songs, made once for the tests below."""
    run_dir = tmp_path_factory.mktemp("listen") / "run"
    options, defaults = fake_pipeline(load_defaults())
    dataset = get_dataset("synthetic")
    said: list[str] = []
    tracks = listen.make(
        run_dir,
        dataset,
        dataset.track_ids()[:2],
        defaults=defaults,
        old_models=OLD,
        device="cpu",
        pipeline=options,
        progress=said.append,
    )
    return run_dir, tracks, said, (dataset, defaults, options)


def test_the_old_config_differs_only_in_the_single_models() -> None:
    _, defaults = fake_pipeline(load_defaults())
    old = listen.old_defaults(defaults, {"drums": "fake_bands_b"})
    assert old.ensembles.slots["drums"].single_best == "fake_bands_b"
    assert old.ensembles.slots["bass"] == defaults.ensembles.slots["bass"]
    assert old.gate == defaults.gate
    assert defaults.ensembles.slots["drums"].single_best != "fake_bands_b"  # not changed in place


def test_make_writes_every_system_for_the_stems_a_song_has_and_builds_the_kit(made) -> None:
    run_dir, tracks, said, _ = made
    assert said[0].startswith("song 1 of 2: ")
    assert said[-1] == "building the kit"
    kit = listen.load_kit(run_dir / "kit")
    key = json.loads((run_dir / "key.json").read_text())
    assert key["systems"] == list(listen.SYSTEMS)
    assert len(kit["items"]) == sum(len(track["stems"]) for track in tracks)
    assert {item["stem"] for item in kit["items"]} >= set(SAME)
    for item in kit["items"]:
        folder = run_dir / "kit" / "audio" / item["track"] / item["stem"]
        entry = key["items"][f"{item['track']}/{item['stem']}"]
        if item["stem"] in SAME:
            # The same model now as before: one clip stands for both, beside the true stem.
            assert sorted(item["samples"]) == ["A", "B"]
            assert sorted(entry["labels"].values()) == ["new", "reference"]
            assert entry["same_as"] == {"old": "new"}
        else:
            assert sorted(item["samples"]) == ["A", "B", "C"]
            assert sorted(entry["labels"].values()) == sorted(listen.SYSTEMS)
            assert "same_as" not in entry
        # The versions, and the same stretch of the whole song for reference.
        assert sorted(p.name for p in folder.iterdir()) == sorted(
            [*(f"{label}.wav" for label in item["samples"]), "song.wav"]
        )
    # Nothing in the folder that is served says which letter is which.
    served = "".join(
        p.read_text() for p in (run_dir / "kit").rglob("*") if p.suffix in (".json", ".md", ".csv")
    )
    assert "reference" not in served
    assert not (run_dir / "kit" / "key.json").exists()
    # The letters do not always mean the same thing.
    orders = {tuple(sorted(e["labels"].items())) for e in key["items"].values()}
    assert len(orders) > 1


def test_a_song_already_made_is_not_made_again(made) -> None:
    run_dir, tracks, _, (dataset, defaults, options) = made

    def never(*args, **kwargs):
        raise AssertionError("a model was built for a song that is already made")

    again = listen.make(
        run_dir,
        dataset,
        [track["track_id"] for track in tracks],
        defaults=defaults,
        old_models=OLD,
        device="cpu",
        pipeline={**options, "factory": never},
    )
    assert again == tracks


@pytest.fixture
def page(made, tmp_path):
    """The server on a free port, with a ratings file of its own."""
    import shutil

    run_dir = tmp_path / "run"
    shutil.copytree(made[0] / "kit", run_dir / "kit")
    shutil.copy(made[0] / "key.json", run_dir / "key.json")
    server = listen.make_server(run_dir, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", run_dir
    server.shutdown()
    server.server_close()


def call(url: str, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=headers or {})
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def marks(item, overall: dict[str, int], **more):
    ratings = {
        s: {"overall": overall[s], "bleed": None, "artifacts": None} for s in item["samples"]
    }
    return {"ratings": ratings, "note": "", **more}


def test_the_page_its_kit_and_its_audio_are_served_and_nothing_else(page) -> None:
    url, _run_dir = page
    status, headers, body = call(url + "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"Listening test" in body
    assert headers["Cache-Control"] == "no-store"
    status, _, body = call(url + "/api/kit")
    kit = json.loads(body)
    first = kit["items"][0]
    assert set(first) == {"id", "track", "stem", "samples"}
    assert kit["ratings"] == ["overall", "bleed", "artifacts"]
    status, headers, audio = call(f"{url}/audio/{first['track']}/{first['stem']}/A.wav")
    assert (status, headers["Content-Type"], audio[:4]) == (200, "audio/wav", b"RIFF")
    assert call(f"{url}/audio/{first['track']}/{first['stem']}/song.wav")[0] == 200
    # The key, the marks file and anything outside the audio folder cannot be fetched.
    for path in ("/key.json", "/audio/../../key.json", "/audio/%2e%2e/%2e%2e/key.json",
                 "/ratings.json", "/audio/T01/vocals/Z.wav", "/kit.json"):  # fmt: skip
        assert call(url + path)[0] == 404, path
    # A page on another site cannot read or write through the visitor's browser.
    assert call(url + "/api/kit", headers={"Origin": "https://example.org"})[0] == 403
    assert call(url + "/api/kit", headers={"Host": "evil.example"})[0] == 403


def test_marks_are_checked_saved_after_every_step_and_can_be_changed(page) -> None:
    url, run_dir = page
    kit = json.loads(call(url + "/api/kit")[2])
    three = [item for item in kit["items"] if len(item["samples"]) == 3]
    first, second = three[0], three[1]
    assert json.loads(call(url + "/api/ratings")[2]) == {"items": {}}
    good = marks(first, {"A": 5, "B": 3, "C": 1}, note="B is muffled")
    assert call(f"{url}/api/ratings/{first['id']}", good)[0] == 200
    saved = json.loads((run_dir / "ratings.json").read_text())
    assert saved["items"][first["id"]]["ratings"]["B"] == {
        "overall": 3,
        "bleed": None,
        "artifacts": None,
    }
    assert saved["items"][first["id"]]["note"] == "B is muffled"
    # Changing one's mind replaces the step's marks; the other steps are untouched.
    assert (
        call(f"{url}/api/ratings/{second['id']}", marks(second, {"A": 2, "B": 2, "C": 2}))[0] == 200
    )
    assert (
        call(f"{url}/api/ratings/{first['id']}", marks(first, {"A": 4, "B": 4, "C": 1}))[0] == 200
    )
    again = json.loads(call(url + "/api/ratings")[2])["items"]
    assert again[first["id"]]["ratings"]["A"]["overall"] == 4
    assert set(again) == {first["id"], second["id"]}
    # A step with one clip for both of the product's versions takes two marks, not three.
    two = next(item for item in kit["items"] if len(item["samples"]) == 2)
    assert call(f"{url}/api/ratings/{two['id']}", marks(two, {"A": 4, "B": 2}))[0] == 200
    status, _, answer = call(
        f"{url}/api/ratings/{two['id']}", marks(first, {"A": 4, "B": 2, "C": 1})
    )
    assert status == 400
    assert "exactly the versions" in json.loads(answer)["error"]
    again = json.loads(call(url + "/api/ratings")[2])["items"]
    # What is refused, and why.
    for body, reason in [
        (marks(first, {"A": 5, "B": 3, "C": 0}), "from 1 to 5"),
        (marks(first, {"A": 5, "B": 3, "C": 2.5}), "from 1 to 5"),
        ({"ratings": {"A": {"overall": 3}}, "note": ""}, "exactly the versions"),
        ({"ratings": {s: {"overall": None} for s in "ABC"}}, "overall must be"),
        (marks(first, {"A": 5, "B": 3, "C": 1}, note="x" * 2001), "at most 2000"),
        ([1, 2, 3], "must be an object"),
    ]:
        status, _, answer = call(f"{url}/api/ratings/{first['id']}", body)
        assert status == 400, body
        assert reason in json.loads(answer)["error"]
    assert call(f"{url}/api/ratings/T99/vocals", good)[0] == 404
    assert (
        call(f"{url}/api/ratings/{first['id']}", good, {"Origin": "https://example.org"})[0] == 403
    )
    # None of the refused requests changed anything.
    assert json.loads(call(url + "/api/ratings")[2])["items"] == again


def test_the_report_unblinds_the_marks(made, tmp_path) -> None:
    run_dir = made[0]
    kit = listen.load_kit(run_dir / "kit")
    key = json.loads((run_dir / "key.json").read_text())
    items = listen.presentation_order(kit)
    score = {"reference": 5, "new": 4, "old": 2}
    ratings = {"items": {}}
    flipped = next(item["id"] for item in items[:-1] if item["stem"] not in SAME)
    for item in items[:-1]:  # the last step is left unmarked
        labels = key["items"][item["id"]]["labels"]
        given = {
            label: dict.fromkeys(listen.RATINGS, score[system]) for label, system in labels.items()
        }
        if item["id"] == flipped:  # one step where the old version was the better one
            for label, system in labels.items():
                given[label]["overall"] = {"reference": 5, "new": 2, "old": 3}[system]
        ratings["items"][item["id"]] = {
            "ratings": given,
            "note": "hiss" if item["id"] == flipped else "",
        }
    analysis = listen.analyse(kit, key, ratings)
    marked = len(items) - 1
    one_clip = sum(1 for item in items[:-1] if item["stem"] in SAME)
    two_clips = marked - one_clip
    assert one_clip > 0
    assert (analysis["steps"], analysis["marked"]) == (len(items), marked)
    # Now against before counts only the steps where the two were two clips.
    assert analysis["new_vs_old"] == {
        "steps": two_clips,
        "higher": two_clips - 1,
        "lower": 1,
        "same": 0,
        "p": listen.sign_test(two_clips - 1, 1),
    }
    assert analysis["same_model"] == {"steps": one_clip, "stems": list(SAME)}
    assert analysis["reference_top"] == marked
    assert analysis["by_system"]["reference"]["overall"] == 5.0
    # Where one clip stood for both, its mark is the old version's mark too.
    assert analysis["by_stem"]["guitar"]["old"] == analysis["by_stem"]["guitar"]["new"] == 4.0
    assert analysis["by_stem"]["bass"]["old"] < analysis["by_stem"]["bass"]["new"]
    assert analysis["notes"] == [{"id": flipped, "note": "hiss"}]
    text = listen.render(analysis, {"run_id": "owner-test", "tracks": 2})
    assert f"Steps marked: {marked} of {len(items)}. **The test was not finished.**" in text
    assert "| The true stem | 5.00 | 5.00 | 5.00 |" in text
    assert f"Over the {two_clips} steps where the two come from different models" in text
    assert f"was marked higher than the one it made before in {two_clips - 1}, lower in 1" in text
    assert f"In {one_clip} of the marked steps (guitar, piano) the product makes the stem" in text
    assert f"- `{flipped}`: hiss" in text
    assert "One listener" in text
    empty = listen.render(listen.analyse(kit, key, {"items": {}}), {"run_id": "r"})
    assert "Nothing has been marked yet." in empty
    assert "the product makes the stem with the same model" not in empty


def test_a_kit_is_not_rebuilt_over_marks_and_differing_files_are_both_played(
    made, tmp_path
) -> None:
    import shutil

    from stems_worker.audio import load_audio, save_audio

    run_dir = tmp_path / "run"
    shutil.copytree(made[0] / "stems", run_dir / "stems")
    tracks = made[1]
    same = {stem: {"old": "new"} for stem in SAME}
    # The config says guitar is the same model, but one song's file says otherwise.
    path = run_dir / "stems" / "old" / tracks[0]["track_id"] / "guitar.flac"
    audio = load_audio(path, 44100).samples
    save_audio(path, audio * 0.5, 44100, "flac")
    listen.build_kit(run_dir, tracks, same_as=same)
    key = json.loads((run_dir / "key.json").read_text())
    assert sorted(key["items"]["T01/guitar"]["labels"].values()) == sorted(listen.SYSTEMS)
    assert sorted(key["items"]["T02/guitar"]["labels"].values()) == sorted(listen.SYSTEMS)
    assert key["items"]["T01/piano"]["same_as"] == {"old": "new"}
    # Once there are marks, the letters must keep their meaning.
    (run_dir / "ratings.json").write_text('{"items": {}}')
    with pytest.raises(listen.ListenError, match="holds marks"):
        listen.build_kit(run_dir, tracks, same_as=same)
    assert json.loads((run_dir / "key.json").read_text()) == key


def test_a_stem_whose_true_recording_is_next_to_nothing_is_left_out(made, tmp_path) -> None:
    import shutil

    from stems_worker.audio import load_audio, save_audio

    run_dir = tmp_path / "run"
    shutil.copytree(made[0] / "stems", run_dir / "stems")
    tracks = made[1]
    limit = load_defaults().gate.rules.empty_below_mix_db
    # Nothing is left out of the synthetic songs as they are.
    assert listen.build_kit(run_dir, tracks, empty_below_mix_db=limit) == []
    assert json.loads((made[0] / "left_out.json").read_text())["steps"] == []
    # One song's true bass turned down by 60 dB: far under the level the product calls empty.
    path = run_dir / "stems" / "reference" / tracks[1]["track_id"] / "bass.flac"
    save_audio(path, load_audio(path, 44100).samples * 0.001, 44100, "flac")
    left = listen.build_kit(run_dir, tracks, empty_below_mix_db=limit)
    assert [(step["track"], step["stem"]) for step in left] == [("T02", "bass")]
    assert left[0]["below_song_db"] < limit
    kit = listen.load_kit(run_dir / "kit")
    steps = {(item["track"], item["stem"]) for item in kit["items"]}
    assert ("T02", "bass") not in steps
    assert ("T01", "bass") in steps
    assert len(steps) == sum(len(track["stems"]) for track in tracks) - 1
    assert json.loads((run_dir / "left_out.json").read_text()) == {
        "empty_below_mix_db": limit,
        "steps": left,
    }
    assert not (run_dir / "kit" / "left_out.json").exists()
    # The report says so.
    key = json.loads((run_dir / "key.json").read_text())
    run = {"run_id": "r", "left_out": left, "empty_below_mix_db": limit}
    text = listen.render(listen.analyse(kit, key, {"items": {}}), run)
    assert "Nothing has been marked yet." in text
    marked = {"items": {}}
    first = listen.presentation_order(kit)[0]
    marked["items"][first["id"]] = {
        "ratings": {s: dict.fromkeys(listen.RATINGS, 3) for s in first["samples"]},
        "note": "",
    }
    text = listen.render(listen.analyse(kit, key, marked), run)
    assert f"more than {-limit:g} dB under the song): `T02/bass` (" in text
    # Without the rule, the step is in the kit.
    assert listen.build_kit(run_dir, tracks) == []
    assert ("T02", "bass") in {
        (i["track"], i["stem"]) for i in listen.load_kit(run_dir / "kit")["items"]
    }


def test_the_sign_test() -> None:
    assert listen.sign_test(0, 0) is None
    assert listen.sign_test(5, 5) == 1.0
    assert listen.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert listen.sign_test(0, 10) == listen.sign_test(10, 0)
    assert listen.sign_test(9, 1) == pytest.approx(2 * 11 / 1024)


def test_the_page_is_one_document_that_talks_only_to_this_server() -> None:
    from stems_bench.listen_page import PAGE

    assert PAGE.lstrip().startswith("<!doctype html>")
    assert "http://" not in PAGE
    assert "https://" not in PAGE
    for route in ("/api/kit", "/api/ratings", "/audio/"):
        assert route in PAGE


def test_commands_say_what_is_missing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(listen, "LISTEN_DIR", tmp_path)
    assert listen.main(["report", "--run-id", "nothing"]) == 2
    assert "run `make` first" in capsys.readouterr().err
    assert listen.main(["serve", "--run-id", "nothing"]) == 2
    assert "run `make` first" in capsys.readouterr().err
