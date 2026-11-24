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

OLD = dict.fromkeys(PRODUCT_STEMS, "fake_bands_b")


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
    for item in kit["items"]:
        assert sorted(item["samples"]) == ["A", "B", "C"]
        folder = run_dir / "kit" / "audio" / item["track"] / item["stem"]
        assert sorted(p.name for p in folder.iterdir()) == ["A.wav", "B.wav", "C.wav"]
        labels = key["items"][f"{item['track']}/{item['stem']}"]["labels"]
        assert sorted(labels.values()) == sorted(listen.SYSTEMS)
    # The song itself is there for reference, once per song.
    assert (run_dir / "kit" / "audio" / "T01" / "mix.wav").is_file()
    assert (run_dir / "kit" / "audio" / "T02" / "mix.wav").is_file()
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
    assert call(f"{url}/audio/{first['track']}/mix.wav")[0] == 200
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
    first, second = kit["items"][0], kit["items"][1]
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
    for number, item in enumerate(items[:-1]):  # the last step is left unmarked
        labels = key["items"][item["id"]]["labels"]
        given = {
            label: dict.fromkeys(listen.RATINGS, score[system]) for label, system in labels.items()
        }
        if number == 0:  # one step where the old version was the better one
            for label, system in labels.items():
                given[label]["overall"] = {"reference": 5, "new": 2, "old": 3}[system]
        ratings["items"][item["id"]] = {"ratings": given, "note": "hiss" if number == 0 else ""}
    analysis = listen.analyse(kit, key, ratings)
    marked = len(items) - 1
    assert (analysis["steps"], analysis["marked"]) == (len(items), marked)
    assert analysis["new_vs_old"]["higher"] == marked - 1
    assert analysis["new_vs_old"]["lower"] == 1
    assert analysis["reference_top"] == marked
    assert analysis["by_system"]["reference"]["overall"] == 5.0
    assert analysis["notes"] == [{"id": items[0]["id"], "note": "hiss"}]
    text = listen.render(analysis, {"run_id": "owner-test", "tracks": 2})
    assert f"Steps marked: {marked} of {len(items)}. **The test was not finished.**" in text
    assert "| The true stem | 5.00 | 5.00 | 5.00 |" in text
    assert f"was marked higher than the one it made before in {marked - 1}, lower in 1" in text
    assert f"- `{items[0]['id']}`: hiss" in text
    assert "One listener" in text
    empty = listen.render(listen.analyse(kit, key, {"items": {}}), {"run_id": "r"})
    assert "Nothing has been marked yet." in empty


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
