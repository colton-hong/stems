"""The smoke test runs every separator, survives failures, and reports both."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_common.synthetic import generate_track  # noqa: E402
from stems_models.registry import ModelEntry, Registry, load_registry  # noqa: E402
from stems_worker import smoke  # noqa: E402
from stems_worker.fakes import OracleSeparator  # noqa: E402
from stems_worker.smoke import FIXTURE_SEED, render_report, run_smoke  # noqa: E402
from stems_worker.timing import TimingLog, read_timings  # noqa: E402


def _entry(model_id, stems, kind="separator", status="available") -> ModelEntry:
    return ModelEntry.model_construct(
        id=model_id, stems=stems, status=status, kind=kind, loader="msst", files=[]
    )


REGISTRY = Registry.model_construct(
    schema_version=1,
    models=[
        _entry("good6", list(PRODUCT_STEMS)),
        _entry("broken", ["vocals", "instrumental"]),
        _entry("wrong_outputs", ["vocals", "drums"]),
        _entry("kara", ["lead_vocals", "backing_vocals"]),
        _entry("dry", ["dry"]),
        _entry("a_tagger", [], kind="tagger"),
        _entry("paper_only", ["vocals"], status="unavailable"),
    ],
)
SECONDS = 2.0


def factory(model_id, stems):
    track = generate_track(seed=FIXTURE_SEED, duration_seconds=SECONDS)
    if model_id == "good6":
        return [OracleSeparator(dict(track.stems), model_id=model_id)]
    if model_id == "broken":
        raise RuntimeError("weights are corrupt")
    if model_id == "wrong_outputs":
        return [OracleSeparator({"vocals": track.stems["vocals"]}, model_id=model_id)]
    if model_id == "kara":
        parts = {"lead_vocals": track.lead_vocals, "backing_vocals": track.backing_vocals}
        return [OracleSeparator(parts, model_id=model_id)]
    return [OracleSeparator({"dry": track.stems["vocals"]}, model_id=model_id)]


@pytest.fixture(scope="module")
def results():
    return {r.model_id: r for r in run_smoke(REGISTRY, factory, "cpu", seconds=SECONDS)}


def test_every_loadable_separator_is_tried_and_nothing_else(results) -> None:
    assert list(results) == ["good6", "broken", "wrong_outputs", "kara", "dry"]


def test_a_working_model_is_reported_with_outputs_times_and_routing(results) -> None:
    good = results["good6"]
    assert good.ok is True
    assert good.error is None
    assert good.outputs == list(PRODUCT_STEMS)
    assert (good.device, good.input, good.audio_seconds) == ("cpu", "mix", SECONDS)
    assert good.run_seconds >= 0
    assert good.load_seconds >= 0
    # A perfect model puts each fixture stem wholly in its own output and nowhere else.
    for output in PRODUCT_STEMS:
        for stem in PRODUCT_STEMS:
            expected = 1.0 if stem == output else 0.0
            assert good.routing[output][stem] == pytest.approx(expected, abs=0.01)


def test_failures_are_recorded_and_do_not_stop_the_run(results) -> None:
    broken = results["broken"]
    assert broken.ok is False
    assert broken.error.startswith("RuntimeError: weights are corrupt")
    wrong = results["wrong_outputs"]
    assert wrong.ok is False
    assert "registry lists ['drums', 'vocals']" in wrong.error
    assert results["kara"].ok is True  # the ones after a failure still ran


def test_vocal_stage_models_are_fed_the_vocals_stem(results) -> None:
    kara = results["kara"]
    assert kara.input == "vocals stem"
    assert kara.routing["lead_vocals"] == {"lead": 1.0, "backing": 0.0}
    assert kara.routing["backing_vocals"] == {"lead": 0.0, "backing": 1.0}
    dry = results["dry"]
    assert dry.input == "vocals stem with reverb"
    assert dry.routing["dry"] == {"dry": 1.0, "reverb": 0.0}


def test_report_lists_every_model_with_its_result(results) -> None:
    header = {
        "command": "python -m stems_worker.smoke",
        "at": "now",
        "device": "cpu",
        "seed": 1,
        "seconds": 2.0,
    }
    report = render_report(list(results.values()), header)
    assert "**3 of 5 models loaded and ran**, 2 failed" in report
    assert "| `good6` | ran | mix | cpu |" in report
    assert "| `broken` | **FAILED** | mix | | | | RuntimeError: weights are corrupt" in report
    assert "### `kara` (vocals stem)" in report
    assert "| lead_vocals | +1.00 |" in report
    assert "python -m stems_worker.smoke" in report
    assert "not how fast it is on real songs" in report


def test_command_writes_the_report_and_fails_if_any_model_failed(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(smoke, "RUNS_DIR", tmp_path / "runs")
    argv = ["--device", "cpu", "--seconds", "2", "--out", str(tmp_path / "smoke")]
    code = smoke.main(argv, factory=factory, registry=REGISTRY)
    assert code == 1
    assert "3 of 5 ran" in capsys.readouterr().out
    report = (tmp_path / "smoke" / "report.md").read_text()
    assert "--device cpu --seconds 2" in report
    data = json.loads((tmp_path / "smoke" / "report.json").read_text())
    assert [r["model_id"] for r in data["results"] if not r["ok"]] == ["broken", "wrong_outputs"]
    # Real runs are logged as smoke runs so they can be told apart later.
    lines = read_timings(tmp_path / "runs" / "timings.jsonl")
    assert {line["stage"] for line in lines} == {"smoke"}
    assert {line["source"] for line in lines} == {"smoke"}

    code = smoke.main([*argv, "good6", "kara", "--no-timings"], factory=factory, registry=REGISTRY)
    assert code == 0
    assert smoke.main([*argv, "a_tagger"], factory=factory, registry=REGISTRY) == 2


def test_shares_are_not_fooled_by_correlated_signals() -> None:
    track = generate_track(seed=FIXTURE_SEED, duration_seconds=SECONDS)
    lead, backing = track.lead_vocals, track.backing_vocals
    # In this short clip the two voices overlap heavily: taken one at a time, most of the
    # backing's energy would be credited to the lead.
    overlap = float(
        np.dot(lead.ravel(), backing.ravel()) / np.dot(backing.ravel(), backing.ravel())
    )
    assert overlap > 0.5
    references = {"lead": lead, "backing": backing}
    assert smoke.shares(lead, references) == {"lead": 1.0, "backing": 0.0}
    assert smoke.shares(0.25 * lead + 0.5 * backing, references) == {"lead": 0.25, "backing": 0.5}
    silent = {"lead": lead, "backing": np.zeros_like(lead)}
    assert smoke.shares(lead, silent) == {"lead": 1.0, "backing": 0.0}


def test_timings_are_optional_in_the_library_call(tmp_path) -> None:
    log = TimingLog(tmp_path / "t.jsonl")
    run_smoke(REGISTRY, factory, "cpu", model_ids=["good6"], seconds=SECONDS, timings=log)
    assert [line["model_id"] for line in read_timings(log.path)] == ["good6"]


def test_the_real_registry_has_separators_to_smoke() -> None:
    separators = [m for m in load_registry().models if m.kind == "separator" and m.is_loadable]
    assert len(separators) == 43
