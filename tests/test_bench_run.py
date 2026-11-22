"""The benchmark run manager: cached per job, resumable, and honest about failures."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("scipy")

from stems_bench import cli  # noqa: E402
from stems_bench import run as run_module  # noqa: E402
from stems_bench.datasets import DatasetError, synthetic, write_frozen  # noqa: E402
from stems_bench.datasets.synthetic import SyntheticDataset  # noqa: E402
from stems_bench.metrics import BssScores, bss_eval_stem, fit_length  # noqa: E402
from stems_bench.run import (  # noqa: E402
    BenchError,
    RunSpec,
    applicability,
    latest_run,
    read_status,
    result_key,
    run_benchmark,
)
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_models.registry import ModelEntry, Registry  # noqa: E402
from stems_worker.cache import audio_sha256  # noqa: E402
from stems_worker.separator import Separator  # noqa: E402
from stems_worker.timing import TimingLog, read_timings  # noqa: E402

TRACKS = ["synth-01", "synth-07", "synth-08"]  # full, no guitar or piano, no vocals


class ReadyDataset(SyntheticDataset):
    """A synthetic dataset frozen in a temporary folder, where it cannot be committed."""

    def check_ready(self) -> None:
        pass


class MixOnlyDataset(ReadyDataset):
    name = "synthetic"
    inputs = ("mix",)
    targets = ("vocals", "drums", "bass", "other_4stem")

    def load(self, track_id):
        track = super().load(track_id)
        refs = {k: v for k, v in track.references.items() if k in self.targets}
        return type(track)(track.track_id, track.sample_rate, {"mix": track.mix}, refs, track.meta)


@pytest.fixture(scope="module")
def frozen_path(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setattr(synthetic, "SPEC", tuple(s for s in synthetic.SPEC if s["id"] in TRACKS))
    patch.setattr(synthetic, "DURATION_SECONDS", 2.0)
    data = synthetic.build_frozen()
    patch.undo()
    path = tmp_path_factory.mktemp("frozen") / "synthetic.json"
    write_frozen(path, data)
    return path


@pytest.fixture(scope="module")
def dataset(frozen_path):
    return ReadyDataset(frozen_path)


@pytest.fixture(scope="module")
def track_of(dataset):
    """Which track a piece of input audio is, so fakes can log and misbehave by track."""
    table = {}
    for track_id in TRACKS:
        track = dataset.load(track_id)
        for audio in track.inputs.values():
            table[audio_sha256(audio, 44100)] = track_id
    return lambda audio: table[audio_sha256(np.ascontiguousarray(audio), 44100)]


def _entry(model_id, stems, kind="separator") -> ModelEntry:
    return ModelEntry.model_construct(
        id=model_id,
        stems=list(stems),
        status="available",
        kind=kind,
        loader="msst",
        files=[],
        config={},
    )


REGISTRY = Registry.model_construct(
    schema_version=1,
    models=[
        _entry("two_stem", ["vocals", "instrumental"]),
        _entry("four_stem", ["vocals", "drums", "bass", "other"]),
        _entry("six_stem", list(PRODUCT_STEMS)),
        _entry("per_stem_bag", ["drums", "bass"]),
        _entry("splitter", ["lead_vocals", "backing_vocals"]),
        _entry("dereverb", ["dry"]),
        _entry("guitar_only", ["guitar"]),
        _entry("unbuildable", ["vocals"]),
        _entry("a_tagger", [], kind="tagger"),
    ],
)


class Scaler(Separator):
    """Every output is the input times a gain. Logs loads and runs, and can misbehave."""

    chunk_size = 44100

    def __init__(self, model_id, gains, world):
        self.model_id = model_id
        self.outputs = tuple(gains)
        self._gains = np.array(list(gains.values()), dtype=np.float32)
        self._world = world

    def load(self, device):
        if self.model_id in self._world.unloadable:
            raise RuntimeError("weights are corrupt")
        self._world.events.append(("load", self.model_id))
        self.device = device

    def prepare(self, audio):
        track_id = self._world.track_of(audio)
        self._world.events.append(("run", self.model_id, track_id))
        self._world.on_run(self.model_id, track_id)
        return audio, None

    def separate_chunks(self, batch):
        return self._gains[None, :, None, None] * batch[:, None]

    def unload(self):
        self._world.events.append(("unload", self.model_id))
        super().unload()


class World:
    """The fake models, and a record of everything they were asked to do."""

    def __init__(self, track_of):
        self.track_of = track_of
        self.events = []
        self.unloadable = set()
        self.on_run = lambda model_id, track_id: None
        self.fingerprints = {}

    def factory(self, model_id, stems):
        if model_id == "unbuildable":
            raise FileNotFoundError("checkpoint is missing")
        if model_id == "per_stem_bag":
            return [Scaler(model_id, {s: 0.5}, self) for s in stems]
        return [Scaler(model_id, dict.fromkeys(REGISTRY.get(model_id).stems, 0.5), self)]

    def fingerprint(self, model_id):
        return self.fingerprints.get(model_id, f"weights-of-{model_id}")

    def runs(self):
        return [e[1:] for e in self.events if e[0] == "run"]


@pytest.fixture
def world(track_of):
    return World(track_of)


def _plain_sdr(reference, estimate, mix, sample_rate, **_):
    """SDR by its definition, per 1 s window. Fast stand-in for museval in these tests."""
    estimate = fit_length(estimate, reference.shape[-1]).astype(np.float64)
    reference = reference.astype(np.float64)
    windows = reference.shape[-1] // sample_rate
    sdr = np.full(windows, np.nan)
    for w in range(windows):
        part = slice(w * sample_rate, (w + 1) * sample_rate)
        signal = float((reference[:, part] ** 2).sum())
        error = float(((estimate - reference)[:, part] ** 2).sum())
        if signal > 0 and error > 0:
            sdr[w] = 10 * np.log10(signal / error)
    scored = sdr[~np.isnan(sdr)]
    median = float(np.median(scored)) if scored.size else float("nan")
    frames = dict.fromkeys(("sdr", "sir", "sar", "isr"), sdr)
    return BssScores(median, median, median, median, windows, int(scored.size), 1.0, frames)


@pytest.fixture(autouse=True)
def quick_metric(monkeypatch):
    # museval takes about 0.3 s per output even on two seconds of audio. One test below
    # asks for the real thing; the rest are about bookkeeping.
    monkeypatch.setattr(run_module, "bss_eval_stem", _plain_sdr)


@pytest.fixture
def real_metric(monkeypatch, quick_metric):
    monkeypatch.setattr(run_module, "bss_eval_stem", bss_eval_stem)


def run(world, dataset, bench_dir, models, run_id="r1", **spec):
    spec = RunSpec(dataset="synthetic", model_ids=tuple(models), device="cpu", **spec)
    return run_benchmark(
        spec,
        dataset=dataset,
        registry=REGISTRY,
        factory=world.factory,
        fingerprint=world.fingerprint,
        run_id=run_id,
        bench_dir=bench_dir,
        command="bench run (test)",
    )


def _result(summary, model_id, track_id):
    job = next(j for j in summary.jobs if (j.model_id, j.track_id) == (model_id, track_id))
    return json.loads(open(job.result).read())


# --- results ------------------------------------------------------------------------------


def test_every_job_gets_a_result_file_with_scores_and_provenance(world, dataset, tmp_path) -> None:
    summary = run(world, dataset, tmp_path, ["two_stem", "four_stem"])
    assert summary.ok
    assert [(j.model_id, j.track_id, j.outcome) for j in summary.jobs] == [
        (m, t, "computed") for m in ("two_stem", "four_stem") for t in TRACKS
    ]
    result = _result(summary, "two_stem", "synth-01")
    assert result["protocol_version"] == 1
    assert result["dataset"] == "synthetic"
    assert result["frozen_sha256"] == dataset.frozen.sha256
    assert (result["model_id"], result["track_id"]) == ("two_stem", "synth-01")
    assert result["fingerprint"] == "weights-of-two_stem"
    assert (result["device"], result["device_requested"], result["overlap"]) == ("cpu", "cpu", 0.25)
    assert result["input"] == "mix"
    assert result["audio_seconds"] == 2.0
    assert result["wall_seconds"] >= 0
    assert result["num_chunks"] >= 2
    assert set(result["git"]) == {"commit", "dirty"}
    assert result["packages"]["museval"] == "0.4.1"
    assert set(result["outputs"]) == {"vocals", "instrumental"}
    vocals = result["outputs"]["vocals"]
    assert vocals["target"] == "vocals"
    assert vocals["reference_present"] is True
    assert set(vocals["bss"]) >= {"sdr", "sir", "sar", "windows", "scored_windows", "frames"}
    assert vocals["signal"]["duration_matches"] is True
    # Each model is loaded once for all its tracks.
    loads = [e[1] for e in world.events if e[0] == "load"]
    assert loads == ["two_stem", "four_stem"]


def test_scores_are_the_metric_of_what_the_model_returned(
    world, dataset, tmp_path, real_metric
) -> None:
    summary = run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-01",))
    track = dataset.load("synth-01")
    expected = bss_eval_stem(track.references["vocals"], 0.5 * track.mix, track.mix, 44100)
    got = _result(summary, "two_stem", "synth-01")["outputs"]["vocals"]
    assert got["bss"]["sdr"] == pytest.approx(expected.sdr)
    assert got["bss"]["frames"]["sdr"] == pytest.approx(expected.frames["sdr"], abs=1e-3)
    estimate_rms = np.sqrt(np.mean((0.5 * track.mix) ** 2))
    reference_rms = np.sqrt(np.mean(track.references["vocals"] ** 2))
    expected_delta = 20 * np.log10(estimate_rms / reference_rms)
    assert got["signal"]["rms_delta_db"] == pytest.approx(expected_delta, abs=0.01)


def test_other_is_scored_against_the_right_kind_of_other(world, dataset, tmp_path) -> None:
    summary = run(world, dataset, tmp_path, ["four_stem", "six_stem"], track_ids=("synth-01",))
    four = _result(summary, "four_stem", "synth-01")["outputs"]
    assert four["other"]["target"] == "other_4stem"
    assert "other_4stem" not in four
    six = _result(summary, "six_stem", "synth-01")["outputs"]
    assert six["other"]["target"] == "other"
    # A six-stem model also answers the four-stem question, as the sum of three outputs.
    assert six["other_4stem"] == {**six["other_4stem"], "target": "other_4stem", "derived": True}
    assert six["guitar"]["derived"] is False


def test_a_stem_absent_from_a_track_is_recorded_not_scored(world, dataset, tmp_path) -> None:
    summary = run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-08",))
    vocals = _result(summary, "two_stem", "synth-08")["outputs"]["vocals"]
    assert vocals["reference_present"] is False
    assert vocals["bss"]["sdr"] is None
    assert vocals["bss"]["scored_windows"] == 0
    assert vocals["signal"]["reference"]["silence_fraction"] == 1.0


def test_stage_models_are_fed_the_vocals_and_scored_on_them(world, dataset, tmp_path) -> None:
    summary = run(world, dataset, tmp_path, ["splitter", "dereverb"], track_ids=("synth-01",))
    assert _result(summary, "splitter", "synth-01")["input"] == "vocals"
    assert set(_result(summary, "splitter", "synth-01")["outputs"]) == {
        "lead_vocals",
        "backing_vocals",
    }
    dry = _result(summary, "dereverb", "synth-01")
    assert dry["input"] == "vocals_wet"
    # Half the wet vocals, scored against the dry vocals: about 6 dB.
    assert dry["outputs"]["dry"]["bss"]["sdr"] == pytest.approx(6.0, abs=0.5)


def test_models_a_dataset_cannot_score_are_not_run(world, frozen_path, tmp_path) -> None:
    mix_only = MixOnlyDataset(frozen_path)
    summary = run(
        world, mix_only, tmp_path, ["dereverb", "guitar_only", "six_stem"], track_ids=("synth-01",)
    )
    outcomes = {j.model_id: (j.outcome, j.detail) for j in summary.jobs}
    assert outcomes["dereverb"] == (
        "not_applicable",
        "needs the 'vocals_wet' input, which synthetic does not have",
    )
    assert outcomes["guitar_only"][0] == "not_applicable"
    assert "no reference for any of its outputs" in outcomes["guitar_only"][1]
    assert outcomes["six_stem"][0] == "computed"
    assert world.runs() == [("six_stem", "synth-01")]
    result = _result(summary, "six_stem", "synth-01")
    assert set(result["outputs"]) == {"vocals", "drums", "bass", "other_4stem"}
    assert result["unscored_outputs"]["guitar"] == "no 'guitar' reference in this dataset"
    assert result["unscored_outputs"]["other"] == "no 'other' reference in this dataset"
    assert summary.ok
    assert applicability(REGISTRY.get("six_stem"), mix_only) is None


def test_a_model_in_several_parts_gives_one_result(world, dataset, tmp_path) -> None:
    summary = run(world, dataset, tmp_path, ["per_stem_bag"], track_ids=("synth-01", "synth-07"))
    assert summary.count("computed") == 2
    result = _result(summary, "per_stem_bag", "synth-07")
    assert set(result["outputs"]) == {"drums", "bass"}
    assert world.runs() == [("per_stem_bag", t) for t in ("synth-01", "synth-07")] * 2
    assert [e[0] for e in world.events].count("load") == 2


# --- resuming -----------------------------------------------------------------------------


def test_a_second_run_does_no_work(world, dataset, tmp_path) -> None:
    run(world, dataset, tmp_path, ["two_stem", "four_stem"])
    done = list(world.events)
    again = run(world, dataset, tmp_path, ["two_stem", "four_stem"])
    assert world.events == done  # nothing was even loaded
    assert again.count("cached") == 6
    assert again.ok
    # Results belong to the job, not the run: a new run id reuses them too.
    other = run(world, dataset, tmp_path, ["two_stem", "four_stem", "six_stem"], run_id="r2")
    assert other.count("cached") == 6
    assert other.count("computed") == 3
    assert {m for m, _ in world.runs()[6:]} == {"six_stem"}


def test_a_killed_run_resumes_without_redoing_finished_jobs(world, dataset, tmp_path) -> None:
    def kill(model_id, track_id):
        if (model_id, track_id) == ("four_stem", "synth-07"):
            raise KeyboardInterrupt

    world.on_run = kill
    with pytest.raises(KeyboardInterrupt):
        run(world, dataset, tmp_path, ["two_stem", "four_stem", "six_stem"])
    status = read_status(tmp_path / "r1")
    assert status["state"] == "interrupted"
    assert (status["finished"], status["computed"], status["total"]) == (4, 4, 9)
    assert ("unload", "four_stem") in world.events  # the model was released on the way out
    lines = (tmp_path / "r1" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 4

    world.on_run = lambda model_id, track_id: None
    world.events.clear()
    summary = run(world, dataset, tmp_path, ["two_stem", "four_stem", "six_stem"])
    assert summary.ok
    assert (summary.count("cached"), summary.count("computed")) == (4, 5)
    assert world.runs() == [
        ("four_stem", "synth-07"),
        ("four_stem", "synth-08"),
        *(("six_stem", t) for t in TRACKS),
    ]
    assert read_status(tmp_path / "r1")["state"] == "done"
    assert len((tmp_path / "r1" / "results.jsonl").read_text().splitlines()) == 9


def test_status_file_follows_the_run(world, dataset, tmp_path) -> None:
    seen = []

    def look(model_id, track_id):
        seen.append(json.loads((tmp_path / "r1" / "status.json").read_text()))

    world.on_run = look
    run(world, dataset, tmp_path, ["two_stem", "four_stem"])
    assert [s["finished"] for s in seen] == [0, 1, 2, 3, 4, 5]
    assert all(s["state"] == "running" and s["total"] == 6 for s in seen)
    assert seen[4]["current"] == {"model_id": "four_stem", "track_id": "synth-07"}
    final = read_status(tmp_path / "r1")
    assert (final["state"], final["finished"], final["current"]) == ("done", 6, None)
    assert final["dataset"] == "synthetic"
    assert latest_run(tmp_path) == tmp_path / "r1"
    run_file = json.loads((tmp_path / "r1" / "run.json").read_text())
    assert run_file["models"] == ["two_stem", "four_stem"]
    assert run_file["tracks"] == TRACKS
    assert run_file["command"] == "bench run (test)"


def test_a_running_status_whose_process_is_gone_reads_as_dead(tmp_path) -> None:
    (tmp_path / "r").mkdir()
    status = {"state": "running", "pid": 2**22 + 12345}
    (tmp_path / "r" / "status.json").write_text(json.dumps(status))
    assert read_status(tmp_path / "r")["state"] == "dead"
    assert latest_run(tmp_path / "nothing") is None


def test_new_weights_or_settings_mean_new_results(world, dataset, tmp_path) -> None:
    run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-01",))
    world.fingerprints["two_stem"] = "retrained"
    summary = run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-01",))
    assert summary.count("computed") == 1
    assert _result(summary, "two_stem", "synth-01")["fingerprint"] == "retrained"
    summary = run(
        world, dataset, tmp_path, ["two_stem"], run_id="r2", track_ids=("synth-01",), overlap=0.5
    )
    assert summary.count("computed") == 1

    base = {
        "dataset": "d",
        "frozen_sha256": "f",
        "track_id": "t",
        "model_id": "m",
        "fingerprint": "w",
        "device": "cpu",
        "overlap": 0.25,
    }
    keys = {result_key(**base)}
    for field, value in [
        ("dataset", "e"),
        ("frozen_sha256", "g"),
        ("track_id", "u"),
        ("model_id", "n"),
        ("fingerprint", "x"),
        ("device", "mps"),
        ("overlap", 0.5),
    ]:
        keys.add(result_key(**{**base, field: value}))
    assert len(keys) == 8
    assert result_key(**base) == result_key(**base)


def test_a_run_id_cannot_be_reused_for_something_else(world, dataset, tmp_path) -> None:
    run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-01",))
    with pytest.raises(BenchError, match="already exists with a different overlap"):
        run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-01",), overlap=0.5)


# --- failures -----------------------------------------------------------------------------


def test_failures_are_recorded_and_retried_next_time(world, dataset, tmp_path) -> None:
    def trouble(model_id, track_id):
        if (model_id, track_id) == ("two_stem", "synth-07"):
            raise RuntimeError("out of memory")

    world.on_run = trouble
    world.unloadable = {"four_stem"}
    models = ["unbuildable", "two_stem", "four_stem", "six_stem"]
    summary = run(world, dataset, tmp_path, models)
    assert not summary.ok
    assert summary.state == "done"
    failures = {(j.model_id, j.track_id): j.detail for j in summary.jobs if j.outcome == "failed"}
    assert failures == {
        **{("unbuildable", t): "FileNotFoundError: checkpoint is missing" for t in TRACKS},
        ("two_stem", "synth-07"): "RuntimeError: out of memory",
        **{("four_stem", t): "RuntimeError: weights are corrupt" for t in TRACKS},
    }
    assert (summary.count("computed"), summary.count("failed")) == (5, 7)
    status = read_status(tmp_path / "r1")
    assert status["failed"] == 7
    oom = {"model_id": "two_stem", "track_id": "synth-07", "error": "RuntimeError: out of memory"}
    assert oom in status["failures"]

    world.on_run = lambda model_id, track_id: None
    world.unloadable = set()
    world.events.clear()
    again = run(world, dataset, tmp_path, models)
    assert (again.count("cached"), again.count("computed"), again.count("failed")) == (5, 4, 3)
    assert world.runs() == [("two_stem", "synth-07"), *(("four_stem", t) for t in TRACKS)]


def test_bad_requests_are_refused_before_any_work(world, dataset, frozen_path, tmp_path) -> None:
    with pytest.raises(DatasetError, match=r"not in synthetic\.json"):
        run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-99",))
    with pytest.raises(BenchError, match="a_tagger is not a loadable separator"):
        run(world, dataset, tmp_path, ["a_tagger"])
    with pytest.raises(KeyError, match="unknown model id"):
        run(world, dataset, tmp_path, ["nope"])
    # The real rule: a frozen list that is not committed cannot be benchmarked on.
    with pytest.raises(DatasetError, match="is not committed"):
        run(world, SyntheticDataset(frozen_path), tmp_path, ["two_stem"])
    assert world.events == []
    assert not (tmp_path / "r1").exists()


# --- kept estimates and timings -----------------------------------------------------------


def test_kept_estimates_are_reused_when_scores_must_be_redone(world, dataset, tmp_path) -> None:
    kept = {"track_ids": ("synth-01",), "keep_estimates": ("two_stem",)}
    first = run(world, dataset, tmp_path, ["two_stem", "four_stem"], **kept)
    assert _result(first, "two_stem", "synth-01")["estimates_from_cache"] is False
    stored = list((tmp_path / "estimates").rglob("*.npy"))
    assert sorted(p.name for p in stored) == ["instrumental.npy", "vocals.npy"]  # two_stem only
    for path in (tmp_path / "results").rglob("*.json"):
        path.unlink()
    world.events.clear()
    second = run(world, dataset, tmp_path, ["two_stem", "four_stem"], **kept)
    assert world.runs() == [("four_stem", "synth-01")]  # two_stem came from its kept audio
    result = _result(second, "two_stem", "synth-01")
    assert result["estimates_from_cache"] is True
    assert result["outputs"] == _result(first, "two_stem", "synth-01")["outputs"]


def test_real_runs_are_logged_as_bench_timings(world, dataset, tmp_path) -> None:
    spec = RunSpec("synthetic", ("two_stem",), "cpu", track_ids=("synth-01", "synth-07"))
    log = TimingLog(tmp_path / "timings.jsonl", {"source": "bench", "run_id": "r1"})
    kwargs = {
        "dataset": dataset,
        "registry": REGISTRY,
        "factory": world.factory,
        "fingerprint": world.fingerprint,
        "run_id": "r1",
        "bench_dir": tmp_path / "bench",
        "timings": log,
    }
    run_benchmark(spec, **kwargs)
    run_benchmark(spec, **kwargs)  # all cached: nothing more is logged
    lines = read_timings(log.path)
    assert [(line["model_id"], line["stage"], line["run_id"]) for line in lines] == [
        ("two_stem", "bench", "r1")
    ] * 2


# --- the command --------------------------------------------------------------------------


def test_command_runs_reports_and_shows_status(world, dataset, tmp_path, capsys) -> None:
    bench_dir = tmp_path / "bench"
    common = {"factory": world.factory, "registry": REGISTRY, "dataset": dataset}
    common["bench_dir"] = bench_dir
    argv = ["run", "--dataset", "synthetic", "--models", "two_stem,six_stem", "--device", "cpu"]
    assert cli.main([*argv, "--run-id", "mine", "--tracks", "synth-01"], **common) == 0
    out = capsys.readouterr().out
    assert "two_stem / synth-01: computed" in out
    assert json.loads(out.splitlines()[-1]) == {
        "run_id": "mine",
        "computed": 2,
        "already_done": 0,
        "failed": 0,
        "not_applicable": 0,
        "run_dir": str(bench_dir / "mine"),
    }
    assert cli.main(["status"], bench_dir=bench_dir) == 0
    out = capsys.readouterr().out
    assert "run mine: done" in out
    assert "2 of 2 jobs: 2 computed, 0 already done, 0 failed, 0 not applicable" in out
    assert cli.main(["status", "mine"], bench_dir=bench_dir) == 0
    assert cli.main(["status", "absent"], bench_dir=bench_dir) == 2
    capsys.readouterr()
    # Without --run-id a run is named after the dataset and the time.
    assert cli.main([*argv, "--tracks", "synth-01", "--quiet"], **common) == 0
    assert json.loads(capsys.readouterr().out)["run_id"].startswith("synthetic-20")
    # Failures and bad requests have their own exit codes.
    world.unloadable = {"four_stem"}
    failing = ["run", "--dataset", "synthetic", "--models", "four_stem", "--device", "cpu"]
    assert cli.main([*failing, "--tracks", "synth-01", "--run-id", "bad", "--quiet"], **common) == 1
    assert cli.main(["status", "bad"], bench_dir=bench_dir) == 1
    shown = capsys.readouterr().out
    assert "FAILED four_stem / synth-01: RuntimeError: weights are corrupt" in shown
    assert cli.main(["run", "--dataset", "synthetic", "--models", "nope"], **common) == 2
    assert "not in the registry" in capsys.readouterr().err


def test_command_all_means_every_loadable_separator(world, dataset, tmp_path, capsys) -> None:
    common = {"factory": world.factory, "registry": REGISTRY, "dataset": dataset}
    argv = ["run", "--dataset", "synthetic", "--all", "--device", "cpu", "--quiet"]
    code = cli.main([*argv, "--tracks", "synth-01"], bench_dir=tmp_path, **common)
    assert code == 1  # "unbuildable" fails; everything else runs
    summary = json.loads(capsys.readouterr().out)
    assert (summary["computed"], summary["failed"]) == (7, 1)
