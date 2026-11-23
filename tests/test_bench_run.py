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
    ensemble_systems,
    latest_run,
    read_status,
    result_key,
    run_benchmark,
)
from stems_common.settings import Defaults, load_defaults  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_models.registry import (  # noqa: E402
    ModelEntry,
    Registry,
    load_registry,
)
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
        _entry("four_stem_b", ["vocals", "drums", "bass", "other"]),
        _entry("piano_only", ["piano"]),
        _entry("splitter_b", ["lead_vocals", "backing_vocals"]),
        _entry("dereverb_b", ["dry"]),
        _entry("a_tagger", [], kind="tagger"),
    ],
)


# Every fake model returns its input times a gain; these differ so blends can be told apart.
GAINS = {"four_stem": 0.3, "six_stem": 0.2, "four_stem_b": 0.4, "splitter_b": 0.1}


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
        gain = GAINS.get(model_id, 0.5)
        return [Scaler(model_id, dict.fromkeys(REGISTRY.get(model_id).stems, gain), self)]

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


def test_a_trial_does_the_same_jobs_again_under_its_own_keys(world, dataset, tmp_path) -> None:
    first = run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-01",))
    world.events.clear()
    again = run(
        world, dataset, tmp_path, ["two_stem"], run_id="r2", track_ids=("synth-01",), trial=1
    )
    assert again.count("computed") == 1
    assert world.runs() == [("two_stem", "synth-01")]
    a, b = _result(first, "two_stem", "synth-01"), _result(again, "two_stem", "synth-01")
    assert a["key"] != b["key"]
    assert (a.get("trial"), b["trial"]) == (None, 1)
    assert a["outputs"]["vocals"]["bss"]["sdr"] == b["outputs"]["vocals"]["bss"]["sdr"]
    assert json.loads((tmp_path / "r2" / "run.json").read_text())["trial"] == 1
    # The first run's keys are what they were before trials existed.
    assert result_key(**KEY_FIELDS) == result_key(**KEY_FIELDS, trial=0)
    assert result_key(**KEY_FIELDS) != result_key(**KEY_FIELDS, trial=1)
    # A run id belongs to one trial.
    with pytest.raises(BenchError, match="already exists with a different trial"):
        run(world, dataset, tmp_path, ["two_stem"], run_id="r2", track_ids=("synth-01",))
    with pytest.raises(BenchError, match="already exists with a different trial"):
        run(world, dataset, tmp_path, ["two_stem"], track_ids=("synth-01",), trial=2)


KEY_FIELDS = {
    "dataset": "d",
    "frozen_sha256": "f",
    "track_id": "t",
    "model_id": "m",
    "fingerprint": "w",
    "device": "cpu",
    "overlap": 0.25,
}


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
    assert (summary["computed"], summary["failed"]) == (11, 1)


# --- blends -------------------------------------------------------------------------------

SLOTS = {
    "vocals": ["two_stem", "four_stem", "six_stem"],
    "drums": ["four_stem", "six_stem", "per_stem_bag"],
    "bass": ["four_stem", "six_stem", "per_stem_bag"],
    "guitar": ["six_stem", "guitar_only"],
    "piano": ["six_stem", "piano_only"],
    "other": ["four_stem", "four_stem_b"],
    "vocal_split": ["splitter", "splitter_b"],
    "dereverb": ["dereverb", "dereverb_b"],
}


def fake_defaults(**changes) -> Defaults:
    raw = load_defaults().model_dump()
    raw["ensembles"]["slots"] = {
        slot: {"single_best": ids[0], "two_model_blend": ids[:2], "full_ensemble": ids}
        for slot, ids in SLOTS.items()
    }
    for slot, fields in changes.items():
        raw["ensembles"]["slots"][slot].update(fields)
    return Defaults.model_validate(raw)


def systems(*slots, **changes):
    return tuple(ensemble_systems(fake_defaults(**changes), REGISTRY, slots or None))


def test_the_real_config_gives_three_modes_per_slot_and_a_composed_other() -> None:
    built = ensemble_systems(load_defaults(), load_registry())
    by_id = {system.id: system for system in built}
    assert len(built) == 8 * 3 + 3
    assert by_id["vocals:single_best"].members == ("mel_roformer_kim_vocals",)
    assert by_id["vocals:two_model_blend"].weights == (0.5, 0.5)
    assert by_id["vocals:full_ensemble"].method == "waveform"
    assert by_id["vocal_split:single_best"].outputs == ("lead_vocals", "backing_vocals")
    assert by_id["vocal_split:single_best"].input == "vocals"
    assert by_id["dereverb:two_model_blend"].input == "vocals_wet"
    # The config's "other" models are four-stem ones...
    assert by_id["other:full_ensemble"].targets == ("other_4stem",)
    # ...so the product's six-stem other is that blend minus the same mode's guitar and piano.
    composed = by_id["other_composed:two_model_blend"]
    assert composed.targets == ("other",)
    assert [sub.id for sub in composed.minus] == ["guitar:two_model_blend", "piano:two_model_blend"]
    assert set(composed.all_members) == {
        "scnet_xl_ihf",
        "scnet_large",
        "bs_roformer_sw_6stem",
        "mel_roformer_guitar_becruily",
        "htdemucs_6s",
    }
    assert composed.needed()["bs_roformer_sw_6stem"] == {"guitar", "piano"}
    assert composed.needed()["scnet_xl_ihf"] == {"other"}


def test_inconsistent_ensembles_are_refused() -> None:
    with pytest.raises(BenchError, match="guitar: two_stem does not emit"):
        systems(guitar={"single_best": "two_stem"})
    mixed = {"two_model_blend": ["four_stem", "six_stem"]}
    with pytest.raises(BenchError, match="mixes four-stem and six-stem 'other' models"):
        systems(other=mixed)


def test_blends_are_scored_from_their_members_kept_audio(world, dataset, tmp_path) -> None:
    summary = run(world, dataset, tmp_path, [], track_ids=("synth-01",), ensembles=systems())
    assert summary.ok
    assert summary.count("computed") == 11 + 27  # every member model, then every system
    # Each member ran once; nothing else did.
    assert sorted(m for m, _ in world.runs()) == sorted(
        [m for m in {x for ids in SLOTS.values() for x in ids} if m != "per_stem_bag"]
        + ["per_stem_bag"] * 2
    )
    track = dataset.load("synth-01")

    single = _result(summary, "vocals:single_best", "synth-01")
    assert single["kind"] == "ensemble"
    assert (single["slot"], single["mode"], single["members"]) == (
        "vocals",
        "single_best",
        ["two_stem"],
    )
    # The single best as a system scores exactly as that model does on its own.
    own = _result(summary, "two_stem", "synth-01")
    assert own["kind"] == "model"
    assert single["outputs"]["vocals"]["bss"] == own["outputs"]["vocals"]["bss"]

    blend = _result(summary, "vocals:two_model_blend", "synth-01")
    assert (blend["members"], blend["weights"], blend["method"]) == (
        ["two_stem", "four_stem"],
        [0.5, 0.5],
        "waveform",
    )
    expected = _plain_sdr(track.references["vocals"], 0.4 * track.mix, track.mix, 44100)
    assert blend["outputs"]["vocals"]["bss"]["sdr"] == pytest.approx(expected.sdr, abs=1e-3)
    full = _result(summary, "vocals:full_ensemble", "synth-01")
    expected = _plain_sdr(track.references["vocals"], track.mix / 3, track.mix, 44100)
    assert full["outputs"]["vocals"]["bss"]["sdr"] == pytest.approx(expected.sdr, abs=1e-3)

    # A blend costs the time of all its members.
    members = [_result(summary, m, "synth-01") for m in blend["members"]]
    total = sum(m["wall_seconds"] for m in members) + blend["blend_seconds"]
    assert blend["wall_seconds"] == pytest.approx(total, abs=1e-3)
    assert blend["device"] == "cpu"

    # Two outputs for the vocal split, fed the vocals stem.
    split = _result(summary, "vocal_split:two_model_blend", "synth-01")
    assert set(split["outputs"]) == {"lead_vocals", "backing_vocals"}
    assert split["input"] == "vocals"
    expected = _plain_sdr(
        track.references["lead_vocals"], 0.3 * track.inputs["vocals"], track.inputs["vocals"], 44100
    )
    assert split["outputs"]["lead_vocals"]["bss"]["sdr"] == pytest.approx(expected.sdr, abs=1e-3)


def test_composed_other_is_the_four_stem_blend_minus_guitar_and_piano(
    world, dataset, tmp_path
) -> None:
    chosen = systems("other", "other_composed", "guitar", "piano")
    summary = run(world, dataset, tmp_path, [], track_ids=("synth-01",), ensembles=chosen)
    track = dataset.load("synth-01")
    four = _result(summary, "other:two_model_blend", "synth-01")
    assert four["outputs"]["other"]["target"] == "other_4stem"
    composed = _result(summary, "other_composed:two_model_blend", "synth-01")
    assert composed["outputs"]["other"]["target"] == "other"
    assert composed["minus"] == [
        {"slot": "guitar", "members": ["six_stem", "guitar_only"], "weights": [0.5, 0.5]},
        {"slot": "piano", "members": ["six_stem", "piano_only"], "weights": [0.5, 0.5]},
    ]
    assert composed["all_members"] == [
        "four_stem",
        "four_stem_b",
        "six_stem",
        "guitar_only",
        "piano_only",
    ]
    # (0.3 + 0.4) / 2 of the mix, minus (0.2 + 0.5) / 2 for guitar and the same for piano.
    gain = 0.35 - 0.35 - 0.35
    expected = _plain_sdr(track.references["other"], gain * track.mix, track.mix, 44100)
    assert composed["outputs"]["other"]["bss"]["sdr"] == pytest.approx(expected.sdr, abs=1e-3)
    walls = [_result(summary, m, "synth-01")["wall_seconds"] for m in composed["all_members"]]
    assert composed["wall_seconds"] >= sum(walls)


def test_only_what_blends_read_is_kept_on_disk(world, dataset, tmp_path) -> None:
    run(world, dataset, tmp_path, [], track_ids=("synth-01",), ensembles=systems())
    kept: dict[str, set[str]] = {}
    for path in (tmp_path / "estimates" / "synthetic").rglob("*.npy"):
        kept.setdefault(path.parts[-3], set()).add(path.stem)
    assert kept["two_stem"] == {"vocals"}  # its instrumental is scored but not kept
    assert kept["six_stem"] == {"vocals", "drums", "bass", "guitar", "piano"}  # not its other
    assert kept["four_stem"] == {"vocals", "drums", "bass", "other"}
    assert kept["splitter"] == {"lead_vocals", "backing_vocals"}


def test_blends_resume_and_reuse_like_any_other_job(world, dataset, tmp_path) -> None:
    chosen = systems("vocals", "drums")
    run(world, dataset, tmp_path, [], ensembles=chosen)
    world.events.clear()
    again = run(world, dataset, tmp_path, [], ensembles=chosen)
    assert world.events == []
    assert again.count("cached") == len(again.jobs) == (4 + 6) * 3
    # Other weights are another blend: only the systems that use them are scored again.
    weighted = systems("vocals", "drums", vocals={"two_model_weights": [3, 1]})
    third = run(world, dataset, tmp_path, [], run_id="r2", ensembles=weighted)
    assert world.events == []
    redone = [(j.model_id, j.track_id) for j in third.jobs if j.outcome == "computed"]
    assert redone == [("vocals:two_model_blend", t) for t in TRACKS]
    result = _result(third, "vocals:two_model_blend", "synth-01")
    assert result["weights"] == [0.75, 0.25]
    track = dataset.load("synth-01")
    expected = _plain_sdr(track.references["vocals"], 0.45 * track.mix, track.mix, 44100)
    assert result["outputs"]["vocals"]["bss"]["sdr"] == pytest.approx(expected.sdr, abs=1e-3)


def test_models_scored_earlier_are_separated_again_but_keep_their_scores(
    world, dataset, tmp_path
) -> None:
    first = run(world, dataset, tmp_path, ["two_stem", "four_stem"], track_ids=("synth-01",))
    before = _result(first, "two_stem", "synth-01")
    world.events.clear()
    chosen = tuple(s for s in systems("vocals") if s.mode != "full_ensemble")
    second = run(world, dataset, tmp_path, [], track_ids=("synth-01",), ensembles=chosen)
    assert second.ok
    assert world.runs() == [("two_stem", "synth-01"), ("four_stem", "synth-01")]
    jobs = {j.model_id: j for j in second.jobs}
    assert jobs["two_stem"].outcome == "cached"
    assert jobs["two_stem"].detail == "scores kept; audio separated again for the blends"
    assert _result(second, "two_stem", "synth-01") == before  # the result file is untouched
    assert jobs["vocals:two_model_blend"].outcome == "computed"


def test_a_blend_with_a_failed_member_fails_and_says_why(world, dataset, tmp_path) -> None:
    world.unloadable = {"four_stem"}
    summary = run(
        world, dataset, tmp_path, [], track_ids=("synth-01",), ensembles=systems("vocals")
    )
    outcomes = {j.model_id: (j.outcome, j.detail) for j in summary.jobs}
    assert outcomes["vocals:single_best"][0] == "computed"  # two_stem alone
    assert outcomes["vocals:two_model_blend"] == (
        "failed",
        "BenchError: four_stem has no separated audio for this track "
        "(it failed, or was not part of this run)",
    )
    assert outcomes["vocals:full_ensemble"][0] == "failed"
    assert not summary.ok


def test_blends_a_dataset_cannot_score_are_not_applicable(world, frozen_path, tmp_path) -> None:
    mix_only = MixOnlyDataset(frozen_path)
    chosen = systems("vocals", "guitar", "vocal_split", "dereverb", "other", "other_composed")
    summary = run(world, mix_only, tmp_path, [], track_ids=("synth-01",), ensembles=chosen)
    outcomes = {j.model_id: (j.outcome, j.detail) for j in summary.jobs if ":" in j.model_id}
    assert outcomes["vocals:full_ensemble"][0] == "computed"
    assert outcomes["other:two_model_blend"][0] == "computed"  # four-stem other is there
    assert outcomes["guitar:single_best"] == ("not_applicable", "synthetic has no guitar reference")
    assert outcomes["other_composed:single_best"] == (
        "not_applicable",
        "synthetic has no other reference",
    )
    assert outcomes["dereverb:single_best"][1].startswith("needs the 'vocals_wet' input")
    assert outcomes["vocal_split:single_best"][1].startswith("needs the 'vocals' input")


def test_audio_is_not_kept_for_blends_the_dataset_cannot_score(
    world, frozen_path, tmp_path
) -> None:
    mix_only = MixOnlyDataset(frozen_path)  # no guitar, piano or six-stem other references
    run(world, mix_only, tmp_path, [], track_ids=("synth-01",), ensembles=systems())
    kept: dict[str, set[str]] = {}
    for path in (tmp_path / "estimates" / "synthetic").rglob("*.npy"):
        kept.setdefault(path.parts[-3], set()).add(path.stem)
    assert kept["six_stem"] == {"vocals", "drums", "bass"}  # not its guitar or piano
    assert kept["four_stem"] == {"vocals", "drums", "bass", "other"}
    assert "guitar_only" not in kept
    assert "splitter" not in kept


def test_command_with_ensembles_adds_their_members(world, dataset, tmp_path, capsys) -> None:
    common = {"factory": world.factory, "registry": REGISTRY, "dataset": dataset}
    common |= {"bench_dir": tmp_path, "defaults": fake_defaults()}
    argv = ["run", "--dataset", "synthetic", "--ensembles", "--device", "cpu", "--quiet"]
    assert cli.main([*argv, "--tracks", "synth-01", "--run-id", "e1"], **common) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["computed"], summary["failed"]) == (11 + 27, 0)
    run_file = json.loads((tmp_path / "e1" / "run.json").read_text())
    assert len(run_file["ensembles"]) == 27
    assert "two_stem" in run_file["models"]
    assert cli.main(["run", "--dataset", "synthetic"], **common) == 2
    assert "give --models, --all or --ensembles" in capsys.readouterr().err
    # The report has the comparison section.
    assert cli.main(["report", "e1"], bench_dir=tmp_path) == 0
    text = (tmp_path / "e1" / "report.md").read_text()
    assert "## Single best, two-model blend, full ensemble" in text
    assert "### other_composed" in text
    assert "| two-model blend | vocals | vocals |" in text
    # Blends are compared in their own section, not mixed into the per-model tables.
    vocals_table = text[text.index("## vocals") : text.index("## drums")]
    assert "vocals:two_model_blend" not in vocals_table
