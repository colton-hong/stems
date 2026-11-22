"""Tuning blends: only on tracks outside the test lists, and reproducibly."""

from __future__ import annotations

import json
import statistics
from typing import ClassVar

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from stems_bench import cli, tune  # noqa: E402
from stems_bench import run as run_module  # noqa: E402
from stems_bench.datasets import synthetic, write_frozen  # noqa: E402
from stems_bench.datasets.synthetic import SyntheticDataset  # noqa: E402
from stems_bench.metrics import BssScores, median_sdr, sdr_frames  # noqa: E402
from stems_bench.run import BenchError  # noqa: E402
from stems_common.settings import Defaults, load_defaults  # noqa: E402
from stems_common.stems import PRODUCT_STEMS  # noqa: E402
from stems_models.registry import ModelEntry, Registry  # noqa: E402
from stems_worker.separator import Separator  # noqa: E402

# model id -> (outputs, gain): every fake returns its input times the gain.
MODELS = {
    "v1": (["vocals", "instrumental"], 0.9),
    "v2": (["vocals", "drums", "bass", "other"], 0.1),
    "s1": (list(PRODUCT_STEMS), 0.3),
    "s2": (list(PRODUCT_STEMS), 0.2),
    "o1": (["vocals", "drums", "bass", "other"], 0.5),
    "k1": (["lead_vocals", "backing_vocals"], 0.8),
    "k2": (["lead_vocals", "backing_vocals"], 0.2),
    "r1": (["dry"], 1.0),
    "r2": (["dry"], 0.6),
}
REGISTRY = Registry.model_construct(
    schema_version=1,
    models=[
        ModelEntry.model_construct(
            id=model_id,
            stems=outputs,
            status="available",
            kind="separator",
            loader="msst",
            files=[],
            config={},
        )
        for model_id, (outputs, _) in MODELS.items()
    ],
)
SLOTS = {
    "vocals": ["v1", "v2", "s1"],
    "drums": ["s1", "s2"],
    "bass": ["s1", "s2"],
    "guitar": ["s1", "s2"],
    "piano": ["s1", "s2"],
    "other": ["v2", "o1"],
    "vocal_split": ["k1", "k2"],
    "dereverb": ["r1", "r2"],
}


def fake_defaults() -> Defaults:
    raw = load_defaults().model_dump()
    raw["ensembles"]["slots"] = {
        slot: {"single_best": ids[0], "two_model_blend": ids[:2], "full_ensemble": ids}
        for slot, ids in SLOTS.items()
    }
    return Defaults.model_validate(raw)


class Gain(Separator):
    chunk_size = 44100
    broken: ClassVar[set] = set()
    loads: ClassVar[list] = []

    def __init__(self, model_id):
        self.model_id = model_id
        self.outputs = tuple(MODELS[model_id][0])

    def load(self, device):
        if self.model_id in Gain.broken:
            raise RuntimeError("weights are corrupt")
        Gain.loads.append(self.model_id)
        self.device = device

    def separate_chunks(self, batch):
        gain = np.float32(MODELS[self.model_id][1])
        return np.repeat((gain * batch)[:, None], len(self.outputs), axis=1)


class Ready(SyntheticDataset):
    def check_ready(self) -> None:
        pass


def _frozen(tmp_path_factory, name, spec):
    patch = pytest.MonkeyPatch()
    attribute = "SPEC" if name == "synthetic" else "TUNING_SPEC"
    patch.setattr(synthetic, attribute, tuple(spec))
    patch.setattr(synthetic, "DURATION_SECONDS", 2.0)
    data = synthetic.build_frozen(name)
    patch.undo()
    path = tmp_path_factory.mktemp(name) / f"{name}.json"
    write_frozen(path, data)
    return Ready(path, name)


@pytest.fixture(scope="module")
def tuning_set(tmp_path_factory):
    return _frozen(tmp_path_factory, "synthetic_tuning", synthetic.TUNING_SPEC[:2])


@pytest.fixture(scope="module")
def test_set(tmp_path_factory):
    return _frozen(tmp_path_factory, "synthetic", synthetic.SPEC[:2])


def _quick_scores(reference, estimate, mix, sample_rate, **_):
    frames = sdr_frames(reference, estimate, sample_rate)
    scored = frames[~np.isnan(frames)]
    median = float(np.median(scored)) if scored.size else float("nan")
    every = dict.fromkeys(("sdr", "sir", "sar", "isr"), frames)
    return BssScores(median, median, median, median, len(frames), int(scored.size), 1.0, every)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    Gain.broken = set()
    Gain.loads = []
    # The member models are scored by the ordinary run first; museval would take most of
    # a minute here and is not what these tests are about.
    monkeypatch.setattr(run_module, "bss_eval_stem", _quick_scores)


def run_tune(tuning_set, test_set, bench_dir, **kwargs):
    kwargs.setdefault("slots", ["vocals"])
    return tune.tune(
        tuning_set,
        defaults=fake_defaults(),
        registry=REGISTRY,
        factory=lambda model_id, stems: [Gain(model_id)],
        fingerprint=lambda model_id: f"w-{model_id}",
        device="cpu",
        run_id="t1",
        bench_dir=bench_dir,
        test_lists=[test_set],
        command="bench tune (test)",
        **kwargs,
    )


# --- candidates ---------------------------------------------------------------------------


def test_weight_grid_covers_the_simplex_in_even_steps() -> None:
    two = tune.weight_grid(2)
    assert len(two) == 11
    assert (0.0, 1.0) in two
    assert (1.0, 0.0) in two
    assert (0.3, 0.7) in two
    three = tune.weight_grid(3)
    assert len(three) == 21
    assert (0.2, 0.2, 0.6) in three
    assert all(sum(w) == pytest.approx(1.0) for w in two + three)
    assert len(tune.weight_grid(4)) == 35
    assert tune.weight_grid(2, steps=2) == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]


def test_candidates_start_with_the_untuned_blend_and_do_not_repeat() -> None:
    tried = tune.candidates(2)
    assert tried[0] == ("waveform", (0.5, 0.5))
    assert len(tried) == len(set(tried)) == 11 + 9 + 2
    # One model alone is tried once, not once per method.
    assert ("waveform", (1.0, 0.0)) in tried
    assert ("stft_magnitude", (1.0, 0.0)) not in tried
    assert ("stft_magnitude", (0.5, 0.5)) in tried
    assert tried[-2:] == [("stft_min", None), ("stft_max", None)]
    three = tune.candidates(3)
    assert three[0][1] == pytest.approx((1 / 3,) * 3)
    assert ("stft_magnitude", three[0][1]) in three  # equal weights, though off the grid


# --- the rule: never tune on test tracks --------------------------------------------------


def test_only_tuning_lists_disjoint_from_the_test_lists_are_accepted(
    tuning_set, test_set, tmp_path_factory
) -> None:
    checked = tune.check_disjoint(tuning_set, [test_set])
    assert checked == [
        {"dataset": "synthetic", "frozen_sha256": test_set.frozen.sha256, "tracks": 2}
    ]
    with pytest.raises(BenchError, match="synthetic is not a tuning list"):
        tune.check_disjoint(test_set, [])
    # A "tuning" list that reuses a test track is refused.
    overlapping = _frozen(
        tmp_path_factory, "synthetic_tuning", [{**synthetic.SPEC[0]}, synthetic.TUNING_SPEC[0]]
    )
    with pytest.raises(BenchError, match=r"shares tracks with synthetic: \['synth-01'\]"):
        tune.check_disjoint(overlapping, [test_set])


def test_the_committed_tuning_list_shares_nothing_with_the_test_list() -> None:
    tuning = SyntheticDataset(name="synthetic_tuning")
    tuning.check_ready()
    checked = tune.check_disjoint(tuning)  # reads the real frozen lists
    assert checked[0]["dataset"] == "synthetic"
    assert tuning.track_ids() == [f"tune-{i:02d}" for i in range(1, 5)]
    assert tuning.load("tune-01").duration_seconds == 12.0


def test_tuning_on_a_test_list_never_starts(test_set, tmp_path) -> None:
    with pytest.raises(BenchError, match="is not a tuning list"):
        run_tune(test_set, test_set, tmp_path)
    assert Gain.loads == []
    assert not (tmp_path / "t1").exists()


# --- what it chooses ----------------------------------------------------------------------


def _brute_force(tuning_set, gains, target="vocals", input_name="mix"):
    """The best waveform weights for two scaled copies of the input, worked out directly."""
    scores = {}
    for weights in [(0.5, 0.5), *tune.weight_grid(2)]:
        gain = weights[0] * gains[0] + weights[1] * gains[1]
        per_track = []
        for track_id in tuning_set.track_ids():
            track = tuning_set.load(track_id)
            estimate = (gain * track.inputs[input_name]).astype(np.float32)
            per_track.append(median_sdr(track.references[target], estimate, 44100))
        scores.setdefault(weights, statistics.median(per_track))
    return scores


def test_the_chosen_weights_are_the_best_on_the_tuning_tracks(
    tuning_set, test_set, tmp_path
) -> None:
    result = run_tune(tuning_set, test_set, tmp_path)
    assert sorted(set(Gain.loads)) == ["s1", "v1", "v2"]
    mode = result["slots"]["vocals"]["modes"]["two_model_blend"]
    assert mode["members"] == ["v1", "v2"]
    assert (mode["tracks"], mode["skipped_tracks"], mode["candidates_tried"]) == (2, [], 22)
    expected = _brute_force(tuning_set, (0.9, 0.1))
    assert mode["baseline"]["weights"] == (0.5, 0.5)
    assert mode["baseline"]["score"] == pytest.approx(expected[0.5, 0.5], abs=1e-3)
    best = max(expected, key=expected.get)
    assert mode["chosen"]["method"] == "waveform"  # in-phase copies: no STFT method beats it
    assert mode["chosen"]["weights"] == best
    assert mode["chosen"]["score"] == pytest.approx(expected[best], abs=1e-3)
    assert mode["chosen"]["score"] > mode["baseline"]["score"]
    assert result["slots"]["vocals"]["method"] == "waveform"
    full = result["slots"]["vocals"]["modes"]["full_ensemble"]
    assert full["candidates_tried"] == len(tune.candidates(3))
    assert full["chosen"]["score"] >= mode["chosen"]["score"] - 1e-3  # a superset of choices
    # What was tuned on is on record.
    assert result["tracks"] == ["tune-01", "tune-02"]
    assert result["disjoint_from"][0]["dataset"] == "synthetic"
    assert result["dataset"] == "synthetic_tuning"
    assert json.loads((tmp_path / "tune" / "t1" / "tuning.json").read_text())["run_id"] == "t1"


def test_a_slot_with_two_outputs_is_tuned_on_their_mean(tuning_set, test_set, tmp_path) -> None:
    result = run_tune(tuning_set, test_set, tmp_path, slots=["vocal_split"])
    mode = result["slots"]["vocal_split"]["modes"]["two_model_blend"]
    lead = _brute_force(tuning_set, (0.8, 0.2), "lead_vocals", "vocals")
    backing = _brute_force(tuning_set, (0.8, 0.2), "backing_vocals", "vocals")
    mean = {w: (lead[w] + backing[w]) / 2 for w in lead}
    assert mode["chosen"]["score"] == pytest.approx(max(mean.values()), abs=1e-3)
    assert mode["baseline"]["score"] == pytest.approx(mean[0.5, 0.5], abs=1e-3)


def test_tuning_resumes_from_its_cached_scores(tuning_set, test_set, tmp_path, monkeypatch) -> None:
    first = run_tune(tuning_set, test_set, tmp_path)
    Gain.loads.clear()
    monkeypatch.setattr(tune, "_score_track", lambda *a, **k: pytest.fail("scored again"))
    second = run_tune(tuning_set, test_set, tmp_path)
    assert Gain.loads == []
    assert second["slots"] == first["slots"]


def test_a_failed_member_is_reported_and_its_blends_left_untuned(
    tuning_set, test_set, tmp_path
) -> None:
    Gain.broken = {"v2"}
    result = run_tune(tuning_set, test_set, tmp_path)
    assert {f["model_id"] for f in result["member_failures"]} == {"v2"}
    mode = result["slots"]["vocals"]["modes"]["two_model_blend"]
    assert (mode["tracks"], mode["chosen"], mode["baseline"]) == (0, None, None)
    assert mode["skipped_tracks"] == ["tune-01", "tune-02"]
    assert result["slots"]["vocals"]["method"] is None
    text = tune.render(result)
    assert "| vocals | two-model blend | v1, v2 | n/a |" in text
    assert "- `v2` on `tune-01`: RuntimeError: weights are corrupt" in text


def test_report_lists_the_tracks_the_check_and_the_config(tuning_set, test_set, tmp_path) -> None:
    result = run_tune(tuning_set, test_set, tmp_path, slots=["vocals", "dereverb"])
    text = tune.render(result)
    assert "# Blend tuning: synthetic_tuning" in text
    assert "2 tracks: `tune-01`, `tune-02`" in text
    assert "- Shares no track with `synthetic` (2 tracks, sha256 `" in text
    assert "- Command: `bench tune (test)`" in text
    assert "Nothing\nhere is a test score" in text
    row = next(line for line in text.splitlines() if line.startswith("| vocals | two-model blend"))
    mode = result["slots"]["vocals"]["modes"]["two_model_blend"]
    assert f"| {mode['baseline']['score']:.2f} | waveform |" in row
    assert f"{mode['chosen']['score'] - mode['baseline']['score']:+.2f}" in row
    # The snippet is valid config for those slots.
    snippet = yaml.safe_load(tune.config_snippet(result))["ensembles"]["slots"]
    assert set(snippet) == {"vocals", "dereverb"}
    assert snippet["vocals"]["blend_method"] == "waveform"
    assert snippet["vocals"]["two_model_weights"] == list(mode["chosen"]["weights"])
    raw = load_defaults().model_dump()
    raw["ensembles"]["slots"] = fake_defaults().model_dump()["ensembles"]["slots"]
    for slot, fields in snippet.items():
        raw["ensembles"]["slots"][slot].update(fields)
    tuned = Defaults.model_validate(raw).ensembles.slots["vocals"]
    assert tuned.weights_for("two_model_blend") == pytest.approx(mode["chosen"]["weights"])


def test_command_tunes_and_writes_the_document(tuning_set, test_set, tmp_path, capsys) -> None:
    out = tmp_path / "docs" / "tuning.md"
    argv = ["tune", "--dataset", "synthetic_tuning", "--slots", "drums", "--device", "cpu"]
    code = cli.main(
        [*argv, "--quiet", "--run-id", "c1", "--out", str(out)],
        factory=lambda model_id, stems: [Gain(model_id)],
        registry=REGISTRY,
        dataset=tuning_set,
        bench_dir=tmp_path / "bench",
        defaults=fake_defaults(),
        test_lists=[test_set],
    )
    assert code == 0
    assert "| drums | two-model blend | `s1`, `s2` |" in out.read_text()
    assert f"wrote {out}" in capsys.readouterr().out
    code = cli.main(
        ["tune", "--dataset", "synthetic", "--device", "cpu", "--quiet"],
        factory=lambda model_id, stems: [Gain(model_id)],
        registry=REGISTRY,
        dataset=test_set,
        bench_dir=tmp_path / "bench",
        defaults=fake_defaults(),
        test_lists=[],
    )
    assert code == 2
    assert "is not a tuning list" in capsys.readouterr().err
