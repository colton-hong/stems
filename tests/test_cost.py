"""The cost-per-song report: arithmetic on measured quantities and dated prices."""

from __future__ import annotations

import json

import pytest

from stems_bench import cost
from stems_common.settings import load_defaults

PRICES = {
    "checked": "2026-11-24",
    "jev": {
        "source": "https://prices.invalid/jev",
        "usd_per_million_input_tokens": 0.05,
        "usd_per_million_output_tokens": 0.0,
    },
    "modal": {
        "source": "https://prices.invalid/gpu",
        "usd_per_second": {"T4": 0.0002, "L4": 0.0003},
        "usd_per_core_second": 0.00001,
        "usd_per_gib_second": 0.000002,
    },
    "r2": {
        "source": "https://prices.invalid/storage",
        "usd_per_gb_month": 0.015,
        "usd_per_gb_egress": 0.0,
        "free_gb_month": 10,
    },
    "not_found": [{"what": "Some other price", "source": "https://prices.invalid/other"}],
}
USAGE = {
    "parse": {"input_tokens": 1000, "output_tokens": 200, "file": "p.jsonl"},
    "plan": {"input_tokens": 2000, "output_tokens": 500, "file": "l.jsonl"},
    "gate": {"input_tokens": 700, "output_tokens": 100, "file": "g.jsonl"},
}
DEFAULTS = load_defaults()


def benchmark(leave_out: str | None = None) -> dict:
    """A benchmark data file that times every model the defaults name at 10 s per minute."""
    models = {m for mode in cost.MODES for m in cost.models_for(DEFAULTS, mode)}
    rows = [
        {"system": m, "seconds_per_audio_minute": 10.0, "devices": ["mps"]}
        for m in sorted(models)
        if m != leave_out
    ]
    rows.append(
        {"system": "vocals:single_best", "seconds_per_audio_minute": 99.0, "devices": ["mps"]}
    )
    return {"run_id": "test-run", "rows": rows}


def test_decision_tokens_are_one_plan_and_a_gate_call_per_stem_attempt() -> None:
    assert cost.decision_tokens(USAGE, 6, parse=False) == (2000 + 6 * 700, 500 + 6 * 100)
    assert cost.decision_tokens(USAGE, 6, parse=True) == (1000 + 2000 + 6 * 700, 200 + 500 + 600)
    assert cost.decision_tokens(USAGE, 2, parse=False, retries_per_stem=2) == (
        2000 + 2 * 3 * 700,
        500 + 2 * 3 * 100,
    )
    # 6,200 input tokens at $0.05 per million; output is free at these prices.
    assert cost.jev_cost((6200, 1100), PRICES) == pytest.approx(0.00031)
    priced_output = {**PRICES, "jev": {**PRICES["jev"], "usd_per_million_output_tokens": 1.0}}
    assert cost.jev_cost((0, 1_000_000), priced_output) == pytest.approx(1.0)


def test_separation_time_is_the_sum_over_the_distinct_models_a_mode_runs() -> None:
    single = cost.models_for(DEFAULTS, "single_best")
    assert len(single) == len(set(single))  # a model serving two stems runs once
    assert set(single) == {DEFAULTS.ensembles.slots[s].single_best for s in cost.PRODUCT_STEMS}
    rates = cost.model_rates(benchmark())
    assert "vocals:single_best" not in rates  # a blend's row is not a model
    seconds, missing = cost.separation_seconds(single, rates, 4.0)
    assert (seconds, missing) == (len(single) * 10.0 * 4.0, [])
    more = cost.models_for(DEFAULTS, "full_ensemble")
    assert set(single) <= set(more)
    assert len(more) > len(single)


def test_a_model_with_no_timing_gives_no_time_and_is_named() -> None:
    single = cost.models_for(DEFAULTS, "single_best")
    rates = cost.model_rates(benchmark(leave_out=single[0]))
    assert cost.separation_seconds(single, rates, 4.0) == (None, [single[0]])
    report = cost.build(
        PRICES, USAGE, benchmark(leave_out=single[0]), DEFAULTS, benchmark_file="b.json"
    )
    assert report["separation"]["single_best"]["seconds"] is None
    text = cost.render(report, PRICES)
    assert f"not known: no timing for `{single[0]}`" in text


def test_an_empty_stem_is_tried_with_the_fallback_and_then_as_a_blend() -> None:
    from stems_models.registry import load_registry
    from stems_worker.decision.plan import config_fallback

    registry = load_registry()
    for stem in ("guitar", "piano"):
        slot = DEFAULTS.ensembles.slots[stem]
        attempts = cost.empty_attempts(stem, DEFAULTS, registry, "single_best")
        assert attempts[0] == [slot.single_best]
        assert attempts[1] == [config_fallback(stem, DEFAULTS, registry)]
        assert attempts[1] != attempts[0]
        assert len(attempts) <= 1 + DEFAULTS.gate.max_retries_per_stem
        assert len({tuple(a) for a in attempts}) == len(attempts)  # nothing is tried twice


def test_what_an_absent_instrument_adds_is_the_models_not_already_run() -> None:
    from stems_models.registry import load_registry

    registry = load_registry()
    base = cost.models_for(DEFAULTS, "single_best")
    tried = {
        stem: cost.empty_attempts(stem, DEFAULTS, registry, "single_best")
        for stem in ("guitar", "piano")
    }
    new = {
        stem: list(dict.fromkeys(m for a in tried[stem][1:] for m in a if m not in base))
        for stem in tried
    }
    everything = {m for attempts in tried.values() for a in attempts for m in a} | set(base)
    rates = {m: {"seconds_per_audio_minute": 7.0, "devices": ["mps"]} for m in everything}
    costs = cost.absent_costs(DEFAULTS, registry, rates, 4.0, "single_best")
    assert list(costs) == ["no guitar", "no piano", "neither"]
    for label, stem in (("no guitar", "guitar"), ("no piano", "piano")):
        assert costs[label]["attempts"] == {stem: tried[stem]}
        assert costs[label]["extra_models"] == new[stem]
        assert costs[label]["seconds"] == len(new[stem]) * 7.0 * 4.0
    both = list(dict.fromkeys(new["guitar"] + new["piano"]))
    assert costs["neither"]["extra_models"] == both
    assert costs["neither"]["seconds"] == len(both) * 7.0 * 4.0
    # A model the benchmark has no time for gives no figure, and is named.
    if new["piano"]:
        short = {m: r for m, r in rates.items() if m != new["piano"][0]}
        missing = cost.absent_costs(DEFAULTS, registry, short, 4.0, "single_best")
        assert missing["no piano"]["seconds"] is None
        assert missing["no piano"]["unmeasured"] == [new["piano"][0]]


def test_the_report_has_the_absent_instrument_table_only_with_a_registry_and_the_rule_on() -> None:
    from stems_models.registry import load_registry

    registry = load_registry()
    with_it = cost.build(
        PRICES, USAGE, benchmark(), DEFAULTS, benchmark_file="b.json", registry=registry
    )
    assert with_it["absent"]["mode"] == DEFAULTS.decision.mode_for_quality["balanced"]
    assert with_it["absent"]["base_seconds"] == with_it["separation"]["single_best"]["seconds"]
    text = cost.render(with_it, PRICES)
    assert "### When the song lacks an instrument" in text
    assert "| no guitar | guitar: `" in text
    assert "| neither | guitar: `" in text
    without = cost.build(PRICES, USAGE, benchmark(), DEFAULTS, benchmark_file="b.json")
    assert without["absent"] is None
    assert "When the song lacks an instrument" not in cost.render(without, PRICES)
    rules = DEFAULTS.gate.rules.model_copy(update={"empty_below_mix_db": None})
    off = DEFAULTS.model_copy(update={"gate": DEFAULTS.gate.model_copy(update={"rules": rules})})
    assert (
        cost.build(PRICES, USAGE, benchmark(), off, benchmark_file="b.json", registry=registry)[
            "absent"
        ]
        is None
    )


def test_the_taggers_time_comes_from_the_timings_of_real_jobs(tmp_path) -> None:
    assert cost.listening_usage(tmp_path / "no-such-run") is None
    results = tmp_path / "results"
    results.mkdir()

    def job(track: str, audio: float, heard: list[float], state: str = "ready") -> None:
        timings = [{"model_id": "m", "stage": "separation", "device": "cpu", "wall_seconds": 500.0}]
        timings += [
            {"model_id": "tagger", "stage": "listening", "device": "cpu", "wall_seconds": w}
            for w in heard
        ]
        record = {"track_id": track, "state": state, "audio_seconds": audio, "timings": timings}
        (results / f"{track}.json").write_text(json.dumps(record))

    job("aaaaaaaa-1", 240.0, [4.0, 4.0, 4.0, 12.0])  # 24 s: 6 s per minute of audio
    job("bbbbbbbb-2", 120.0, [10.0, 10.0])  # 10 s per minute
    job("cccccccc-3", 60.0, [2.0])  # 2 s per minute
    job("dddddddd-4", 60.0, [])  # the tagger was off: says nothing
    job("eeeeeeee-5", 60.0, [9.0], state="failed")
    usage = cost.listening_usage(tmp_path)
    assert [j["track_id"] for j in usage["jobs"]] == ["aaaaaaaa-1", "bbbbbbbb-2", "cccccccc-3"]
    assert usage["jobs"][0]["tagger_seconds"] == 24.0
    assert usage["jobs"][0]["times_listened"] == 4
    assert usage["median_seconds_per_audio_minute"] == 6.0

    report = cost.build(
        PRICES, USAGE, benchmark(), DEFAULTS, benchmark_file="b.json", listening=usage
    )
    assert report["listening"]["seconds"] == 24.0  # four minutes at the median
    assert report["listening"]["enabled"] is DEFAULTS.listening.enabled
    text = cost.render(report, PRICES)
    assert "### The tagger" in text
    assert "| `aaaaaaaa` | 240 s | 4 | 24 | 6.0 s |" in text
    assert "a 4-minute song costs 24 s of tagger time" in text
    state = "on" if DEFAULTS.listening.enabled else "off"
    assert f"`listening.enabled` is **{state}**" in text
    plain = cost.build(PRICES, USAGE, benchmark(), DEFAULTS, benchmark_file="b.json")
    assert plain["listening"] is None
    assert "### The tagger" not in cost.render(plain, PRICES)


def test_storage_is_an_upper_bound_from_the_format() -> None:
    # One minute of 24-bit 44.1 kHz stereo is 15,876,000 bytes.
    assert cost.WAV_BYTES_PER_SECOND * 60 == 15_876_000
    # The upload, six stems and a zip of the six: thirteen stem-sized files.
    assert cost.storage_gb(6, 4.0) == pytest.approx(13 * 4 * 15_876_000 / 1e9)
    report = cost.build(PRICES, USAGE, benchmark(), DEFAULTS, benchmark_file="b.json")
    expected = cost.storage_gb(6, 4.0) * 0.015 * DEFAULTS.retention.days / 30
    assert report["storage"]["usd"] == pytest.approx(expected)
    assert report["storage"]["days"] == 7


def test_the_report_cites_every_source_and_gives_no_gpu_cost() -> None:
    report = cost.build(PRICES, USAGE, benchmark(), DEFAULTS, benchmark_file="docs/b.json")
    text = cost.render(report, PRICES)
    for needle in (
        "Prices were read on 2026-11-24",
        "https://prices.invalid/jev, read 2026-11-24",
        "https://prices.invalid/gpu, read 2026-11-24",
        "https://prices.invalid/storage, read 2026-11-24",
        "`docs/b.json` (run `test-run`, device mps)",
        "| plan | 2,000 | 500 | `l.jsonl` |",
        "| six ticked stems, no retries | 6,200 | $0.000310 |",
        "| Separation on a rented GPU | not known | no GPU timing exists (BLOCKED.md B4) |",
        "**What this costs on a rented GPU is not known.**",
        "There is no total",
        "Some other price: looked for at https://prices.invalid/other and not found there.",
    ):
        assert needle in text, needle
    # The GPU prices appear only as prices per second: no dollar figure for a song's GPU time.
    gpu_section = text.split("## Separation")[1].split("## Storage")[0]
    dollars = [word for word in gpu_section.split() if word.startswith("$")]
    assert dollars == ["$0.0002", "$0.0003"]


def test_prices_must_name_their_source_and_day(tmp_path) -> None:
    import yaml

    good = tmp_path / "prices.yaml"
    good.write_text(yaml.safe_dump(PRICES))
    assert cost.load_prices(good)["jev"]["usd_per_million_input_tokens"] == 0.05
    for change in (
        lambda p: p["jev"].pop("source"),
        lambda p: p.pop("checked"),
        lambda p: p.pop("r2"),
    ):
        broken = json.loads(json.dumps(PRICES))
        change(broken)
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(cost.CostError):
            cost.load_prices(path)


def test_the_committed_prices_and_recordings_make_a_report() -> None:
    prices = cost.load_prices()
    assert str(prices["checked"]) == "2026-11-24"
    for section in ("jev", "modal", "r2"):
        assert prices[section]["source"].startswith("https://")
    usage = cost.recorded_usage()
    assert {kind: used["input_tokens"] for kind, used in usage.items()} == {
        "parse": 1094,
        "plan": 2316,
        "gate": 767,
    }
    assert all(used["file"].startswith("tests/fixtures/jev/") for used in usage.values())
    if not cost.BENCHMARK_PATH.is_file():
        pytest.skip("no MoisesDB benchmark report in this checkout")
    data = json.loads(cost.BENCHMARK_PATH.read_text())
    report = cost.build(
        prices, usage, data, DEFAULTS, benchmark_file="docs/benchmarks/moisesdb.json"
    )
    assert report["decisions"]["six ticked stems, no retries"]["tokens"][0] == 2316 + 6 * 767
    assert report["separation"]["single_best"]["seconds"] is not None
