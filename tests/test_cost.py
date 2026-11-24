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
        "gate": 744,
    }
    assert all(used["file"].startswith("tests/fixtures/jev/") for used in usage.values())
    if not cost.BENCHMARK_PATH.is_file():
        pytest.skip("no MoisesDB benchmark report in this checkout")
    data = json.loads(cost.BENCHMARK_PATH.read_text())
    report = cost.build(
        prices, usage, data, DEFAULTS, benchmark_file="docs/benchmarks/moisesdb.json"
    )
    assert report["decisions"]["six ticked stems, no retries"]["tokens"][0] == 2316 + 6 * 744
    assert report["separation"]["single_best"]["seconds"] is not None
