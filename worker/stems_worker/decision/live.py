"""The live check: one real request to Jev per kind of decision, kept as a recording.

    .venv/bin/python -m stems_worker.decision.live          # needs TYPESAFE_API_KEY

It asks Jev to parse one request, plan one track and assess one stem, with fixed inputs
that are written out below, and saves each exchange (request and reply, never the key) as
``tests/fixtures/jev/<kind>.recorded.jsonl``. Those files are what the hand-written
fixtures stand in for: the service's real wire format, which the tests then replay.

A kind that already has a recording is not asked again, so running this twice costs
nothing; ``--force`` records afresh. The three calls are the only cost: a few hundred
input tokens each.

What the answers are is not judged here. This shows that a call succeeds and can be
read; whether Jev's choices are good is a question for the benchmark and the calibration.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stems_common.paths import REPO_ROOT
from stems_common.settings import Defaults, load_defaults
from stems_models.registry import Registry, load_registry
from stems_worker.decision.assess import assess_stem
from stems_worker.decision.jev import KEY_ENV, HttpTransport, JevClient, Transport
from stems_worker.decision.mock import RecordingTransport, read_exchanges
from stems_worker.decision.parse import from_selection, parse_request
from stems_worker.decision.plan import plan_ensemble
from stems_worker.decision.record import Decision

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "jev"
KINDS = ("parse", "plan", "gate")
JOB_ID = "5b0d9a0e-3c1f-4f7a-8d2e-9a6c4b1e7f20"

# -- the fixed inputs ----------------------------------------------------------------------

PARSE_TEXT = "just the vocals and the drums please, as flac"
PLAN_STEMS = ("vocals", "drums")
# A description of a track, as the listening layer would give it. Made up: no audio.
TRACK_FEATURES: dict[str, Any] = {
    "duration_seconds": 200.0,
    "sample_rate_hz": 44100,
    "channels": 2,
    "vocal_presence": "lead_vocals",
    "detected_instruments": ["vocals", "drums", "bass"],
    "genre_hint": "Rock music",
    "reverb_estimate": None,
    "loudness_lufs": -9.5,
}
GATE_STEM = "vocals"
GATE_MODEL = "mel_roformer_kim_vocals"
# Measurements of a stem, as the worker would make them. Made up: no audio.
GATE_METRICS: dict[str, Any] = {
    "sdr_db": None,
    "sir_db": None,
    "sar_db": None,
    "duration_seconds": 200.0,
    "rms_dbfs": -20.0,
    "peak_dbfs": -1.0,
    "silence_fraction": 0.1,
    "bleed_estimate": {"suspected_source": "drums", "level": "slight"},
    "tagger_notes": "live check: made-up measurements, no audio behind them",
}
GATE_AVAILABLE: dict[str, str | None] = {
    "retry_fallback_model": "bs_roformer_viperx_vocals",
    "retry_other_ensemble": "two_model_blend",
    "retry_with_dereverb": None,
}


def recording_path(kind: str, directory: Path = FIXTURE_DIR) -> Path:
    return directory / f"{kind}.recorded.jsonl"


def decide(kind: str, client: JevClient | None, defaults: Defaults, registry: Registry) -> Decision:
    """Make the one decision of ``kind`` from the fixed inputs above."""
    if kind == "parse":
        return parse_request(
            PARSE_TEXT, client=client, settings=defaults.decision, request_id=JOB_ID
        )
    if kind == "plan":
        parsed = from_selection(PLAN_STEMS, settings=defaults.decision, request_id=JOB_ID).value
        return plan_ensemble(
            parsed,
            TRACK_FEATURES,
            client=client,
            defaults=defaults,
            registry=registry,
            job_id=JOB_ID,
            rates={},
        )
    if kind == "gate":
        return assess_stem(
            GATE_STEM,
            GATE_MODEL,
            GATE_METRICS,
            client=client,
            defaults=defaults,
            job_id=JOB_ID,
            present=True,
            available=GATE_AVAILABLE,
        )
    raise ValueError(f"unknown kind {kind!r}")


def answered(record: dict[str, Any]) -> bool:
    """Did the service give a usable reply? (Its answer may still be set aside as unsure.)"""
    reason = str(record.get("fallback_reason") or "")
    return bool(record.get("asked")) and not reason.startswith(("jev_failed", "no_key"))


def run_live(
    transport: Transport,
    *,
    defaults: Defaults | None = None,
    registry: Registry | None = None,
    directory: Path = FIXTURE_DIR,
    force: bool = False,
    say: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Record each kind that has no recording yet. Returns kind -> decision record."""
    defaults = defaults or load_defaults()
    registry = registry or load_registry()
    records: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        path = recording_path(kind, directory)
        if path.exists() and not force:
            say(f"{kind}: already recorded ({len(read_exchanges(path))} exchange), not asked again")
            continue
        partial = path.with_suffix(".partial")
        partial.unlink(missing_ok=True)
        client = JevClient(RecordingTransport(transport, partial), defaults.decision)
        record = decide(kind, client, defaults, registry).record
        records[kind] = record
        if answered(record) and partial.exists():
            partial.replace(path)  # only a call that worked becomes a fixture
        else:
            partial.unlink(missing_ok=True)
        say(
            f"{kind}: provider={record['provider']} model={record.get('model')} "
            f"tokens={record.get('input_tokens')}+{record.get('output_tokens')} "
            f"latency_ms={record.get('latency_ms')} attempts={record.get('attempts')} "
            f"fallback_reason={record.get('fallback_reason')}"
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stems_worker.decision.live",
        description="Make one real Jev request per decision type and record it.",
    )
    parser.add_argument("--force", action="store_true", help="record again over what is there")
    args = parser.parse_args(argv)
    transport = HttpTransport.from_env()
    if transport is None:
        print(f"live check: {KEY_ENV} is not set; nothing was sent", file=sys.stderr)
        return 2
    records = run_live(transport, force=args.force)
    failed = [kind for kind, record in records.items() if not answered(record)]
    missing = [kind for kind in KINDS if not recording_path(kind).exists()]
    if failed or missing:
        print(f"live check: no usable reply for {', '.join(failed or missing)}", file=sys.stderr)
        return 1
    print(f"live check: all {len(KINDS)} kinds have a recording in {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
