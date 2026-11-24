"""Choose a default single model for each stem, on the tuning tracks (P10.2).

    python -m stems_bench.recommend --out docs/benchmarks/recommendation.md

The models the product uses by default were picked before any benchmark existed. This
picks them from measurements, on tracks that no test list contains, and then shows what
the test lists say about the pick. Blends are not chosen here: that is the tuner's job
(P2.12) and the rest of P2.13.

**The rule, fixed before the tuning scores were looked at:**

1. Only the tuning list decides (``moisesdb_tuning``, 30 tracks). A model is a candidate
   for a stem if the benchmark scored its own output of that name against that stem on at
   least ``MIN_TRACKS`` of them. Blends and derived outputs are not candidates.
2. A stem's best candidate is the one with the highest median SDR. Every candidate within
   ``MARGIN_DB`` of it is acceptable for that stem: a difference that small is not worth
   another model's running time.
3. Among acceptable candidates, the fewest models win. The model acceptable for the most
   stems still without one is taken first and given all of those stems; then again for
   the stems that are left. Ties go to the model that gives up the least SDR in total
   against each stem's best, then to the faster one, then to the name.

Time enters only through rule 3: one model that serves five stems runs once. The test
lists (``moisesdb_subset``, ``musdb18hq_test``) are read afterwards and change nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from stems_common.paths import REPO_ROOT
from stems_common.stems import PRODUCT_STEMS

BENCHMARKS = REPO_ROOT / "docs" / "benchmarks"
TUNING_PATH = BENCHMARKS / "tuning-models-moisesdb.json"
# The single models the config named before this recommendation, kept so that the report
# can still say what was replaced after the config has changed.
IN_USE_PATH = BENCHMARKS / "defaults-before-recommendation.json"
TEST_PATHS = {
    "moisesdb_subset": BENCHMARKS / "moisesdb.json",
    "musdb18hq_test": BENCHMARKS / "musdb18hq.json",
}

# -- fixed before the tuning scores were looked at -----------------------------------------
MARGIN_DB = 0.3
MIN_TRACKS = 10
# So that a difference of exactly the margin is inside it, whatever the floats say.
_EPSILON = 1e-9


class RecommendError(ValueError):
    """The recommendation cannot be made or reported as asked."""


def candidates(rows: Sequence[Mapping[str, Any]], stem: str) -> dict[str, dict[str, Any]]:
    """Model -> its median SDR, tracks and time, for the models that may serve ``stem``."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sdr = (row.get("medians") or {}).get("sdr")
        if (
            ":" in row["system"]  # a blend
            or row["output"] != stem
            or row["target"] != stem
            or row.get("derived")
            or sdr is None
            or row["tracks_scored"] < MIN_TRACKS
        ):
            continue
        out[row["system"]] = {
            "sdr": float(sdr),
            "tracks": int(row["tracks_scored"]),
            "seconds_per_audio_minute": row.get("seconds_per_audio_minute"),
        }
    return out


def choose(
    rows: Sequence[Mapping[str, Any]], stems: Sequence[str] = PRODUCT_STEMS
) -> dict[str, dict[str, Any]]:
    """The chosen model per stem, with what it was chosen from. A stem with no candidate
    gets ``model: None``."""
    table = {stem: candidates(rows, stem) for stem in stems}
    result: dict[str, dict[str, Any]] = {}
    acceptable: dict[str, dict[str, float]] = {}
    for stem in stems:
        found = table[stem]
        if not found:
            result[stem] = {"model": None, "best_model": None, "candidates": {}}
            continue
        best = max(found, key=lambda m: (found[m]["sdr"], m))
        top = found[best]["sdr"]
        # What each acceptable model gives up against the best, in dB (0 for the best).
        acceptable[stem] = {
            m: top - c["sdr"] for m, c in found.items() if top - c["sdr"] <= MARGIN_DB + _EPSILON
        }
        result[stem] = {"model": None, "best_model": best, "best_sdr": top, "candidates": found}

    def speed(model: str) -> float:
        rates = [
            table[stem][model]["seconds_per_audio_minute"]
            for stem in stems
            if model in table[stem] and table[stem][model]["seconds_per_audio_minute"] is not None
        ]
        return min(rates) if rates else float("inf")

    remaining = [stem for stem in stems if stem in acceptable]
    while remaining:
        models = sorted({m for stem in remaining for m in acceptable[stem]})

        def rank(model: str) -> tuple[float, float, float, str]:
            served = [stem for stem in remaining if model in acceptable[stem]]
            given_up = sum(acceptable[stem][model] for stem in served)
            return (-len(served), given_up, speed(model), model)

        winner = min(models, key=rank)
        for stem in [stem for stem in remaining if winner in acceptable[stem]]:
            entry = result[stem]
            entry["model"] = winner
            entry["sdr"] = entry["candidates"][winner]["sdr"]
            entry["gives_up_db"] = acceptable[stem][winner]
            remaining.remove(stem)
    return result


def scores_on(rows: Sequence[Mapping[str, Any]], model: str, stem: str) -> dict[str, Any] | None:
    """One model's score for one stem in a benchmark data file; no track minimum here."""
    for row in rows:
        if row["system"] == model and row["output"] == stem and row["target"] == stem:
            sdr = (row.get("medians") or {}).get("sdr")
            if sdr is not None:
                return {"sdr": float(sdr), "tracks": int(row["tracks_scored"])}
    return None


BLEND_MODES = ("two_model_blend", "full_ensemble")


def blends_beside(
    rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The config's blends for each stem, as the benchmark scored them, beside the chosen
    single model. For information: the rule above does not choose blends."""
    out: dict[str, dict[str, Any]] = {}
    for stem, entry in chosen.items():
        if entry["model"] is None:
            continue
        modes = {}
        for mode in BLEND_MODES:
            row = next(
                (
                    r
                    for r in rows
                    if r["system"] == f"{stem}:{mode}"
                    and r["target"] == stem
                    and (r.get("medians") or {}).get("sdr") is not None
                ),
                None,
            )
            if row is not None:
                modes[mode] = {
                    "sdr": float(row["medians"]["sdr"]),
                    "difference_db": float(row["medians"]["sdr"]) - entry["sdr"],
                    "seconds_per_audio_minute": row.get("seconds_per_audio_minute"),
                }
        if modes:
            out[stem] = modes
    return out


def seconds_per_minute(rows: Sequence[Mapping[str, Any]], models: Sequence[str]) -> float | None:
    """Seconds per minute of audio to run each of ``models`` once; None if one is untimed."""
    total = 0.0
    for model in dict.fromkeys(models):
        rates = [
            row["seconds_per_audio_minute"]
            for row in rows
            if row["system"] == model and row.get("seconds_per_audio_minute") is not None
        ]
        if not rates:
            return None
        total += rates[0]
    return total


def build(
    tuning: Mapping[str, Any],
    tests: Mapping[str, Mapping[str, Any]],
    in_use: Mapping[str, str],
) -> dict[str, Any]:
    """Everything the report states. ``in_use`` maps each stem to the model used until now."""
    chosen = choose(tuning["rows"], tuple(in_use))
    checks: dict[str, dict[str, Any]] = {}
    for stem, entry in chosen.items():
        checks[stem] = {}
        for name, data in tests.items():
            new = scores_on(data["rows"], entry["model"], stem) if entry["model"] else None
            old = scores_on(data["rows"], in_use[stem], stem)
            composed = False
            if old is None:
                # What the product delivered when its model's own output is a different
                # thing: the six-stem "other" made from a four-stem one (PLAN.md, "Other").
                old = scores_on(data["rows"], f"{stem}_composed:single_best", stem)
                composed = old is not None
            checks[stem][name] = {
                "chosen": new,
                "in_use": old,
                "in_use_composed": composed,
                "difference_db": None if new is None or old is None else new["sdr"] - old["sdr"],
            }
    picked = [entry["model"] for entry in chosen.values() if entry["model"]]
    return {
        "tuning_run": tuning["run_id"],
        "tuning_dataset": tuning["dataset"],
        "test_runs": {name: data["run_id"] for name, data in tests.items()},
        "chosen": chosen,
        "blends": blends_beside(tuning["rows"], chosen),
        "in_use": dict(in_use),
        "checks": checks,
        "time": {
            "chosen_models": list(dict.fromkeys(picked)),
            "chosen": seconds_per_minute(tuning["rows"], picked),
            "in_use_models": list(dict.fromkeys(in_use.values())),
            "in_use": seconds_per_minute(tuning["rows"], list(in_use.values())),
        },
    }


def _db(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def render(report: Mapping[str, Any]) -> str:
    chosen, checks, time = report["chosen"], report["checks"], report["time"]
    lines = [
        "# Default models: a single model per stem",
        "",
        "Written by `python -m stems_bench.recommend`. The choice is made on the tuning "
        f"tracks only (`{report['tuning_dataset']}`, run `{report['tuning_run']}`, "
        "`docs/benchmarks/tuning-models-moisesdb.md`), by a rule that was committed before "
        "those scores were looked at. The test lists are shown afterwards and decided "
        "nothing. Blends are not chosen here.",
        "",
        "## The rule",
        "",
        f"1. A model is a candidate for a stem if the benchmark scored its own output of "
        f"that name against that stem on at least {MIN_TRACKS} tuning tracks.",
        f"2. The best candidate has the highest median SDR. Every candidate within "
        f"{MARGIN_DB:g} dB of it is acceptable.",
        "3. Among acceptable candidates the fewest models win: the model acceptable for the "
        "most stems is taken first. Ties go to the one that gives up the least SDR, then "
        "to the faster one.",
        "",
        "## The choice",
        "",
        "SDR in dB, median over the tuning tracks, on this Mac's GPU.",
        "",
        "| Stem | Chosen | Its SDR | Best on the tuning tracks | Best SDR | Given up | "
        "In use until now | Its SDR |",
        "|---|---|---:|---|---:|---:|---|---:|",
    ]
    for stem, entry in chosen.items():
        used = report["in_use"][stem]
        used_score = entry["candidates"].get(used, {}).get("sdr")
        if entry["model"] is None:
            lines.append(f"| {stem} | no candidate | - | - | - | - | `{used}` | - |")
            continue
        lines.append(
            f"| {stem} | `{entry['model']}` | {_db(entry['sdr'])} | `{entry['best_model']}` | "
            f"{_db(entry['best_sdr'])} | {_db(entry['gives_up_db'])} | `{used}` | "
            f"{_db(used_score)} |"
        )
    lines += [
        "",
        'A dash under "In use until now" means the benchmark did not score that model\'s '
        'own output against this stem (a four-stem model\'s "other" is a different thing '
        "from the six-stem one).",
        "",
        "Every candidate, per stem:",
        "",
        "| Stem | Model | SDR | Tracks | s per min of audio | Acceptable |",
        "|---|---|---:|---:|---:|---|",
    ]
    for stem, entry in chosen.items():
        ranked = sorted(entry["candidates"].items(), key=lambda item: -item[1]["sdr"])
        for model, scored in ranked:
            ok = entry["best_sdr"] - scored["sdr"] <= MARGIN_DB + _EPSILON
            rate = scored["seconds_per_audio_minute"]
            lines.append(
                f"| {stem} | `{model}` | {_db(scored['sdr'])} | {scored['tracks']} | "
                f"{'-' if rate is None else f'{rate:.2f}'} | {'yes' if ok else 'no'} |"
            )

    def models(names: Sequence[str]) -> str:
        return ", ".join(f"`{m}`" for m in names)

    def seconds(value: float | None) -> str:
        return "not known" if value is None else f"{value:.1f} s"

    if report.get("blends"):
        lines += [
            "",
            "## The configured blends beside it",
            "",
            "The config also names a two-model blend and a full ensemble for each stem, with "
            "equal weights; nobody has tuned them yet (P2.12). On the same tuning tracks, "
            "against the chosen single model:",
            "",
            "| Stem | Chosen model | Two-model blend | Against it | Full ensemble | Against it |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        best_gain = None
        for stem, modes in report["blends"].items():
            cells = []
            for mode in BLEND_MODES:
                scored = modes.get(mode)
                cells += [
                    _db(scored and scored["sdr"]),
                    _db(scored and scored["difference_db"], signed=True),
                ]
                if scored and (best_gain is None or scored["difference_db"] > best_gain):
                    best_gain = scored["difference_db"]
            lines.append(f"| {stem} | {_db(chosen[stem]['sdr'])} | " + " | ".join(cells) + " |")
        lines += [
            "",
            f"The most any of them gains on the single model is {_db(best_gain, signed=True)} dB, "
            "and each costs more models. The six-stem `other` has no blend row here: its "
            "blends are four-stem ones, scored against a different target.",
        ]
    lines += [
        "",
        "## Time",
        "",
        "Seconds to run each model once per minute of audio, from the same run:",
        "",
        f"- chosen: {models(time['chosen_models'])}: {seconds(time['chosen'])}",
        f"- in use until now: {models(time['in_use_models'])}: {seconds(time['in_use'])}",
        "",
        "## What the test lists show",
        "",
        "These tracks took no part in the choice. SDR in dB, median over the list's tracks, "
        "for the chosen model and for the one in use until now.",
        "",
        "| Stem | List | Chosen | In use until now | Difference | Tracks |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stem, lists in checks.items():
        for name, check in lists.items():
            new, old = check["chosen"], check["in_use"]
            if new is None and old is None:
                continue
            lines.append(
                f"| {stem} | `{name}` | {_db(new and new['sdr'])} | "
                f"{_db(old and old['sdr'])}{' (composed)' if check['in_use_composed'] else ''} | "
                f"{_db(check['difference_db'], signed=True)} | "
                f"{(new or old)['tracks']} |"
            )
    lines += [
        "",
        '"Composed" is the six-stem other the product has been delivering: a four-stem '
        "model's other with the guitar and piano stems taken out, which the benchmark scores "
        "as its own system. MUSDB18-HQ has no guitar or piano references, so those stems "
        "appear for MoisesDB only. Test runs: "
        + ", ".join(f"`{name}` from run `{run}`" for name, run in report["test_runs"].items())
        + ".",
        "",
        "## Limits",
        "",
        "- One tuning list of 30 tracks from one dataset decides. MUSDB18-HQ's training "
        "tracks have not been scored yet.",
        "- Only the models the config named when the benchmark started are candidates.",
        "- Which songs each checkpoint was trained on is not known here. A model that met "
        "these datasets' songs in training would score higher on them than on new music, "
        "and nothing in this repository can tell.",
        "- SDR is not how it sounds. No listening test has been run.",
        "- The vocal split and de-reverb models are not chosen here, and neither are blends.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from stems_common.settings import load_defaults

    parser = argparse.ArgumentParser(
        prog="python -m stems_bench.recommend",
        description="Choose a default single model per stem on the tuning tracks.",
    )
    parser.add_argument("--tuning", type=Path, default=TUNING_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--in-use",
        type=Path,
        default=IN_USE_PATH if IN_USE_PATH.is_file() else None,
        help="JSON file mapping each stem to the model in use until now (default: "
        f"{IN_USE_PATH.name} beside the reports, else single_best in config/defaults.yaml)",
    )
    args = parser.parse_args(argv)
    try:
        if not args.tuning.is_file():
            raise RecommendError(f"{args.tuning} is missing: run `bench report` on the tuning run")
        tuning = json.loads(args.tuning.read_text(encoding="utf-8"))
        tests = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in TEST_PATHS.items()
            if path.is_file()
        }
        if args.in_use:
            in_use = json.loads(args.in_use.read_text(encoding="utf-8"))
        else:
            slots = load_defaults().ensembles.slots
            in_use = {stem: slots[stem].single_best for stem in PRODUCT_STEMS}
        report = build(tuning, tests, in_use)
    except RecommendError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = render(report)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
