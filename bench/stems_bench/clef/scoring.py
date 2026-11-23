"""Statistics and pass/fail verdicts for the Clef-omni test.

Plain functions over plain data, so each can be checked by hand:

- accuracy of present/absent decisions at a fixed threshold, overall and per stem;
- AUC, the probability that a bleed clip gets a higher bleed probability than a clean one
  (ties count half);
- Spearman's rank correlation between a bleed score and the true bleed level;
- the share of clean clips called bleeding;
- each pre-registered threshold applied to its measured value.

A model's results are two lists. Part A: one item per (track, stem) with ``truth`` (bool)
and ``probability`` of presence. Part B: one item per clip with ``level`` (0 clean, 1 to 4
for +30 to +6 dB SIR), ``bleed_probability``, and ``bleed_score`` when the model gives one
(Clef-omni's rubric score; BEATs has none, and its probability is used in its place).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from stems_common.stems import PRODUCT_STEMS

PRESENT_THRESHOLD = 0.5


def accuracy(truth: Sequence[bool], predicted: Sequence[bool]) -> float | None:
    """The share of decisions that match. None for no decisions."""
    if len(truth) != len(predicted):
        raise ValueError("truth and predictions differ in length")
    if not truth:
        return None
    return sum(1 for t, p in zip(truth, predicted, strict=True) if t == p) / len(truth)


def auc(negatives: Sequence[float], positives: Sequence[float]) -> float | None:
    """The chance that a positive scores above a negative; a tie counts half.

    None when either group is empty.
    """
    if not negatives or not positives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, with tied values sharing the mean of their ranks."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson's correlation of the ranks. None if either side is constant or too short."""
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def best_f1_threshold(truth: Sequence[bool], scores: Sequence[float]) -> dict[str, float] | None:
    """The cut-off with the best F1, tried at every observed score. Secondary analysis only:
    it is chosen after seeing the answers. None when nothing is truly present."""
    if not any(truth):
        return None
    best: dict[str, float] | None = None
    for cut in sorted(set(scores)):
        predicted = [s >= cut for s in scores]
        tp = sum(1 for t, p in zip(truth, predicted, strict=True) if t and p)
        fp = sum(1 for t, p in zip(truth, predicted, strict=True) if not t and p)
        fn = sum(1 for t, p in zip(truth, predicted, strict=True) if t and not p)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": cut, "f1": f1, "accuracy": accuracy(truth, predicted) or 0.0}
    return best


def _check(value: float | None, minimum: float) -> bool:
    return value is not None and value >= minimum


def score_part_a(
    items: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """A1, A2 and A3 from presence decisions at probability 0.5."""
    truth = [bool(i["truth"]) for i in items]
    predicted = [i["probability"] >= PRESENT_THRESHOLD for i in items]
    a1 = accuracy(truth, predicted)
    per_stem: dict[str, Any] = {}
    for stem in PRODUCT_STEMS:
        chosen = [n for n, i in enumerate(items) if i["stem"] == stem]
        stem_truth = [truth[n] for n in chosen]
        value = accuracy(stem_truth, [predicted[n] for n in chosen])
        minimum = thresholds["A2"]["min_accuracy"][stem]
        per_stem[stem] = {
            "value": value,
            "threshold": minimum,
            "pass": _check(value, minimum),
            "n": len(chosen),
            "present": sum(stem_truth),
            "best_f1": best_f1_threshold(stem_truth, [items[n]["probability"] for n in chosen]),
        }
    vocals = per_stem["vocals"]
    always = vocals["present"] / vocals["n"] if vocals["n"] else None
    return {
        "A1": {
            "value": a1,
            "threshold": thresholds["A1"]["min_accuracy"],
            "pass": _check(a1, thresholds["A1"]["min_accuracy"]),
            "n": len(items),
            "correct": sum(1 for t, p in zip(truth, predicted, strict=True) if t == p),
        },
        "A2": {"per_stem": per_stem, "pass": all(s["pass"] for s in per_stem.values())},
        "A3": {
            "value": vocals["value"],
            "threshold": thresholds["A3"]["min_accuracy"],
            "pass": _check(vocals["value"], thresholds["A3"]["min_accuracy"]),
            "n": vocals["n"],
            "absent": vocals["n"] - vocals["present"],
            # What answering "present" every time would score: how little A3 can show
            # when few excerpts lack vocals.
            "always_present_accuracy": always,
        },
    }


def score_part_b(
    items: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """B1, B2 and B3 from the clean and bleed clips."""
    clean = [i for i in items if i["level"] == 0]
    bleed = [i for i in items if i["level"] > 0]
    hardest = [i for i in items if i["level"] == 1]

    def p(group: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(i["bleed_probability"]) for i in group]

    def rating(item: Mapping[str, Any]) -> float:
        given = item.get("bleed_score")
        return float(item["bleed_probability"] if given is None else given)

    pooled, hard = auc(p(clean), p(bleed)), auc(p(clean), p(hardest))
    rho = spearman([rating(i) for i in items], [float(i["level"]) for i in items])
    flagged = [
        i
        for i in clean
        if i["bleed_probability"] >= PRESENT_THRESHOLD
        or (i.get("bleed_score") is not None and i["bleed_score"] >= 1.0)
    ]
    rate = len(flagged) / len(clean) if clean else None
    b1, b2, b3 = thresholds["B1"], thresholds["B2"], thresholds["B3"]
    return {
        "B1": {
            "auc_pooled": pooled,
            "auc_hardest": hard,
            "threshold_pooled": b1["min_auc_pooled"],
            "threshold_hardest": b1["min_auc_hardest"],
            "pass": _check(pooled, b1["min_auc_pooled"]) and _check(hard, b1["min_auc_hardest"]),
            "clean": len(clean),
            "bleed": len(bleed),
            "hardest": len(hardest),
        },
        "B2": {
            "value": rho,
            "threshold": b2["min_spearman"],
            "pass": _check(rho, b2["min_spearman"]),
            "n": len(items),
            "rated_by": "bleed_score"
            if all(i.get("bleed_score") is not None for i in items)
            else "bleed_probability",
        },
        "B3": {
            "value": rate,
            "threshold": b3["max_false_positive_rate"],
            "pass": rate is not None and rate <= b3["max_false_positive_rate"],
            "clean": len(clean),
            "flagged": len(flagged),
        },
    }


def evaluate(
    part_a: Sequence[Mapping[str, Any]],
    part_b: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Every threshold for one model, and whether Parts A and B pass as wholes."""
    result = {**score_part_a(part_a, thresholds), **score_part_b(part_b, thresholds)}
    result["passes_a"] = all(result[k]["pass"] for k in ("A1", "A2", "A3"))
    result["passes_b"] = all(result[k]["pass"] for k in ("B1", "B2", "B3"))
    return result


def compare(
    clef: Mapping[str, Any], beats: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Threshold C: Clef-omni's margin over BEATs on A1 and B1, in percentage points."""
    minimum = thresholds["C"]["min_margin_points"]

    def margin(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else round((a - b) * 100.0, 6)

    a1 = margin(clef["A1"]["value"], beats["A1"]["value"])
    b1 = margin(clef["B1"]["auc_pooled"], beats["B1"]["auc_pooled"])
    return {
        "A1_margin_points": a1,
        "B1_margin_points": b1,
        "threshold": minimum,
        "pass": _check(a1, minimum) and _check(b1, minimum),
    }


def verdict(clef: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-registered decision rule, applied."""
    if clef["passes_a"] and clef["passes_b"]:
        use = "gating" if comparison["pass"] else "passes, but BEATs preferred"
    elif clef["passes_a"]:
        use = "presence and routing only"
    else:
        use = "rejected"
    return {
        "clef_omni": use,
        "listening_layer": "Clef-omni" if use == "gating" else "BEATs with CLAP as second opinion",
        "passes_a": clef["passes_a"],
        "passes_b": clef["passes_b"],
        "beats_margin_met": comparison["pass"],
    }
