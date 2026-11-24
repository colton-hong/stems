# Gate calibration: the listening thresholds

Generated from `runs/calibration/gate-v1/scores.jsonl` (scored 2026-11-24T10:45:59+00:00, tagger `beats_iter3plus_as2m_cpt2` on cpu, commit `603308b67da8`, with uncommitted changes). Reproduce with `python -m stems_bench.calibrate collect` and then `report`.

The rules that pick each threshold, the bleed level each mixing ratio counts as, and the bar for switching the gate on were written into `bench/stems_bench/calibrate.py` at commit `7d58a494a794`, before any of these clips was scored.

## What was scored

| Set | Tracks | Mix excerpts | Stem excerpts (clean and bled) |
|---|---:|---:|---:|
| Calibration (`moisesdb_tuning`) | 30 | 30 | 450 |
| Held out (`moisesdb_subset`) | 50 | 50 | 740 |

Each excerpt is the loudest 60 seconds of its track. From each track up to 3 stems were taken clean and with another stem of the same track mixed in at 30, 20, 12 and 6 dB below it, counted as slight, moderate, moderate and severe bleed.

## Chosen values

| Setting | Was (a guess) | Chosen on the calibration set |
|---|---:|---:|
| `listening.presence.present_min` | 0.5 | 0.6466 |
| `listening.presence.absent_max` | 0.2 | 0.0207 |
| `listening.bleed.slight_min` | 0.2 | 0.0347 |
| `listening.bleed.moderate_min` | 0.5 | 0.0405 |
| `listening.bleed.severe_min` | 0.8 | 0.0418 |

## Presence: is the instrument there?

The layer answers present, absent, or cannot tell. "Committed" is how often it gives one of the first two; "right" is how often that answer is correct.

| Where | Decisions | Committed | Right when committed | Said present | Said absent |
|---|---:|---:|---:|---:|---:|
| Stems, calibration | 900 | 58.3% | 95.0% | 142 | 383 |
| Mixes, calibration | 180 | 55.6% | 29.0% | 0 | 100 |
| Stems, held out | 1480 | 57.5% | 97.1% | 232 | 619 |
| Mixes, held out | 300 | 56.7% | 42.9% | 1 | 169 |

## Bleed: how much of another instrument is in the stem?

| Set | Clips | AUC | Balanced accuracy | Unacceptable caught | Acceptable passed | Clean flagged | Exact level |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibration | 450 | 0.628 | 63.2% | 80.4% | 46.1% | 48.9% | 28.7% |
| Held out | 740 | 0.644 | 61.4% | 77.3% | 45.6% | 50.0% | 26.8% |

Held out, by true level: the median bleed score, and what the chosen thresholds call it.

| True level | Clips | Median score | Called none | slight | moderate | severe |
|---|---:|---:|---:|---:|---:|---:|
| none | 148 | 0.0404 | 70 | 4 | 0 | 74 |
| slight | 148 | 0.0587 | 57 | 4 | 0 | 87 |
| moderate | 296 | 0.1099 | 67 | 11 | 1 | 217 |
| severe | 148 | 0.1771 | 19 | 4 | 2 | 123 |

The trade-off on the held-out set: move the moderate threshold and this is what is caught and what is wrongly flagged.

| Threshold | Unacceptable caught | Acceptable passed | Balanced accuracy |
|---:|---:|---:|---:|
| 0.0100 | 95.7% | 21.3% | 58.5% |
| 0.0200 | 89.2% | 29.4% | 59.3% |
| 0.0405 (chosen) | 77.3% | 45.6% | 61.4% |
| 0.0500 | 72.3% | 51.0% | 61.7% |
| 0.1000 | 56.8% | 66.2% | 61.5% |
| 0.2000 | 36.0% | 77.0% | 56.5% |
| 0.3000 | 26.4% | 79.7% | 53.0% |
| 0.5000 | 13.3% | 87.8% | 50.6% |
| 0.8000 | 0.5% | 99.7% | 50.1% |

## Against the values used until now (held out)

| Thresholds | Stems: committed | right | Mixes: committed | right | Bleed balanced accuracy | Clean flagged |
|---|---:|---:|---:|---:|---:|---:|
| Until now | 83.2% | 85.2% | 90.7% | 29.0% | 50.6% | 11.5% |
| Chosen | 57.5% | 97.1% | 56.7% | 42.9% | 61.4% | 50.0% |

On mixes neither set of thresholds works: an instrument that is playing in a full mix scores far lower than the same instrument on its own stem, so "absent" is usually the wrong answer there. The thresholds were chosen on stems, which is what the gate judges.

## The bar for switching the gate on (held out, stems)

| Check | Needed | Measured | Result |
|---|---:|---:|---|
| Presence answers that are right, at least | 90.0% | 97.1% | pass |
| Presence answered at all, at least | 50.0% | 57.5% | pass |
| Bleed AUC, at least | 0.80 | 0.644 | fail |
| Bleed balanced accuracy, at least | 75.0% | 61.4% | fail |
| Clean stems flagged, at most | 10.0% | 50.0% | fail |

## Verdict

3 of 5 checks failed. `listening.calibrated` stays **false**: the gate keeps delivering stems it cannot vouch for with a note, and does not retry on the tagger's word.
