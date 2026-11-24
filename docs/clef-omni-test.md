# Clef-omni against BEATs: test report

Generated from `runs/clef_test/clef-omni-v1/` (finished 2026-11-24T08:54:44+00:00), pre-registration version 1 at commit `705155b69ea9`. Reproduce with `python -m stems_bench.clef.run`.

## Clef-omni

| Test | Measured | Needed | Result |
|---|---|---|---|
| A1 presence accuracy, 300 decisions | 81.3% | 85.0% | fail |
| A2 vocals (44 of 50 present) | 96.0% | 80.0% | pass |
| A2 drums (48 of 50 present) | 86.0% | 80.0% | pass |
| A2 bass (48 of 50 present) | 80.0% | 80.0% | pass |
| A2 guitar (45 of 50 present) | 92.0% | 80.0% | pass |
| A2 piano (14 of 50 present) | 76.0% | 80.0% | fail |
| A2 other (24 of 50 present) | 58.0% | 70.0% | fail |
| A3 vocals (6 of 50 absent; always answering 'present' scores 88.0%) | 96.0% | 90.0% | pass |
| B1 AUC, 20 clean against 80 bleed | 0.901 | 0.850 | fail |
| B1 AUC, clean against +30 dB SIR | 0.757 | 0.800 | |
| B2 Spearman, bleed_score against bleed level | 0.703 | 0.700 | pass |
| B3 clean clips called bleeding (4 of 20) | 20.0% | at most 10.0% | fail |

183 of 183 requests answered, 137930 input tokens.

Exploratory, not part of the decision: lead or backing answered correctly on 39.1% of 23 clips; reverb clips against their clean versions, AUC 0.680 (5 pairs).

## BEATs

| Test | Measured | Needed | Result |
|---|---|---|---|
| A1 presence accuracy, 300 decisions | 26.0% | 85.0% | fail |
| A2 vocals (44 of 50 present) | 12.0% | 80.0% | fail |
| A2 drums (48 of 50 present) | 4.0% | 80.0% | fail |
| A2 bass (48 of 50 present) | 4.0% | 80.0% | fail |
| A2 guitar (45 of 50 present) | 12.0% | 80.0% | fail |
| A2 piano (14 of 50 present) | 72.0% | 80.0% | fail |
| A2 other (24 of 50 present) | 52.0% | 70.0% | fail |
| A3 vocals (6 of 50 absent; always answering 'present' scores 88.0%) | 12.0% | 90.0% | fail |
| B1 AUC, 20 clean against 80 bleed | 0.642 | 0.850 | fail |
| B1 AUC, clean against +30 dB SIR | 0.598 | 0.800 | |
| B2 Spearman, bleed_probability against bleed level | 0.231 | 0.700 | fail |
| B3 clean clips called bleeding (2 of 20) | 10.0% | at most 10.0% | pass |

## Clef-omni against BEATs (threshold C)

Margin on A1: 55.333333 points. Margin on B1: 25.875 points. Needed on both: 5.0. Result: pass.

## Verdict

Clef-omni: **rejected**. Listening layer: **BEATs with CLAP as second opinion**.
