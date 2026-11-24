# The listening layer on real separator output

Generated from `runs/calibration/gate-outputs-v1/scores.jsonl` (scored 2026-11-24T11:38:31+00:00, tagger `beats_iter3plus_as2m_cpt2` on cpu, commit `4ac33f743007`). Reproduce with `python -m stems_bench.gate_outputs collect` and then `report`; that needs the benchmark's kept outputs under `runs/bench/estimates/`, which are not in the repository.

The two questions, the rules tried, how each threshold is chosen and the bar each must clear were written into `bench/stems_bench/gate_outputs.py` at commit `5b63f0807f6e`, before any of these outputs was scored.

P7.1 (`docs/gate-calibration.md`) tested the tagger on reference stems with another stem mixed in by hand. This tests it on what the gate really sees: outputs of real separators, with their quality as the benchmark measured it.

## What was scored

| Set | Tracks | Models | Outputs | Of them empty |
|---|---:|---:|---:|---:|
| Calibration (`moisesdb_tuning`) | 30 | 8 | 360 | 26 |
| Held out (`moisesdb_subset`) | 50 | 9 | 700 | 111 |

Each output is cut to the loudest 60 seconds of its track. "Empty" means the reference stem is not present in that minute: the song has no such instrument there, and the output is leftovers. Only outputs the benchmark kept are here (the members of its blends), so weak models are under-represented.

Models, held out: `bs_polarformer_vocals`, `bs_roformer_sw_6stem`, `bs_roformer_viperx_vocals`, `htdemucs_6s`, `htdemucs_ft`, `mel_roformer_guitar_becruily`, `mel_roformer_kim_vocals`, `scnet_large`, `scnet_xl_ihf`.

## Question 1: can an empty stem be told from a real one?

Each rule calls an output empty when its signal is at or below the threshold. "Right" is how many of the outputs it calls empty are; "found" is how many of the empty outputs it calls empty. AUC is the chance that a real stem's signal is above an empty one's.

| Rule | Threshold | AUC | Called empty | Right | Found | Result |
|---|---:|---:|---:|---:|---:|---|
| Level: the output is this far below the mix, or further | -19.8 dB | 0.997 | 103 | 95.1% | 88.3% | pass |
| Tagger: own-instrument score at most the configured `absent_max` | 0.0207 | 0.951 | 99 | 87.9% | 78.4% | fail |
| Tagger: own-instrument score, threshold chosen on these outputs | 0.0002 | 0.951 | 79 | 100.0% | 71.2% | pass |

Needed on the held-out outputs: right at least 90.0%, found at least 50.0%, with at least 10 outputs called empty and 10 empty ones. There were 111 empty outputs among 700.

For comparison, the rule the gate already acts on (an output that is silent for at least 98.0% of its length is 'mostly silent') calls 77 of these outputs empty; 77 of those are, which is 69.4% of the empty ones.

The same rules on the outputs their thresholds were chosen on:

| Rule | AUC | Called empty | Right | Found |
|---|---:|---:|---:|---:|
| Level: the output is this far below the mix, or further | 1.000 | 26 | 96.2% | 96.2% |
| Tagger: own-instrument score at most the configured `absent_max` | 0.987 | 29 | 75.9% | 84.6% |
| Tagger: own-instrument score, threshold chosen on these outputs | 0.987 | 22 | 95.5% | 80.8% |

## Question 2: of two outputs of the same stem, which is better?

Pairs of outputs of one stem of one track from two models, whose measured SDR in the excerpt differs by at least 3 dB. A rule is right when it prefers the one with the higher SDR. Guessing gives 50%.

| Rule | Pairs | Prefers the better one |
|---|---:|---:|
| Lower bleed score (the rule on trial) | 91 | 64.8% |
| Higher own-instrument score | 91 | 62.6% |
| Higher own-instrument score minus bleed score | 91 | 67.0% |
| Louder output | 91 | 72.5% |

Needed: at least 80.0% on at least 30 pairs, for the bleed score. The other rules are shown for comparison and decide nothing.

The bleed score by stem:

| Stem | Pairs | Prefers the better one |
|---|---:|---:|
| vocals | 4 | 100.0% |
| drums | 4 | 50.0% |
| bass | 18 | 55.6% |
| guitar | 54 | 66.7% |
| piano | 11 | 63.6% |

## What the configured thresholds say about real outputs (held out)

Outputs whose reference is present, by stem. "Called present" and "called absent" use `listening.presence`; "called bled" is a bleed score at or above `listening.bleed.moderate_min`. The last two columns are rank correlations of the bleed score with the measured SIR and SDR: a bleed score that worked would be strongly negative in both.

| Stem | Outputs | Median SDR | Median SIR | Median own score | Called present | Called absent | Median bleed score | Called bled | Bleed vs SIR | Bleed vs SDR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocals | 141 | 10.34 | 24.73 | 0.378 | 18 | 0 | 0.003 | 0 | -0.44 | -0.52 |
| drums | 144 | 10.55 | 21.66 | 0.711 | 84 | 2 | 0.024 | 0 | 0.09 | 0.05 |
| bass | 144 | 13.24 | 28.25 | 0.828 | 107 | 7 | 0.602 | 95 | -0.07 | -0.10 |
| guitar | 132 | 6.16 | 14.35 | 0.572 | 48 | 0 | 0.063 | 0 | -0.04 | -0.03 |
| piano | 28 | 3.93 | 14.42 | 0.212 | 6 | 3 | 0.161 | 0 | -0.35 | -0.23 |

## Verdict

**Empty stems: the rule to use is `level`** (level: the output is this far below the mix, or further; threshold -19.8 dB). It is the first of the three, in the order fixed beforehand, that cleared the bar.

**The better of two: failed.** The bleed score preferred the better output in 64.8% of 91 pairs (80.0% needed). A retry chosen on the tagger's word would not reliably pick the better stem, so the gate keeps not acting on it.

## The empty-stem rule against the whole song

Question 1 called an output empty when its reference is absent in the loudest minute, and the gate's note for such a stem says the instrument may not be in the song. Here the truth is the whole song: the instrument is in it if its reference sounds for at least 5 seconds in total, anywhere. The rules below and what is done with the result were added to this file's code at commit `fcd175726d06`, after the results above were known and before any whole-song figure was computed. No tagger is involved; the levels were read from the same kept outputs.

Of the 700 held-out outputs, 111 have a reference that is absent in the loudest minute, and 90 have an instrument that is not in the song at all. The difference, 21 outputs, is instruments that play only outside the loudest minute; the rule in force calls 13 of them empty.

| Rule | Threshold | AUC | Called empty | Right | Found | Result |
|---|---:|---:|---:|---:|---:|---|
| In force: level against the mix over the mix's loudest minute | -19.8 dB | 0.989 | 103 | 82.5% | 94.4% | fail |
| Level against the mix over the whole song | -21.7 dB | 0.996 | 92 | 90.2% | 92.2% | pass |

"Right" and "found" are against the whole-song truth. Needed, as before: right at least 90.0%, found at least 50.0%.

**The rule is replaced.** Measured over the loudest minute it fails against the whole song; measured over the whole song it passes, and that is what the gate now uses.

## Limits

- One minute per track, not the whole stem; a job scores the whole stem.
- Kept outputs only: the models the blends are made of. A model bad enough to be left out of every blend is not here.
- One dataset (MoisesDB), one tagger, outputs made on this Mac's GPU.
- "Empty" is decided from the reference in that minute. An instrument that plays elsewhere in the song counts as absent here.
