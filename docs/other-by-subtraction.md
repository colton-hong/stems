# `other`, made the other way

Written by `python -m stems_bench.other_made report` from the run `other-made-v1`, made at commit `3b37fcd`. No model was run: the stems here are put together from the benchmark's kept outputs and read beside the true stems. What is counted and what is done with it were written into `bench/stems_bench/other_made.py` and committed (`3b37fcd`) before any kept output was read for this.

The product can make `other` in two ways from models. It can take a six-stem model's own (`bs_roformer_sw_6stem`, what it does now). Or it can take a four-stem model's `other`, which still holds the guitar and the piano, and subtract the guitar and piano stems from it (what it did before Phase 10). The first is silent for a seventh of the time the true `other` sounds (`docs/what-is-lost.md`). This counts the second, for every four-stem `other` the benchmark kept, with the default model's guitar and piano taken out.

A stem is **dropped** for a tenth of a second when the true stem sounds in it and the made one does not; a dropout is **long** from 5 seconds. A song **has an `other`** when its true stem sounds for 5 seconds or more. Where it has none, the table gives how far the made stem is below the song and how often the product's own rule would note it as silent or very quiet: there, anything in the stem is not `other`.

## Beside the default's own, on the product's 20 tracks

The default's own `other` was not kept by the benchmark. Its row is from the product's own run (`docs/what-is-lost.md`), with the SDR that run measured; the run did not record the stem's level, so that cell is empty.

| How `other` is made | Songs with an `other` | True stem sounds (min) | Dropped (min) | Share | With a long dropout | SDR (dB, median) | Songs with none | There: dB under the song (median) | There: noted as empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_roformer_sw_6stem`'s own (the default) | 16 | 34.0 | 5.5 | 16.1% | 6 | 2.80 | 4 | - | 4 of 4 |
| `htdemucs_ft minus guitar and piano` | 16 | 34.0 | 0.5 | 1.3% | 0 | - | 4 | 18.8 | 0 of 4 |
| `scnet_large minus guitar and piano` | 16 | 34.0 | 0.1 | 0.2% | 0 | - | 4 | 18.2 | 0 of 4 |
| `scnet_xl_ihf minus guitar and piano` | 16 | 34.0 | 0.4 | 1.2% | 0 | 1.27 | 4 | 21.7 | 3 of 4 |

## MoisesDB test: 50 tracks

| How `other` is made | Songs with an `other` | True stem sounds (min) | Dropped (min) | Share | With a long dropout | SDR (dB, median) | Songs with none | There: dB under the song (median) | There: noted as empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `htdemucs_ft minus guitar and piano` | 33 | 79.1 | 0.7 | 0.8% | 0 | - | 17 | 17.2 | 0 of 17 |
| `scnet_large minus guitar and piano` | 33 | 79.1 | 0.1 | 0.2% | 0 | - | 17 | 17.6 | 1 of 17 |
| `scnet_xl_ihf minus guitar and piano` | 33 | 79.1 | 0.7 | 0.9% | 0 | 1.99 | 17 | 20.2 | 5 of 17 |

## MoisesDB tuning: 30 tracks

| How `other` is made | Songs with an `other` | True stem sounds (min) | Dropped (min) | Share | With a long dropout | SDR (dB, median) | Songs with none | There: dB under the song (median) | There: noted as empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `htdemucs_ft minus guitar and piano` | 27 | 66.3 | 0.9 | 1.4% | 2 | - | 3 | 17.9 | 1 of 3 |
| `scnet_large minus guitar and piano` | 27 | 66.3 | 0.3 | 0.4% | 0 | - | 3 | 16.2 | 1 of 3 |
| `scnet_xl_ihf minus guitar and piano` | 27 | 66.3 | 0.9 | 1.4% | 1 | - | 3 | 20.8 | 1 of 3 |

## What this is and is not

- A count, and it decides nothing: `other` stays the default model's own (BLOCKED.md D15).
- The guitar and piano taken out are the default model's, as the product would take them out today. Before Phase 10 they were the same model's.
- A stem that is never silent has not thereby got the instrument: subtraction leaves in `other` whatever the models disagree about. The columns for songs with no `other`, and SDR, are there for that.
- SDR is this benchmark's and this product run's, a median over the songs counted in the row. It is shown only where one of them measured exactly that stem.
- One dataset. Nothing in the product was changed.
