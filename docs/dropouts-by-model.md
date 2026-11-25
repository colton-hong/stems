# Dropouts by model

Written by `python -m stems_bench.dropouts_by_model report` from the run `by-model-v1`, made at commit `af0a950`. No model was run: these are the benchmark's kept outputs, read beside the true stems. What is counted and what is done with it were written into `bench/stems_bench/dropouts_by_model.py` and committed (`af0a950`) before any kept output was read for this.

The default model of every stem was chosen by SDR. SDR hardly moves when a stem falls silent for a while as its instrument plays, and the default model does that (`docs/dropouts.md`, `docs/what-is-lost.md`). This puts the same count beside SDR for every model whose output the benchmark kept, and for each stem's configured two-model blend. A stem is **dropped** for a tenth of a second when the true stem sounds in it and the choice's stem does not; a dropout is **long** from 5 seconds. Only songs that have the instrument are counted.

## The choice that drops least, beside the default

Named on the MoisesDB tuning tracks, where choices are made in this project, and shown on the two test lists. Nothing is decided by it.

| Stem | Drops least on the tuning tracks | List | Its share dropped | Its SDR | The default's share dropped | The default's SDR |
|---|---|---|---:|---:|---:|---:|
| vocals | `blend of mel_roformer_kim_vocals and bs_roformer_viperx_vocals` | MoisesDB tuning | 6.0% | - | not kept | - |
| vocals | `blend of mel_roformer_kim_vocals and bs_roformer_viperx_vocals` | MoisesDB test | 4.7% | - | not kept | - |
| vocals | `blend of mel_roformer_kim_vocals and bs_roformer_viperx_vocals` | MUSDB18-HQ test | 6.3% | - | not kept | - |
| drums | `scnet_xl_ihf` | MoisesDB tuning | 2.7% | 11.26 | not kept | - |
| drums | `scnet_xl_ihf` | MoisesDB test | 2.3% | 11.50 | not kept | - |
| drums | `scnet_xl_ihf` | MUSDB18-HQ test | 1.2% | 11.92 | not kept | - |
| bass | `bs_roformer_sw_6stem` (the default) | MoisesDB tuning | 1.2% | 15.64 | 1.2% | 15.64 |
| bass | `bs_roformer_sw_6stem` (the default) | MoisesDB test | 1.7% | 15.09 | 1.7% | 15.09 |
| bass | `bs_roformer_sw_6stem` (the default) | MUSDB18-HQ test | 3.1% | 12.47 | 3.1% | 12.47 |
| guitar | `blend of bs_roformer_sw_6stem and mel_roformer_guitar_becruily` | MoisesDB tuning | 1.3% | - | 2.0% | 7.85 |
| guitar | `blend of bs_roformer_sw_6stem and mel_roformer_guitar_becruily` | MoisesDB test | 0.9% | - | 1.4% | 9.19 |
| guitar | `blend of bs_roformer_sw_6stem and mel_roformer_guitar_becruily` | MUSDB18-HQ test | - | - | not kept | - |
| piano | `bs_roformer_sw_6stem` (the default) | MoisesDB tuning | 9.2% | 8.63 | 9.2% | 8.63 |
| piano | `bs_roformer_sw_6stem` (the default) | MoisesDB test | 9.6% | 5.80 | 9.6% | 5.80 |
| piano | `bs_roformer_sw_6stem` (the default) | MUSDB18-HQ test | - | - | not kept | - |

## The default against the rest

The default beside whichever other choice drops least on the same songs. This table was added after the run; every figure in it is in a table below.

| Stem | Songs | The default drops | With a long dropout | The least another drops | With a long dropout | That choice | Its SDR | The default's SDR |
|---|---|---:|---:|---:|---:|---|---:|---:|
| vocals | 19 on the product's 20 tracks | 4.0% | 0 | 3.9% | 0 | `bs_roformer_viperx_vocals` | 13.10 | - |
| drums | 19 on the product's 20 tracks | 2.7% | 0 | 2.8% | 2 | `scnet_xl_ihf` | 11.60 | - |
| bass | 48 on MoisesDB test | 1.7% | 1 | 1.7% | 1 | `blend of htdemucs_ft and bs_roformer_sw_6stem` | - | 15.09 |
| guitar | 45 on MoisesDB test | 1.4% | 1 | 0.9% | 0 | `blend of bs_roformer_sw_6stem and mel_roformer_guitar_becruily` | - | 9.19 |
| piano | 20 on MoisesDB test | 9.6% | 6 | 8.8% | 5 | `blend of bs_roformer_sw_6stem and htdemucs_6s` | - | 5.80 |

## MoisesDB test: 50 tracks

| Stem | Choice | Songs with it | True stem sounds (min) | Dropped (min) | Share | With a long dropout | SDR (dB, median) |
|---|---|---:|---:|---:|---:|---:|---:|
| vocals | `blend of mel_roformer_kim_vocals and bs_roformer_viperx_vocals` | 49 | 121.5 | 5.7 | 4.7% | 1 | - |
| vocals | `bs_polarformer_vocals` | 49 | 121.5 | 6.2 | 5.1% | 1 | 10.22 |
| vocals | `bs_roformer_viperx_vocals` | 49 | 121.5 | 5.5 | 4.5% | 2 | 10.78 |
| vocals | `mel_roformer_kim_vocals` | 49 | 121.5 | 6.0 | 4.9% | 1 | 10.82 |
| drums | `blend of scnet_xl_ihf and htdemucs_ft` | 48 | 154.3 | 4.4 | 2.8% | 1 | - |
| drums | `htdemucs_ft` | 48 | 154.3 | 5.4 | 3.5% | 1 | 10.51 |
| drums | `scnet_large` | 48 | 154.3 | 4.0 | 2.6% | 3 | 10.21 |
| drums | `scnet_xl_ihf` | 48 | 154.3 | 3.5 | 2.3% | 4 | 11.50 |
| bass | `blend of htdemucs_ft and bs_roformer_sw_6stem` | 48 | 159.7 | 2.7 | 1.7% | 1 | - |
| bass | `bs_roformer_sw_6stem` (the default) | 48 | 159.7 | 2.8 | 1.7% | 1 | 15.09 |
| bass | `htdemucs_ft` | 48 | 159.7 | 3.0 | 1.9% | 1 | 13.19 |
| bass | `scnet_xl_ihf` | 48 | 159.7 | 2.7 | 1.7% | 1 | 12.99 |
| guitar | `blend of bs_roformer_sw_6stem and mel_roformer_guitar_becruily` | 45 | 159.8 | 1.4 | 0.9% | 0 | - |
| guitar | `bs_roformer_sw_6stem` (the default) | 45 | 159.8 | 2.2 | 1.4% | 1 | 9.19 |
| guitar | `htdemucs_6s` | 45 | 159.8 | 5.5 | 3.4% | 4 | 5.17 |
| guitar | `mel_roformer_guitar_becruily` | 45 | 159.8 | 2.1 | 1.3% | 1 | 5.44 |
| piano | `blend of bs_roformer_sw_6stem and htdemucs_6s` | 20 | 53.2 | 4.7 | 8.8% | 5 | - |
| piano | `bs_roformer_sw_6stem` (the default) | 20 | 53.2 | 5.1 | 9.6% | 6 | 5.80 |
| piano | `htdemucs_6s` | 20 | 53.2 | 19.2 | 36.0% | 13 | 0.23 |

## MoisesDB tuning: 30 tracks

| Stem | Choice | Songs with it | True stem sounds (min) | Dropped (min) | Share | With a long dropout | SDR (dB, median) |
|---|---|---:|---:|---:|---:|---:|---:|
| vocals | `blend of mel_roformer_kim_vocals and bs_roformer_viperx_vocals` | 30 | 72.8 | 4.4 | 6.0% | 1 | - |
| vocals | `bs_polarformer_vocals` | 30 | 72.8 | 4.8 | 6.6% | 2 | 11.20 |
| vocals | `bs_roformer_viperx_vocals` | 30 | 72.8 | 4.4 | 6.1% | 2 | 11.15 |
| vocals | `mel_roformer_kim_vocals` | 30 | 72.8 | 4.8 | 6.6% | 2 | 11.09 |
| drums | `blend of scnet_xl_ihf and htdemucs_ft` | 30 | 91.0 | 2.8 | 3.1% | 1 | - |
| drums | `htdemucs_ft` | 30 | 91.0 | 3.4 | 3.7% | 1 | 10.57 |
| drums | `scnet_large` | 30 | 91.0 | 3.0 | 3.3% | 3 | 10.46 |
| drums | `scnet_xl_ihf` | 30 | 91.0 | 2.5 | 2.7% | 3 | 11.26 |
| bass | `blend of htdemucs_ft and bs_roformer_sw_6stem` | 30 | 90.7 | 1.1 | 1.3% | 1 | - |
| bass | `bs_roformer_sw_6stem` (the default) | 30 | 90.7 | 1.1 | 1.2% | 1 | 15.64 |
| bass | `htdemucs_ft` | 30 | 90.7 | 1.6 | 1.7% | 1 | 12.62 |
| bass | `scnet_xl_ihf` | 30 | 90.7 | 1.8 | 2.0% | 2 | 12.89 |
| guitar | `blend of bs_roformer_sw_6stem and mel_roformer_guitar_becruily` | 27 | 82.4 | 1.1 | 1.3% | 1 | - |
| guitar | `bs_roformer_sw_6stem` (the default) | 27 | 82.4 | 1.6 | 2.0% | 1 | 7.85 |
| guitar | `htdemucs_6s` | 27 | 82.4 | 4.8 | 5.8% | 3 | 3.27 |
| guitar | `mel_roformer_guitar_becruily` | 27 | 82.4 | 1.3 | 1.6% | 1 | 3.59 |
| piano | `blend of bs_roformer_sw_6stem and htdemucs_6s` | 15 | 46.5 | 5.4 | 11.6% | 2 | - |
| piano | `bs_roformer_sw_6stem` (the default) | 15 | 46.5 | 4.3 | 9.2% | 2 | 8.63 |
| piano | `htdemucs_6s` | 15 | 46.5 | 13.1 | 28.2% | 9 | 1.97 |

## MUSDB18-HQ test: 50 tracks

| Stem | Choice | Songs with it | True stem sounds (min) | Dropped (min) | Share | With a long dropout | SDR (dB, median) |
|---|---|---:|---:|---:|---:|---:|---:|
| vocals | `blend of mel_roformer_kim_vocals and bs_roformer_viperx_vocals` | 50 | 131.1 | 8.3 | 6.3% | 1 | - |
| vocals | `bs_polarformer_vocals` | 50 | 131.1 | 9.0 | 6.9% | 4 | 11.86 |
| vocals | `bs_roformer_viperx_vocals` | 50 | 131.1 | 8.2 | 6.3% | 2 | 12.29 |
| vocals | `mel_roformer_kim_vocals` | 50 | 131.1 | 8.9 | 6.8% | 1 | 11.95 |
| drums | `blend of scnet_xl_ihf and htdemucs_ft` | 50 | 174.5 | 2.8 | 1.6% | 1 | - |
| drums | `htdemucs_ft` | 50 | 174.5 | 3.8 | 2.2% | 1 | 10.28 |
| drums | `scnet_large` | 50 | 174.5 | 1.8 | 1.1% | 0 | 10.73 |
| drums | `scnet_xl_ihf` | 50 | 174.5 | 2.1 | 1.2% | 0 | 11.92 |
| bass | `blend of htdemucs_ft and bs_roformer_sw_6stem` | 50 | 155.5 | 2.4 | 1.5% | 3 | - |
| bass | `bs_roformer_sw_6stem` (the default) | 50 | 155.5 | 4.8 | 3.1% | 6 | 12.47 |
| bass | `htdemucs_ft` | 50 | 155.5 | 4.8 | 3.1% | 7 | 9.95 |
| bass | `scnet_xl_ihf` | 50 | 155.5 | 3.8 | 2.4% | 4 | 10.60 |

## The default where the benchmark did not keep it

For these stems the default model's output was not kept. Its row is from the product's own run on the first twenty test tracks (`docs/what-is-lost.md`), and the other choices are counted on the same tracks. The product's run does not carry SDR.

| Stem | Choice | Songs with it | True stem sounds (min) | Dropped (min) | Share | With a long dropout | SDR (dB, median) |
|---|---|---:|---:|---:|---:|---:|---:|
| vocals | `bs_roformer_sw_6stem` (the default) | 19 | 48.4 | 1.9 | 4.0% | 0 | - |
| vocals | `blend of mel_roformer_kim_vocals and bs_roformer_viperx_vocals` | 19 | 48.4 | 2.1 | 4.2% | 0 | - |
| vocals | `bs_polarformer_vocals` | 19 | 48.4 | 2.2 | 4.5% | 0 | 12.80 |
| vocals | `bs_roformer_viperx_vocals` | 19 | 48.4 | 1.9 | 3.9% | 0 | 13.10 |
| vocals | `mel_roformer_kim_vocals` | 19 | 48.4 | 2.2 | 4.6% | 0 | 12.60 |
| drums | `bs_roformer_sw_6stem` (the default) | 19 | 60.0 | 1.6 | 2.7% | 0 | - |
| drums | `blend of scnet_xl_ihf and htdemucs_ft` | 19 | 60.0 | 2.1 | 3.5% | 0 | - |
| drums | `htdemucs_ft` | 19 | 60.0 | 2.7 | 4.5% | 0 | 10.77 |
| drums | `scnet_large` | 19 | 60.0 | 2.2 | 3.6% | 1 | 10.85 |
| drums | `scnet_xl_ihf` | 19 | 60.0 | 1.7 | 2.8% | 2 | 11.60 |

## What this is and is not

- A count, and it decides nothing: the defaults are as Phase 10's rule left them.
- Only the outputs the benchmark kept, which are the members of its blends: a model that is in no blend for a stem is not here. No output was kept for: other.
- Dropped time says a stem is silent, not what is in it when it sounds: a model can drop nothing because it lets everything through. SDR is beside it for that reason.
- A blend is silent only where all of its members are. Its SDR is not shown: the benchmark's scores for blends are in its own reports.
- SDR is this benchmark's, a median over the songs counted here. It is not comparable with figures published elsewhere.
- Nothing in the product was changed.
