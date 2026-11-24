# Default models: a single model per stem

Written by `python -m stems_bench.recommend`. The choice is made on the tuning tracks only (`moisesdb_tuning`, run `moisesdb-tuning-v1`, `docs/benchmarks/tuning-models-moisesdb.md`), by a rule that was committed before those scores were looked at. The test lists are shown afterwards and decided nothing. Blends are not chosen here.

## The rule

1. A model is a candidate for a stem if the benchmark scored its own output of that name against that stem on at least 10 tuning tracks.
2. The best candidate has the highest median SDR. Every candidate within 0.3 dB of it is acceptable.
3. Among acceptable candidates the fewest models win: the model acceptable for the most stems is taken first. Ties go to the one that gives up the least SDR, then to the faster one.

## The choice

SDR in dB, median over the tuning tracks, on this Mac's GPU.

| Stem | Chosen | Its SDR | Best on the tuning tracks | Best SDR | Given up | In use until now | Its SDR |
|---|---|---:|---|---:|---:|---|---:|
| vocals | `bs_roformer_sw_6stem` | 11.48 | `bs_roformer_sw_6stem` | 11.48 | 0.00 | `mel_roformer_kim_vocals` | 11.09 |
| drums | `bs_roformer_sw_6stem` | 12.14 | `bs_roformer_sw_6stem` | 12.14 | 0.00 | `scnet_xl_ihf` | 11.26 |
| bass | `bs_roformer_sw_6stem` | 15.64 | `bs_roformer_sw_6stem` | 15.64 | 0.00 | `htdemucs_ft` | 12.62 |
| guitar | `bs_roformer_sw_6stem` | 7.85 | `bs_roformer_sw_6stem` | 7.85 | 0.00 | `bs_roformer_sw_6stem` | 7.85 |
| piano | `bs_roformer_sw_6stem` | 8.63 | `bs_roformer_sw_6stem` | 8.63 | 0.00 | `bs_roformer_sw_6stem` | 8.63 |
| other | `bs_roformer_sw_6stem` | 2.61 | `bs_roformer_sw_6stem` | 2.61 | 0.00 | `scnet_xl_ihf` | - |

A dash under "In use until now" means the benchmark did not score that model's own output against this stem (a four-stem model's "other" is a different thing from the six-stem one).

Every candidate, per stem:

| Stem | Model | SDR | Tracks | s per min of audio | Acceptable |
|---|---|---:|---:|---:|---|
| vocals | `bs_roformer_sw_6stem` | 11.48 | 30 | 18.25 | yes |
| vocals | `bs_polarformer_vocals` | 11.20 | 30 | 29.97 | yes |
| vocals | `bs_roformer_viperx_vocals` | 11.15 | 30 | 29.88 | no |
| vocals | `mel_roformer_kim_vocals` | 11.09 | 30 | 12.51 | no |
| vocals | `scnet_xl_ihf` | 11.02 | 30 | 25.25 | no |
| vocals | `scnet_large` | 9.29 | 30 | 7.83 | no |
| vocals | `htdemucs_ft` | 8.78 | 30 | 8.24 | no |
| vocals | `htdemucs_6s` | 8.65 | 30 | 3.77 | no |
| drums | `bs_roformer_sw_6stem` | 12.14 | 30 | 18.25 | yes |
| drums | `scnet_xl_ihf` | 11.26 | 30 | 25.25 | no |
| drums | `htdemucs_ft` | 10.57 | 30 | 8.24 | no |
| drums | `scnet_large` | 10.46 | 30 | 7.83 | no |
| drums | `htdemucs_6s` | 10.29 | 30 | 3.77 | no |
| bass | `bs_roformer_sw_6stem` | 15.64 | 30 | 18.25 | yes |
| bass | `scnet_xl_ihf` | 12.89 | 30 | 25.25 | no |
| bass | `htdemucs_ft` | 12.62 | 30 | 8.24 | no |
| bass | `htdemucs_6s` | 12.33 | 30 | 3.77 | no |
| bass | `scnet_large` | 11.58 | 30 | 7.83 | no |
| guitar | `bs_roformer_sw_6stem` | 7.85 | 27 | 18.25 | yes |
| guitar | `mel_roformer_guitar_becruily` | 3.59 | 27 | 6.54 | no |
| guitar | `htdemucs_6s` | 3.27 | 27 | 3.77 | no |
| piano | `bs_roformer_sw_6stem` | 8.63 | 15 | 18.25 | yes |
| piano | `htdemucs_6s` | 1.97 | 15 | 3.77 | no |
| other | `bs_roformer_sw_6stem` | 2.61 | 27 | 18.25 | yes |
| other | `htdemucs_6s` | 0.01 | 27 | 3.77 | no |

## The configured blends beside it

The config also names a two-model blend and a full ensemble for each stem, with equal weights; nobody has tuned them yet (P2.12). On the same tuning tracks, against the chosen single model:

| Stem | Chosen model | Two-model blend | Against it | Full ensemble | Against it |
|---|---:|---:|---:|---:|---:|
| vocals | 11.48 | 11.38 | -0.10 | 11.51 | +0.03 |
| drums | 12.14 | 11.26 | -0.88 | 11.22 | -0.92 |
| bass | 15.64 | 14.73 | -0.91 | 13.77 | -1.87 |
| guitar | 7.85 | 6.36 | -1.49 | 5.68 | -2.17 |
| piano | 8.63 | 5.70 | -2.92 | 5.70 | -2.92 |

The most any of them gains on the single model is +0.03 dB, and each costs more models. The six-stem `other` has no blend row here: its blends are four-stem ones, scored against a different target.

## Time

Seconds to run each model once per minute of audio, from the same run:

- chosen: `bs_roformer_sw_6stem`: 18.3 s
- in use until now: `mel_roformer_kim_vocals`, `scnet_xl_ihf`, `htdemucs_ft`, `bs_roformer_sw_6stem`: 64.3 s

## What the test lists show

These tracks took no part in the choice. SDR in dB, median over the list's tracks, for the chosen model and for the one in use until now.

| Stem | List | Chosen | In use until now | Difference | Tracks |
|---|---|---:|---:|---:|---:|
| vocals | `moisesdb_subset` | 10.55 | 10.82 | -0.27 | 49 |
| vocals | `musdb18hq_test` | 12.30 | 11.95 | +0.34 | 50 |
| drums | `moisesdb_subset` | 12.95 | 11.50 | +1.44 | 48 |
| drums | `musdb18hq_test` | 13.35 | 11.92 | +1.43 | 50 |
| bass | `moisesdb_subset` | 15.09 | 13.19 | +1.90 | 48 |
| bass | `musdb18hq_test` | 12.47 | 9.95 | +2.52 | 50 |
| guitar | `moisesdb_subset` | 9.19 | 9.19 | +0.00 | 45 |
| piano | `moisesdb_subset` | 5.80 | 5.80 | +0.00 | 20 |
| other | `moisesdb_subset` | 2.97 | 1.99 (composed) | +0.98 | 33 |

"Composed" is the six-stem other the product has been delivering: a four-stem model's other with the guitar and piano stems taken out, which the benchmark scores as its own system. MUSDB18-HQ has no guitar or piano references, so those stems appear for MoisesDB only. Test runs: `moisesdb_subset` from run `moisesdb-v1`, `musdb18hq_test` from run `musdb18hq-v1`.

## The vocal-split model

The same rule on the two outputs of the vocal split, lead and backing vocals. In the benchmark these models are fed the reference vocals stem, not a separated one, and only tracks that have both kinds of vocal are scored.

| Output | Chosen | Its SDR | Tracks | In use until now | Its SDR |
|---|---|---:|---:|---|---:|
| lead_vocals | `mel_roformer_karaoke_aufr33_viperx` | 12.32 | 14 | `mel_roformer_karaoke_aufr33_viperx` | 12.32 |
| backing_vocals | `mel_roformer_karaoke_aufr33_viperx` | 4.92 | 14 | `mel_roformer_karaoke_aufr33_viperx` | 4.92 |

Every candidate:

| Output | Model | SDR | Tracks |
|---|---|---:|---:|
| lead_vocals | `mel_roformer_karaoke_aufr33_viperx` | 12.32 | 14 |
| lead_vocals | `uvr_mdxnet_kara_2` | 10.99 | 14 |
| lead_vocals | `uvr_vr_5hp_karaoke` | 10.06 | 14 |
| backing_vocals | `mel_roformer_karaoke_aufr33_viperx` | 4.92 | 14 |
| backing_vocals | `uvr_vr_5hp_karaoke` | 1.09 | 14 |
| backing_vocals | `uvr_mdxnet_kara_2` | 0.84 | 14 |

**No change.** The rule names `mel_roformer_karaoke_aufr33_viperx` for both outputs, which is the model already in use.

On the test lists, which decided nothing:

| Output | List | Chosen | In use until now | Tracks |
|---|---|---:|---:|---:|
| lead_vocals | `moisesdb_subset` | 10.53 | 10.53 | 26 |
| backing_vocals | `moisesdb_subset` | 4.28 | 4.28 | 26 |

## Limits

- One tuning list of 30 tracks from one dataset decides. MUSDB18-HQ's training tracks have not been scored yet.
- Only the models the config named when the benchmark started are candidates.
- Which songs each checkpoint was trained on is not known here. A model that met these datasets' songs in training would score higher on them than on new music, and nothing in this repository can tell.
- SDR is not how it sounds. No listening test has been run.
- The de-reverb model is not chosen here (no list has reverberant vocals to score it on), and neither are blends.
