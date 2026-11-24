# The product beside the benchmark

Generated from `runs/product/product-v1/results/` (finished 2026-11-24T14:07:54+00:00, commit `5eec3568b9cf, with uncommitted changes`). Reproduce with `python -m stems_bench.product_check run --tracks 3 --device cpu` and then `report`; that needs MoisesDB and the model weights.

The benchmark runs each model directly on a track. This sends the same tracks down the road a user's upload takes (a file in storage, the worker, its plan, the gate and its retries, FLAC files back in storage) with the real models, and scores what comes out with the benchmark's own scorer against the same references. No Jev key was used, so the rules decided. The audio was deleted after each track was scored.

Tracks: the first 3 of the frozen `moisesdb_subset` list. Device here: cpu; the benchmark's figures are from this Mac's GPU (`docs/benchmarks/moisesdb.md`), so a small difference between the two columns is expected (`docs/benchmarks/device-spread-synthetic.md`).

## Stem by stem

SDR in dB, the median over one-second windows, higher is better. "Benchmark" is the benchmark's figure for the same model on the same track, where one model made the stem and the benchmark scored that output against the same stem. "In song" is whether the reference sounds for at least 5 seconds.

| Track | Stem | Made by | Attempts | In song | Product SDR | Benchmark SDR | Difference | Gate's note |
|---|---|---|---:|---|---:|---:|---:|---|
| `014f3712` | vocals | `mel_roformer_kim_vocals` | 1 | yes | 11.25 | 11.25 | 0.00 | not checked |
| `014f3712` | drums | `scnet_xl_ihf` | 1 | yes | 15.87 | 15.87 | 0.00 | not checked |
| `014f3712` | bass | `htdemucs_ft` | 1 | yes | 16.15 | 16.15 | 0.00 | not checked |
| `014f3712` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 9.56 | 9.56 | -0.00 | not checked |
| `014f3712` | piano | `bs_roformer_sw_6stem` | 3 | no | - | - | - | probably empty |
| `014f3712` | other | `scnet_xl_ihf` | 1 | yes | 3.40 | - | - | not checked |
| `01c8ba69` | vocals | `mel_roformer_kim_vocals` | 1 | yes | 10.82 | 10.82 | 0.00 | not checked |
| `01c8ba69` | drums | `scnet_xl_ihf` | 1 | yes | 11.52 | 11.52 | 0.00 | not checked |
| `01c8ba69` | bass | `htdemucs_ft` | 1 | yes | 10.28 | 10.28 | -0.00 | not checked |
| `01c8ba69` | guitar | `mel_roformer_guitar_becruily` | 2 | no | - | - | - | not checked |
| `01c8ba69` | piano | `bs_roformer_sw_6stem` | 1 | yes | 6.53 | 6.53 | -0.00 | not checked |
| `01c8ba69` | other | `scnet_xl_ihf` | 1 | yes | 2.13 | - | - | not checked |
| `028c6f6a` | vocals | `mel_roformer_kim_vocals` | 1 | yes | 7.35 | 7.27 | 0.07 | not checked |
| `028c6f6a` | drums | `scnet_xl_ihf` | 1 | yes | 9.54 | 9.54 | 0.00 | not checked |
| `028c6f6a` | bass | `htdemucs_ft` | 1 | yes | 14.35 | 14.35 | -0.00 | not checked |
| `028c6f6a` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 5.87 | 5.87 | 0.00 | not checked |
| `028c6f6a` | piano | `bs_roformer_sw_6stem` | 3 | no | - | - | - | probably empty |
| `028c6f6a` | other | `scnet_xl_ihf` | 1 | yes | 1.46 | - | - | not checked |

A stem with no SDR has a reference that is silent throughout: there is nothing to score it against.

## Does the product deliver what the benchmark measured?

12 delivered stems could be set beside a benchmark figure. 12 of them are within 0.5 dB of it. The median difference (product minus benchmark) is 0.00 dB and the largest 0.07 dB.

## Did the gate flag the right stems?

Of 3 stems whose instrument is not in the song, the gate flagged 2 as probably empty. Of 15 stems whose instrument is in the song, it flagged 0.
- `01c8ba69` guitar: not in the song, not flagged: an earlier attempt was found empty, the retry with `mel_roformer_guitar_becruily` was not, and the retry is what was delivered, without the note. The song has no such instrument, so what the retry found is something else.

## Time

| Track | Audio | Wall time | Models | Tagger | Everything else |
|---|---:|---:|---:|---:|---:|
| `014f3712` | 344 s | 1820 s | 1765 s | 44 s | 11 s |
| `01c8ba69` | 249 s | 1298 s | 1261 s | 30 s | 8 s |
| `028c6f6a` | 240 s | 1600 s | 1541 s | 36 s | 24 s |

All on cpu, on a machine that was also running a benchmark and at times this project's tests; these are single runs, not speed measurements. "Models" counts each model once (a blend that reuses an output costs nothing more). "Everything else" is decoding, the gate's level checks, writing FLAC and peaks, and the zip. Scoring is not included.

## Limits

- 3 tracks. That is enough to see whether the road works and far too few to say how often it does.
- One dataset, the product's default models, no Jev.
- The worker's entry point was called directly. The API, the browser and the download links are not in this path; `make web-e2e-real` covers those on a synthetic signal.
