# The product beside the benchmark

Generated from `runs/product/product-gpu-v1/results/` (finished 2026-11-24T18:00:00+00:00, commit `d0405ae2199e`). Reproduce with `python -m stems_bench.product_check run --tracks 3 --device mps` and then `report`; that needs MoisesDB and the model weights.

The benchmark runs each model directly on a track. This sends the same tracks down the road a user's upload takes (a file in storage, the worker, its plan, the gate and its retries, FLAC files back in storage) with the real models, and scores what comes out with the benchmark's own scorer against the same references. No Jev key was used, so the rules decided. The audio was deleted after each track was scored.

Tracks: the first 3 of the frozen `moisesdb_subset` list. Device here: mps; the benchmark's figures are from this Mac's GPU (`docs/benchmarks/moisesdb.md`), the same device.

## Stem by stem

SDR in dB, the median over one-second windows, higher is better. "Benchmark" is the benchmark's figure for the same model on the same track, where one model made the stem and the benchmark scored that output against the same stem. "In song" is whether the reference sounds for at least 5 seconds.

| Track | Stem | Made by | Attempts | In song | Product SDR | Benchmark SDR | Difference | Gate's note |
|---|---|---|---:|---|---:|---:|---:|---|
| `014f3712` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 11.26 | 11.26 | 0.00 | not checked |
| `014f3712` | drums | `bs_roformer_sw_6stem` | 1 | yes | 16.46 | 16.44 | 0.01 | not checked |
| `014f3712` | bass | `bs_roformer_sw_6stem` | 1 | yes | 16.89 | 16.89 | 0.00 | not checked |
| `014f3712` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 9.56 | 9.56 | 0.00 | not checked |
| `014f3712` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `014f3712` | other | `bs_roformer_sw_6stem` | 1 | yes | 6.66 | 6.66 | 0.00 | not checked |
| `01c8ba69` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 10.69 | 10.69 | 0.00 | not checked |
| `01c8ba69` | drums | `bs_roformer_sw_6stem` | 1 | yes | 13.18 | 13.18 | 0.00 | not checked |
| `01c8ba69` | bass | `bs_roformer_sw_6stem` | 1 | yes | 13.98 | 13.98 | 0.00 | not checked |
| `01c8ba69` | guitar | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `01c8ba69` | piano | `bs_roformer_sw_6stem` | 1 | yes | 6.53 | 6.53 | 0.00 | not checked |
| `01c8ba69` | other | `bs_roformer_sw_6stem` | 1 | yes | 5.80 | 5.80 | 0.00 | not checked |
| `028c6f6a` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 7.26 | 7.25 | 0.02 | not checked |
| `028c6f6a` | drums | `bs_roformer_sw_6stem` | 1 | yes | 10.01 | 10.00 | 0.01 | not checked |
| `028c6f6a` | bass | `bs_roformer_sw_6stem` | 1 | yes | 15.65 | 15.63 | 0.02 | not checked |
| `028c6f6a` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 5.87 | 5.87 | 0.00 | not checked |
| `028c6f6a` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `028c6f6a` | other | `bs_roformer_sw_6stem` | 1 | yes | 2.63 | 2.63 | 0.00 | not checked |

A stem with no SDR has a reference that is silent throughout: there is nothing to score it against.

## Does the product deliver what the benchmark measured?

15 delivered stems could be set beside a benchmark figure. 15 of them are within 0.5 dB of it. The median difference (product minus benchmark) is 0.00 dB and the largest 0.02 dB.

## Did the gate flag the right stems?

Of 3 stems whose instrument is not in the song, the gate flagged 3 as probably empty. Of 15 stems whose instrument is in the song, it flagged 0.

## Time

| Track | Audio | Wall time | Models | Tagger | Everything else | Wall time on cpu |
|---|---:|---:|---:|---:|---:|---:|
| `014f3712` | 344 s | 120 s | 108 s | 0 s | 11 s | 411 s |
| `01c8ba69` | 249 s | 94 s | 84 s | 0 s | 10 s | 303 s |
| `028c6f6a` | 240 s | 88 s | 77 s | 0 s | 11 s | 310 s |

The last column is the same songs in run `product-v2` (`runs/product/product-v2/results/`).

All on mps, on a machine that was also running a benchmark and at times this project's tests; these are single runs, not speed measurements. "Models" counts each model once (a blend that reuses an output costs nothing more). "Everything else" is decoding, the gate's level checks, writing FLAC and peaks, and the zip. Scoring is not included.

## Limits

- 3 tracks. That is enough to see whether the road works and far too few to say how often it does.
- One dataset, the product's default models, no Jev.
- The worker's entry point was called directly. The API, the browser and the download links are not in this path; `make web-e2e-real` covers those on a synthetic signal.
