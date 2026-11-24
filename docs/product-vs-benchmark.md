# The product beside the benchmark

Generated from `runs/product/product-v2/results/` (finished 2026-11-24T17:28:50+00:00, commit `ab6d0825b948`). Reproduce with `python -m stems_bench.product_check run --tracks 20 --device cpu` and then `report`; that needs MoisesDB and the model weights.

The benchmark runs each model directly on a track. This sends the same tracks down the road a user's upload takes (a file in storage, the worker, its plan, the gate and its retries, FLAC files back in storage) with the real models, and scores what comes out with the benchmark's own scorer against the same references. No Jev key was used, so the rules decided. The audio was deleted after each track was scored.

Tracks: the first 20 of the frozen `moisesdb_subset` list. Device here: cpu; the benchmark's figures are from this Mac's GPU (`docs/benchmarks/moisesdb.md`), so a small difference between the two columns is expected (`docs/benchmarks/device-spread-synthetic.md`).

**1 of 20 jobs did not finish.**

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
| `07c17c22` | - | - | - | - | - | - | - | job failed: The separation failed inside the server (FileNotFoundError). The details are in the server's log. |
| `0f5fb60c` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 12.83 | 12.73 | 0.10 | not checked |
| `0f5fb60c` | drums | `bs_roformer_sw_6stem` | 1 | yes | 13.72 | 13.57 | 0.15 | not checked |
| `0f5fb60c` | bass | `bs_roformer_sw_6stem` | 1 | yes | 13.93 | 13.92 | 0.01 | not checked |
| `0f5fb60c` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 5.33 | 5.20 | 0.14 | not checked |
| `0f5fb60c` | piano | `bs_roformer_sw_6stem` | 1 | yes | 7.11 | 7.04 | 0.07 | not checked |
| `0f5fb60c` | other | `bs_roformer_sw_6stem` | 1 | yes | 3.62 | 3.62 | 0.00 | not checked |
| `11845abc` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 15.48 | 15.48 | 0.00 | not checked |
| `11845abc` | drums | `bs_roformer_sw_6stem` | 1 | yes | 20.44 | 20.44 | 0.00 | not checked |
| `11845abc` | bass | `bs_roformer_sw_6stem` | 1 | yes | 0.00 | 0.00 | 0.00 | not checked |
| `11845abc` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 12.86 | 12.86 | 0.00 | not checked |
| `11845abc` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `11845abc` | other | `bs_roformer_sw_6stem` | 1 | yes | -1.43 | -1.43 | 0.00 | not checked |
| `1921a83e` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 16.94 | 16.86 | 0.08 | not checked |
| `1921a83e` | drums | `bs_roformer_sw_6stem` | 1 | yes | 12.62 | 12.58 | 0.04 | not checked |
| `1921a83e` | bass | `bs_roformer_sw_6stem` | 1 | yes | 18.90 | 18.85 | 0.05 | not checked |
| `1921a83e` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 12.67 | 12.67 | 0.00 | not checked |
| `1921a83e` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `1921a83e` | other | `bs_roformer_sw_6stem` | 1 | yes | 6.00 | 6.00 | 0.00 | not checked |
| `1e957e76` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 18.81 | 18.81 | 0.00 | not checked |
| `1e957e76` | drums | `bs_roformer_sw_6stem` | 1 | yes | 13.49 | 13.48 | 0.01 | not checked |
| `1e957e76` | bass | `bs_roformer_sw_6stem` | 1 | yes | 16.12 | 16.12 | 0.00 | not checked |
| `1e957e76` | guitar | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `1e957e76` | piano | `bs_roformer_sw_6stem` | 1 | yes | 15.84 | 15.84 | 0.00 | not checked |
| `1e957e76` | other | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `1ee489e8` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 10.80 | 10.78 | 0.02 | not checked |
| `1ee489e8` | drums | `bs_roformer_sw_6stem` | 1 | yes | 12.00 | 12.00 | 0.00 | not checked |
| `1ee489e8` | bass | `bs_roformer_sw_6stem` | 1 | yes | 20.88 | 20.88 | 0.00 | not checked |
| `1ee489e8` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 7.05 | 7.05 | 0.00 | not checked |
| `1ee489e8` | piano | `bs_roformer_sw_6stem` | 1 | yes | 5.54 | 5.54 | 0.00 | probably empty |
| `1ee489e8` | other | `bs_roformer_sw_6stem` | 1 | yes | 0.23 | 0.23 | 0.00 | probably empty |
| `21ce1331` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 8.65 | 8.65 | 0.00 | not checked |
| `21ce1331` | drums | `bs_roformer_sw_6stem` | 1 | yes | 18.83 | 18.83 | 0.00 | not checked |
| `21ce1331` | bass | `bs_roformer_sw_6stem` | 1 | yes | 20.51 | 20.50 | 0.01 | not checked |
| `21ce1331` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 7.36 | 7.36 | 0.00 | not checked |
| `21ce1331` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `21ce1331` | other | `bs_roformer_sw_6stem` | 1 | yes | 4.51 | 4.51 | 0.00 | not checked |
| `24f8a652` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 12.77 | 12.77 | 0.00 | not checked |
| `24f8a652` | drums | `bs_roformer_sw_6stem` | 1 | yes | 17.09 | 17.09 | 0.00 | not checked |
| `24f8a652` | bass | `bs_roformer_sw_6stem` | 1 | yes | 19.81 | 19.81 | 0.00 | not checked |
| `24f8a652` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 8.69 | 8.69 | 0.00 | not checked |
| `24f8a652` | piano | `bs_roformer_sw_6stem` | 1 | yes | -0.09 | -0.09 | 0.00 | probably empty |
| `24f8a652` | other | `bs_roformer_sw_6stem` | 1 | yes | 0.00 | 0.00 | 0.00 | probably empty |
| `2973adc9` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 16.28 | 16.28 | 0.00 | not checked |
| `2973adc9` | drums | `bs_roformer_sw_6stem` | 1 | yes | 11.61 | 11.61 | 0.00 | not checked |
| `2973adc9` | bass | `bs_roformer_sw_6stem` | 1 | yes | 15.79 | 15.79 | 0.00 | not checked |
| `2973adc9` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 10.10 | 10.10 | 0.00 | not checked |
| `2973adc9` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `2973adc9` | other | `bs_roformer_sw_6stem` | 1 | yes | 3.33 | 3.33 | 0.00 | not checked |
| `2e5d996d` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 18.16 | 18.07 | 0.09 | not checked |
| `2e5d996d` | drums | `bs_roformer_sw_6stem` | 1 | yes | 15.27 | 15.27 | 0.00 | not checked |
| `2e5d996d` | bass | `bs_roformer_sw_6stem` | 1 | yes | 13.83 | 13.83 | 0.00 | not checked |
| `2e5d996d` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 14.47 | 14.47 | 0.00 | not checked |
| `2e5d996d` | piano | `bs_roformer_sw_6stem` | 1 | yes | 16.00 | 16.00 | 0.00 | not checked |
| `2e5d996d` | other | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `334ed0e5` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 14.29 | 14.23 | 0.05 | not checked |
| `334ed0e5` | drums | `bs_roformer_sw_6stem` | 1 | yes | 12.77 | 12.77 | 0.00 | not checked |
| `334ed0e5` | bass | `bs_roformer_sw_6stem` | 1 | yes | 16.28 | 16.28 | 0.00 | not checked |
| `334ed0e5` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 12.14 | 12.14 | 0.00 | not checked |
| `334ed0e5` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `334ed0e5` | other | `bs_roformer_sw_6stem` | 1 | yes | 1.20 | 1.20 | 0.00 | probably empty |
| `35a19148` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 14.25 | 14.25 | 0.00 | not checked |
| `35a19148` | drums | `bs_roformer_sw_6stem` | 1 | yes | 17.23 | 17.23 | 0.00 | not checked |
| `35a19148` | bass | `bs_roformer_sw_6stem` | 1 | yes | 17.81 | 17.81 | 0.00 | not checked |
| `35a19148` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 10.66 | 10.66 | 0.00 | not checked |
| `35a19148` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `35a19148` | other | `bs_roformer_sw_6stem` | 1 | yes | 11.69 | 11.69 | 0.00 | not checked |
| `3c557409` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 0.00 | 0.00 | 0.00 | not checked |
| `3c557409` | drums | `bs_roformer_sw_6stem` | 1 | yes | 10.63 | 10.61 | 0.02 | not checked |
| `3c557409` | bass | `bs_roformer_sw_6stem` | 1 | yes | 12.36 | 12.36 | 0.00 | not checked |
| `3c557409` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 8.80 | 8.80 | 0.00 | not checked |
| `3c557409` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `3c557409` | other | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `3e389000` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 9.30 | 9.30 | 0.00 | not checked |
| `3e389000` | drums | `bs_roformer_sw_6stem` | 1 | yes | 14.57 | 14.57 | 0.00 | not checked |
| `3e389000` | bass | `bs_roformer_sw_6stem` | 1 | yes | 15.44 | 15.44 | 0.00 | not checked |
| `3e389000` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 8.90 | 8.90 | 0.00 | not checked |
| `3e389000` | piano | `bs_roformer_sw_6stem` | 1 | yes | 0.00 | 0.00 | 0.00 | not checked |
| `3e389000` | other | `bs_roformer_sw_6stem` | 1 | yes | 2.97 | 2.97 | 0.00 | not checked |
| `3e7985e5` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 14.95 | 14.95 | 0.00 | not checked |
| `3e7985e5` | drums | `bs_roformer_sw_6stem` | 1 | yes | 13.67 | 13.67 | 0.00 | not checked |
| `3e7985e5` | bass | `bs_roformer_sw_6stem` | 1 | yes | 18.08 | 18.08 | 0.00 | not checked |
| `3e7985e5` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 10.52 | 10.52 | 0.00 | not checked |
| `3e7985e5` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `3e7985e5` | other | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `3f5233cb` | vocals | `bs_roformer_sw_6stem` | 1 | yes | 14.47 | 14.47 | 0.00 | not checked |
| `3f5233cb` | drums | `bs_roformer_sw_6stem` | 1 | yes | 10.40 | 10.40 | 0.00 | not checked |
| `3f5233cb` | bass | `bs_roformer_sw_6stem` | 1 | yes | 14.24 | 14.22 | 0.02 | not checked |
| `3f5233cb` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 10.95 | 10.95 | 0.00 | not checked |
| `3f5233cb` | piano | `bs_roformer_sw_6stem` | 1 | yes | 8.20 | 8.20 | 0.00 | not checked |
| `3f5233cb` | other | `bs_roformer_sw_6stem` | 1 | yes | 2.42 | 2.42 | 0.00 | not checked |
| `46bc5393` | vocals | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `46bc5393` | drums | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `46bc5393` | bass | `bs_roformer_sw_6stem` | 1 | yes | 26.78 | 26.78 | 0.00 | not checked |
| `46bc5393` | guitar | `bs_roformer_sw_6stem` | 1 | yes | 22.77 | 22.77 | 0.00 | not checked |
| `46bc5393` | piano | `bs_roformer_sw_6stem` | 1 | no | - | - | - | probably empty |
| `46bc5393` | other | `bs_roformer_sw_6stem` | 1 | yes | 0.00 | 0.00 | 0.00 | probably empty |

A stem with no SDR has a reference that is silent throughout: there is nothing to score it against.

## Does the product deliver what the benchmark measured?

95 delivered stems could be set beside a benchmark figure. 95 of them are within 0.5 dB of it. The median difference (product minus benchmark) is 0.00 dB and the largest 0.15 dB.

## Did the gate flag the right stems?

Of 19 stems whose instrument is not in the song, the gate flagged 19 as probably empty. Of 95 stems whose instrument is in the song, it flagged 6.
- `1ee489e8` piano: in the song, flagged; its SDR is 5.54 dB.
- `1ee489e8` other: in the song, flagged; its SDR is 0.23 dB.
- `24f8a652` piano: in the song, flagged; its SDR is -0.09 dB.
- `24f8a652` other: in the song, flagged; its SDR is 0.00 dB.
- `334ed0e5` other: in the song, flagged; its SDR is 1.20 dB.
- `46bc5393` other: in the song, flagged; its SDR is 0.00 dB.

## Time

| Track | Audio | Wall time | Models | Tagger | Everything else |
|---|---:|---:|---:|---:|---:|
| `014f3712` | 344 s | 411 s | 395 s | 0 s | 16 s |
| `01c8ba69` | 249 s | 303 s | 294 s | 0 s | 9 s |
| `028c6f6a` | 240 s | 310 s | 296 s | 0 s | 14 s |
| `0f5fb60c` | 242 s | 294 s | 284 s | 0 s | 10 s |
| `11845abc` | 229 s | 322 s | 312 s | 0 s | 10 s |
| `1921a83e` | 102 s | 132 s | 126 s | 0 s | 6 s |
| `1e957e76` | 132 s | 156 s | 150 s | 0 s | 6 s |
| `1ee489e8` | 192 s | 246 s | 236 s | 0 s | 10 s |
| `21ce1331` | 235 s | 257 s | 248 s | 0 s | 9 s |
| `24f8a652` | 257 s | 282 s | 273 s | 0 s | 9 s |
| `2973adc9` | 75 s | 111 s | 105 s | 0 s | 6 s |
| `2e5d996d` | 218 s | 262 s | 253 s | 0 s | 9 s |
| `334ed0e5` | 92 s | 118 s | 112 s | 0 s | 6 s |
| `35a19148` | 184 s | 220 s | 212 s | 0 s | 7 s |
| `3c557409` | 598 s | 748 s | 726 s | 0 s | 22 s |
| `3e389000` | 228 s | 263 s | 254 s | 0 s | 9 s |
| `3e7985e5` | 229 s | 260 s | 251 s | 0 s | 9 s |
| `3f5233cb` | 229 s | 279 s | 270 s | 0 s | 9 s |
| `46bc5393` | 160 s | 222 s | 214 s | 0 s | 8 s |

All on cpu, on a machine that was also running a benchmark and at times this project's tests; these are single runs, not speed measurements. "Models" counts each model once (a blend that reuses an output costs nothing more). "Everything else" is decoding, the gate's level checks, writing FLAC and peaks, and the zip. Scoring is not included.

## Limits

- 20 tracks. That is enough to see whether the road works and far too few to say how often it does.
- One dataset, the product's default models, no Jev.
- The worker's entry point was called directly. The API, the browser and the download links are not in this path; `make web-e2e-real` covers those on a synthetic signal.
