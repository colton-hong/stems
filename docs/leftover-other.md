# Is what is left over a better `other`?

Written by `python -m stems_bench.product_check leftover`. The default model has an `other` output of its own, and it is the weakest stem the product delivers. The alternative costs nothing: the mix minus the five other delivered stems. Both were scored against the `other` reference with the benchmark's scorer, on songs put through the worker's own path. The comparison and what counts as better were written into `bench/stems_bench/product_check.py` before any leftover was scored.

A track counts if its `other` sounds for at least 5 seconds. The leftover is better if its median SDR is at least 0.3 dB above the model's own and it is the higher of the two on more than half of the tracks; worse is the mirror image; at least 10 tracks are needed.

| Tracks | Counted | Model's own `other`, median SDR | Leftover, median SDR | Leftover higher on | Own higher on |
|---|---:|---:|---:|---:|---:|
| Tuning (`leftover-tuning-v1`) | 27 | 2.61 | 2.56 | 2 | 25 |
| Test (`leftover-test-v1`), decides nothing | 16 | 2.80 | 2.76 | 4 | 12 |

**No clear difference** on the tuning tracks.

## When the song has no `other`

Then both should be close to silence. Their level against the mix, median over such tracks (lower is quieter):

| Tracks | Songs without `other` | Model's own | Leftover |
|---|---:|---:|---:|
| Tuning | 3 | -78.3 dB | -34.2 dB |
| Test | 4 | -81.3 dB | -35.8 dB |

The leftover holds everything the other five stems missed or got wrong, so it is rarely near silence; the gate's empty-stem note depends on that level.

## What happens with this

Nothing changes in the product. PLAN.md decides that a six-stem model's `other` is used as it is; this result is put to the owner in BLOCKED.md.

## Limits

- One model, one dataset, SDR only. The leftover makes the six stems add up to the song exactly, which SDR does not reward and a listener might.
- The runs were on this Mac, with no Jev and the default settings.
