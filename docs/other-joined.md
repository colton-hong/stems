# The two ways of making `other`, joined

Written by `python -m stems_bench.other_joined report` from the run `other-joined-v1`, made at commit `5c539a1`. The stems are the benchmark's kept outputs, read beside the true stems. The rule for joining, its bar and what each outcome leads to were written into `bench/stems_bench/other_joined.py` and committed (`5c539a1`) before the default model's kept `other` was read for this.

**Own** is `bs_roformer_sw_6stem`'s `other`. **Subtracted** is `scnet_xl_ihf`'s four-stem `other` less the default's guitar and piano. A **gap** is an unbroken stretch in which own is silent while the song sounds for 5 seconds or more; it is a **dropout** when the true `other` sounds for that long inside it, and a **rest** otherwise. The **joined** stem is own, with the subtracted stem put into every gap over which the subtracted stem is at least T dB against the song.

## The answer

**No level is reliable on the tuning tracks, so nothing was joined: the subtracted stem's loudness does not tell a dropout of the default's own `other` from a rest.**

A level is reliable on a list when it fills 10 gaps or more and at least 90% of the gaps it fills are dropouts. On the tuning tracks: no level is. The nearest: at -20.19 dB, 17 gaps filled, 12 of them dropouts (71%).

| List | Tracks | Gaps | Dropouts | Filled | Dropouts among them | Share | Of all dropouts | Reliable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tuning | 30 | 64 | 25 | 0 | 0 | - | 0 of 25 | no: no level found |
| test | 50 | 111 | 28 | 0 | 0 | - | 0 of 28 | no: no level found |

## What each way of making `other` does

Where the song has an `other`: the time its stem is silent while the true one sounds, and the SDR of the benchmark's protocol. That protocol cannot score a one-second window in which the estimate is silent, so the column after it says how many windows the SDR stands on. Where the song has none: how far the stem is under the song, and whether the product's rule would note it as silent or very quiet.

| List | How `other` is made | Songs with an `other` | Dropped | With a long dropout | SDR (dB, median) | Windows scored | Songs with none | There: dB under the song (median) | There: noted as empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tuning | the default's own | 27 | 22.3% | 13 | - | - | 3 | 78.3 | 3 of 3 |
| tuning | made by subtraction | 27 | 1.4% | 1 | - | - | 3 | 20.8 | 1 of 3 |
| test | the default's own | 33 | 13.6% | 10 | - | - | 17 | 84.7 | 17 of 17 |
| test | made by subtraction | 33 | 0.9% | 0 | - | - | 17 | 20.2 | 5 of 17 |

No stem was scored in this run: scoring is part of joining, and nothing was joined. `docs/other-by-subtraction.md` has the SDR of the two ways.

## What this is and is not

- One dataset, one default model, one four-stem model. The level is chosen on the tuning tracks and tried on the test tracks once.
- The join switches on the tenth of a second with no fade, which a product would not do; what is counted here does not depend on it.
- A higher SDR for a stem that drops out is partly the protocol's doing: the windows it is silent in are not scored. The windows column is there for that.
- Nothing in the product was changed. The choice this bears on is BLOCKED.md D15.
