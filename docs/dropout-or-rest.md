# Can the product tell a dropout from a rest?

Written by `python -m stems_bench.gaps report` from the run `gaps-v1`, made at commit `131fbfa`. No model was run: these are the benchmark's kept outputs for the 30 tuning tracks and the 50 test tracks of MoisesDB, read beside the true stems. The definitions, the three rules and what would be done with the result were written into `bench/stems_bench/gaps.py` and committed (`131fbfa`) before any kept output was read for this.

A stem can be silent for a long stretch while its instrument plays (`docs/dropouts.md`, `docs/what-is-lost.md`), and the product says nothing. It has no true stem to know it by. It has the stem it made, the song, and the second model its plan names for the stem. Can it tell from those?

Counted: bass (`bs_roformer_sw_6stem`, then `htdemucs_ft`), guitar (`bs_roformer_sw_6stem`, then `mel_roformer_guitar_becruily`), piano (`bs_roformer_sw_6stem`, then `htdemucs_6s`) (the default model, then the second). Not counted, because the benchmark did not keep both models' outputs for every track: vocals, drums, other.

## The answer

**No rule is reliable on the tuning tracks, so none was tried on the test tracks: nothing here shows that the product can tell a dropout from a rest.**

A **gap** is an unbroken stretch in which the default model's stem is silent (under -60 dBFS) while the song sounds for 5 seconds or more. It is a **dropout** when the true stem sounds for that long inside it, and a **rest** when it does not. A stem that already carries the gate's note for the whole song has no gaps counted.

## The gaps

| List | Stem | Stems | Noted already | With a gap | With a dropout | Gaps | Dropouts | Rests |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tuning | bass | 30 | 0 | 19 | 1 | 30 | 1 | 29 |
| tuning | guitar | 30 | 3 | 13 | 1 | 33 | 5 | 28 |
| tuning | piano | 30 | 15 | 7 | 1 | 18 | 3 | 15 |
| test | bass | 50 | 2 | 36 | 1 | 53 | 3 | 50 |
| test | guitar | 50 | 5 | 15 | 1 | 29 | 2 | 27 |
| test | piano | 50 | 32 | 10 | 4 | 28 | 6 | 22 |

## The three rules

A rule is reliable on a list when it flags 10 gaps or more and at least 90% of them are dropouts. For the third rule the level was looked for on the tuning tracks: no level is reliable there, so the rule flags nothing. The nearest any level comes: at -19.48 dB it flags 12 gaps, of which 3 are dropouts (25%).

| Rule | Flags | List | Flagged | Dropouts among them | Share | Of all dropouts | Reliable |
|---|---|---|---:|---:|---:|---:|---|
| `silence` | every gap | tuning | 81 | 9 | 11% | 9 of 9 | no: wrong too often |
| `silence` | every gap | test | 110 | 11 | 10% | 11 of 11 | no: wrong too often |
| `second` | gaps in which the second model's stem sounds | tuning | 18 | 4 | 22% | 4 of 9 | no: wrong too often |
| `second` | gaps in which the second model's stem sounds | test | 27 | 5 | 19% | 5 of 11 | no: wrong too often |
| `loud_second` | gaps over which the second model's stem is loud against the song | tuning | 0 | 0 | - | 0 of 9 | no: no level found |
| `loud_second` | gaps over which the second model's stem is loud against the song | test | 0 | 0 | - | 0 of 11 | no: no level found |

No rule was tried on the test tracks; their rows are for information and decide nothing.

## What the second model has in a gap

For information. Its stem sounds inside a gap when it does so for 5 seconds or more; its level is against the song, over the gap; the share is how much of the true stem it holds over a dropout (`<true, second> / <true, true>`, as in `docs/what-is-lost.md`).

| List | Dropouts | It sounds in | Its level there (median) | Share of the true stem (median) | Rests | It sounds in | Its level there (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuning | 9 | 4 | -26.7 dB | 4% | 72 | 14 | -38.0 dB |
| test | 11 | 5 | -25.1 dB | 11% | 99 | 22 | -47.6 dB |

## What this is and is not

- 3 of the six stems, one dataset, one default model and one second model for each stem. It says nothing about the stems that are not counted.
- The benchmark's outputs, not a job's. `docs/product-vs-benchmark.md` compares the two.
- A rest may still hold some of the instrument: up to 5 seconds of it, by the definition.
- There are few dropouts to go by, and they come in clusters: 9 on the tuning tracks, in 2 songs, and 11 on the test tracks, in 6. Gaps in one song are not independent of one another.
- With 9 dropouts on the tuning tracks, a rule clears the bar there only by flagging exactly 10 gaps and missing almost no dropout. The bar was set before the count; the rules above are nowhere near it.
- A stem that carries the gate's note is not counted. Of the stems left out that way, 1 on the tuning tracks and 2 on the test tracks hold a stretch that would have been a dropout.
- Nothing in the product was changed.
