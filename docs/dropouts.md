# Where the instrument plays and the product delivers silence

Written by `python -m stems_bench.dropouts` from the audio of the listening run `owner-v1`: 6 songs of the frozen MoisesDB test list, each put through the product's own job code with the models it uses now and with the ones it used before the recommendation of Phase 10, and the true stems beside them. SDR gives a stem one number and the gate looks at its level over the whole song; this asks instead whether a stem is simply missing for part of the song.

A tenth of a second is **dropped** when the true stem sounds in it (reaches -60 dBFS, the project's silence level) and the product's stem does not. A **dropout** is one unbroken stretch of the product's silence, as long as the true stem sounds inside it; it is **long** from 5 seconds, the time that makes an instrument count as being in a song. These definitions were written into `bench/stems_bench/dropouts.py` and committed (`dc9ee6c`) before the count was run; one case had been seen by then (the bass of the sixth song).

| | Stems | True stem sounds (min) | Dropped (min) | Share | With a long dropout |
|---|---:|---:|---:|---:|---:|
| The product now | 31 | 101.7 | 5.8 | 5.7% | 3 |
| The product before | 31 | 101.7 | 3.7 | 3.7% | 2 |

For guitar and piano the product uses the same model now as before, so those rows of the two versions are the same audio.

## By stem

| Stem, version | Stems | True stem sounds (min) | Dropped (min) | Share | With a long dropout |
|---|---:|---:|---:|---:|---:|
| vocals, now (`bs_roformer_sw_6stem`) | 6 | 18.2 | 1.0 | 5.5% | 0 |
| vocals, before (`mel_roformer_kim_vocals`) | 6 | 18.2 | 1.1 | 6.2% | 0 |
| drums, now (`bs_roformer_sw_6stem`) | 6 | 23.0 | 0.4 | 1.8% | 0 |
| drums, before (`scnet_xl_ihf`) | 6 | 23.0 | 0.5 | 2.1% | 1 |
| bass, now (`bs_roformer_sw_6stem`) | 6 | 22.1 | 2.1 | 9.4% | 1 |
| bass, before (`htdemucs_ft`) | 6 | 22.1 | 2.0 | 8.9% | 1 |
| guitar, now (`bs_roformer_sw_6stem`) | 5 | 20.8 | 0.0 | 0.2% | 0 |
| piano, now (`bs_roformer_sw_6stem`) | 2 | 5.2 | 0.0 | 0.8% | 0 |
| other, now (`bs_roformer_sw_6stem`) | 6 | 12.4 | 2.2 | 18.0% | 2 |
| other, before (`scnet_xl_ihf`) | 6 | 12.4 | 0.1 | 0.5% | 0 |

## Every long dropout

| Song | Stem | Version | True stem sounds (s) | Dropped (s) | Longest (s) | True stem under the song (dB) | Product's stem under the song (dB) | The product's note |
|---|---|---|---:|---:|---:|---:|---:|---|
| T04 | other | now | 83.5 | 83.4 | 69.2 | 35.5 | 63.8 | yes |
| T06 | bass | now | 144.0 | 110.0 | 54.4 | 7.1 | 15.6 | no |
| T06 | bass | before | 144.0 | 93.7 | 46.1 | 7.1 | 14.6 | no |
| T03 | other | now | 203.9 | 24.1 | 14.0 | 11.8 | 15.8 | no |
| T02 | drums | before | 231.0 | 12.2 | 7.5 | 6.2 | 6.5 | no |

"The product's note" is the one it puts on a stem that is silent or very quiet against the whole song (more than 21.68 dB under it). 1 of these 5 stems would carry it. The others are delivered without a word: over the whole song they are loud enough.

In 1 of them (T04's other) the true stem is itself more than 21.68 dB under the song. By the product's own rule there was next to nothing to find there, and near-silence is close to the right answer; the count above includes them all the same, as it was defined.

## What this is and is not

- 6 songs, one dataset. It says the thing happens and how it looks; it is not a rate for music in general.
- Silence in the product's stem is counted only against the true stem, which the product never has. Nothing here can be turned into a check inside the product as it is.
- What the dataset files under an instrument's name is not always what a model was taught to call that instrument (a synthesiser playing the bass line, say). A dropout can be the model's miss or a difference of naming; the listener hears the same thing either way.
- The opposite fault, sound in the product's stem where the true stem is silent, is bleed. It is not counted here.
