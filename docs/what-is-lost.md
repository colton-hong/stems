# What the six stems lose

Written by `python -m stems_bench.lost` from the run `lost-v1`: 20 songs of the frozen `moisesdb_subset` list put through the worker's own path with the real default model on `mps`, all six stems asked for, and measured against the true stems before the audio was deleted. The measures were written into `bench/stems_bench/lost.py` and committed (`1790879`) before any real song was measured; the run's own record shows it was made at that commit, with nothing uncommitted.

## How long a stem is silent while its instrument plays

As in `docs/dropouts.md`: a tenth of a second is dropped when the true stem sounds in it and the product's stem does not; a dropout is long from 5 seconds of true-stem sound inside one stretch of silence.

| Stem | Songs with it | True stem sounds (min) | Dropped (min) | Share | With a long dropout |
|---|---:|---:|---:|---:|---:|
| vocals | 19 | 48.4 | 1.9 | 4.0% | 0 |
| drums | 19 | 60.0 | 1.6 | 2.7% | 0 |
| bass | 20 | 64.8 | 2.3 | 3.5% | 1 |
| guitar | 18 | 64.2 | 1.8 | 2.9% | 1 |
| piano | 8 | 21.5 | 3.0 | 13.9% | 3 |
| other | 16 | 34.0 | 5.5 | 16.1% | 6 |
| all | | 292.9 | 16.1 | 5.5% | 11 |

## Where the instrument is when its stem is silent

For the dropped time of each true stem: the share of it found in each of the six delivered stems, and in none of them (the song minus the six). A share is `<true, place> / <true, true>`; shares over several songs are weighted by dropped time. The last column is their sum, which is 100% when the true stems add up to the song and do not resemble one another.

| True stem | in vocals | in drums | in bass | in guitar | in piano | in other | in none | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vocals | 6% | 29% | 19% | 29% | 4% | 12% | 4% | 102% |
| drums | 21% | 22% | 4% | 31% | 1% | 10% | 4% | 92% |
| bass | 1% | 9% | 3% | 6% | 1% | 79% | 0% | 99% |
| guitar | 31% | 2% | 9% | 5% | 11% | 37% | 5% | 100% |
| piano | 22% | 1% | 1% | 10% | 1% | 65% | 4% | 103% |
| other | 11% | 68% | 5% | 11% | 1% | 0% | 1% | 98% |
| all | 14% | 31% | 6% | 14% | 2% | 30% | 2% | 99% |

Every stem with 5 seconds or more of dropped time:

| Song | True stem | Dropped (s) | in vocals | in drums | in bass | in guitar | in piano | in other | in none | Sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `11845abc` | bass | 110.0 | 0% | 1% | 0% | 3% | 0% | 97% | 0% | 100% |
| `24f8a652` | piano | 91.5 | 35% | 1% | 0% | 11% | 0% | 52% | 4% | 102% |
| `07c17c22` | other | 83.4 | 9% | 86% | 0% | 3% | 0% | 0% | 1% | 99% |
| `24f8a652` | other | 62.9 | 22% | 79% | 0% | 0% | 0% | 0% | 0% | 101% |
| `3e389000` | piano | 62.7 | 1% | 0% | 0% | 5% | 0% | 92% | 2% | 101% |
| `46bc5393` | other | 56.1 | 0% | 88% | 0% | 12% | 0% | 0% | 0% | 100% |
| `24f8a652` | guitar | 49.3 | 24% | 3% | 2% | 1% | 0% | 72% | 3% | 104% |
| `3f5233cb` | other | 34.9 | 8% | 5% | 2% | 70% | 5% | 0% | 1% | 91% |
| `1ee489e8` | guitar | 30.6 | 35% | 16% | 33% | 3% | 35% | 2% | 4% | 129% |
| `1ee489e8` | other | 29.0 | 12% | 81% | 2% | 1% | 0% | 1% | 1% | 98% |
| `028c6f6a` | other | 24.1 | 8% | 87% | 2% | 1% | 0% | 0% | 1% | 99% |
| `014f3712` | vocals | 22.5 | 11% | 32% | 1% | 51% | 0% | 4% | 2% | 100% |
| `21ce1331` | guitar | 22.4 | 35% | -25% | 0% | 14% | 0% | 15% | 4% | 44% |
| `1ee489e8` | piano | 19.8 | 25% | 4% | 0% | 19% | 2% | 48% | 4% | 102% |
| `01c8ba69` | other | 16.7 | 6% | 4% | 83% | 0% | 1% | 0% | 4% | 97% |
| `35a19148` | vocals | 16.3 | 3% | -9% | 79% | 3% | 0% | 6% | 2% | 83% |
| `0f5fb60c` | vocals | 14.1 | 2% | 8% | 0% | 33% | 8% | 25% | 3% | 79% |
| `3c557409` | drums | 13.9 | 1% | 11% | 26% | 74% | 0% | 0% | 3% | 113% |
| `014f3712` | bass | 12.7 | 1% | 67% | 12% | 30% | 0% | 0% | 3% | 112% |
| `3c557409` | vocals | 12.1 | 1% | 52% | 6% | 40% | 0% | 0% | 2% | 100% |
| `3e7985e5` | drums | 10.7 | 9% | 25% | 15% | 37% | 0% | 0% | 1% | 87% |
| `11845abc` | vocals | 9.8 | 6% | 60% | 0% | 19% | 0% | 17% | 3% | 104% |
| `21ce1331` | other | 9.5 | 43% | 55% | 1% | 0% | 0% | 1% | 1% | 101% |
| `01c8ba69` | vocals | 9.1 | 4% | 21% | 29% | 0% | 2% | 31% | 8% | 95% |
| `3e389000` | vocals | 8.5 | 7% | 16% | 6% | 93% | -2% | 26% | 4% | 149% |
| `11845abc` | drums | 7.8 | 11% | 10% | 2% | 55% | 0% | 47% | 7% | 132% |
| `2e5d996d` | drums | 7.8 | 11% | 46% | 20% | -14% | -14% | 0% | 1% | 51% |
| `35a19148` | drums | 7.8 | 47% | 15% | 7% | 33% | 0% | 10% | 5% | 116% |
| `3f5233cb` | drums | 7.4 | 25% | 33% | -3% | 17% | 8% | 3% | 4% | 87% |
| `1e957e76` | drums | 6.3 | 13% | 43% | 2% | 0% | 19% | 0% | 2% | 77% |
| `3e389000` | drums | 6.3 | 24% | 11% | -19% | 72% | -3% | 6% | 3% | 95% |
| `0f5fb60c` | other | 6.2 | 15% | 0% | 7% | 4% | 30% | 3% | 3% | 62% |
| `0f5fb60c` | drums | 6.1 | 25% | 16% | -79% | 11% | 10% | 19% | 19% | 23% |
| `01c8ba69` | drums | 5.7 | 43% | 3% | 7% | 0% | 1% | 42% | 3% | 99% |

How to read the sum (this paragraph was added after the run; it changes no figure). The six delivered stems and what is in none of them add up to the song, so a row's sum is `<true, song> / <true, true>`. It leaves 100% by as much as the true stem resembles the other true stems over its dropped time, by sound they share or by chance. A row far from 100%, or with a share below zero, cannot be read share by share. 19 of the 34 stems listed are within 10 points of 100%, and they hold 11.9 of the 16.1 minutes dropped; 7 have a share below zero. A stem's own column is not zero because a stem that counts as silent (under -60 dBFS) is not empty.

**In none of the six: 2% of what is dropped**, over all stems. The most for one stem listed is 19% (`0f5fb60c` drums, a row that adds up to 23%); among the rows within 10 points of 100% it is 8% (`01c8ba69` vocals).

A check that the share means what it should: the same figure for each stem's own output over the time it is not dropped, median over songs. vocals 96%, drums 95%, bass 97%, guitar 90%, piano 80%, other 61%.

## How much of a song is in none of its six stems

What is left when the six delivered stems are taken away from the song, as a level against the song over its whole length: median 34.2 dB under it (from 28.4 to 43.7 dB under, over 20 songs).

The product calls a stem very quiet when it is more than 21.68 dB under the song. Tenth of a second by tenth of a second, what is in no stem is louder than that for 0.0% of a song's sounding time (median; the most in one song is 1.8%).

## What this is and is not

- 20 songs, one dataset, one model. It is not a rate for music in general.
- A share is a correlation, not a listening judgement: two instruments that play the same notes resemble one another, and then the shares do not add up to 100%.
- The dataset's six true stems add up to its mix: what they leave out is 89.7 dB under the song or lower in every song (median 96.7 dB).
- Everything here needs the true stems except the last section: what is in no stem can be worked out from the song and its six stems alone.
- Nothing in the product was changed. The choice this bears on is BLOCKED.md D15.
