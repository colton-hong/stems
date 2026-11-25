# A listening test by one listener

Written by `python -m stems_bench.listen report` from `runs/listening/owner-v1/ratings.json`. One person, the owner of this project, listened blind: for each stem of each song, 30 seconds of every version, matched in level, under letters whose meaning changed from stem to stem. Marks are from 1 (bad) to 5 (excellent).

Songs: 6, the first of the frozen `moisesdb_subset` list. Steps marked: 28 of 30. The others are not counted, for the reason below.

**Not counted:** `T01/bass`, `T05/bass`. Their marks were saved at 02:21 UTC, before the page could be heard on the phone, and every mark in them is a 3; the listener chose to leave them out.

**How the clips were heard.** 28 steps were marked on an iPhone, from 320 kbps MP3 copies of the clips on a private page.

| Version | Overall | Bleed | Artifacts |
|---|---:|---:|---:|
| The true stem | 3.71 | 3.71 | 3.71 |
| The product now | 2.79 | 2.79 | 2.79 |
| The product before | 2.64 | 2.64 | 2.64 |

Bleed and artifacts were optional and are means over the steps where they were given. Higher is better in every column.

In 7 of the marked steps (guitar, piano) the product makes the stem with the same model now as before. The two are then the same audio; it was played once, and its mark counts for both versions above and in the table by stem.

## Now against before

Over the 21 steps where the two come from different models, the version the product makes now was marked higher than the one it made before in 10, lower in 7 and the same in 4 (a split at least this uneven would happen by chance 63% of the time, by a sign test on the steps that differ).

## By stem

Mean overall mark, and in how many steps the version the product makes now was marked above, below or the same as the one it made before (a dash where the two are the same audio).

| Stem | Steps | The true stem | The product now | The product before | Now above | Now below | The same |
|---|---:|---:|---:|---:|---:|---:|---:|
| vocals | 6 | 3.83 | 2.67 | 3.33 | 1 | 4 | 1 |
| drums | 6 | 3.83 | 3.00 | 2.67 | 3 | 2 | 1 |
| bass | 4 | 3.75 | 2.25 | 2.00 | 1 | 1 | 2 |
| guitar | 5 | 3.80 | 2.80 | 2.80 | - | - | - |
| piano | 2 | 4.50 | 2.00 | 2.00 | - | - | - |
| other | 5 | 3.00 | 3.40 | 2.40 | 5 | 0 | 0 |

## Could the true stem be told from the separated ones?

The true stem had the highest mark, alone or shared, in 21 of 28 steps. The product's version was marked below it in 18, the same in 4 and above it in 6.

## What this is and is not

- One listener, who knows the project and what was being compared, though not which letter was which.
- The 28 steps were marked in 17 minutes, about 36 seconds a step. A step has two or three clips of 30 seconds, so that is less than hearing each clip through once.
- In all 77 clips the marks for bleed and artifacts are the same as the overall mark, so those two columns say nothing of their own.
- Songs from one dataset; 30 seconds of each stem, chosen where its true recording is loudest.
- Left out, because the true recording is so quiet that the product itself would call such a stem empty (more than 21.68 dB under the song): `T04/other` (35.5 dB under).
- Not the test the research report describes (five listeners, two paid tools). That one still waits (BLOCKED.md B8).
