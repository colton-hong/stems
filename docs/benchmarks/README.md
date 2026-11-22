# Benchmarks

Every file here is written by the `bench` command from the result files of a run in this
repository. Nothing is typed in by hand, and no figure from a paper or a leaderboard appears
anywhere: other people's numbers were measured under other protocols and are not comparable.

| File | What it is | Written by |
|---|---|---|
| `synthetic.md` | Every separator and every blend on the synthetic set | `make bench-synthetic` |
| `device-spread-synthetic.md` | The same models run twice per device, and across devices | `make bench-device-spread` |
| `musdb18hq.md` | MUSDB18-HQ test | **Not run.** Needs the dataset (BLOCKED.md B2) |
| `moisesdb.md` | The frozen MoisesDB subset | **Not run.** Needs the dataset (BLOCKED.md B3) |
| `recommendation.md` | Default model or blend per stem | **Not written.** Needs both of the above |

Each `.md` has a `.json` beside it with the same figures and the SHA-256 of every result file
they came from.

**The synthetic set says nothing about quality on music.** Its tracks are sine tones and
noise with exactly known stems. It exists so the whole harness can be run and checked with
no music on disk. No model has been chosen from it, and none should be.

## Protocol (version 1)

The rules live in `bench/stems_bench/protocol.py`; the version number is stored in every
frozen track list and every result, and neither is used under another version.

- Audio is scored at 44.1 kHz stereo, full length, as the dataset provides it.
- SDR, SIR and SAR are museval 0.4.1's BSS Eval v4 on non-overlapping 1-second windows. A
  track's score is the median over its windows; a model's score is the median over tracks.
  A track where the stem is absent is not scored.
- Every stem is scored as the pair (target, input minus target), for single-stem and
  multi-stem models alike, so SIR and SAR are comparable between them. SDR is the same as
  in museval's usual joint evaluation; SIR and SAR are not the joint four-source figures.
- A model's `other` is scored against four-stem other (guitar, piano and the rest) unless
  the model separates guitar and piano itself. A six-stem model is also scored on four-stem
  other, as the sum of its guitar, piano and other.
- Vocal-split models are fed the reference vocals stem and de-reverb models the reference
  vocals with reverb, so they are scored on their own and not on top of another model's
  errors.
- Also recorded per output: RMS, peak, silence fraction, length difference, and the energy
  at and above 8 kHz relative to the reference's.
- Track lists are frozen in `bench/frozen/` and committed before anything is scored on
  them. A run refuses to start on a list that is not committed, and a track whose audio no
  longer matches its list is refused.
- A report is built from one run, a run asks for one device, and a dataset is never
  averaged with another.

## Running it

```bash
make bench-synthetic        # safe to interrupt and repeat
```

```bash
make bench-device-spread    # after the above
```

Or step by step:

```bash
.venv/bin/bench run --dataset synthetic --all --ensembles --run-id synthetic-v1
```

```bash
.venv/bin/bench status synthetic-v1
```

```bash
.venv/bin/bench report synthetic-v1 --out docs/benchmarks/synthetic.md
```

A job is one model on one track. Its result is a file under `runs/bench/results/`, named by
a hash of the protocol version, the dataset and its frozen list, the track, the model and
its pinned weights, the device asked for and the chunk overlap. A job with a result file is
never run again, so a stopped run continues where it was and a second run costs nothing.

`--ensembles` adds each slot of `config/defaults.yaml` in its three modes (single best,
two-model blend, full ensemble). The audio a blend needs from its members is kept under
`runs/bench/estimates/` (about 21 MB per minute of audio per output, as float32 stereo) and
can be deleted afterwards.

## When the datasets arrive

MUSDB18-HQ, once `MUSDB18HQ_PATH` is set in `.env`:

```bash
.venv/bin/python -m stems_bench.datasets.musdb
```

freezes `bench/frozen/musdb18hq_test.json`. Commit it, then:

```bash
.venv/bin/bench run --dataset musdb18hq_test --all --ensembles --run-id musdb18hq-v1 --background
```

MoisesDB, once `MOISESDB_PATH` is set:

```bash
.venv/bin/python -m stems_bench.datasets.moisesdb --freeze
```

selects and freezes the 50-track subset in `bench/frozen/moisesdb_subset.json`. Commit it,
then run with `--dataset moisesdb_subset`.
