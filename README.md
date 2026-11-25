# Stems

Upload a song, get back its vocals, drums, bass, guitar, piano and the rest as separate
files, with an optional lead and backing vocal split and a de-reverbed vocal. A personal,
non-commercial project: the best separation open models can give, shown with benchmarks
anyone can re-run.

This page was last brought up to date on 2026-11-24, and every command on it was run that
day. [PROGRESS.md](PROGRESS.md) is the full record; [BLOCKED.md](BLOCKED.md) is what waits
on the owner.

## What works

- **The web app, the API and the worker, on one machine.** Upload a song, watch the job,
  play the stems together, download them; everything is deleted after 7 days or when you
  say so. No accounts: a cookie identifies the browser.
- **The command line.** `separate` runs the same models on a file.
- **The benchmark.** Every model and blend named in the config, scored on the MUSDB18-HQ
  test set and on a frozen 50-track subset of MoisesDB.

## Run it

```
make setup                                  # Python: creates .venv and installs the project
make node-setup && make web-setup           # Node.js into .tools/node, then the web app's packages
make api                                    # terminal 1: the API on http://127.0.0.1:8000
make web-dev                                # terminal 2: the app on http://localhost:3000
```

The first job with a model downloads its weights (all of them are about 17 GB;
`make models-status` shows what is on disk). To try everything without any weights, start
the API with `STEMS_WORKER_FAKE_MODELS=1`: the "stems" are then frequency bands, not music.

```
.venv/bin/separate song.wav --stems all     # the same pipeline on one file
make check                                  # lint and the Python tests
make web-check                              # web lint, types, unit tests, build
make web-e2e                                # browser tests in Chrome, stand-in models
make web-e2e-real                           # one job with the real models through the browser (slow)
.venv/bin/bench status                      # progress of the benchmark run in flight
make listen                                 # the blind listening test for one listener (see below)
```

[docs/deployment.md](docs/deployment.md) covers settings, HTTPS, proxies, buckets and
Modal, and says for each step whether it has ever been run.

## What has been measured

Every figure is from a run in this repository; the report named beside it has the command.

| What | Result | Report |
|---|---|---|
| Best single model per stem, MoisesDB subset, SDR in dB | vocals 10.82, drums 12.95, bass 15.09, guitar 9.19, piano 5.80, other 2.97 | [docs/benchmarks/moisesdb.md](docs/benchmarks/moisesdb.md) |
| Best single model per stem, MUSDB18-HQ test, SDR in dB | vocals 12.30, drums 13.35, bass 12.47 | [docs/benchmarks/musdb18hq.md](docs/benchmarks/musdb18hq.md) |
| What a user downloads against what the benchmark scored | 100 of 100 stems of 20 real songs within 0.15 dB | [docs/product-vs-benchmark.md](docs/product-vs-benchmark.md) |
| The same on this Mac's GPU, three songs | 15 of 15 stems within 0.02 dB; 88 to 120 s a song, where the CPU took 303 to 411 s | [docs/product-on-gpu.md](docs/product-on-gpu.md) |
| The longest upload the limits allow, 20 minutes, through a worker process | finished in 568 s on the GPU, every stem the right length; the worker held at most 6.46 GB | [docs/deployment.md](docs/deployment.md) |
| The gate's note on those 20 songs | all 20 stems whose instrument is not in the song flagged; 7 of 100 that are in it flagged too, 6 of them stems the model found almost nothing for | [docs/product-vs-benchmark.md](docs/product-vs-benchmark.md) |
| The default model, chosen on 30 tuning tracks by a rule fixed first | one model for all six stems, `bs_roformer_sw_6stem`; against the models it replaced, on the test lists: drums +1.4 dB, bass +1.9 and +2.5 dB, vocals -0.27 and +0.34 dB | [docs/benchmarks/recommendation.md](docs/benchmarks/recommendation.md) |
| Time for a 4-minute song, six stems, the default model, this Mac's GPU | 69 s | [docs/cost-per-song.md](docs/cost-per-song.md) |
| `other` as the model makes it, against the song minus the five other stems | no clear difference by SDR: 2.61 against 2.56 dB on 27 tuning tracks, 2.80 against 2.76 on 16 test tracks | [docs/leftover-other.md](docs/leftover-other.md) |
| Where the instrument plays and the product's stem is silent, six songs | 5.7% of the time the true stems sound; 3 of 31 stems have a stretch of 5 s or more, and 2 of those 3 are delivered with no note | [docs/dropouts.md](docs/dropouts.md) |
| Where the instrument is when its stem is silent, and what is in no stem at all, 20 songs | a stem is silent for 5.5% of the time its instrument plays; 2% of the instrument is then in none of the six stems and 93% in a stem of another name; what is in no stem is 34.2 dB under the song (median) | [docs/what-is-lost.md](docs/what-is-lost.md) |
| Telling a dropout from an instrument that rests, without the true stem: bass, guitar and piano on 80 tracks | not shown. A silent stretch of 5 s or more is a dropout 1 time in 10 (9 of 81 on the tuning tracks, 11 of 110 on the test tracks); where a second model sounds in it, 2 in 10 | [docs/dropout-or-rest.md](docs/dropout-or-rest.md) |
| Telling a stem whose instrument is not in the song, by its level | right 90.2% of the time, finds 92.2% | [docs/gate-on-real-output.md](docs/gate-on-real-output.md) |
| Whether the audio tagger can judge a stem's quality | no: it fails on bleed twice over, and on naming instruments in a mix | [docs/gate-calibration.md](docs/gate-calibration.md), [docs/gate-on-real-output.md](docs/gate-on-real-output.md) |
| Clef-omni as the listening model | rejected by its pre-registered test | [docs/clef-omni-test.md](docs/clef-omni-test.md) |
| Decisions by Jev, per song | $0.000291 | [docs/cost-per-song.md](docs/cost-per-song.md) |
| Security of uploads and downloads; known vulnerabilities in dependencies | 12 findings fixed; none known on 2026-11-24 | [docs/security-review.md](docs/security-review.md) |

The benchmark figures are medians over tracks, on this Mac's GPU, for the 14 models the
config names; the protocol is in [docs/benchmarks/README.md](docs/benchmarks/README.md).
They are not comparable with figures published elsewhere.

## What is not known

- **How it sounds to people.** Nobody has listened yet. A blind test for one listener is
  ready: `make listen` serves 30 steps from six songs, what the product makes now and what
  it made before against the true stems, and `python -m stems_bench.listen pack` writes
  the same test for a phone, with the clips as 320 kbps MP3. It waits for the owner; the
  test with a panel and two paid tools waits on B8.
- **When a stem is missing its instrument for part of a song.** On 20 songs, 11 of the
  100 stems whose instrument is in the song are silent for 5 seconds or more while it
  plays. The product does not say so, and nothing tried so far lets it tell such a stretch
  from the instrument resting. The instrument is then in another of the six stems.
- **Whether any blend of models beats the single default.** None of the untuned ones
  does, so "fastest", "balanced" and "best quality" all give the same model for now. The
  tuner that could change that is still running.
- **What each model was trained on.** A checkpoint that met these datasets' songs in
  training scores higher on them than it would on new music. Nothing here can tell.
- **What it costs on a rented GPU.** Nothing has run on one.
- **Anything beyond one machine.** No bucket, no Modal, no real proxy, no real phone.
- **How it does on songs outside one dataset.** Twenty MoisesDB tracks have been through
  the product's own path; a person has not yet uploaded a song in the browser and
  listened.

## Waiting on the owner

[BLOCKED.md](BLOCKED.md) has the details: a Modal account (B4), bucket credentials (B7),
listeners and reference purchases for the listening test (B8), Azure quota (B9), and
fifteen decisions made for you that you can overrule (D1 to D15).

## Where things are

`common/` shared code and schemas, `models/` the registry and downloads, `worker/` the
separation pipeline and the quality gate, `bench/` the benchmark and the measurement
tools, `api/` the FastAPI service, `web/` the Next.js app, `config/` every tunable number,
`docs/` reports, `runs/` logs and results (not in the repository).
[PLAN.md](PLAN.md) is the design; [docs/corrections.md](docs/corrections.md) lists where
the research report it started from was wrong.
