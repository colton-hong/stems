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
```

[docs/deployment.md](docs/deployment.md) covers settings, HTTPS, proxies, buckets and
Modal, and says for each step whether it has ever been run.

## What has been measured

Every figure is from a run in this repository; the report named beside it has the command.

| What | Result | Report |
|---|---|---|
| Best single model per stem, MoisesDB subset, SDR in dB | vocals 10.82, drums 12.95, bass 15.09, guitar 9.19, piano 5.80, other 2.97 | [docs/benchmarks/moisesdb.md](docs/benchmarks/moisesdb.md) |
| Best single model per stem, MUSDB18-HQ test, SDR in dB | vocals 12.30, drums 13.35, bass 12.47 | [docs/benchmarks/musdb18hq.md](docs/benchmarks/musdb18hq.md) |
| What a user downloads against what the benchmark scored | 12 of 12 stems of 3 real songs within 0.07 dB | [docs/product-vs-benchmark.md](docs/product-vs-benchmark.md) |
| Time for a 4-minute song, six stems, default models, this Mac's GPU | 230 s | [docs/cost-per-song.md](docs/cost-per-song.md) |
| Telling a stem whose instrument is not in the song, by its level | right 90.2% of the time, finds 92.2% | [docs/gate-on-real-output.md](docs/gate-on-real-output.md) |
| Whether the audio tagger can judge a stem's quality | no: it fails on bleed twice over, and on naming instruments in a mix | [docs/gate-calibration.md](docs/gate-calibration.md), [docs/gate-on-real-output.md](docs/gate-on-real-output.md) |
| Clef-omni as the listening model | rejected by its pre-registered test | [docs/clef-omni-test.md](docs/clef-omni-test.md) |
| Decisions by Jev, per song | $0.000291 | [docs/cost-per-song.md](docs/cost-per-song.md) |
| Security of uploads and downloads; known vulnerabilities in dependencies | 12 findings fixed; none known on 2026-11-24 | [docs/security-review.md](docs/security-review.md) |

The benchmark figures are medians over tracks, on this Mac's GPU, for the 14 models the
config names; the protocol is in [docs/benchmarks/README.md](docs/benchmarks/README.md).
They are not comparable with figures published elsewhere.

## What is not known

- **How it sounds to people.** No listening test has been run. The kit for one is built.
- **Which models should be the defaults.** The ones in use are provisional. The benchmark
  run that settles it (blend tuning, then a recommendation) is still going.
- **What it costs on a rented GPU.** Nothing has run on one.
- **Anything beyond one machine.** No bucket, no Modal, no real proxy, no real phone.
- **How often things go wrong on real songs.** Three have been through the product's own
  path; a person has not yet uploaded one in the browser and listened.

## Waiting on the owner

[BLOCKED.md](BLOCKED.md) has the details: a Modal account (B4), bucket credentials (B7),
listeners and reference purchases for the listening test (B8), Azure quota (B9), and
thirteen decisions made for you that you can overrule (D1 to D13).

## Where things are

`common/` shared code and schemas, `models/` the registry and downloads, `worker/` the
separation pipeline and the quality gate, `bench/` the benchmark and the measurement
tools, `api/` the FastAPI service, `web/` the Next.js app, `config/` every tunable number,
`docs/` reports, `runs/` logs and results (not in the repository).
[PLAN.md](PLAN.md) is the design; [docs/corrections.md](docs/corrections.md) lists where
the research report it started from was wrong.
