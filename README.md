# Stems

Upload a song, get back separated stems: vocals, drums, bass, guitar, piano and other, with
optional lead/backing vocal split and de-reverb. A personal, non-commercial project aiming
for the best quality open models can give, with benchmarks you can re-run.

**Status:** Phases 0 to 2 are built as far as they can be without the datasets. The
`separate` command works with real models, in single-model and blend modes, and the
benchmark harness runs end to end on a synthetic set. **No quality number on real music
exists yet**: the MUSDB18-HQ and MoisesDB benchmarks wait for the datasets. There is no web
app yet.

- [PLAN.md](PLAN.md): architecture and what each phase delivers
- [PROGRESS.md](PROGRESS.md): task checklist and current state
- [BLOCKED.md](BLOCKED.md): what is waiting on the owner
- [models/registry.yaml](models/registry.yaml): every model checkpoint the pipeline can use
- [docs/corrections.md](docs/corrections.md): where the research report was wrong
- [docs/benchmarks/](docs/benchmarks/README.md): benchmark protocol, commands and generated reports

## Setup

```
make setup     # Python 3.11+; creates .venv
make check     # lint and tests
make registry  # print the model registry

make node-setup && make web-setup && make web-check   # web app (Node.js, project-local)
```

## Separating a file

```
make setup
.venv/bin/separate song.wav --stems vocals,drums,bass,guitar,piano,other
.venv/bin/separate song.mp3 --stems all --format flac --out ./stems
.venv/bin/separate song.wav --stems vocals --split-vocals --dereverb
.venv/bin/separate song.wav --stems all --mode two_model_blend
```

The first run downloads the models it needs (a few GB). Stems and a `manifest.json` are
written next to the input, in `<name>_stems/`. Model outputs are cached under `runs/cache/`,
so repeating a run, or asking for another format, does not run the models again.

## Benchmarking

```
make bench-synthetic                  # every separator and blend on the synthetic set
.venv/bin/bench status                # progress of the latest run
.venv/bin/bench report RUN --out FILE # a report built only from that run's result files
```

Runs are resumable: one result file per model and track, never computed twice. See
[docs/benchmarks/README.md](docs/benchmarks/README.md) for the protocol and for what to run
when the datasets arrive.

Copy `.env.example` to `.env` when you have keys or dataset paths. Nothing in it is needed
to run the tests.

## Layout

`common/` shared code, `models/` registry and downloads, `worker/` separation pipeline,
`bench/` benchmark harness, `api/` FastAPI service, `web/` Next.js app, `config/` tunable
defaults, `docs/` research and reports, `runs/` logs and results (git-ignored).
