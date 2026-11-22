# Stems

Upload a song, get back separated stems: vocals, drums, bass, guitar, piano and other, with
optional lead/backing vocal split and de-reverb. A personal, non-commercial project aiming
for the best quality open models can give, with benchmarks you can re-run.

**Status:** Phase 1 (separation core) is complete. The `separate` command works with real
models. There is no benchmark yet and no web app yet.

- [PLAN.md](PLAN.md): architecture and what each phase delivers
- [PROGRESS.md](PROGRESS.md): task checklist and current state
- [BLOCKED.md](BLOCKED.md): what is waiting on the owner
- [models/registry.yaml](models/registry.yaml): every model checkpoint the pipeline can use
- [docs/corrections.md](docs/corrections.md): where the research report was wrong

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
```

The first run downloads the models it needs (a few GB). Stems and a `manifest.json` are
written next to the input, in `<name>_stems/`. Model outputs are cached under `runs/cache/`,
so repeating a run, or asking for another format, does not run the models again.

Copy `.env.example` to `.env` when you have keys or dataset paths. Nothing in it is needed
to run the tests.

## Layout

`common/` shared code, `models/` registry and downloads, `worker/` separation pipeline,
`bench/` benchmark harness, `api/` FastAPI service, `web/` Next.js app, `config/` tunable
defaults, `docs/` research and reports, `runs/` logs and results (git-ignored).
