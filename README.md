# Stems

Upload a song, get back separated stems: vocals, drums, bass, guitar, piano and other, with
optional lead/backing vocal split and de-reverb. A personal, non-commercial project aiming
for the best quality open models can give, with benchmarks you can re-run.

**Status:** Phase 0 (plan and scaffold). There is no separation code yet.

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
```

Copy `.env.example` to `.env` when you have keys or dataset paths. Nothing in it is needed
to run the tests.

## Layout

`common/` shared code, `models/` registry and downloads, `worker/` separation pipeline,
`bench/` benchmark harness, `api/` FastAPI service, `web/` Next.js app, `config/` tunable
defaults, `docs/` research and reports, `runs/` logs and results (git-ignored).
