# Working in this repo

A web-based audio stem separator. Personal, non-commercial. The goal is the best separation
quality open models can give, shown with reproducible benchmarks.

## Every iteration

1. Read `PROGRESS.md`. It is the single source of truth. Then `BLOCKED.md`.
2. Take the first unticked, unblocked task in the current phase. One task per iteration.
3. Implement it, run `make check`, commit with a clear message, tick the task with a
   one-line note. Tick only when the acceptance check passes.
4. If the phase has no unticked unblocked tasks left, write the phase summary in
   `PROGRESS.md`, then start the next phase. If no phase is left, add one to `PROGRESS.md`
   with tasks and acceptance checks, and work it. (The owner asked for this on 2026-11-24;
   before that the rule was to stop and wait for a go-ahead.)
5. If the same task fails three iterations in a row, write what you tried in `BLOCKED.md`
   and move on.

`PLAN.md` holds the design and the decisions already made. Do not reopen those decisions.
`docs/research-report.pdf` is the research behind them; `docs/corrections.md` lists where it
was wrong. Where the report and `PLAN.md` disagree, `PLAN.md` wins.

## Rules

- Never invent a benchmark number, timing or price. Every figure in a report comes from a
  run in this repo, with the command that produced it. If a run has not happened, say so.
- Never copy the report's published SDR figures into our results or compare our numbers
  with them. The protocols differ.
- Model choice is by measured quality only. `licence_note` in the registry is informational
  and must never gate anything.
- Do not sign up for, buy or submit anything. Accounts, keys, dataset access and quota
  requests are the owner's. Read keys from environment variables. Never commit secrets or
  print them in logs.
- Datasets are read from the paths in `.env`. On 2026-11-22 the owner asked for MUSDB18-HQ
  and MoisesDB to be downloaded for them, and confirmed what went on MoisesDB's request
  form. They live in `~/Datasets/`, outside the repo. That covers those two only: do not
  obtain any other dataset or request any other access without being asked.
- Never commit model weights, dataset audio or any real music. Tests and examples use the
  synthetic fixture only.
- The Jev model is always the pinned `jev-1.14.0`, never an alias. Every Jev-driven
  decision needs a deterministic fallback that works with no key.
- If the report is wrong about something, add it to `docs/corrections.md` with its source
  and proceed on the corrected fact.
- Commit only after `make check` has exited 0. Check the exit code itself: piping the output
  (`make check | tail`) hides a failure, and so does chaining the commit with `;`.
- Tasks must be idempotent. Check for existing downloads, cached results and finished runs
  before redoing work.
- Long jobs run in the background, log under `runs/`, write a `status.json`, and are
  checked by a later iteration instead of being waited on.

## Commands

```
make setup              # create .venv, install the project and dev tools
make check              # ruff + pytest; run before every commit
make test-slow          # tests that load real checkpoints (need downloaded weights)
make smoke              # run every separator on the fixture; report in runs/smoke/
make env                # write runs/env/worker.json (devices, package versions)
.venv/bin/separate FILE --stems all   # run the pipeline on one file
.venv/bin/python -m stems_worker.timing   # median speed per model and device, from runs/timings.jsonl
.venv/bin/bench run --dataset synthetic --all   # benchmark; resumable, results in runs/bench/
.venv/bin/bench status  # progress of the latest benchmark run
.venv/bin/bench report RUN --out docs/benchmarks/NAME.md   # report from a run's result files
make bench-synthetic    # every separator and blend on the synthetic set, then its report
.venv/bin/python -m stems_bench.recommend --out docs/benchmarks/recommendation.md   # default single models, chosen on the tuning tracks
make bench-device-spread   # repeat runs per device, then docs/benchmarks/device-spread-synthetic.md
python -m stems_bench.clef.prereg   # show the committed Clef-omni pre-registration (.venv/bin/python)
python -m stems_bench.clef.run      # run the Clef-omni test; --report FILE writes its report
.venv/bin/python -m stems_worker.decision.job FILE --request "vocals and drums"   # one job through the decision layer; stems + job.json
.venv/bin/python -m stems_bench.calibrate collect   # score clean and bled stems with the tagger (long; resumable)
.venv/bin/python -m stems_bench.calibrate report --out docs/gate-calibration.md   # choose thresholds, check them held out
.venv/bin/python -m stems_bench.gate_outputs collect   # score the benchmark's kept outputs with the tagger (long; resumable)
.venv/bin/python -m stems_bench.gate_outputs levels    # the same outputs' levels over whole songs (no tagger; resumable)
.venv/bin/python -m stems_bench.gate_outputs report --out docs/gate-on-real-output.md   # can the listening layer judge real output?
.venv/bin/python -m stems_bench.cost --out docs/cost-per-song.md   # cost per song from measured usage and config/prices.yaml
.venv/bin/python -m stems_bench.product_check run --tracks 3 --device cpu   # real songs through the worker's own path, scored (long; resumable)
.venv/bin/python -m stems_bench.product_check report --out docs/product-vs-benchmark.md   # the product beside the benchmark
.venv/bin/python -m stems_bench.long_upload --device mps   # the longest allowed upload through a worker process: time, memory, lengths
.venv/bin/python -m stems_bench.product_check report --run-id product-gpu-v1 --beside product-v2 --out docs/product-on-gpu.md   # the GPU run, with the CPU run's times beside it
.venv/bin/python -m stems_bench.listening_kit build MANIFEST.yaml --out KIT --key KEY.json   # blind listening-test kit
.venv/bin/python -m stems_bench.listening_kit score KIT --key KEY.json --sheets DIR   # unblind returned sheets
.venv/bin/python -m stems_worker.decision.live   # one real Jev request per decision type, recorded as fixtures (needs the key)
.venv/bin/python -m stems_bench.listen make --tracks 6 --device mps   # the owner's listening test: make the audio and the kit (long; resumable)
make listen             # serve the blind listening page on http://127.0.0.1:8765
.venv/bin/python -m stems_bench.listen take MARKS --heard "on a phone, from ..."   # bring in marks made on another device (a file the page copied out, or a folder of its documents)
.venv/bin/python -m stems_bench.listen report --out docs/listening-owner.md   # unblind the marks
.venv/bin/python -m stems_bench.dropouts --out docs/dropouts.md   # where the true stem plays and the product's is silent (needs the listening run's audio)
.venv/bin/python -m stems_bench.product_check run --run-id lost-v1 --tracks 20 --device mps --lost   # the same, plus where a dropped instrument is and what is in no stem (long; resumable)
.venv/bin/python -m stems_bench.lost --run-id lost-v1 --out docs/what-is-lost.md   # its report
.venv/bin/python -m stems_bench.gaps collect   # a dropout or a rest? counted on the benchmark's kept outputs (no model runs; resumable)
.venv/bin/python -m stems_bench.gaps report --out docs/dropout-or-rest.md   # its report
make api                # serve the API on http://127.0.0.1:8000 (python -m stems_api)
.venv/bin/python -m stems_api.db.migrate    # bring the database to the newest schema (the API does this at startup)
.venv/bin/python -m stems_worker.jobs JOB_ID   # run one job from storage: what the local provider starts
make registry           # validate models/registry.yaml and print it
make schemas            # regenerate the EnsemblePlan schema's model lists after a registry edit
make registry-sources   # HEAD every model URL and refresh sizes (network)
make models-status      # which model files are downloaded or in progress
make node-setup         # project-local Node.js into .tools/node (once per clone)
make web-setup          # npm ci in web/
make web-check          # web lint, typecheck, unit tests, build
make web-e2e            # browser tests in Chrome against the real API with stand-in models
make web-e2e-real       # one job with the real models through the API and Chrome (slow; needs the weights)
```

## Environment notes

- The interpreter is `.venv/bin/python` (python.org 3.14, no system CA bundle: use
  `certifi` for HTTPS).
- Node.js lives in `.tools/node` (project-local, git-ignored). Run web commands through
  `make`, or put `.tools/node/bin` on `PATH` first.
- No ffmpeg or Homebrew is installed. Do not install system software; ask in `BLOCKED.md`.
- Apple Silicon, no NVIDIA GPU. Local timings are MPS or CPU and must be labelled as such.
