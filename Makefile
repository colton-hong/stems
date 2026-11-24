# Every target is safe to re-run.
PY := .venv/bin/python
# Use the project-local Node.js if tools/install_node.sh has installed one.
export PATH := $(CURDIR)/.tools/node/bin:$(PATH)

.PHONY: setup lint format test test-slow api smoke bench-synthetic bench-device-spread check env registry schemas registry-sources models-status node-setup web-setup web-check

setup: ## create .venv and install the project with dev tools
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[worker,bench,api,dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

test-slow: ## tests that load real checkpoints (need downloaded weights)
	$(PY) -m pytest -m slow -rs

api: ## serve the API on http://127.0.0.1:8000 (migrates the database first)
	$(PY) -m stems_api

smoke: ## load and run every separator on the fixture; writes runs/smoke/report.md
	$(PY) -m stems_worker.smoke

check: lint test ## what every loop iteration runs before committing

bench-synthetic: ## every separator and every blend on the synthetic set; writes docs/benchmarks/synthetic.md
	-$(PY) -m stems_bench run --dataset synthetic --all --ensembles --run-id synthetic-v1
	$(PY) -m stems_bench report synthetic-v1 --out docs/benchmarks/synthetic.md
	$(PY) -m stems_bench status synthetic-v1

# The provisional default model of every slot: what the device comparison is run on.
SPREAD_MODELS := mel_roformer_kim_vocals,scnet_xl_ihf,htdemucs_ft,bs_roformer_sw_6stem,mel_roformer_karaoke_aufr33_viperx,mel_roformer_dereverb_anvuew

bench-device-spread: ## the same models twice per device on the synthetic set; needs bench-synthetic first
	-$(PY) -m stems_bench run --dataset synthetic --models $(SPREAD_MODELS) --trial 1 --run-id synthetic-mps-again
	-$(PY) -m stems_bench run --dataset synthetic --models $(SPREAD_MODELS) --device cpu --run-id synthetic-cpu
	-$(PY) -m stems_bench run --dataset synthetic --models $(SPREAD_MODELS) --device cpu --trial 1 --run-id synthetic-cpu-again
	$(PY) -m stems_bench compare --pair synthetic-v1 synthetic-mps-again --pair synthetic-cpu synthetic-cpu-again \
		--pair synthetic-v1 synthetic-cpu --out docs/benchmarks/device-spread-synthetic.md

env: ## record this machine's devices and package versions in runs/env/worker.json
	$(PY) -m stems_worker.env

registry: ## validate models/registry.yaml and print the summary table
	$(PY) -m stems_models

schemas: ## regenerate the EnsemblePlan schema's model lists from the registry
	$(PY) -m stems_models.plan_schema --write

registry-sources: ## HEAD every download URL and refresh size_bytes (network)
	$(PY) -m stems_models.check_sources --write

models-status: ## show which model files are downloaded or in progress
	$(PY) -m stems_models.download --status

node-setup: ## install Node.js into .tools/node (skip if you have Node >= 22.13 already)
	sh tools/install_node.sh

web-setup: ## install web dependencies exactly as locked
	cd web && npm ci --no-fund --no-audit

web-check:
	cd web && npm run lint && npm run typecheck && npm run build
