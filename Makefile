# Every target is safe to re-run.
PY := .venv/bin/python
# Use the project-local Node.js if tools/install_node.sh has installed one.
export PATH := $(CURDIR)/.tools/node/bin:$(PATH)

.PHONY: setup lint format test test-slow check env registry registry-sources models-status node-setup web-setup web-check

setup: ## create .venv and install the project with dev tools
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[worker,dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

test-slow: ## tests that load real checkpoints (need downloaded weights)
	$(PY) -m pytest -m slow

check: lint test ## what every loop iteration runs before committing

env: ## record this machine's devices and package versions in runs/env/worker.json
	$(PY) -m stems_worker.env

registry: ## validate models/registry.yaml and print the summary table
	$(PY) -m stems_models

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
