# Every target is safe to re-run.
PY := .venv/bin/python

.PHONY: setup lint format test check registry registry-sources web-setup web-check

setup: ## create .venv and install the project with dev tools
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

check: lint test ## what every loop iteration runs before committing

registry: ## validate models/registry.yaml and print the summary table
	$(PY) -m stems_models

registry-sources: ## HEAD every download URL and refresh size_bytes (network)
	$(PY) -m stems_models.check_sources --write

web-setup: ## needs Node >= 22 (see BLOCKED.md B1)
	cd web && npm install

web-check:
	cd web && npm run lint && npm run typecheck && npm run build
