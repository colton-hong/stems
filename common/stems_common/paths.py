"""Repository paths. Everything else resolves files relative to REPO_ROOT."""

from pathlib import Path

# common/stems_common/paths.py -> repo root is two levels above the package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH = REPO_ROOT / "models" / "registry.yaml"
DEFAULTS_PATH = REPO_ROOT / "config" / "defaults.yaml"
RUNS_DIR = REPO_ROOT / "runs"
