"""The worker as a Modal app.

    modal deploy worker/stems_worker/modal_app.py      # the owner runs this, once per change

It defines one function, ``run_job(job_id)``, which is the same entry point the local
provider runs in a subprocess: ``stems_worker.jobs.main``. The API's Modal provider
(``stems_api.providers.modal``) spawns a call of it per job. The container gets:

- the repository at ``/app`` (so ``config/defaults.yaml`` and ``models/registry.yaml`` are
  where the code looks for them) with the ``worker`` and ``api`` extras installed;
- model weights on the Modal Volume ``stems-weights``, mounted at ``/weights``
  (``STEMS_MODELS_DIR``); a checkpoint is downloaded on first use and kept there;
- the Modal Secret ``stems``, which must hold the storage settings (``STEMS_STORAGE_BACKEND``
  and that backend's variables) and, if Jev is used, ``TYPESAFE_API_KEY``.

NOT RUN. There is no Modal account yet (BLOCKED.md B4), so this file has never been
deployed: the image has not been built, the GPU type and timeout are first choices, and
the first deploy is likely to need small fixes. A test checks only that it imports and
declares the function under the names the provider looks up.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "stems-worker"
FUNCTION_NAME = "run_job"
VOLUME_NAME = "stems-weights"
SECRET_NAME = "stems"
WEIGHTS_DIR = "/weights"
GPU = "T4"
# Longest job the worker is given: a 20-minute upload through a full ensemble with retries.
TIMEOUT_SECONDS = 3600

_REPO = Path(__file__).resolve().parents[2]
_IGNORE = [
    ".git",
    ".venv",
    ".tools",
    ".env",
    ".env.*",
    "runs",
    "web",
    "docs",
    "tests",
    "third_party",
    "models/weights",
    "**/__pycache__",
    "*.egg-info",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install_from_pyproject(
        str(_REPO / "pyproject.toml"), optional_dependencies=["worker", "api"]
    )
    .env(
        {
            "PYTHONPATH": "/app/common:/app/models:/app/worker:/app/api",
            "STEMS_MODELS_DIR": WEIGHTS_DIR,
        }
    )
    .workdir("/app")
    .add_local_dir(_REPO, "/app", ignore=_IGNORE)
)

app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    name=FUNCTION_NAME,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={WEIGHTS_DIR: weights},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def run_job(job_id: str) -> str:
    """Run one job from storage to storage. Returns the job's final state."""
    from stems_api.storage import read_json, storage_from_env
    from stems_common import jobs as contract
    from stems_worker.jobs import main

    code = main([job_id])
    weights.commit()  # keep any checkpoint this job downloaded
    if code != 0:
        raise RuntimeError(f"the worker could not run job {job_id}")
    return str(read_json(storage_from_env(), contract.result_key(job_id))["state"])
