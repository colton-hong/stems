"""Shared parts for the API tests: a provider driven by hand, a clock, uploads."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import soundfile as sf
from sqlalchemy import select

from stems_api.app import Deps
from stems_api.db import migrate
from stems_api.db.engine import make_engine, session_factory
from stems_api.db.models import Job
from stems_api.providers.base import ProviderError, ProviderStatus
from stems_api.storage import LocalStorage
from stems_common import jobs as contract
from stems_common.settings import Defaults, load_defaults

SECRET = b"0123456789abcdef0123456789abcdef-test-only"
START = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
SR = 44100
MAX_BYTES = 1_500_000
MAX_SECONDS = 10


class Clock:
    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class ManualProvider:
    """A provider that starts nothing: the test decides what each worker does."""

    name = "manual"

    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.states: dict[str, str] = {}
        self.cancelled: list[str] = []
        self.refuse: Exception | None = None

    def submit(self, spec: dict[str, Any]) -> str:
        if self.refuse is not None:
            raise self.refuse
        contract.validate_spec(spec)
        self.submitted.append(spec)
        handle = f"manual:{spec['job_id']}"
        self.states[handle] = "queued"
        return handle

    def status(self, handle: str) -> ProviderStatus:
        return ProviderStatus(self.states.get(handle, "unknown"))  # type: ignore[arg-type]

    def cancel(self, handle: str) -> None:
        self.cancelled.append(handle)
        if self.states.get(handle) in ("queued", "running"):
            self.states[handle] = "cancelled"


def refusal() -> ProviderError:
    return ProviderError("no capacity")


def test_defaults(**updates: Any) -> Defaults:
    """The real defaults with small upload limits and quick timers, so tests stay fast."""
    base = load_defaults()
    changed = {
        "uploads": base.uploads.model_copy(
            update={"max_bytes": MAX_BYTES, "max_duration_seconds": MAX_SECONDS}
        ),
        "api": base.api.model_copy(update={"poll_interval_seconds": 0.05}),
        **updates,
    }
    return base.model_copy(update=changed)


test_defaults.__test__ = False  # a helper, not a test


def make_deps(tmp_path, *, provider=None, clock: Clock | None = None, defaults=None) -> Deps:
    clock = clock or Clock()
    engine = make_engine(f"sqlite:///{tmp_path / 'api.db'}")
    migrate.upgrade(engine)
    storage = LocalStorage(tmp_path / "store", SECRET, clock=lambda: clock().timestamp())
    return Deps(
        defaults=defaults or test_defaults(),
        sessions=session_factory(engine),
        storage=storage,
        provider=provider or ManualProvider(),
        clock=clock,
        web_origins=("http://localhost:3000",),
        # Checked in this process: the same check, without a process start per upload.
        # tests/test_api_hardening.py runs it the way the server does.
        isolate_decoding=False,
    )


def wav_bytes(seconds: float = 1.0, sample_rate: int = SR, channels: int = 2) -> bytes:
    """A real WAV file: quiet noise, 16-bit."""
    rng = np.random.default_rng(7)
    data = (rng.standard_normal((int(seconds * sample_rate), channels)) * 0.05).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def upload(client, data: bytes | None = None, name: str = "My Song.wav", **form: Any):
    form.setdefault("stems", "vocals,drums") if "request_text" not in form else None
    body = wav_bytes() if data is None else data
    return client.post("/api/jobs", files={"file": (name, body, "audio/wav")}, data=form)


def stored_keys(deps: Deps) -> list[str]:
    """Every object key in the local storage, sorted."""
    root = deps.storage.root / "objects"
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def all_jobs(deps: Deps) -> list[Job]:
    with deps.sessions() as session:
        jobs = list(session.scalars(select(Job).order_by(Job.created_at, Job.id)))
        for job in jobs:
            _ = job.stems  # load before the session closes
        return jobs
