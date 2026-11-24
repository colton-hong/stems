"""The HTTP API (PLAN.md section 8).

    python -m stems_api             # serve on http://127.0.0.1:8000

The API owns the database and never loads a model. A job's audio and its spec go to
storage, a worker is started through the provider, and from then on the API only reads
what the worker writes. Nothing here waits for a worker.

Every job route is scoped to the browser's owner cookie. A job that belongs to someone
else gets the same 404 as one that does not exist.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from stems_api.db import system
from stems_api.db.models import Job, Owner
from stems_api.db.store import JobStore, create_owner, find_owner, new_id
from stems_api.providers.base import ProviderError, WorkerProvider
from stems_api.ratelimit import Limit, RateLimited, RateLimiter
from stems_api.storage import (
    LinkExpired,
    LinkInvalid,
    ObjectNotFound,
    Storage,
    StorageError,
    content_disposition,
    put_json,
    read_json,
)
from stems_api.sweeper import purge, sweep
from stems_api.sync import JobSync
from stems_api.uploads import UploadRejected, check_name, check_size, display_name, inspect
from stems_api.views import download_name, job_view
from stems_common import jobs as contract
from stems_common.settings import QUALITY_TARGETS, Defaults
from stems_common.stems import PRODUCT_STEMS

log = logging.getLogger("stems_api")

OWNER_COOKIE = "stems_owner"
FORMATS = ("wav", "flac", "mp3_320", "mp3_192")
QUALITY_MODES = (*QUALITY_TARGETS, "unspecified")
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Room for the multipart framing and the form fields around the file.
BODY_OVERHEAD_BYTES = 1024 * 1024
UPLOAD_FAILED = "The upload could not be stored. Please try again."
SUBMIT_FAILED = "The job could not be started. Please try again."
NOT_FOUND = "Not found."


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class Deps:
    """Everything the routes use. Tests build one with their own parts."""

    defaults: Defaults
    sessions: sessionmaker[Session]
    storage: Storage
    provider: WorkerProvider
    clock: Callable[[], datetime] = utcnow
    limiter: RateLimiter = field(default_factory=RateLimiter)
    web_origins: tuple[str, ...] = ()
    # The Jev client for POST /requests/parse; None means the keyword rules answer.
    parse_client: Any | None = None
    # Uploads are checked by decoders running in a child process. Tests that make many
    # uploads turn this off to save the process start; the check itself is the same.
    isolate_decoding: bool = True
    sync: JobSync = field(init=False)

    def __post_init__(self) -> None:
        self.sync = JobSync(self.storage, self.provider, self.clock)


def deps_from_env() -> Deps:
    from stems_api.db import migrate
    from stems_api.db.engine import make_engine, session_factory
    from stems_api.providers import provider_from_env
    from stems_api.storage import storage_from_env
    from stems_common.env import get_env
    from stems_common.settings import load_defaults
    from stems_worker.decision.jev import JevClient

    defaults = load_defaults()
    engine = make_engine()
    migrate.upgrade(engine)
    origin = get_env("STEMS_WEB_ORIGIN") or "http://localhost:3000"
    return Deps(
        defaults=defaults,
        sessions=session_factory(engine),
        storage=storage_from_env(base_url="/api/files"),
        provider=provider_from_env(),
        web_origins=tuple(o.strip().rstrip("/") for o in origin.split(",") if o.strip()),
        parse_client=JevClient.from_env(defaults.decision),
    )


# -- guard: body size and cross-site requests ----------------------------------------------


class Guard:
    """Refuses request bodies over the limit, and unsafe requests from another site.

    The size is enforced while the body is read, so a client that sends no (or a false)
    ``Content-Length`` is still cut off. The origin check backs up the cookie's SameSite
    rule: a state-changing request that names an ``Origin`` must name this API or the
    configured web app.
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int, allowed_origins: tuple[str, ...]):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.allowed_origins = frozenset(allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in UNSAFE_METHODS:
            await self.app(scope, receive, send)
            return
        headers = {
            name.decode("latin-1"): value.decode("latin-1") for name, value in scope["headers"]
        }
        origin = headers.get("origin")
        if origin is not None:
            own = f"{scope.get('scheme', 'http')}://{headers.get('host', '')}"
            if origin.rstrip("/") not in self.allowed_origins and origin != own:
                await _plain(send, 403, "Requests from this origin are not allowed.")
                return
        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await _plain(send, 413, "The upload is larger than the limit.")
            return
        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(413, "The upload is larger than the limit.")
            return message

        await self.app(scope, counted, send)


class SecurityHeaders:
    """Headers every reply carries: do not guess content types, do not send this site's
    addresses to others as a referrer, and do not keep answers about a browser's jobs in
    any cache. A route that sets its own caching (the peaks, the downloads) keeps it."""

    ALWAYS = (
        (b"x-content-type-options", b"nosniff"),
        (b"referrer-policy", b"no-referrer"),
    )

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                headers += [(name, value) for name, value in self.ALWAYS if name not in present]
                if b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_headers)


async def _plain(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# -- background work -----------------------------------------------------------------------


class Maintenance:
    """One thread that keeps jobs in progress synced and runs the sweeper on a timer."""

    def __init__(self, deps: Deps) -> None:
        self._deps = deps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="stems-maintenance", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _loop(self) -> None:
        deps, settings = self._deps, self._deps.defaults.api
        next_sweep = time.monotonic() + settings.sweep_interval_seconds
        while not self._stop.wait(settings.poll_interval_seconds):
            try:
                deps.sync.sync_all(deps.sessions)
                if time.monotonic() >= next_sweep:
                    next_sweep = time.monotonic() + settings.sweep_interval_seconds
                    sweep(deps.sessions, deps.storage, deps.provider, deps.clock())
            except Exception:
                log.exception("maintenance pass failed; it will run again")


# -- request bodies ------------------------------------------------------------------------


class ParseBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _options(
    stems: str | None,
    request_text: str | None,
    output_format: str | None,
    quality_mode: str,
) -> tuple[list[str] | None, str | None, str | None]:
    """Check the form's options. Returns (stems, text, format); raises 422 otherwise."""
    text = (request_text or "").strip() or None
    chosen: list[str] | None = None
    if stems is not None and stems.strip():
        names = [name.strip().lower() for name in stems.split(",") if name.strip()]
        if names == ["all"]:
            names = list(PRODUCT_STEMS)
        unknown = sorted(set(names) - set(PRODUCT_STEMS))
        if unknown:
            raise HTTPException(422, f"Unknown stems: {', '.join(unknown)}.")
        chosen = [stem for stem in PRODUCT_STEMS if stem in names]
    if (chosen is None) == (text is None):
        raise HTTPException(422, "Send either the stems you want or a request in words.")
    if text is not None and len(text) > 2000:
        raise HTTPException(422, "The request text is longer than 2000 characters.")
    fmt = (output_format or "").strip().lower() or None
    if fmt is not None and fmt not in FORMATS:
        raise HTTPException(422, f"Unknown format. Choose one of: {', '.join(FORMATS)}.")
    if quality_mode not in QUALITY_MODES:
        raise HTTPException(
            422, f"Unknown quality mode. Choose one of: {', '.join(QUALITY_MODES)}."
        )
    return chosen, text, fmt


# -- the app -------------------------------------------------------------------------------


def create_app(deps: Deps | None = None, *, background: bool = True) -> FastAPI:
    """The application. ``background=False`` leaves out the maintenance thread, for tests
    that drive syncing and sweeping themselves."""
    deps = deps or deps_from_env()
    defaults = deps.defaults
    maintenance = Maintenance(deps)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(deps.sync.recover, deps.sessions)
        await run_in_threadpool(sweep, deps.sessions, deps.storage, deps.provider, deps.clock())
        if background:
            maintenance.start()
        try:
            yield
        finally:
            maintenance.stop()

    app = FastAPI(
        title="Stems",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.deps = deps
    app.add_middleware(
        Guard,
        max_body_bytes=defaults.uploads.max_bytes + BODY_OVERHEAD_BYTES,
        allowed_origins=deps.web_origins,
    )
    app.add_middleware(SecurityHeaders)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(deps.web_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["Retry-After"],
    )

    @app.exception_handler(RateLimited)
    async def _rate_limited(_request: Request, exc: RateLimited) -> JSONResponse:
        return JSONResponse(
            {"detail": f"Too many requests. Try again in {exc.retry_after} seconds."},
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(UploadRejected)
    async def _upload_rejected(_request: Request, exc: UploadRejected) -> JSONResponse:
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code)

    router = APIRouter(prefix="/api")

    # -- helpers ---------------------------------------------------------------------------

    def owner_of(request: Request, session: Session) -> Owner | None:
        return find_owner(session, request.cookies.get(OWNER_COOKIE), deps.clock())

    def ensure_owner(request: Request, response: Response, session: Session) -> Owner:
        """The browser's owner, made on its first write. The cookie holds the only copy of
        the secret: HttpOnly, SameSite, and Secure whenever the request came over HTTPS."""
        owner = owner_of(request, session)
        if owner is not None:
            return owner
        owner, token = create_owner(session, deps.clock())
        response.set_cookie(
            OWNER_COOKIE,
            token,
            max_age=defaults.api.owner_cookie_days * 86400,
            path="/api",
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
        return owner

    def client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def owned_job(request: Request, session: Session, job_id: str) -> tuple[JobStore, Job]:
        owner = owner_of(request, session)
        job = None if owner is None else JobStore(session, owner.id).get(job_id)
        if owner is None or job is None:
            raise HTTPException(404, NOT_FOUND)
        return JobStore(session, owner.id), job

    def refresh(request: Request, job_id: str) -> None:
        """Bring the owner's job up to date with its worker before it is read. Whose job
        it is comes first: nothing is synced for a browser that does not own it."""
        with deps.sessions.begin() as session:
            owned_job(request, session, job_id)
        deps.sync.refresh(deps.sessions, job_id)

    def view(job: Job) -> dict[str, Any]:
        return job_view(job, deps.sync.live(job.id), deps.clock())

    def absolute(request: Request, url: str) -> str:
        return str(request.base_url).rstrip("/") + url if url.startswith("/") else url

    # -- public ----------------------------------------------------------------------------

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/config")
    def config() -> dict[str, Any]:
        return {
            "stems": list(PRODUCT_STEMS),
            "extra_stems": list(contract.EXTRA_STEMS),
            "formats": list(FORMATS),
            "default_format": defaults.audio.output_format,
            "quality_modes": list(QUALITY_MODES),
            # What each setting gives when the rules plan a job: a quality named by no one
            # is planned as "balanced". With Jev planning ("decided_by": "jev") a setting
            # other than "fastest" may also be given a blend, so the settings can differ
            # even when these are all the same.
            "quality_gives": {
                mode: defaults.decision.mode_for_quality[
                    "balanced" if mode == "unspecified" else mode
                ]
                for mode in QUALITY_MODES
            },
            "decided_by": "jev" if deps.parse_client is not None else "rules",
            "uploads": {
                "max_bytes": defaults.uploads.max_bytes,
                "max_duration_seconds": defaults.uploads.max_duration_seconds,
                "allowed_extensions": list(defaults.uploads.allowed_extensions),
            },
            "retention_days": defaults.retention.days,
            "link_ttl_seconds": defaults.retention.signed_url_ttl_seconds,
        }

    @router.post("/requests/parse")
    def parse(body: ParseBody, request: Request) -> dict[str, Any]:
        from stems_worker.decision.parse import parse_request

        limits = defaults.rate_limits
        checks = [Limit(f"parse:ip:{client_ip(request)}", limits.parses_per_ip_per_minute, 60)]
        with deps.sessions.begin() as session:
            owner = owner_of(request, session)
            if owner is not None:
                checks.append(
                    Limit(f"parse:owner:{owner.id}", limits.parses_per_owner_per_minute, 60)
                )
        deps.limiter.hit(checks)
        try:
            decision = parse_request(
                body.text, client=deps.parse_client, settings=defaults.decision
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {
            "parsed": decision.value,
            "provider": decision.record["provider"],
            "fallback_reason": decision.record.get("fallback_reason"),
        }

    # -- jobs ------------------------------------------------------------------------------

    @router.post("/jobs", status_code=201)
    def create_job(
        request: Request,
        response: Response,
        file: UploadFile = File(...),  # noqa: B008
        stems: str | None = Form(None),
        request_text: str | None = Form(None),
        split_vocals: bool = Form(False),
        dereverb: bool = Form(False),
        output_format: str | None = Form(None, alias="format"),
        quality_mode: str = Form("unspecified"),
    ) -> dict[str, Any]:
        limits, uploads = defaults.rate_limits, defaults.uploads
        now = deps.clock()
        # Every attempt counts, refused or not: a refused upload still cost the read.
        # A browser with no cookie yet is counted by its address alone.
        checks = [Limit(f"upload:ip:{client_ip(request)}", limits.uploads_per_ip_per_hour, 3600)]
        active: list[Job] = []
        with deps.sessions.begin() as session:
            known = owner_of(request, session)
            if known is not None:
                per_owner = limits.uploads_per_owner_per_hour
                checks.append(Limit(f"upload:owner:{known.id}", per_owner, 3600))
                mine = JobStore(session, known.id).list()
                active = [job for job in mine if job.status in ("queued", "running")]
        deps.limiter.hit(checks)
        with deps.sessions.begin() as session:
            everyone = len(system.jobs_in_progress(session))
        if everyone >= limits.max_jobs_in_progress:
            raise HTTPException(
                503,
                "The server is busy with other jobs. Please try again in a few minutes.",
                headers={"Retry-After": "120"},
            )
        if len(active) >= limits.max_active_jobs_per_owner:
            raise HTTPException(
                429,
                f"You already have {len(active)} jobs in progress. Wait for one to finish.",
                headers={"Retry-After": "60"},
            )
        chosen, text, fmt = _options(stems, request_text, output_format, quality_mode)
        check_name(file.filename, uploads)
        if file.size is not None:
            check_size(file.size, uploads)

        job_id = new_id()
        with tempfile.TemporaryDirectory(prefix="stems-upload-") as work:
            local = Path(work) / "upload"
            with local.open("wb") as out:
                while chunk := file.file.read(1024 * 1024):
                    out.write(chunk)
            inspected = inspect(local, uploads, isolate=deps.isolate_decoding)
            spec = contract.build_spec(
                job_id,
                created_at=now,
                stems=chosen,
                text=text,
                split_vocals=split_vocals,
                dereverb=dereverb,
                fmt=fmt,
                quality_mode=quality_mode,
                source_bytes=inspected.bytes,
            )
            # The row first: an object in storage always has a row that will expire it.
            # The owner is made here and not before, so a refused upload leaves nothing.
            with deps.sessions.begin() as session:
                owner = ensure_owner(request, response, session)
                owner_id = owner.id
                if known is None:
                    per_owner = limits.uploads_per_owner_per_hour
                    deps.limiter.hit([Limit(f"upload:owner:{owner_id}", per_owner, 3600)])
                store = JobStore(session, owner_id)
                job = store.create(
                    job_id=job_id,
                    request=spec["request"],
                    source_key=contract.source_key(job_id),
                    now=now,
                    expires_at=now + timedelta(days=defaults.retention.days),
                    display_name=display_name(file.filename),
                    source_bytes=inspected.bytes,
                    source_duration_s=inspected.info.duration_seconds,
                    source_sample_rate_hz=inspected.info.sample_rate,
                    source_channels=inspected.info.channels,
                )
                system.add_event(session, job, at=now, stage="uploaded")
            try:
                with local.open("rb") as handle:
                    deps.storage.put(
                        contract.source_key(job_id), handle, "application/octet-stream"
                    )
                put_json(deps.storage, contract.spec_key(job_id), spec)
            except StorageError:
                log.exception("job %s: the upload could not be stored", job_id)
                _abandon(job_id, owner_id, UPLOAD_FAILED)
                raise HTTPException(503, UPLOAD_FAILED, headers=_cookie(response)) from None
        try:
            handle_ = deps.provider.submit(spec)
        except (ProviderError, contract.ContractError, OSError):
            log.exception("job %s: the provider refused it", job_id)
            _abandon(job_id, owner_id, SUBMIT_FAILED)
            raise HTTPException(503, SUBMIT_FAILED, headers=_cookie(response)) from None
        with deps.sessions.begin() as session:
            job = JobStore(session, owner_id).get(job_id)
            assert job is not None
            job.provider, job.provider_handle = deps.provider.name, handle_
            job.stage, job.updated_at = "queued", deps.clock()
            system.add_event(session, job, at=deps.clock(), stage="queued")
            return view(job)

    def _cookie(response: Response) -> dict[str, str] | None:
        """The owner cookie just set, for an error reply: a job that failed to start is
        still its owner's to see."""
        cookie = response.headers.get("set-cookie")
        return None if cookie is None else {"set-cookie": cookie}

    def _abandon(job_id: str, owner_id: str, error: str) -> None:
        """A job that never started: remove what was stored and say why on its row."""
        try:
            deps.storage.delete_prefix(contract.job_prefix(job_id))
        except StorageError:
            log.exception("job %s: its objects could not be removed", job_id)
        with deps.sessions.begin() as session:
            job = JobStore(session, owner_id).get(job_id)
            if job is not None:
                job.status, job.stage, job.error = "failed", "failed", error
                job.updated_at = deps.clock()

    @router.get("/jobs")
    def list_jobs(request: Request) -> dict[str, Any]:
        with deps.sessions.begin() as session:
            owner = owner_of(request, session)
            if owner is None:
                return {"jobs": []}
            return {"jobs": [view(job) for job in JobStore(session, owner.id).list()]}

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        refresh(request, job_id)
        with deps.sessions.begin() as session:
            _, job = owned_job(request, session, job_id)
            return view(job)

    @router.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        try:
            after = int(request.headers.get("last-event-id") or 0)
        except ValueError:
            after = 0

        def read(last_id: int) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
            refresh(request, job_id)
            with deps.sessions.begin() as session:
                store, job = owned_job(request, session, job_id)
                events = [
                    {
                        "id": event.id,
                        "at": contract.timestamp(event.at),
                        "stage": event.stage,
                        "stem": event.stem,
                        "message": event.message,
                        "progress": event.progress,
                    }
                    for event in store.events(job_id, after_id=last_id)
                ]
                return view(job), events, job.status not in ("queued", "running")

        # Answer 404 before the stream starts, like every other job route.
        first = await run_in_threadpool(read, after)

        async def stream() -> AsyncIterator[str]:
            last_id, sent, current = after, None, first
            while True:
                snapshot, events, over = current
                for event in events:
                    last_id = event["id"]
                    yield f"id: {event['id']}\nevent: progress\ndata: {json.dumps(event)}\n\n"
                if snapshot != sent:
                    sent = snapshot
                    yield f"event: job\ndata: {json.dumps(snapshot)}\n\n"
                if over:
                    yield "event: end\ndata: {}\n\n"
                    return
                await anyio.sleep(defaults.api.poll_interval_seconds)
                if await request.is_disconnected():
                    return
                try:
                    current = await run_in_threadpool(read, last_id)
                except HTTPException:  # deleted while it was being watched
                    yield "event: end\ndata: {}\n\n"
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.post("/jobs/{job_id}/links")
    def links(job_id: str, request: Request) -> dict[str, Any]:
        ttl = defaults.retention.signed_url_ttl_seconds
        refresh(request, job_id)
        with deps.sessions.begin() as session:
            _, job = owned_job(request, session, job_id)
            if job.status != "ready":
                raise HTTPException(409, "This job has no files to download.")
            files = []
            for stem in job.stems:
                extension = stem.object_key.rsplit(".", 1)[-1]
                name = download_name(job.display_name, stem.name, extension)
                url = deps.storage.signed_url(stem.object_key, ttl, filename=name)
                files.append({"name": stem.name, "filename": name, "url": absolute(request, url)})
            archive = None
            zip_key = contract.zip_key(job.id)
            if deps.storage.exists(zip_key):
                name = download_name(job.display_name, "stems", "zip")
                url = deps.storage.signed_url(zip_key, ttl, filename=name)
                archive = {"filename": name, "url": absolute(request, url)}
            return {
                "expires_in": ttl,
                "expires_at": contract.timestamp(deps.clock() + timedelta(seconds=ttl)),
                "stems": files,
                "zip": archive,
            }

    @router.get("/jobs/{job_id}/stems/{name}/peaks")
    def peaks(job_id: str, name: str, request: Request) -> JSONResponse:
        with deps.sessions.begin() as session:
            store, job = owned_job(request, session, job_id)
            if store.stem(job.id, name) is None:
                raise HTTPException(404, NOT_FOUND)
        try:
            document = read_json(deps.storage, contract.peaks_key(job_id, name))
        except (ObjectNotFound, contract.ContractError):
            raise HTTPException(404, NOT_FOUND) from None
        return JSONResponse(document, headers={"Cache-Control": "private, max-age=3600"})

    @router.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, request: Request) -> Response:
        with deps.sessions.begin() as session:
            owner = owner_of(request, session)
            store = None if owner is None else JobStore(session, owner.id)
            job = None if store is None else store.get(job_id, include_ended=False)
            if store is None or job is None:
                raise HTTPException(404, NOT_FOUND)
            # Objects first: if that fails the row is untouched and the delete can be retried.
            purge(deps.storage, deps.provider, job)
            store.mark_deleted(job.id, deps.clock())
            deps.sync.forget(job.id)
        return Response(status_code=204)

    @router.delete("/me", status_code=204)
    def delete_me(request: Request) -> Response:
        with deps.sessions.begin() as session:
            owner = owner_of(request, session)
            if owner is not None:
                store = JobStore(session, owner.id)
                for job in store.list(include_ended=True):
                    purge(deps.storage, deps.provider, job)
                    deps.sync.forget(job.id)
                store.forget_everything()
        response = Response(status_code=204)
        response.delete_cookie(OWNER_COOKIE, path="/api")
        return response

    # -- downloads from local storage ------------------------------------------------------

    @router.api_route("/files/{token}", methods=["GET", "HEAD"])
    def download(token: str) -> FileResponse:
        verify = getattr(deps.storage, "verify", None)
        stat = getattr(deps.storage, "stat", None)
        if verify is None or stat is None:  # S3 and Azure links go straight to the bucket
            raise HTTPException(404, NOT_FOUND)
        try:
            link = verify(token)
            stored = stat(link.key)
        except LinkExpired:
            raise HTTPException(410, "This link has expired. Ask for a new one.") from None
        except (LinkInvalid, ObjectNotFound):
            raise HTTPException(404, NOT_FOUND) from None
        return FileResponse(
            stored.path,
            media_type=stored.content_type,
            headers={
                "Content-Disposition": content_disposition(link.filename),
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    app.include_router(router)
    return app
