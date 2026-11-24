"""What the API says about a job: the JSON a browser gets."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePath
from typing import Any

from stems_api.db.models import Job, Stem
from stems_common import jobs as contract

REQUEST_FIELDS = ("stems", "text", "split_vocals", "dereverb", "format", "quality_mode")


def _quality(stem: Stem) -> dict[str, Any] | None:
    assessment = (stem.assessment or {}).get("assessment")
    if not isinstance(assessment, dict):
        return None
    return {
        "score": assessment.get("score"),
        "score_label": assessment.get("score_label"),
        "confidence": assessment.get("confidence"),
        "reason_code": assessment.get("reason_code"),
    }


def _stems(job: Job, live: dict[str, Any] | None) -> list[dict[str, Any]]:
    if job.status == "ready":
        return [
            {
                "name": stem.name,
                "state": "delivered_with_warning" if stem.warning else "delivered",
                "attempt": stem.attempts - 1,
                "format": stem.format,
                "bytes": stem.bytes,
                "warning": stem.warning,
                "quality": _quality(stem),
                "model_ids": stem.model_ids,
                "ensemble_mode": stem.ensemble_mode,
            }
            for stem in job.stems
        ]
    if job.status not in ("queued", "running"):
        return []
    asked = job.request.get("stems") or []
    states = (live or {}).get("stems") or {}
    names = list(states) or list(asked)
    return [
        {
            "name": name,
            "state": states.get(name, {}).get("state", "waiting"),
            "attempt": states.get(name, {}).get("attempt", 0),
        }
        for name in names
    ]


def job_view(job: Job, live: dict[str, Any] | None, now: datetime) -> dict[str, Any]:
    """The job as its owner sees it. ``live`` is the worker's latest status, if any."""
    in_progress = job.status in ("queued", "running")
    view: dict[str, Any] = {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "name": job.display_name,
        "request": {field: job.request.get(field) for field in REQUEST_FIELDS},
        "created_at": contract.timestamp(job.created_at),
        "updated_at": contract.timestamp(job.updated_at),
        "expires_at": contract.timestamp(job.expires_at),
        "seconds_until_expiry": max(0, int((job.expires_at - now).total_seconds())),
        "source": {
            "bytes": job.source_bytes,
            "duration_seconds": job.source_duration_s,
            "sample_rate_hz": job.source_sample_rate_hz,
            "channels": job.source_channels,
        },
        "progress": (live or {}).get("progress") if in_progress else None,
        "stems": _stems(job, live),
        "error": job.error,
    }
    if "parsed_request" in job.request:
        view["parsed_request"] = job.request["parsed_request"]
    return view


def download_name(display_name: str | None, label: str, extension: str) -> str:
    """``My Song (vocals).flac``: the uploaded file's name without its extension, then
    what this download is."""
    base = PurePath(display_name).stem if display_name else ""
    return f"{base or 'stems'} ({label}).{extension}"
