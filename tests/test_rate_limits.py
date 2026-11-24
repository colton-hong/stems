"""Rate limits: the limiter itself, then the routes that use it."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

import api_helpers as h
from fastapi.testclient import TestClient

from stems_api.app import create_app
from stems_api.ratelimit import Limit, RateLimited, RateLimiter
from stems_common.settings import RateLimitSettings, load_defaults


class Ticks:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# -- the limiter -------------------------------------------------------------------------


def test_n_events_per_window_then_refused_until_the_oldest_leaves() -> None:
    ticks = Ticks()
    limiter = RateLimiter(ticks)
    limit = Limit("upload:owner:a", 3, 60)
    for second in (0, 10, 20):
        ticks.now = 1000.0 + second
        limiter.hit([limit])
    ticks.now = 1030.0
    with pytest.raises(RateLimited) as caught:
        limiter.hit([limit])
    assert caught.value.retry_after == 30  # the first event leaves the window at 1060
    assert caught.value.limit == limit
    ticks.now = 1059.5
    with pytest.raises(RateLimited) as caught:
        limiter.hit([limit])
    assert caught.value.retry_after == 1  # rounded up, never zero
    ticks.now = 1060.0
    limiter.hit([limit])  # one slot is free again
    with pytest.raises(RateLimited) as caught:
        limiter.hit([limit])
    assert caught.value.retry_after == 10  # until the event at 1010 leaves


def test_a_refused_request_does_not_use_up_any_limit() -> None:
    ticks = Ticks()
    limiter = RateLimiter(ticks)
    owner, address = Limit("owner:a", 5, 60), Limit("ip:1.2.3.4", 1, 60)
    limiter.hit([owner, address])
    for _ in range(10):
        with pytest.raises(RateLimited) as caught:
            limiter.hit([owner, address])
        assert caught.value.limit == address
    # The owner limit was charged once, not eleven times.
    for _ in range(4):
        limiter.hit([owner])
    with pytest.raises(RateLimited):
        limiter.hit([owner])


def test_keys_are_separate_and_the_longest_wait_is_reported() -> None:
    ticks = Ticks()
    limiter = RateLimiter(ticks)
    limiter.hit([Limit("owner:a", 1, 60)])
    limiter.hit([Limit("owner:b", 1, 60)])  # another key: unaffected
    ticks.now += 5
    limiter.hit([Limit("ip:x", 1, 600)])
    ticks.now += 5
    with pytest.raises(RateLimited) as caught:
        limiter.hit([Limit("owner:a", 1, 60), Limit("ip:x", 1, 600)])
    assert caught.value.retry_after == 595
    assert caught.value.limit.key == "ip:x"


def test_a_limit_of_zero_refuses_everything_and_old_keys_are_forgotten() -> None:
    ticks = Ticks()
    limiter = RateLimiter(ticks)
    with pytest.raises(RateLimited) as caught:
        limiter.hit([Limit("parse:ip:x", 0, 60)])
    assert caught.value.retry_after == 60
    for index in range(100):
        limiter.hit([Limit(f"ip:{index}", 5, 60)])
    assert limiter.tracked_keys() == 100
    ticks.now += 61
    for index in range(100):
        limiter.hit([Limit(f"ip:{index}", 5, 60)])
    assert limiter.tracked_keys() == 100  # the old timestamps were dropped, not piled up
    assert len(limiter._events["ip:0"]) == 1


# -- the routes --------------------------------------------------------------------------


def limited(tmp_path, **limits: int):
    base = load_defaults().rate_limits.model_dump()
    defaults = h.test_defaults(rate_limits=RateLimitSettings(**{**base, **limits}))
    return h.make_deps(tmp_path, defaults=defaults)


def finish(deps) -> None:
    """Mark every job as over, so the in-progress limit is out of the picture."""
    with deps.sessions.begin() as session:
        from sqlalchemy import update

        from stems_api.db.models import Job

        session.execute(update(Job).values(status="failed", stage="failed"))


def test_uploads_per_owner_come_from_config_and_answer_429_with_retry_after(tmp_path) -> None:
    deps = limited(tmp_path, uploads_per_owner_per_hour=2, uploads_per_ip_per_hour=100)
    limiter_clock = Ticks()
    deps.limiter = RateLimiter(limiter_clock)
    with TestClient(create_app(deps, background=False)) as client:
        for _ in range(2):
            assert h.upload(client).status_code == 201
            finish(deps)
            limiter_clock.now += 600
        response = h.upload(client)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "2400"  # the first upload was 1200 s ago
        assert response.json() == {"detail": "Too many requests. Try again in 2400 seconds."}
        assert len(h.all_jobs(deps)) == 2
        # Another browser has its own allowance.
        with TestClient(create_app(deps, background=False)) as other:
            assert h.upload(other).status_code == 201
        limiter_clock.now += 2400
        assert h.upload(client).status_code == 201


def test_refused_uploads_count_too(tmp_path) -> None:
    deps = limited(tmp_path, uploads_per_owner_per_hour=3, uploads_per_ip_per_hour=100)
    with TestClient(create_app(deps, background=False)) as client:
        assert h.upload(client).status_code == 201  # the browser now has its cookie
        finish(deps)
        assert h.upload(client, name="notes.pdf").status_code == 415
        assert h.upload(client, b"junk" * 1000, name="junk.mp3").status_code == 415
        assert h.upload(client).status_code == 429


def test_uploads_per_address_hold_across_browsers(tmp_path) -> None:
    """Clearing cookies gives a new owner, and the same address."""
    deps = limited(tmp_path, uploads_per_owner_per_hour=100, uploads_per_ip_per_hour=3)
    app = create_app(deps, background=False)
    statuses = []
    for _ in range(5):
        with TestClient(app) as fresh_browser:
            statuses.append(h.upload(fresh_browser).status_code)
    assert statuses == [201, 201, 201, 429, 429]
    assert len({job.owner_id for job in h.all_jobs(deps)}) == 3
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200  # reading is not limited
        assert client.get("/api/jobs").status_code == 200


def test_jobs_in_progress_per_browser_are_capped(tmp_path) -> None:
    deps = limited(tmp_path, max_active_jobs_per_owner=2)
    with TestClient(create_app(deps, background=False)) as client:
        assert h.upload(client).status_code == 201
        assert h.upload(client).status_code == 201
        response = h.upload(client)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "60"
        assert "2 jobs in progress" in response.json()["detail"]
        finish(deps)
        assert h.upload(client).status_code == 201


def test_parse_is_limited_per_owner_and_per_address(tmp_path) -> None:
    deps = limited(tmp_path, parses_per_ip_per_minute=3, parses_per_owner_per_minute=2)
    with TestClient(create_app(deps, background=False)) as client:
        ask = lambda: client.post("/api/requests/parse", json={"text": "vocals"})  # noqa: E731
        assert [ask().status_code for _ in range(4)] == [200, 200, 200, 429]
    deps = limited(tmp_path / "second", parses_per_ip_per_minute=50, parses_per_owner_per_minute=2)
    (tmp_path / "second").mkdir(exist_ok=True)
    with TestClient(create_app(deps, background=False)) as client:
        h.upload(client)  # now the browser is a known owner
        ask = lambda: client.post("/api/requests/parse", json={"text": "vocals"})  # noqa: E731
        responses = [ask() for _ in range(3)]
        assert [r.status_code for r in responses] == [200, 200, 429]
        assert int(responses[-1].headers["retry-after"]) in range(1, 61)


def test_the_shipped_limits_are_the_documented_ones() -> None:
    limits = load_defaults().rate_limits
    assert limits.model_dump() == {
        "uploads_per_owner_per_hour": 20,
        "uploads_per_ip_per_hour": 40,
        "parses_per_owner_per_minute": 20,
        "parses_per_ip_per_minute": 40,
        "max_active_jobs_per_owner": 3,
    }
