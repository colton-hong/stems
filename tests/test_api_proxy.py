"""Behind a reverse proxy: whose address the rate limits count, and when a request is HTTPS.

The API is wrapped exactly as ``python -m stems_api`` has uvicorn wrap it, with uvicorn's
own middleware and the options ``proxy_options`` gives.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("uvicorn")

import api_helpers as h
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from stems_api import __main__ as serve
from stems_api.app import OWNER_COOKIE, create_app
from stems_common.settings import RateLimitSettings, load_defaults

PROXY = "10.0.0.5"
ALICE, BOB = "203.0.113.7", "198.51.100.9"


def served(tmp_path, trusted: str | None, **limits: int):
    """The app as it is served with ``--trusted-proxies trusted``."""
    base = load_defaults().rate_limits.model_dump()
    defaults = h.test_defaults(rate_limits=RateLimitSettings(**{**base, **limits}))
    app = create_app(h.make_deps(tmp_path, defaults=defaults), background=False)
    options = serve.proxy_options(trusted)
    if options["proxy_headers"]:
        app = ProxyHeadersMiddleware(app, trusted_hosts=options["forwarded_allow_ips"])
    return app


def parse(app, connection: str, **headers: str) -> int:
    """One request-parse from a fresh browser (no cookie) over a connection from ``connection``."""
    with TestClient(app, client=(connection, 40000)) as client:
        return client.post(
            "/api/requests/parse", json={"text": "vocals please"}, headers=headers
        ).status_code


def test_the_setting_is_a_list_of_addresses_and_nothing_else() -> None:
    assert serve.proxy_options(None) == {"proxy_headers": False}
    assert serve.proxy_options("  ") == {"proxy_headers": False}
    assert serve.proxy_options(" 10.0.0.5 , 10.1.0.0/16,::1") == {
        "proxy_headers": True,
        "forwarded_allow_ips": "10.0.0.5,10.1.0.0/16,::1",
    }
    for bad in ("*", "proxy.internal", "10.0.0.5, *"):
        with pytest.raises(ValueError, match="not an IP address or a network"):
            serve.proxy_options(bad)


def test_with_no_proxy_named_forwarded_headers_change_nothing(tmp_path) -> None:
    app = served(tmp_path, None, parses_per_ip_per_minute=1, parses_per_owner_per_minute=100)
    assert parse(app, PROXY, **{"x-forwarded-for": ALICE}) == 200
    # A different claimed address, the same connection: still the one address, already used up.
    assert parse(app, PROXY, **{"x-forwarded-for": BOB}) == 429
    # Even from this machine, which uvicorn would believe if left to its default.
    assert parse(app, "127.0.0.1", **{"x-forwarded-for": ALICE}) == 200
    assert parse(app, "127.0.0.1", **{"x-forwarded-for": BOB}) == 429


@pytest.mark.parametrize("trusted", [PROXY, "10.0.0.0/24", f"192.0.2.77, {PROXY}"])
def test_behind_a_trusted_proxy_each_visitor_has_their_own_allowance(tmp_path, trusted) -> None:
    app = served(tmp_path, trusted, parses_per_ip_per_minute=1, parses_per_owner_per_minute=100)
    assert parse(app, PROXY, **{"x-forwarded-for": ALICE}) == 200
    assert parse(app, PROXY, **{"x-forwarded-for": BOB}) == 200
    assert parse(app, PROXY, **{"x-forwarded-for": ALICE}) == 429
    assert parse(app, PROXY, **{"x-forwarded-for": BOB}) == 429


def test_a_visitor_cannot_pick_their_address_by_writing_the_header(tmp_path) -> None:
    app = served(tmp_path, PROXY, parses_per_ip_per_minute=1, parses_per_owner_per_minute=100)
    # Straight to the API, not through the proxy: the header is ignored.
    assert parse(app, ALICE, **{"x-forwarded-for": "192.0.2.1"}) == 200
    assert parse(app, ALICE, **{"x-forwarded-for": "192.0.2.2"}) == 429
    # Through the proxy, which appends the address it saw to whatever the visitor sent:
    # the last entry, the one the proxy wrote, is the one that counts.
    assert parse(app, PROXY, **{"x-forwarded-for": f"192.0.2.3, {BOB}"}) == 200
    assert parse(app, PROXY, **{"x-forwarded-for": f"192.0.2.4, {BOB}"}) == 429


def test_the_cookie_is_secure_when_a_trusted_proxy_says_the_visitor_used_https(tmp_path) -> None:
    def cookie(app, connection: str, **headers: str) -> str:
        with TestClient(app, client=(connection, 40000)) as client:
            response = client.post(
                "/api/jobs",
                files={"file": ("My Song.wav", h.wav_bytes(), "audio/wav")},
                data={"stems": "vocals"},
                headers=headers,
            )
            assert response.status_code == 201
            return next(
                value
                for name, value in response.headers.multi_items()
                if name == "set-cookie" and value.startswith(OWNER_COOKIE)
            )

    app = served(tmp_path, PROXY)
    forwarded = {"x-forwarded-for": ALICE, "x-forwarded-proto": "https"}
    assert "Secure" in cookie(app, PROXY, **forwarded)
    assert "Secure" not in cookie(app, PROXY, **{"x-forwarded-for": BOB})
    # The same claim from anywhere else is not believed.
    assert "Secure" not in cookie(app, "192.0.2.50", **forwarded)
    assert "Secure" not in cookie(served(tmp_path / "plain", None), PROXY, **forwarded)
