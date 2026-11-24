"""``python -m stems_api``: serve the API with uvicorn.

Behind a reverse proxy the connection the API sees is the proxy's, so every visitor would
share one address for the rate limits and no request would look like HTTPS. A proxy says
who it is forwarding for in the ``X-Forwarded-For`` and ``X-Forwarded-Proto`` headers,
and anyone can write those headers, so they are believed only when the connection comes
from an address named in ``--trusted-proxies`` (or ``STEMS_TRUSTED_PROXIES``). With
nothing named they are ignored.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
from typing import Any

TRUSTED_PROXIES_ENV = "STEMS_TRUSTED_PROXIES"


def proxy_options(trusted: str | None) -> dict[str, Any]:
    """uvicorn's settings for forwarded headers, from a comma-separated list of proxies.

    Each entry is an IP address or a network in CIDR form. Anything else is refused,
    ``*`` included: trusting every address would let any client choose the address its
    requests are counted under.
    """
    entries = [part.strip() for part in (trusted or "").split(",") if part.strip()]
    if not entries:
        # Said outright: uvicorn's own default is to believe 127.0.0.1.
        return {"proxy_headers": False}
    for entry in entries:
        try:
            ipaddress.ip_network(entry, strict=False)
        except ValueError:
            raise ValueError(
                f"{TRUSTED_PROXIES_ENV}: {entry!r} is not an IP address or a network "
                "(like 10.0.0.5 or 10.0.0.0/24)"
            ) from None
    return {"proxy_headers": True, "forwarded_allow_ips": ",".join(entries)}


def main(argv: list[str] | None = None) -> int:
    from stems_common.env import get_env

    parser = argparse.ArgumentParser(prog="python -m stems_api", description="Serve the Stems API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--trusted-proxies",
        default=get_env(TRUSTED_PROXIES_ENV),
        help="comma-separated addresses or networks of the reverse proxies in front of the "
        f"API; forwarded headers are believed only from them (default: ${TRUSTED_PROXIES_ENV}, "
        "or none)",
    )
    args = parser.parse_args(argv)
    try:
        options = proxy_options(args.trusted_proxies)
    except ValueError as error:
        parser.error(str(error))

    import uvicorn

    from stems_api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info", **options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
