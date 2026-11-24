"""An in-memory Azure Blob service at the HTTP level, for tests.

It is an azure-core transport: the real ``azure-storage-blob`` client builds, signs and
sends its requests as usual, and this answers them from a dict. So the tests exercise
the SDK's real request building and response parsing for the five operations the backend
uses (put blob, get blob, blob properties, delete blob, list blobs). It does not check
the shared-key signature and knows nothing else about the service.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit
from xml.sax.saxutils import escape

from azure.core.pipeline.transport import HttpResponse, HttpTransport

ACCOUNT = "devaccount"
# Not a real key: base64 of "fake-key-used-only-in-tests-0000000000".
ACCOUNT_KEY = "ZmFrZS1rZXktdXNlZC1vbmx5LWluLXRlc3RzLTAwMDAwMDAwMDA="
CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT};AccountKey={ACCOUNT_KEY};"
    "EndpointSuffix=core.windows.net"
)
_DATE = "Mon, 23 Nov 2026 00:00:00 GMT"
_REASONS = {200: "OK", 201: "Created", 202: "Accepted", 206: "Partial Content", 404: "Not Found"}
_NOT_FOUND = (
    b'<?xml version="1.0" encoding="utf-8"?><Error><Code>BlobNotFound</Code>'
    b"<Message>The specified blob does not exist.</Message></Error>"
)


class _Stream:
    """What the SDK iterates to read a download."""

    def __init__(self, response: _Response, body: bytes) -> None:
        self.response = response
        self.request = response.request
        self.content_length = len(body)
        self._parts = iter([body] if body else [])

    def __len__(self) -> int:
        return self.content_length

    def __iter__(self) -> _Stream:
        return self

    def __next__(self) -> bytes:
        return next(self._parts)


class _Response(HttpResponse):
    def __init__(self, request: Any, status: int, headers: dict[str, str], body: bytes) -> None:
        super().__init__(request, None)
        self.status_code = status
        self.reason = _REASONS.get(status, "")
        self.headers = {"x-ms-request-id": "fake", "Date": _DATE, **headers}
        self.content_type = self.headers.get("Content-Type")
        self._body = body

    def body(self) -> bytes:
        return self._body

    def read(self) -> bytes:
        return self._body

    def stream_download(self, pipeline: Any, **kwargs: Any) -> _Stream:
        return _Stream(self, self._body)


class FakeBlobTransport(HttpTransport):
    def __init__(self) -> None:
        self.blobs: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.requests: list[tuple[str, str]] = []
        # Blob names that vanish just before their DELETE arrives (a race with a sweeper).
        self.vanish_on_delete: set[str] = set()

    def open(self) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> FakeBlobTransport:
        return self

    def __exit__(self, *args: object) -> None: ...

    def sleep(self, duration: float) -> None: ...

    def send(self, request: Any, **kwargs: Any) -> _Response:
        parts = urlsplit(request.url)
        query = parse_qs(parts.query)
        container, _, name = unquote(parts.path).lstrip("/").partition("/")
        self.requests.append((request.method, name))
        if request.method == "GET" and query.get("comp") == ["list"]:
            return self._list(request, container, query.get("prefix", [""])[0])
        if request.method == "PUT":
            data = request.data if request.data is not None else request.body
            if hasattr(data, "read"):
                data = data.read()
            content_type = request.headers.get("x-ms-blob-content-type", "application/octet-stream")
            self.blobs[container, name] = (bytes(data or b""), content_type)
            return _Response(request, 201, {"ETag": '"0x1"', "Last-Modified": _DATE}, b"")
        if request.method == "DELETE" and name in self.vanish_on_delete:
            self.blobs.pop((container, name), None)
        if (container, name) not in self.blobs:
            headers = {"x-ms-error-code": "BlobNotFound", "Content-Type": "application/xml"}
            return _Response(request, 404, headers, _NOT_FOUND)
        data, content_type = self.blobs[container, name]
        common = {
            "Content-Type": content_type,
            "ETag": '"0x1"',
            "Last-Modified": _DATE,
            "x-ms-blob-type": "BlockBlob",
            "Accept-Ranges": "bytes",
        }
        if request.method == "HEAD":
            return _Response(request, 200, {**common, "Content-Length": str(len(data))}, b"")
        if request.method == "GET":
            return self._get(request, common, data)
        if request.method == "DELETE":
            del self.blobs[container, name]
            return _Response(request, 202, {}, b"")
        raise AssertionError(f"unexpected request: {request.method} {parts.path}")

    def _get(self, request: Any, common: dict[str, str], data: bytes) -> _Response:
        wanted = request.headers.get("x-ms-range") or request.headers.get("Range")
        if not wanted:
            return _Response(request, 200, {**common, "Content-Length": str(len(data))}, data)
        first, _, last = wanted.removeprefix("bytes=").partition("-")
        start = int(first)
        end = min(int(last), len(data) - 1) if last else len(data) - 1
        if start >= len(data):
            headers = {"x-ms-error-code": "InvalidRange", "Content-Range": f"bytes */{len(data)}"}
            return _Response(request, 416, headers, b"")
        chunk = data[start : end + 1]
        headers = {
            **common,
            "Content-Length": str(len(chunk)),
            "Content-Range": f"bytes {start}-{end}/{len(data)}",
        }
        return _Response(request, 206, headers, chunk)

    def _list(self, request: Any, container: str, prefix: str) -> _Response:
        names = sorted(n for (c, n) in self.blobs if c == container and n.startswith(prefix))
        items = "".join(
            f"<Blob><Name>{escape(n)}</Name><Properties>"
            f"<Content-Length>{len(self.blobs[container, n][0])}</Content-Length>"
            "<BlobType>BlockBlob</BlobType></Properties></Blob>"
            for n in names
        )
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            f'<EnumerationResults ServiceEndpoint="https://{ACCOUNT}.blob.core.windows.net/" '
            f'ContainerName="{container}"><Prefix>{escape(prefix)}</Prefix>'
            f"<Blobs>{items}</Blobs><NextMarker /></EnumerationResults>"
        )
        return _Response(request, 200, {"Content-Type": "application/xml"}, xml.encode("utf-8"))
