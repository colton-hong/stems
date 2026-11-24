"""In-memory stand-ins for the cloud storage clients, used by the storage contract tests.

Each one keeps objects in a dict and behaves like the real client for the calls the
backend makes, including the shape of its errors. ``tests/test_storage_s3.py`` and
``tests/test_storage_azure.py`` check the same calls against the real client libraries.
"""

from __future__ import annotations

import io
from typing import Any


class FakeClientError(Exception):
    """Shaped like ``botocore.exceptions.ClientError``."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeBody(io.BytesIO):
    """Stands in for botocore's ``StreamingBody``."""


class FakeS3Client:
    def __init__(self, page_size: int = 1000) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.page_size = page_size
        self.calls: list[str] = []
        self.delete_batches: list[int] = []
        self.undeletable: set[str] = set()

    # -- the six calls S3Storage makes ---------------------------------------------------

    def upload_fileobj(self, fileobj, bucket: str, key: str, ExtraArgs=None) -> None:
        self.calls.append("upload_fileobj")
        content_type = (ExtraArgs or {}).get("ContentType", "binary/octet-stream")
        self.objects[bucket, key] = (fileobj.read(), content_type)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append("get_object")
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("NoSuchKey", 404)
        data, content_type = self.objects[Bucket, Key]
        return {"Body": FakeBody(data), "ContentType": content_type, "ContentLength": len(data)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append("head_object")
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404", 404)
        data, content_type = self.objects[Bucket, Key]
        return {"ContentType": content_type, "ContentLength": len(data)}

    def get_paginator(self, operation: str) -> FakeS3Client:
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str):
        self.calls.append("list_objects_v2")
        keys = sorted(k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix))
        if not keys:
            yield {"KeyCount": 0}  # the real service sends no "Contents" for an empty page
            return
        for start in range(0, len(keys), self.page_size):
            yield {"Contents": [{"Key": k} for k in keys[start : start + self.page_size]]}

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("delete_objects")
        keys = [item["Key"] for item in Delete["Objects"]]
        assert 0 < len(keys) <= 1000, "DeleteObjects takes 1 to 1000 keys"
        self.delete_batches.append(len(keys))
        errors = []
        for key in keys:
            if key in self.undeletable:
                errors.append({"Key": key, "Code": "AccessDenied"})
            else:
                self.objects.pop((Bucket, key), None)
        return {"Errors": errors} if errors else {}

    def generate_presigned_url(self, operation: str, *, Params: dict, ExpiresIn: int) -> str:
        assert operation == "get_object"
        return (
            f"https://fake-s3.invalid/{Params['Bucket']}/{Params['Key']}?X-Amz-Expires={ExpiresIn}"
        )
