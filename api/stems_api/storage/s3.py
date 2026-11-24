"""Storage in an S3-compatible bucket (AWS S3, Cloudflare R2, Backblaze B2, MinIO).

The backend is given a boto3 S3 client and a bucket name and uses six client calls:
``upload_fileobj``, ``get_object``, ``head_object``, ``list_objects_v2`` (paginated),
``delete_objects`` and ``generate_presigned_url``. Links are presigned GET URLs, so a
download goes straight to the bucket and never through the API.

boto3 is imported only by ``s3_from_env``; the class itself needs nothing but the client.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, BinaryIO

from stems_api.storage.base import (
    ObjectNotFound,
    StorageError,
    check_key,
    check_prefix,
    check_ttl,
    content_disposition,
)
from stems_common.env import get_env

BUCKET_ENV = "S3_BUCKET"
ENDPOINT_ENV = "S3_ENDPOINT_URL"
KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
SECRET_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
REGION_ENVS = ("AWS_REGION", "AWS_DEFAULT_REGION")
# The most keys one DeleteObjects request may carry.
DELETE_BATCH = 1000
_MISSING = frozenset({"404", "NoSuchKey", "NotFound"})

Getter = Callable[[str], str | None]


def _error_code(exc: Exception) -> str | None:
    """The S3 error code of a botocore ``ClientError``, without importing botocore."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    code = response.get("Error", {}).get("Code")
    return None if code is None else str(code)


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class S3Storage:
    """The ``Storage`` protocol on one bucket."""

    def __init__(self, client: Any, bucket: str) -> None:
        if not bucket:
            raise ValueError("a bucket name is required")
        self._client = client
        self.bucket = bucket

    def __repr__(self) -> str:
        return f"S3Storage(bucket={self.bucket!r})"

    def put(self, key: str, data: BinaryIO, content_type: str) -> None:
        self._client.upload_fileobj(
            data, self.bucket, check_key(key), ExtraArgs={"ContentType": content_type}
        )

    def open(self, key: str) -> BinaryIO:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=check_key(key))
        except Exception as exc:
            if _error_code(exc) in _MISSING:
                raise ObjectNotFound(key) from exc
            raise
        return response["Body"]

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=check_key(key))
        except Exception as exc:
            if _error_code(exc) in _MISSING:
                return False
            raise
        return True

    def delete_prefix(self, prefix: str) -> int:
        trimmed = check_prefix(prefix)
        keys = [trimmed] if self.exists(trimmed) else []
        paginator = self._client.get_paginator("list_objects_v2")
        # The slash keeps "jobs/abc" from matching "jobs/abcd/...".
        for page in paginator.paginate(Bucket=self.bucket, Prefix=trimmed + "/"):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        for batch in _batches(keys, DELETE_BATCH):
            response = self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )
            failed = response.get("Errors") or []
            if failed:
                first = failed[0]
                raise StorageError(
                    f"{len(failed)} object(s) under {trimmed!r} could not be deleted; "
                    f"first: {first.get('Key')!r} ({first.get('Code')})"
                )
        return len(keys)

    def signed_url(self, key: str, ttl_seconds: int, filename: str | None = None) -> str:
        params = {"Bucket": self.bucket, "Key": check_key(key)}
        if filename is not None:
            params["ResponseContentDisposition"] = content_disposition(filename)
        return self._client.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=check_ttl(ttl_seconds)
        )


def s3_from_env(getenv: Getter = get_env) -> S3Storage:
    """An ``S3Storage`` from ``S3_BUCKET``, ``S3_ENDPOINT_URL`` and the AWS credentials.

    Credentials given in ``.env`` are passed to the client; with none there, boto3 looks
    in its usual places (environment, shared config, instance role).
    """
    from stems_api.storage.factory import StorageConfigError

    bucket = getenv(BUCKET_ENV)
    if not bucket:
        raise StorageConfigError(f"the s3 storage backend needs {BUCKET_ENV}")
    key_id, secret_key = getenv(KEY_ID_ENV), getenv(SECRET_KEY_ENV)
    if bool(key_id) != bool(secret_key):
        raise StorageConfigError(f"set both {KEY_ID_ENV} and {SECRET_KEY_ENV}, or neither")
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise StorageConfigError(
            "the s3 storage backend needs boto3: pip install -e '.[api]'"
        ) from exc
    endpoint = getenv(ENDPOINT_ENV)
    region = next((value for name in REGION_ENVS if (value := getenv(name))), None)
    options: dict[str, Any] = {"signature_version": "s3v4"}
    if endpoint:
        # What MinIO and most S3-compatible services expect: the bucket in the path, and
        # no checksum headers beyond the ones an operation requires (several services
        # reject the extra ones that newer boto3 sends by default).
        options["s3"] = {"addressing_style": "path"}
        options["request_checksum_calculation"] = "when_required"
        options["response_checksum_validation"] = "when_required"
    config = Config(**options)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region or ("auto" if endpoint else None),
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
        config=config,
    )
    return S3Storage(client, bucket)
