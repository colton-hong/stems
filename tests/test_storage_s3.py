"""The S3 backend against the real boto3 client, with no network.

botocore's ``Stubber`` checks every request's parameters against the S3 service model
and replays a canned reply, so these tests fail if the backend calls an operation that
does not exist or passes it a wrong parameter. Presigned URLs are computed locally by
boto3, so they are the real thing. What none of this proves is that a bucket accepts the
credentials: that is the live check (BLOCKED.md B7).
"""

from __future__ import annotations

import io
from urllib.parse import parse_qs, urlsplit

import pytest
import storage_fakes

from stems_api.storage import ObjectNotFound, StorageConfigError, StorageError, storage_from_env
from stems_api.storage.s3 import S3Storage, s3_from_env

boto3 = pytest.importorskip("boto3")
from botocore.config import Config  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402
from botocore.stub import ANY, Stubber  # noqa: E402

BUCKET = "stems-test"
KEY = "jobs/abc/stems/vocals.wav"
FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"
FAKE_SECRET = "fake-secret-key-used-only-in-tests-0000"


def real_client():
    return boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id=FAKE_KEY_ID,
        aws_secret_access_key=FAKE_SECRET,
        config=Config(signature_version="s3v4"),  # what s3_from_env sets
    )


@pytest.fixture
def stubbed():
    client = real_client()
    with Stubber(client) as stubber:
        yield S3Storage(client, BUCKET), stubber
        stubber.assert_no_pending_responses()


def test_exists_uses_head_object(stubbed) -> None:
    storage, stubber = stubbed
    stubber.add_response("head_object", {"ContentLength": 3}, {"Bucket": BUCKET, "Key": KEY})
    stubber.add_client_error(
        "head_object", "404", http_status_code=404, expected_params={"Bucket": BUCKET, "Key": KEY}
    )
    assert storage.exists(KEY) is True
    assert storage.exists(KEY) is False


def test_other_errors_are_not_mistaken_for_missing(stubbed) -> None:
    storage, stubber = stubbed
    stubber.add_client_error("head_object", "403", http_status_code=403)
    stubber.add_client_error("get_object", "AccessDenied", http_status_code=403)
    with pytest.raises(ClientError):
        storage.exists(KEY)
    with pytest.raises(ClientError):
        storage.open(KEY)


def test_open_reads_the_body_and_reports_a_missing_key(stubbed) -> None:
    storage, stubber = stubbed
    stubber.add_response("get_object", {"Body": io.BytesIO(b"abc")}, {"Bucket": BUCKET, "Key": KEY})
    stubber.add_client_error("get_object", "NoSuchKey", http_status_code=404)
    with storage.open(KEY) as body:
        assert body.read() == b"abc"
    with pytest.raises(ObjectNotFound):
        storage.open(KEY)


def test_put_uploads_with_the_content_type(stubbed) -> None:
    storage, stubber = stubbed
    # boto3 adds a checksum algorithm of its own to the upload; what matters here is
    # that the key and the content type arrive.
    expected = {
        "Bucket": BUCKET,
        "Key": KEY,
        "Body": ANY,
        "ContentType": "audio/wav",
        "ChecksumAlgorithm": ANY,
    }
    stubber.add_response("put_object", {}, expected)
    storage.put(KEY, io.BytesIO(b"RIFF...."), "audio/wav")


def test_delete_prefix_lists_under_the_folder_and_deletes_in_one_request(stubbed) -> None:
    storage, stubber = stubbed
    stubber.add_client_error("head_object", "404", http_status_code=404)
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "jobs/abc/source"}, {"Key": KEY}], "IsTruncated": False},
        {"Bucket": BUCKET, "Prefix": "jobs/abc/"},
    )
    stubber.add_response(
        "delete_objects",
        {},
        {
            "Bucket": BUCKET,
            "Delete": {"Objects": [{"Key": "jobs/abc/source"}, {"Key": KEY}], "Quiet": True},
        },
    )
    assert storage.delete_prefix("jobs/abc/") == 2


def test_delete_prefix_follows_pages_and_skips_the_delete_when_empty(stubbed) -> None:
    storage, stubber = stubbed
    stubber.add_client_error("head_object", "404", http_status_code=404)
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "jobs/abc/a"}], "IsTruncated": True, "NextContinuationToken": "t"},
        {"Bucket": BUCKET, "Prefix": "jobs/abc/"},
    )
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "jobs/abc/b"}], "IsTruncated": False},
        {"Bucket": BUCKET, "Prefix": "jobs/abc/", "ContinuationToken": "t"},
    )
    stubber.add_response("delete_objects", {}, None)
    assert storage.delete_prefix("jobs/abc") == 2
    # Nothing there: one existence check, one listing, no delete request.
    stubber.add_client_error("head_object", "404", http_status_code=404)
    stubber.add_response("list_objects_v2", {"KeyCount": 0, "IsTruncated": False}, None)
    assert storage.delete_prefix("jobs/abc") == 0


def test_delete_prefix_batches_by_a_thousand_and_reports_failures() -> None:
    client = storage_fakes.FakeS3Client()
    storage = S3Storage(client, BUCKET)
    for index in range(2301):
        client.objects[BUCKET, f"jobs/abc/part-{index:05d}"] = (b"x", "text/plain")
    assert storage.delete_prefix("jobs/abc") == 2301
    assert client.delete_batches == [1000, 1000, 301]
    assert client.objects == {}
    client.objects[BUCKET, "jobs/abc/locked"] = (b"x", "text/plain")
    client.undeletable.add("jobs/abc/locked")
    with pytest.raises(StorageError, match="could not be deleted"):
        storage.delete_prefix("jobs/abc")


def test_presigned_url_is_a_real_sigv4_link_with_expiry_and_download_name() -> None:
    storage = S3Storage(real_client(), BUCKET)
    url = storage.signed_url(KEY, 900, filename='Ma Chanson "été".wav')
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.scheme == "https"
    assert parts.path.endswith("/" + KEY)
    assert BUCKET in parts.netloc + parts.path
    assert query["X-Amz-Expires"] == ["900"]
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert len(query["X-Amz-Signature"][0]) == 64
    assert query["response-content-disposition"] == [
        'attachment; filename="Ma Chanson _t_.wav"; '
        "filename*=UTF-8''Ma%20Chanson%20%C3%A9t%C3%A9.wav"
    ]
    assert FAKE_SECRET not in url
    # No download name: nothing about disposition is signed into the link.
    assert "response-content-disposition" not in storage.signed_url(KEY, 60)


def env(values: dict[str, str]):
    return lambda name: values.get(name) or None


def test_factory_builds_a_client_for_a_custom_endpoint() -> None:
    storage = s3_from_env(
        env(
            {
                "S3_BUCKET": BUCKET,
                "S3_ENDPOINT_URL": "https://objects.example.invalid",
                "AWS_ACCESS_KEY_ID": FAKE_KEY_ID,
                "AWS_SECRET_ACCESS_KEY": FAKE_SECRET,
            }
        )
    )
    assert isinstance(storage, S3Storage)
    assert storage.bucket == BUCKET
    url = storage.signed_url(KEY, 60)
    # Path-style: the bucket is in the path, not the host name.
    assert url.startswith(f"https://objects.example.invalid/{BUCKET}/{KEY}?")
    assert FAKE_SECRET not in repr(storage)
    assert FAKE_SECRET not in url


def test_factory_reports_what_is_missing() -> None:
    with pytest.raises(StorageConfigError, match="S3_BUCKET"):
        s3_from_env(env({}))
    with pytest.raises(StorageConfigError, match="both"):
        s3_from_env(env({"S3_BUCKET": BUCKET, "AWS_ACCESS_KEY_ID": FAKE_KEY_ID}))
    with pytest.raises(ValueError, match="bucket"):
        S3Storage(real_client(), "")


def test_storage_from_env_selects_s3() -> None:
    values = {
        "STEMS_STORAGE_BACKEND": "s3",
        "S3_BUCKET": BUCKET,
        "AWS_ACCESS_KEY_ID": FAKE_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": FAKE_SECRET,
        "AWS_REGION": "eu-west-1",
    }
    storage = storage_from_env(env(values))
    assert isinstance(storage, S3Storage)
    assert "eu-west-1" in storage.signed_url(KEY, 60)
