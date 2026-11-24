"""Checks on an upload before it becomes a job.

Three questions, cheapest first: is the name one of the allowed audio types, is the file
small enough, and is it really audio of an allowed length. The last is answered by the
decoders, not by the name: the header is parsed and the first second is decoded, so a
PDF renamed to ``.mp3`` is refused. Tags are not read and nothing fingerprints the audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stems_common.settings import UploadSettings
from stems_worker.audio import AudioError, SourceInfo, decode_head, probe

MAX_DISPLAY_NAME = 255


class UploadRejected(Exception):
    """The upload is refused. ``status_code`` is the HTTP status that says why."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class Inspected:
    info: SourceInfo
    bytes: int


def megabytes(count: int) -> str:
    return f"{count / (1024 * 1024):.0f} MB"


def display_name(filename: str | None) -> str | None:
    """The uploaded file's own name, without any folder, for showing back to the user."""
    if not filename:
        return None
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:MAX_DISPLAY_NAME] or None


def check_name(filename: str | None, settings: UploadSettings) -> str:
    """The file's extension if it is an allowed one; 415 otherwise."""
    extension = Path(filename or "").suffix.lower().lstrip(".")
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise UploadRejected(415, f"Unsupported file type. Upload one of: {allowed}.")
    return extension


def check_size(size: int, settings: UploadSettings) -> None:
    if size <= 0:
        raise UploadRejected(400, "The file is empty.")
    if size > settings.max_bytes:
        raise UploadRejected(
            413, f"The file is larger than the limit of {megabytes(settings.max_bytes)}."
        )


def inspect(path: Path, settings: UploadSettings) -> Inspected:
    """Confirm ``path`` holds audio within the limits. Raises ``UploadRejected`` if not."""
    size = path.stat().st_size
    check_size(size, settings)
    try:
        info = probe(path)
        if info.sample_rate <= 0 or info.channels <= 0:
            raise AudioError("no usable audio stream")
        decode_head(path)
    except AudioError:
        raise UploadRejected(
            415, "The file could not be read as audio. It may be damaged or not an audio file."
        ) from None
    if info.duration_seconds is not None and info.duration_seconds > settings.max_duration_seconds:
        minutes = settings.max_duration_seconds / 60
        raise UploadRejected(422, f"The audio is longer than the limit of {minutes:.0f} minutes.")
    return Inspected(info=info, bytes=size)
