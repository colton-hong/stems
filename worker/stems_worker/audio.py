"""Audio decoding, resampling and encoding.

Everything inside the pipeline is a ``float32`` array shaped ``(channels, samples)`` at
44.1 kHz stereo. This module is the only place that converts between files and that form.

Decoding tries libsndfile first (WAV, FLAC, AIFF, OGG, MP3) and falls back to PyAV, whose
wheels bundle FFmpeg's codecs (M4A/AAC, Opus and the rest), so no system ffmpeg is needed.
Stereo is never summed: mono is duplicated to both channels, and material with more than
two channels is downmixed to stereo by FFmpeg's resampler.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np
import soundfile as sf
import soxr

PIPELINE_SAMPLE_RATE = 44100

OutputFormat = Literal["wav", "flac", "mp3_320", "mp3_192"]

_EXTENSIONS: dict[str, str] = {"wav": ".wav", "flac": ".flac", "mp3_320": ".mp3", "mp3_192": ".mp3"}
_MP3_BITRATES: dict[str, int] = {"mp3_320": 320_000, "mp3_192": 192_000}
_MP3_FRAME = 1152


class AudioError(Exception):
    """Base class for audio I/O problems that are the input's fault, not a bug."""


class UnsupportedAudioError(AudioError):
    """The file is not decodable audio."""


class AudioTooLongError(AudioError):
    """The audio is longer than the caller allows."""


@dataclass(frozen=True)
class SourceInfo:
    """What the file contained before any conversion."""

    sample_rate: int
    channels: int
    # None when the container does not say and the file has not been decoded yet.
    num_samples: int | None
    duration_seconds: float | None
    codec: str
    decoder: Literal["soundfile", "pyav"]


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio in pipeline form, plus what it was decoded from."""

    samples: np.ndarray  # float32, (2, n)
    sample_rate: int
    source: SourceInfo

    @property
    def duration_seconds(self) -> float:
        return self.samples.shape[1] / self.sample_rate


@dataclass(frozen=True)
class SaveResult:
    path: Path
    format: str
    peak: float
    # Samples beyond full scale. Integer formats clip these; callers decide whether to
    # lower the gain first (the same gain for every stem, so stems still sum to the mix).
    clipped_samples: int


def extension_for(fmt: str) -> str:
    try:
        return _EXTENSIONS[fmt]
    except KeyError:
        raise ValueError(f"unknown output format {fmt!r}") from None


# --- probing and decoding -----------------------------------------------------------------


def probe(path: Path | str) -> SourceInfo:
    """Read the file's header without decoding the audio."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        info = sf.info(str(path))
    except sf.SoundFileError:
        return _probe_pyav(path)
    return SourceInfo(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        num_samples=int(info.frames),
        duration_seconds=info.frames / info.samplerate if info.samplerate else None,
        codec=f"{info.format}/{info.subtype}",
        decoder="soundfile",
    )


def _probe_pyav(path: Path) -> SourceInfo:
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise UnsupportedAudioError(f"{path.name}: no audio stream")
            stream = container.streams.audio[0]
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            rate = int(stream.sample_rate)
            return SourceInfo(
                sample_rate=rate,
                channels=int(stream.channels),
                num_samples=round(duration * rate) if duration is not None else None,
                duration_seconds=duration,
                codec=stream.codec_context.name,
                decoder="pyav",
            )
    except av.error.FFmpegError as e:
        raise UnsupportedAudioError(f"{path.name}: not decodable audio ({e})") from e


def load_audio(
    path: Path | str,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
    *,
    max_duration_seconds: float | None = None,
) -> LoadedAudio:
    """Decode a file to float32 stereo at ``sample_rate``.

    ``max_duration_seconds`` is enforced while decoding, so a file that lies about its
    length in the header cannot make this function allocate without bound.
    """
    path = Path(path)
    info = probe(path)
    if info.sample_rate <= 0 or info.channels <= 0:
        raise UnsupportedAudioError(f"{path.name}: invalid sample rate or channel count")
    if (
        max_duration_seconds is not None
        and info.duration_seconds is not None
        and info.duration_seconds > max_duration_seconds
    ):
        raise AudioTooLongError(
            f"{path.name}: {info.duration_seconds:.1f} s exceeds {max_duration_seconds:.1f} s"
        )

    if info.decoder == "soundfile" and info.channels <= 2:
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        samples = np.ascontiguousarray(data.T)
    else:
        # PyAV also handles the downmix of anything wider than stereo.
        samples = _decode_pyav(path, max_duration_seconds)

    if samples.shape[1] == 0:
        raise UnsupportedAudioError(f"{path.name}: contains no audio samples")
    if not np.isfinite(samples).all():
        raise UnsupportedAudioError(f"{path.name}: contains non-finite samples")

    source = SourceInfo(
        sample_rate=info.sample_rate,
        channels=info.channels,
        num_samples=int(samples.shape[1]),
        duration_seconds=samples.shape[1] / info.sample_rate,
        codec=info.codec,
        decoder=info.decoder if info.channels <= 2 else "pyav",
    )
    samples = resample(to_stereo(samples), info.sample_rate, sample_rate)
    return LoadedAudio(samples=samples, sample_rate=sample_rate, source=source)


def _decode_pyav(path: Path, max_duration_seconds: float | None) -> np.ndarray:
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise UnsupportedAudioError(f"{path.name}: no audio stream")
            stream = container.streams.audio[0]
            rate = int(stream.sample_rate)
            layout = "mono" if stream.channels == 1 else "stereo"
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
            limit = None if max_duration_seconds is None else int(max_duration_seconds * rate)
            chunks: list[np.ndarray] = []
            total = 0

            def take(frames) -> None:
                nonlocal total
                for frame in frames:
                    block = frame.to_ndarray()
                    chunks.append(block)
                    total += block.shape[1]
                    if limit is not None and total > limit:
                        raise AudioTooLongError(
                            f"{path.name}: longer than {max_duration_seconds:.1f} s"
                        )

            for frame in container.decode(stream):
                take(resampler.resample(frame))
            take(resampler.resample(None))
    except av.error.FFmpegError as e:
        raise UnsupportedAudioError(f"{path.name}: not decodable audio ({e})") from e
    if not chunks:
        return np.zeros((1, 0), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float32)


# --- channel and rate conversion ----------------------------------------------------------


def to_stereo(samples: np.ndarray) -> np.ndarray:
    """Return ``(2, n)``. Mono is copied to both channels at unchanged level."""
    if samples.ndim != 2:
        raise ValueError(f"expected (channels, samples), got shape {samples.shape}")
    if samples.shape[0] == 2:
        return samples
    if samples.shape[0] == 1:
        return np.repeat(samples, 2, axis=0)
    raise ValueError(f"cannot convert {samples.shape[0]} channels to stereo here")


def resampled_length(num_samples: int, source_rate: int, target_rate: int) -> int:
    return round(num_samples * target_rate / source_rate)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited resampling (soxr, very high quality). Channels stay independent.

    The output length is exactly ``resampled_length(...)``.
    """
    if samples.ndim != 2:
        raise ValueError(f"expected (channels, samples), got shape {samples.shape}")
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if source_rate == target_rate:
        return samples
    out = soxr.resample(samples.T, source_rate, target_rate, quality="VHQ").T
    want = resampled_length(samples.shape[1], source_rate, target_rate)
    if out.shape[1] > want:
        out = out[:, :want]
    elif out.shape[1] < want:
        out = np.pad(out, ((0, 0), (0, want - out.shape[1])))
    return np.ascontiguousarray(out, dtype=np.float32)


# --- encoding -----------------------------------------------------------------------------


def save_audio(
    path: Path | str,
    samples: np.ndarray,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
    fmt: OutputFormat = "wav",
) -> SaveResult:
    """Write ``samples`` to ``path`` as 24-bit WAV, 24-bit FLAC, or constant-bitrate MP3.

    The file is written to a temporary name and moved into place, so a crash never leaves
    a half-written file that a re-run would mistake for a finished one.
    """
    path = Path(path)
    if fmt not in _EXTENSIONS:
        raise ValueError(f"unknown output format {fmt!r}")
    if samples.ndim != 2 or samples.shape[0] not in (1, 2):
        raise ValueError(f"expected (1 or 2, samples), got shape {samples.shape}")
    if samples.shape[1] == 0:
        raise ValueError("refusing to write an empty audio file")
    if not np.isfinite(samples).all():
        raise ValueError("refusing to write non-finite samples")

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    magnitude = np.abs(samples)
    peak = float(magnitude.max())
    clipped = int(np.count_nonzero(magnitude > 1.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        if fmt in ("wav", "flac"):
            sf.write(
                str(partial),
                _to_pcm24(samples).T,
                sample_rate,
                format=fmt.upper(),
                subtype="PCM_24",
            )
        else:
            _write_mp3(partial, np.clip(samples, -1.0, 1.0), sample_rate, _MP3_BITRATES[fmt])
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return SaveResult(path=path, format=fmt, peak=peak, clipped_samples=clipped)


def _to_pcm24(samples: np.ndarray) -> np.ndarray:
    """Round to the nearest 24-bit step and clip, as int32 with the value in the top 24 bits.

    Done here rather than left to libsndfile, whose float conversion truncates instead of
    rounding and so can be off by a whole step.
    """
    scaled = np.round(samples.astype(np.float64) * 2.0**23)
    return np.clip(scaled, -(2**23), 2**23 - 1).astype(np.int32) << 8


def _write_mp3(path: Path, samples: np.ndarray, sample_rate: int, bit_rate: int) -> None:
    layout = "mono" if samples.shape[0] == 1 else "stereo"
    with av.open(str(path), "w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=sample_rate)
        stream.codec_context.format = "fltp"
        stream.codec_context.layout = layout
        stream.codec_context.bit_rate = bit_rate
        for start in range(0, samples.shape[1], _MP3_FRAME):
            block = np.ascontiguousarray(samples[:, start : start + _MP3_FRAME])
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
