"""Synthetic test tracks: sine-and-noise stems whose mix is known exactly.

Tests, the benchmark harness and examples use these instead of real music. Nothing here is
meant to sound good; each stem just has a distinct, predictable character so that a test
can tell them apart (bass is all low frequency, drums carry the energy above 8 kHz, and so
on) and so that silence, panning and reverb are all present somewhere.

Audio convention used throughout the project: ``float32`` arrays shaped
``(channels, samples)``, here always 2 channels.

Generation is deterministic for a given seed, numpy version and platform. Tracks are built
in memory only; no audio file is ever written to the repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stems_common.stems import PRODUCT_STEMS

DEFAULT_SAMPLE_RATE = 44100

# RMS level of each stem before mixing. Low enough that the mix never clips.
_STEM_RMS: dict[str, float] = {
    "vocals": 0.06,
    "drums": 0.05,
    "bass": 0.06,
    "guitar": 0.035,
    "piano": 0.035,
    "other": 0.03,
}

# Backing vocals enter this far into the track, rounded to a beat.
_BACKING_ENTRY_FRACTION = 0.5

_PENTATONIC = (0, 3, 5, 7, 10, 12)


@dataclass(frozen=True)
class SyntheticTrack:
    """A generated track. ``mix`` is the exact float32 sum of ``stems``."""

    track_id: str
    seed: int
    sample_rate: int
    stems: dict[str, np.ndarray]
    # vocals == lead_vocals + backing_vocals, exactly.
    lead_vocals: np.ndarray
    backing_vocals: np.ndarray
    # The vocals stem with synthetic reverb added (the dry stem is the reference).
    vocals_wet: np.ndarray

    @property
    def mix(self) -> np.ndarray:
        return _sum32([self.stems[name] for name in PRODUCT_STEMS])

    @property
    def mix_wet(self) -> np.ndarray:
        """The mix with reverberant vocals in place of the dry ones."""
        parts = [self.vocals_wet if n == "vocals" else self.stems[n] for n in PRODUCT_STEMS]
        return _sum32(parts)

    @property
    def num_samples(self) -> int:
        return int(self.stems["vocals"].shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sample_rate


def generate_track(
    seed: int = 0,
    duration_seconds: float = 6.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    active_stems: Sequence[str] = PRODUCT_STEMS,
) -> SyntheticTrack:
    """Build one track. Stems not in ``active_stems`` are exact silence.

    Each stem depends only on ``seed`` and its own name, so switching another stem off
    never changes it.
    """
    unknown = sorted(set(active_stems) - set(PRODUCT_STEMS))
    if unknown:
        raise ValueError(f"unknown stems {unknown}")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    n = round(duration_seconds * sample_rate)
    song = np.random.default_rng([seed, 0])
    beat = round(sample_rate * 60.0 / float(song.uniform(92.0, 132.0)))
    key = int(song.integers(0, 12))  # semitones above C

    silence = np.zeros((2, n), dtype=np.float32)
    builders = {
        "drums": _drums,
        "bass": _bass,
        "guitar": _guitar,
        "piano": _piano,
        "other": _other,
    }
    stems: dict[str, np.ndarray] = {}
    lead = backing = silence
    for index, name in enumerate(PRODUCT_STEMS, start=1):
        if name not in active_stems:
            stems[name] = silence.copy()
            continue
        rng = np.random.default_rng([seed, index])
        if name == "vocals":
            raw_lead, raw_backing = _vocals(rng, n, sample_rate, beat, key)
            gain = _gain_for(raw_lead + raw_backing, _STEM_RMS[name])
            lead = _finish(raw_lead * gain, sample_rate)
            backing = _finish(raw_backing * gain, sample_rate)
            stems[name] = lead + backing
        else:
            raw = builders[name](rng, n, sample_rate, beat, key)
            stems[name] = _finish(raw * _gain_for(raw, _STEM_RMS[name]), sample_rate)

    if "vocals" not in active_stems:
        lead, backing = silence.copy(), silence.copy()
    wet = add_reverb(stems["vocals"], sample_rate, seed=seed)
    return SyntheticTrack(
        track_id=f"synthetic-{seed:04d}",
        seed=seed,
        sample_rate=sample_rate,
        stems=stems,
        lead_vocals=lead,
        backing_vocals=backing,
        vocals_wet=wet,
    )


def add_reverb(
    audio: np.ndarray,
    sample_rate: int,
    *,
    seed: int = 0,
    rt60_seconds: float = 1.2,
    wet_db: float = -12.0,
) -> np.ndarray:
    """Add a synthetic reverb tail. The output has the same length as the input.

    The added (wet) part has an RMS ``wet_db`` relative to the dry signal, which matches the
    reverb probe in the research report (clean stem plus reverb at -12 dB).
    """
    channels, n = audio.shape
    dry = audio.astype(np.float64)
    if not np.any(dry):
        return audio.astype(np.float32, copy=True)

    rng = np.random.default_rng([seed, 99])
    ir_len = max(1, round(rt60_seconds * sample_rate))
    t = np.arange(ir_len) / sample_rate
    decay = np.exp(-6.9078 * t / rt60_seconds)  # -60 dB at rt60
    size = _fast_len(n + ir_len - 1)
    wet = np.empty_like(dry)
    for ch in range(channels):
        ir = rng.standard_normal(ir_len) * decay
        ir[0] = 0.0  # the direct sound is the dry signal itself
        spectrum = np.fft.rfft(dry[ch], size) * np.fft.rfft(ir, size)
        wet[ch] = np.fft.irfft(spectrum, size)[:n]
    wet *= _rms(dry) / _rms(wet) * 10.0 ** (wet_db / 20.0)
    return (dry + wet).astype(np.float32)


# --- helpers ------------------------------------------------------------------------------


def _sum32(parts: Sequence[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(parts[0], dtype=np.float32)
    for part in parts:
        total += part
    return total


def _fast_len(n: int) -> int:
    """The next power of two: FFTs of awkward lengths are many times slower."""
    return 1 << max(0, n - 1).bit_length()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _gain_for(x: np.ndarray, target_rms: float) -> float:
    level = _rms(x)
    return target_rms / level if level > 0 else 0.0


def _finish(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Fade the very ends (no clicks at the boundaries) and convert to float32."""
    n = x.shape[1]
    fade = min(n // 2, round(0.01 * sample_rate))
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        x = x.copy()
        x[:, :fade] *= ramp
        x[:, n - fade :] *= ramp[::-1]
    return x.astype(np.float32)


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _pan(mono: np.ndarray, position: float) -> np.ndarray:
    """Constant-power pan. -1 is hard left, 0 centre, +1 hard right."""
    angle = (position + 1.0) * np.pi / 4.0
    return np.stack([mono * np.cos(angle), mono * np.sin(angle)])


def _place(buffer: np.ndarray, start: int, signal: np.ndarray) -> None:
    """Add ``signal`` (channels, k) into ``buffer`` at ``start``, clipping at the end."""
    if start >= buffer.shape[1]:
        return
    k = min(signal.shape[1], buffer.shape[1] - start)
    buffer[:, start : start + k] += signal[:, :k]


def _edges(n: int, sample_rate: int, attack: float, release: float) -> np.ndarray:
    """An envelope that is 1 in the middle with raised-cosine ends."""
    env = np.ones(n)
    a = min(n // 2, max(1, round(attack * sample_rate)))
    r = min(n // 2, max(1, round(release * sample_rate)))
    env[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    env[n - r :] = 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
    return env


def _band_noise(rng: np.random.Generator, n: int, sample_rate: int, lo: float, hi: float):
    """White noise limited to [lo, hi] Hz."""
    size = _fast_len(n)
    spectrum = np.fft.rfft(rng.standard_normal(size))
    freqs = np.fft.rfftfreq(size, 1.0 / sample_rate)
    spectrum[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spectrum, size)[:n]


def _harmonic_tone(freq: np.ndarray | float, n: int, sample_rate: int, weights: np.ndarray):
    """Sum of harmonics of ``freq`` (a constant or a per-sample array), below Nyquist."""
    if np.isscalar(freq):
        phase = 2.0 * np.pi * float(freq) * np.arange(n) / sample_rate
        top = float(freq)
    else:
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        top = float(np.max(freq))
    tone = np.zeros(n)
    for k, weight in enumerate(weights, start=1):
        if k * top < 0.45 * sample_rate:
            tone += weight * np.sin(k * phase)
    return tone


# --- stems --------------------------------------------------------------------------------


def _bass(rng: np.random.Generator, n: int, sr: int, beat: int, key: int) -> np.ndarray:
    """One low, mostly sinusoidal note per beat. Centred. Almost nothing above 300 Hz."""
    out = np.zeros((2, n))
    weights = np.array([1.0, 0.35, 0.12])
    root = 28 + key  # E1 and up: fundamentals between 41 and 78 Hz
    for start in range(0, n, beat):
        freq = _hz(root + int(rng.choice(_PENTATONIC[:5])))
        t = np.arange(beat) / sr
        note = _harmonic_tone(freq, beat, sr, weights) * np.exp(-t / 0.45)
        _place(out, start, _pan(note * _edges(beat, sr, 0.005, 0.02), 0.0))
    return out


def _drums(rng: np.random.Generator, n: int, sr: int, beat: int, key: int) -> np.ndarray:
    """Kick on every beat, snare on the off-beats, hats on every half beat.

    The hats carry nearly all of the fixture's energy above 8 kHz.
    """
    out = np.zeros((2, n))
    half = max(1, beat // 2)

    k_len = round(0.30 * sr)
    t = np.arange(k_len) / sr
    sweep = 48.0 + 75.0 * np.exp(-t / 0.03)
    kick = np.sin(2.0 * np.pi * np.cumsum(sweep) / sr) * np.exp(-t / 0.11)

    s_len = round(0.22 * sr)
    t = np.arange(s_len) / sr
    snare_env = np.exp(-t / 0.06)

    h_len = round(0.07 * sr)
    hat_env = np.exp(-np.arange(h_len) / sr / 0.018)

    for index, start in enumerate(range(0, n, half)):
        on_beat = index % 2 == 0
        beat_number = index // 2
        if on_beat:
            _place(out, start, _pan(kick, 0.0))
            if beat_number % 2 == 1:
                body = 0.4 * np.sin(2.0 * np.pi * 190.0 * t)
                snare = (_band_noise(rng, s_len, sr, 180.0, 5000.0) + body) * snare_env
                _place(out, start, _pan(0.7 * snare, 0.1))
        hat = _band_noise(rng, h_len, sr, 8500.0, 16000.0) * hat_env
        _place(out, start, _pan(0.5 * hat, 0.35 if on_beat else -0.35))
    return out


def _vocals(rng: np.random.Generator, n: int, sr: int, beat: int, key: int):
    """A lead line with vibrato and rests, and panned backing voices in the later part."""
    formants = np.array([700.0, 1200.0, 2600.0])

    def voiced(freq: float, length: int, harmonics: int, vibrato: float) -> np.ndarray:
        t = np.arange(length) / sr
        f = freq * (1.0 + vibrato * np.sin(2.0 * np.pi * 5.5 * t))
        k = np.arange(1, harmonics + 1)
        weights = 0.05 + np.exp(-(((k[:, None] * freq - formants) / 260.0) ** 2)).sum(axis=1)
        return _harmonic_tone(f, length, sr, weights / k**0.5)

    lead = np.zeros((2, n))
    note_len = 2 * beat
    root = 57 + key  # A3 and up
    for index, start in enumerate(range(0, n, note_len)):
        if index % 4 == 3 or rng.random() < 0.15:
            continue  # a rest: the lead is silent here
        freq = _hz(root + int(rng.choice(_PENTATONIC)))
        tone = voiced(freq, note_len, 14, 0.006)
        breath = 0.03 * _band_noise(rng, note_len, sr, 4000.0, 7500.0)
        _place(lead, start, _pan((tone + breath) * _edges(note_len, sr, 0.03, 0.08), 0.0))

    backing = np.zeros((2, n))
    bar = 4 * beat
    entry = max(beat, round(n * _BACKING_ENTRY_FRACTION / beat) * beat)
    for start in range(entry, n, bar):
        degree = int(rng.choice((0, 5, 7)))
        for interval, position in ((4, -0.6), (7, 0.6)):
            tone = voiced(_hz(root + degree + interval), bar, 6, 0.003)
            _place(backing, start, _pan(0.5 * tone * _edges(bar, sr, 0.08, 0.15), position))
    return lead, backing


def _guitar(rng: np.random.Generator, n: int, sr: int, beat: int, key: int) -> np.ndarray:
    """Plucked, quickly decaying harmonic notes, panned left."""
    out = np.zeros((2, n))
    half = max(1, beat // 2)
    length = round(0.8 * sr)
    t = np.arange(length) / sr
    for start in range(0, n, half):
        if rng.random() > 0.7:
            continue
        freq = _hz(52 + key + int(rng.choice(_PENTATONIC)))
        note = np.zeros(length)
        for k in range(1, 11):
            if k * freq < 0.45 * sr:
                note += np.sin(2.0 * np.pi * k * freq * t) / k * np.exp(-t * k**0.7 / 0.45)
        _place(out, start, _pan(note * _edges(length, sr, 0.002, 0.02), -0.5))
    return out


def _piano(rng: np.random.Generator, n: int, sr: int, beat: int, key: int) -> np.ndarray:
    """Slowly decaying chords with slightly stretched partials, panned right."""
    out = np.zeros((2, n))
    length = round(1.5 * sr)
    t = np.arange(length) / sr
    inharmonicity = 4e-4
    for start in range(0, n, 2 * beat):
        root = 60 + key + int(rng.choice((0, 5, 7)))
        third = 3 if rng.random() < 0.5 else 4
        chord = np.zeros(length)
        for midi in (root, root + third, root + 7):
            freq = _hz(midi)
            for k in range(1, 9):
                partial = k * freq * np.sqrt(1.0 + inharmonicity * k * k)
                if partial < 0.45 * sr:
                    chord += np.sin(2.0 * np.pi * partial * t) / k**1.2 * np.exp(-t * k**0.5 / 1.1)
        _place(out, start, _pan(chord * _edges(length, sr, 0.004, 0.05), 0.5))
    return out


def _other(rng: np.random.Generator, n: int, sr: int, beat: int, key: int) -> np.ndarray:
    """A sustained detuned pad plus a quiet noise wash, different in each channel."""
    t = np.arange(n) / sr
    weights = 1.0 / np.arange(1, 7)
    out = np.zeros((2, n))
    for ch in range(2):
        pad = np.zeros(n)
        for midi, detune in ((60 + key, 0.997), (60 + key, 1.004), (67 + key, 1.0)):
            pad += _harmonic_tone(_hz(midi) * detune * (1.0 + 0.002 * ch), n, sr, weights)
        tremolo = 0.75 + 0.25 * np.sin(2.0 * np.pi * 0.3 * t + ch * np.pi / 2.0)
        wash = _band_noise(rng, n, sr, 300.0, 3000.0)
        out[ch] = pad * tremolo + 0.3 * _rms(pad) / max(_rms(wash), 1e-12) * wash
    return out
