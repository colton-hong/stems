"""The benchmark protocol, as one version number and the rules it stands for.

Bump ``PROTOCOL_VERSION`` whenever a rule below changes. Frozen track lists and cached
results carry the version they were made under, and neither is used under another one.

Version 1:

- Audio is scored at 44.1 kHz stereo, at full length, exactly as the dataset provides it.
- BSS Eval v4 from museval 0.4.1: image metrics, one distortion filter per track,
  non-overlapping 1-second windows, median over the windows museval scores.
- Every output is scored as the pair (target, input minus target); see
  ``stems_bench.metrics``.
- A model's ``other`` is scored against ``other_4stem`` (guitar, piano and the rest) if
  the model has no guitar and piano outputs of its own, and against the six-stem ``other``
  if it has.
- Vocal-split models are fed the reference vocals stem, de-reverb models the reference
  vocals with reverb; everything else is fed the mix. Stage models are therefore scored on
  their own, not on top of another model's errors.
- Across tracks, the reported figure is the median of the per-track medians; a track where
  the target is absent contributes nothing.
"""

from __future__ import annotations

PROTOCOL_VERSION = 1
SAMPLE_RATE = 44100

# Everything a dataset can hold a reference for.
TARGETS: tuple[str, ...] = (
    "vocals",
    "drums",
    "bass",
    "guitar",
    "piano",
    "other",
    "other_4stem",
    "instrumental",
    "lead_vocals",
    "backing_vocals",
    "dry",
)

# What a model is fed, by the outputs it emits.
INPUT_MIX = "mix"
INPUT_VOCALS = "vocals"
INPUT_VOCALS_WET = "vocals_wet"


def input_for(outputs: tuple[str, ...] | list[str]) -> str:
    """Which input a model with these outputs is fed."""
    if "lead_vocals" in outputs or "backing_vocals" in outputs:
        return INPUT_VOCALS
    if "dry" in outputs:
        return INPUT_VOCALS_WET
    return INPUT_MIX


def target_for(output: str, *, other_contains_guitar_piano: bool) -> str:
    """The reference an output is scored against."""
    if output == "other" and other_contains_guitar_piano:
        return "other_4stem"
    return output
