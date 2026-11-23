"""The retry state machine for one stem: assess, then deliver, retry, or warn.

    assessing ──pass──────────────────────────────▶ delivered
        │ ▲
        │ └──retry (a different attempt, at most ``max_retries_per_stem`` times)
        │
        └──nothing left to try, cap reached, or unverifiable──▶ delivered_with_warning

Each attempt is described by a ``Spec``: which models produce the stem and whether
de-reverb is applied to it. The three retry actions change it in three ways:

- ``retry_fallback_model``: the stem plan's fallback model, alone;
- ``retry_other_ensemble``: the next larger configured blend for the stem;
- ``retry_with_dereverb``: the same attempt with de-reverb applied (vocals only).

An action is on offer only if it leads to an attempt that has not been made. When the
stem ends in a warning, the attempt delivered is the best one made, not the last: highest
score, then highest pass probability, then the earliest.

The machine does no audio work. The caller runs ``machine.current``, has the result
assessed with ``machine.retries_used`` and ``machine.available()``, and hands the
assessment to ``machine.step``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from stems_common.settings import GateSettings
from stems_worker.decision import gate

BLEND_MODES: tuple[str, ...] = ("two_model_blend", "full_ensemble")
MODES: tuple[str, ...] = ("single_best", *BLEND_MODES)
RUNNING = "assessing"
DELIVERED = "delivered"
WARNED = "delivered_with_warning"


class GateError(RuntimeError):
    """The machine was driven in a way its states do not allow."""


@dataclass(frozen=True)
class Spec:
    """How one attempt at a stem is produced."""

    stem: str
    # One model, or the members of a blend in their configured order.
    models: tuple[str, ...]
    mode: str = "single_best"
    # The de-reverb model applied to the result, if any.
    dereverb: str | None = None

    def __post_init__(self) -> None:
        if not self.models or self.mode not in MODES:
            raise GateError(f"bad attempt for {self.stem}: {self.models} in mode {self.mode}")

    @property
    def model_used(self) -> str:
        """The attempt as one string, for ``model_used`` in schema (c)."""
        text = "+".join(self.models)
        return text if self.dereverb is None else f"{text}>{self.dereverb}"


@dataclass(frozen=True)
class Attempt:
    spec: Spec
    # The StemQualityAssessment made of it.
    assessment: dict[str, Any]


@dataclass(frozen=True)
class Step:
    action: str
    # The attempt to make next; None when the stem is finished.
    next: Spec | None
    # Index of the attempt that is delivered; None while still retrying.
    chosen: int | None


class StemGate:
    def __init__(
        self,
        first: Spec,
        *,
        settings: GateSettings,
        fallback_model: str | None = None,
        blends: Mapping[str, Sequence[str]] | None = None,
        dereverb_model: str | None = None,
    ) -> None:
        self.stem = first.stem
        self.settings = settings
        self.current = first
        self.state = RUNNING
        self.attempts: list[Attempt] = []
        self.chosen: int | None = None
        self._fallback = None if fallback_model in (None, "none") else fallback_model
        self._blends = {mode: tuple(models) for mode, models in (blends or {}).items()}
        self._dereverb = dereverb_model

    @property
    def retries_used(self) -> int:
        """Retries made before the current attempt."""
        return len(self.attempts) if self.state == RUNNING else len(self.attempts) - 1

    def _tried(self, spec: Spec) -> bool:
        # A blend is the same attempt whatever its mode is called.
        seen = [(a.spec.models, a.spec.dereverb) for a in self.attempts]
        seen.append((self.current.models, self.current.dereverb))
        return (spec.models, spec.dereverb) in seen

    def options(self) -> dict[str, Spec]:
        """The retry actions that lead to an attempt not made yet, and that attempt."""
        spec, found = self.current, {}
        if self._fallback is not None:
            other = Spec(self.stem, (self._fallback,), "single_best", spec.dereverb)
            if not self._tried(other):
                found["retry_fallback_model"] = other
        for mode in BLEND_MODES[
            BLEND_MODES.index(spec.mode) + 1 if spec.mode in BLEND_MODES else 0 :
        ]:
            members = self._blends.get(mode, ())
            blend = Spec(self.stem, members, mode, spec.dereverb) if len(members) > 1 else None
            if blend is not None and not self._tried(blend):
                found["retry_other_ensemble"] = blend
                break
        if self.stem == "vocals" and spec.dereverb is None and self._dereverb is not None:
            dried = replace(spec, dereverb=self._dereverb)
            if not self._tried(dried):
                found["retry_with_dereverb"] = dried
        return found

    def available(self) -> dict[str, str | None]:
        """``options`` as ``gate.verdict`` takes them: action to the model it would bring in."""
        models = {
            "retry_fallback_model": lambda s: s.models[0],
            "retry_other_ensemble": lambda s: None,
            "retry_with_dereverb": lambda s: s.dereverb,
        }
        return {action: models[action](spec) for action, spec in self.options().items()}

    def best(self) -> int:
        """The attempt to deliver when none passed."""

        def rank(index: int) -> tuple[float, float, int]:
            assessment = self.attempts[index].assessment["assessment"]
            return (assessment["score"], assessment["pass_probability"], -index)

        return max(range(len(self.attempts)), key=rank)

    def step(self, assessment: Mapping[str, Any]) -> Step:
        """Take the assessment of the current attempt and move to the next state."""
        if self.state != RUNNING:
            raise GateError(f"{self.stem} is already {self.state}")
        options = self.options()
        action = assessment["verdict"]["action"]
        self.attempts.append(Attempt(self.current, dict(assessment)))
        if action == gate.DELIVER:
            self.state, self.chosen = DELIVERED, len(self.attempts) - 1
            return Step(action, None, self.chosen)
        if action in gate.RETRY_ACTIONS:
            if len(self.attempts) > self.settings.max_retries_per_stem:
                raise GateError(f"{self.stem}: a retry was asked for beyond the cap")
            if action not in options:
                raise GateError(f"{self.stem}: {action} is not possible from here")
            self.current = options[action]
            return Step(action, self.current, None)
        self.state, self.chosen = WARNED, self.best()
        return Step(gate.WARN, None, self.chosen)

    @property
    def delivered(self) -> Attempt:
        if self.chosen is None:
            raise GateError(f"{self.stem} is still being assessed")
        return self.attempts[self.chosen]

    @property
    def warning(self) -> str | None:
        """What the user is told about the delivered stem; None when it passed."""
        if self.state != WARNED:
            return None
        return gate.warning(self.delivered.assessment["assessment"], self.settings)
