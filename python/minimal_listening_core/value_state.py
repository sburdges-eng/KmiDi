"""ValueState: PRESENT, MISSING, INVALID. Derive, validate, preserve absence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Union


class ValueStateKind(Enum):
    PRESENT = "present"
    MISSING = "missing"
    INVALID = "invalid"


class ValueType(Enum):
    TEMPO_BIAS = "tempo_bias"
    HARMONIC_TENSION = "harmonic_tension"
    RHYTHMIC_DENSITY = "rhythmic_density"
    DYNAMIC_RANGE = "dynamic_range"


# HARMONIC ESCALATION RULE (codified):
#
# One branch may choose:
#   - triad + inversion
#   OR
#   - triad + 7th
#
# Never both.
# Never more than one added harmonic degree.
#
# Forbidden: 7ths+inversions, multiple extensions (9th, 11th), cascading "well since we can..."


class HarmonicEscalationBranch(Enum):
    """Each branch = exactly one added harmonic degree."""

    SEVENTHS = "sevenths"  # triad + 7th. One added degree.
    ONE_INVERSION = "one_inversion"  # triad + inversion. One added degree.

    def added_degrees(self) -> int:
        """Returns added harmonic degrees. Must be <= MAX_ADDED_DEGREES_PER_BRANCH (1)."""
        return 1  # Each valid variant = exactly one.


class HarmonicEscalationBranches:
    """Mutually exclusive branches; pick one. Never more than one added degree per branch."""

    MAX_BRANCHES = 2
    MAX_ADDED_DEGREES_PER_BRANCH = 1

    def __init__(self, branches: Optional[list] = None) -> None:
        self._branches = list(branches) if branches else []

    @classmethod
    def empty(cls) -> "HarmonicEscalationBranches":
        return cls([])

    @classmethod
    def more_harmonic_color(cls) -> "HarmonicEscalationBranches":
        return cls([
            HarmonicEscalationBranch.SEVENTHS,
            HarmonicEscalationBranch.ONE_INVERSION,
        ])

    def satisfies_invariant(self) -> bool:
        """Invariant: at most one choice per branch; never more than one added degree per branch."""
        if len(self._branches) > self.MAX_BRANCHES:
            return False
        for branch in self._branches:
            if branch.added_degrees() > self.MAX_ADDED_DEGREES_PER_BRANCH:
                return False
        return True

    def __iter__(self):
        return iter(self._branches)

    def __len__(self) -> int:
        return len(self._branches)


# RHYTHMIC ESCALATION RULE (codified):
#
# MICRO_TIMING_ONLY: push/pull, swing bias, humanization window.
#
# Does NOT: add notes, change pattern, increase density, add fills,
# change velocity curves beyond a small window.
#
# Branches may differ by: forward push, backward layback, neutral.
# No branch may be "busier".


class RhythmicEscalationBranch(Enum):
    """Each branch = timing only. No added content."""

    FORWARD_PUSH = "forward_push"
    BACKWARD_LAYBACK = "backward_layback"
    NEUTRAL = "neutral"


class RhythmicEscalationBranches:
    """Mutually exclusive branches; pick one. MICRO_TIMING_ONLY."""

    MAX_BRANCHES = 3

    def __init__(self, branches: Optional[list] = None) -> None:
        self._branches = list(branches) if branches else []

    @classmethod
    def empty(cls) -> "RhythmicEscalationBranches":
        return cls([])

    @classmethod
    def more_groove(cls) -> "RhythmicEscalationBranches":
        return cls([
            RhythmicEscalationBranch.FORWARD_PUSH,
            RhythmicEscalationBranch.BACKWARD_LAYBACK,
            RhythmicEscalationBranch.NEUTRAL,
        ])

    def satisfies_invariant(self) -> bool:
        """Invariant: timing only. No branch may be "busier"."""
        return len(self._branches) <= self.MAX_BRANCHES

    def __iter__(self):
        return iter(self._branches)

    def __len__(self) -> int:
        return len(self._branches)


@dataclass(frozen=True)
class ValueState:
    """ValueState<T>: PRESENT(T) | MISSING(reason) | INVALID(reason)."""

    kind: ValueStateKind
    value: float = 0.0
    reason: str = ""

    @classmethod
    def present(cls, value: float) -> ValueState:
        return cls(ValueStateKind.PRESENT, value, "")

    @classmethod
    def missing(cls, reason: str) -> ValueState:
        return cls(ValueStateKind.MISSING, 0.0, reason)

    @classmethod
    def invalid(cls, reason: str) -> ValueState:
        return cls(ValueStateKind.INVALID, 0.0, reason)

    def is_present(self) -> bool:
        return self.kind == ValueStateKind.PRESENT

    def is_missing(self) -> bool:
        return self.kind == ValueStateKind.MISSING

    def is_invalid(self) -> bool:
        return self.kind == ValueStateKind.INVALID

    def get_value(self) -> float:
        return self.value if self.kind == ValueStateKind.PRESENT else 0.0

    def get_reason(self) -> str:
        return self.reason

    @classmethod
    def derive(
        cls, value_type: ValueType, intent_magnitude: float, constrained: bool
    ) -> ValueState:
        if constrained:
            return cls.missing("constrained")
        if intent_magnitude < 0.0 or intent_magnitude > 1.0:
            return cls.invalid("magnitude_out_of_range")
        return cls.present(intent_magnitude)

    def validate(self, value_type: ValueType) -> bool:
        if self.kind == ValueStateKind.INVALID:
            return False
        if self.kind == ValueStateKind.PRESENT:
            return 0.0 <= self.value <= 1.0
        return True


@dataclass
class ValueStateSet:
    """Set of ValueStates for common types."""

    _states: Dict[ValueType, ValueState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._states:
            for vt in ValueType:
                self._states[vt] = ValueState.missing("not_derived")

    @classmethod
    def create(cls) -> ValueStateSet:
        return cls()

    def set(self, value_type: ValueType, state: ValueState) -> None:
        self._states[value_type] = state

    def get(self, value_type: ValueType) -> ValueState:
        return self._states.get(
            value_type, ValueState.missing("invalid_type")
        )

    def preserve_absence(self, other: ValueStateSet) -> None:
        for vt, state in other._states.items():
            if state.is_missing() or state.is_invalid():
                self._states[vt] = state
