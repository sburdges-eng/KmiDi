"""Constraint, ConstraintSet, enforcement rules, violation reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .value_state import ValueState, ValueStateSet, ValueType


class ConstraintKind(Enum):
    FIX_VALUE = "fix_value"
    FORBID_ASPECT = "forbid_aspect"
    LIMIT_RANGE = "limit_range"


@dataclass
class Constraint:
    kind: ConstraintKind
    target_type: ValueType
    min_val: float = 0.0
    max_val: float = 1.0
    fixed_val: float = 0.0
    forbid: bool = False
    reason: str = ""


@dataclass
class ConstraintSet:
    _constraints: List[Constraint] = field(default_factory=list)

    def add(self, c: Constraint) -> None:
        self._constraints.append(c)

    def clear(self) -> None:
        self._constraints.clear()

    def iter_constraints(self):
        return iter(self._constraints)


@dataclass
class Violation:
    value_type: ValueType
    message: str


@dataclass
class ViolationList:
    _violations: List[Violation] = field(default_factory=list)

    def add(self, value_type: ValueType, message: str) -> None:
        self._violations.append(Violation(value_type=value_type, message=message))

    def clear(self) -> None:
        self._violations.clear()

    def is_empty(self) -> bool:
        return len(self._violations) == 0

    def __len__(self) -> int:
        return len(self._violations)

    def __iter__(self):
        return iter(self._violations)


def constraint_overrides_intent(c: Constraint, value_type: ValueType) -> bool:
    return (
        c.target_type == value_type
        and c.kind
        in (ConstraintKind.FIX_VALUE, ConstraintKind.FORBID_ASPECT)
    )


def constraints_apply(
    constraints: ConstraintSet,
    states: ValueStateSet,
    violations: ViolationList,
) -> None:
    """Apply constraints. Constraints override intent and derivation."""
    for c in constraints.iter_constraints():
        state = states.get(c.target_type)
        if c.kind == ConstraintKind.FIX_VALUE:
            states.set(c.target_type, ValueState.present(c.fixed_val))
        elif c.kind == ConstraintKind.FORBID_ASPECT:
            if state.is_present():
                reason = c.reason if c.reason else "forbidden"
                states.set(c.target_type, ValueState.invalid(reason))
                violations.add(c.target_type, "Constraint forbids this aspect")
        elif c.kind == ConstraintKind.LIMIT_RANGE:
            if state.is_present() and (
                state.value < c.min_val or state.value > c.max_val
            ):
                clamped = c.min_val if state.value < c.min_val else c.max_val
                states.set(c.target_type, ValueState.present(clamped))
                violations.add(c.target_type, "Value clamped to constraint range")
