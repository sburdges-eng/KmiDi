"""IntentResult: MusicalIntent, ValueStates, Violations, Metadata. Immutable."""

from __future__ import annotations

from dataclasses import dataclass

from .value_state import HarmonicEscalationBranches, RhythmicEscalationBranches, ValueStateSet, ValueType
from .musical_intent import MusicalIntent
from .constraints import ViolationList


@dataclass(frozen=True)
class IntentResult:
    """Immutable result. Must not be modified after creation."""

    intent: MusicalIntent
    value_states: ValueStateSet
    violations: ViolationList
    harmonic_escalation: HarmonicEscalationBranches
    rhythmic_escalation: RhythmicEscalationBranches  # MICRO_TIMING_ONLY: timing only, no added content
    metadata: str

    def get_intent(self) -> MusicalIntent:
        return self.intent

    def get_value_states(self) -> ValueStateSet:
        return self.value_states

    def get_violations(self) -> ViolationList:
        return self.violations

    def get_harmonic_escalation(self) -> HarmonicEscalationBranches:
        return self.harmonic_escalation

    def get_rhythmic_escalation(self) -> RhythmicEscalationBranches:
        return self.rhythmic_escalation

    def get_metadata(self) -> str:
        return self.metadata

    def is_valid(self) -> bool:
        for vt in ValueType:
            if self.value_states.get(vt).is_invalid():
                return False
        return True

    def has_violations(self) -> bool:
        return not self.violations.is_empty()
