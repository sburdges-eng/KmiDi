"""Intent engine: infer_intent, derive_value_states, validate_invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .musical_intent import IntentAspect, IntentComponent, MusicalIntent
from .value_state import HarmonicEscalationBranches, RhythmicEscalationBranches, ValueState, ValueStateSet, ValueType
from .constraints import (
    ConstraintSet,
    ViolationList,
    constraint_overrides_intent,
    constraints_apply,
)
from .intent_result import IntentResult


def _aspect_from_input_type(input_type: str) -> IntentAspect:
    if input_type in ("chord_change", "harmonic"):
        return IntentAspect.HARMONIC
    if input_type == "rhythmic":
        return IntentAspect.RHYTHMIC
    if input_type == "dynamic":
        return IntentAspect.DYNAMIC
    if input_type == "tempo":
        return IntentAspect.TEMPO
    return IntentAspect.HARMONIC


def _value_type_from_aspect(aspect: IntentAspect) -> ValueType:
    mapping = {
        IntentAspect.HARMONIC: ValueType.HARMONIC_TENSION,
        IntentAspect.RHYTHMIC: ValueType.RHYTHMIC_DENSITY,
        IntentAspect.DYNAMIC: ValueType.DYNAMIC_RANGE,
        IntentAspect.TEMPO: ValueType.TEMPO_BIAS,
    }
    return mapping.get(aspect, ValueType.HARMONIC_TENSION)


@dataclass
class AbstractInput:
    input_type: str
    aspect: IntentAspect
    magnitude: float
    escalation_token: Optional[str] = None  # e.g. "more_harmonic_color" — only affects harmony


def infer_intent(input: AbstractInput) -> MusicalIntent:
    """Infer intent from abstract input. No audio, MIDI, or text parsing."""
    aspect = (
        _aspect_from_input_type(input.input_type)
        if input.input_type
        else input.aspect
    )
    return MusicalIntent.create_with_components(
        [IntentComponent(aspect=aspect, magnitude=input.magnitude, reason="inferred")]
    )


def derive_value_states(
    intent: MusicalIntent,
    constraints: Optional[ConstraintSet],
    out_states: ValueStateSet,
    out_violations: ViolationList,
) -> None:
    """Derive ValueStates from intent. Constraints override."""
    aspect_to_type = [
        (IntentAspect.TEMPO, ValueType.TEMPO_BIAS),
        (IntentAspect.HARMONIC, ValueType.HARMONIC_TENSION),
        (IntentAspect.RHYTHMIC, ValueType.RHYTHMIC_DENSITY),
        (IntentAspect.DYNAMIC, ValueType.DYNAMIC_RANGE),
    ]

    for aspect, vt in aspect_to_type:
        has_intent = intent.has_aspect(aspect)
        constrained = False
        if constraints:
            for c in constraints.iter_constraints():
                if constraint_overrides_intent(c, vt):
                    constrained = True
                    break

        if not has_intent:
            out_states.set(vt, ValueState.missing("no_intent"))
        else:
            mag = intent.get_magnitude(aspect)
            out_states.set(vt, ValueState.derive(vt, mag, constrained))

    if constraints:
        constraints_apply(constraints, out_states, out_violations)


def validate_invariants(result: IntentResult) -> bool:
    """Validate invariants. Fail explicitly on violation."""
    if not result.get_harmonic_escalation().satisfies_invariant():
        return False
    if not result.get_rhythmic_escalation().satisfies_invariant():
        return False
    for vt in ValueType:
        state = result.value_states.get(vt)
        if not state.validate(vt):
            return False
    return True


def _is_empty_input(input: AbstractInput) -> bool:
    return not input.input_type


def _compute_harmonic_escalation(input: AbstractInput, intent: MusicalIntent) -> HarmonicEscalationBranches:
    """Escalation v0: 'more_harmonic_color' only. Two mutually exclusive branches: 7ths OR one inversion."""
    if input.escalation_token != "more_harmonic_color":
        return HarmonicEscalationBranches.empty()
    if not intent.has_aspect(IntentAspect.HARMONIC):
        return HarmonicEscalationBranches.empty()
    return HarmonicEscalationBranches.more_harmonic_color()


def _compute_rhythmic_escalation(input: AbstractInput, intent: MusicalIntent) -> RhythmicEscalationBranches:
    """Escalation v1: 'more_groove' only. Three branches: forward push, backward layback, neutral."""
    if input.escalation_token != "more_groove":
        return RhythmicEscalationBranches.empty()
    if not intent.has_aspect(IntentAspect.RHYTHMIC):
        return RhythmicEscalationBranches.empty()
    return RhythmicEscalationBranches.more_groove()


def intent_engine_process(
    input: AbstractInput,
    constraints: Optional[ConstraintSet] = None,
) -> IntentResult:
    """Process input through intent inference and value state derivation."""
    if _is_empty_input(input):
        return IntentResult(
            intent=MusicalIntent.create(),
            value_states=ValueStateSet.create(),
            violations=ViolationList(),
            harmonic_escalation=HarmonicEscalationBranches.empty(),
            rhythmic_escalation=RhythmicEscalationBranches.empty(),
            metadata="abstain",
        )

    intent = infer_intent(input)
    states = ValueStateSet.create()
    violations = ViolationList()
    derive_value_states(intent, constraints, states, violations)
    h_esc = _compute_harmonic_escalation(input, intent)
    r_esc = _compute_rhythmic_escalation(input, intent)

    result = IntentResult(
        intent=intent,
        value_states=states,
        violations=violations,
        harmonic_escalation=h_esc,
        rhythmic_escalation=r_esc,
        metadata="processed",
    )

    if not validate_invariants(result):
        return result

    return result
