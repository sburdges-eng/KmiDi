use crate::musical_intent::{IntentAspect, IntentComponent, MusicalIntent};
use crate::value_state::{HarmonicEscalationBranches, RhythmicEscalationBranches, ValueState, ValueStateSet, ValueType};
use crate::constraints::{constraint_overrides_intent, constraints_apply, ConstraintSet, ViolationList};
use crate::intent_result::IntentResult;

#[derive(Debug, Clone)]
pub struct AbstractInput {
    pub input_type: String,
    pub aspect: IntentAspect,
    pub magnitude: f64,
    /// e.g. "more_harmonic_color" — only affects harmony
    pub escalation_token: Option<String>,
}

fn aspect_from_input_type(input_type: &str) -> IntentAspect {
    match input_type {
        "chord_change" | "harmonic" => IntentAspect::Harmonic,
        "rhythmic" => IntentAspect::Rhythmic,
        "dynamic" => IntentAspect::Dynamic,
        "tempo" => IntentAspect::Tempo,
        _ => IntentAspect::Harmonic,
    }
}

pub fn infer_intent(input: &AbstractInput) -> MusicalIntent {
    let aspect = if input.input_type.is_empty() {
        input.aspect
    } else {
        aspect_from_input_type(&input.input_type)
    };
    MusicalIntent::with_components([IntentComponent {
        aspect,
        magnitude: input.magnitude,
        reason: "inferred".to_string(),
    }])
}

pub fn derive_value_states(
    intent: &MusicalIntent,
    constraints: Option<&ConstraintSet>,
    out_states: &mut ValueStateSet,
    out_violations: &mut ViolationList,
) {
    *out_states = ValueStateSet::new();

    let value_types = [
        (ValueType::TempoBias, IntentAspect::Tempo),
        (ValueType::HarmonicTension, IntentAspect::Harmonic),
        (ValueType::RhythmicDensity, IntentAspect::Rhythmic),
        (ValueType::DynamicRange, IntentAspect::Dynamic),
    ];

    for (vt, aspect) in value_types {
        let has_intent = intent.has_aspect(aspect);
        let constrained = constraints
            .map(|c| c.iter().any(|constraint| constraint_overrides_intent(constraint, vt)))
            .unwrap_or(false);

        let state = if !has_intent {
            ValueState::missing("no_intent")
        } else {
            let mag = intent.get_magnitude(aspect);
            ValueState::derive(vt, mag, constrained)
        };
        out_states.set(vt, state);
    }

    if let Some(c) = constraints {
        constraints_apply(c, out_states, out_violations);
    }
}

pub fn validate_invariants(result: &IntentResult) -> bool {
    if !result.harmonic_escalation().satisfies_invariant() {
        return false;
    }
    if !result.rhythmic_escalation().satisfies_invariant() {
        return false;
    }
    for vt in [
        ValueType::TempoBias,
        ValueType::HarmonicTension,
        ValueType::RhythmicDensity,
        ValueType::DynamicRange,
    ] {
        let state = result.value_states().get(vt);
        if !state.validate(vt) {
            return false;
        }
    }
    true
}

fn is_empty_input(input: &AbstractInput) -> bool {
    input.input_type.is_empty()
}

/// Escalation v0: "more_harmonic_color" only. Two mutually exclusive branches: 7ths OR one inversion.
fn compute_harmonic_escalation(input: &AbstractInput, intent: &MusicalIntent) -> HarmonicEscalationBranches {
    if input.escalation_token.as_deref() != Some("more_harmonic_color") {
        return HarmonicEscalationBranches::empty();
    }
    if !intent.has_aspect(IntentAspect::Harmonic) {
        return HarmonicEscalationBranches::empty();
    }
    HarmonicEscalationBranches::more_harmonic_color()
}

/// Escalation v1: "more_groove" only. Three branches: forward push, backward layback, neutral.
fn compute_rhythmic_escalation(input: &AbstractInput, intent: &MusicalIntent) -> RhythmicEscalationBranches {
    if input.escalation_token.as_deref() != Some("more_groove") {
        return RhythmicEscalationBranches::empty();
    }
    if !intent.has_aspect(IntentAspect::Rhythmic) {
        return RhythmicEscalationBranches::empty();
    }
    RhythmicEscalationBranches::more_groove()
}

pub fn intent_engine_process(
    input: &AbstractInput,
    constraints: Option<&ConstraintSet>,
) -> IntentResult {
    if is_empty_input(input) {
        return IntentResult::new(
            MusicalIntent::new(),
            ValueStateSet::new(),
            ViolationList::new(),
            "abstain",
            HarmonicEscalationBranches::empty(),
            RhythmicEscalationBranches::empty(),
        );
    }

    let intent = infer_intent(input);
    let mut states = ValueStateSet::new();
    let mut violations = ViolationList::new();
    derive_value_states(&intent, constraints, &mut states, &mut violations);
    let h_esc = compute_harmonic_escalation(input, &intent);
    let r_esc = compute_rhythmic_escalation(input, &intent);

    IntentResult::new(intent, states, violations, "processed", h_esc, r_esc)
}
