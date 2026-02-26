//! Invariant violation: magnitude out of range [0,1] -> INVALID result.

use minimal_listening_core::{
    intent_engine_process, AbstractInput, ConstraintSet, IntentAspect, ValueType,
};

fn main() {
    let input = AbstractInput {
        input_type: "chord_change".to_string(),
        aspect: IntentAspect::Harmonic,
        magnitude: 1.5,
        escalation_token: None,
    };

    let constraints = ConstraintSet::new();
    let result = intent_engine_process(&input, Some(&constraints));

    let states = result.value_states();
    let harmonic = states.get(ValueType::HarmonicTension);

    if harmonic.is_invalid()
        || (harmonic.is_present() && harmonic.get_value() > 1.0)
    {
        eprintln!("invariant violation: harmonic value out of range [0,1]");
        std::process::exit(1);
    }
}
