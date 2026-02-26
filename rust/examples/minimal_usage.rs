//! Minimal usage: chord_change -> harmonic change, no added density.

use minimal_listening_core::{
    intent_engine_process, AbstractInput, ConstraintSet, IntentAspect, ValueType,
};

fn main() {
    let input = AbstractInput {
        input_type: "chord_change".to_string(),
        aspect: IntentAspect::Harmonic,
        magnitude: 0.7,
        escalation_token: None,
    };

    let constraints = ConstraintSet::new();
    let result = intent_engine_process(&input, Some(&constraints));

    let states = result.value_states();
    let harmonic = states.get(ValueType::HarmonicTension);
    let density = states.get(ValueType::RhythmicDensity);

    print!("chord_change input -> harmonic_tension: ");
    if harmonic.is_present() {
        println!("PRESENT({:.2})", harmonic.get_value());
    } else {
        println!("MISSING/INVALID");
    }

    print!("chord_change input -> rhythmic_density: ");
    if density.is_missing() {
        println!("MISSING(no_intent) - preserved absence, no added density");
    } else if density.is_present() {
        println!("PRESENT - ERROR: should not add density");
    }
}
