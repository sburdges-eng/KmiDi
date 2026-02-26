pub mod musical_intent;
pub mod value_state;
pub mod constraints;
pub mod intent_result;
pub mod intent_engine;

pub use musical_intent::{IntentAspect, MusicalIntent};
pub use value_state::{
    HarmonicEscalationBranch, HarmonicEscalationBranches,
    HARMONIC_ESCALATION_MAX_ADDED_DEGREES_PER_BRANCH,
    RhythmicEscalationBranch, RhythmicEscalationBranches,
    ValueState, ValueStateSet, ValueType,
};
pub use constraints::{Constraint, ConstraintSet, Violation, ViolationList};
pub use intent_result::IntentResult;
pub use intent_engine::{infer_intent, derive_value_states, validate_invariants, intent_engine_process, AbstractInput};
