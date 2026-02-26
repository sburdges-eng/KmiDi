//! Escalation v1: "more_groove" yields three mutually exclusive branches.

use minimal_listening_core::{
    intent_engine_process, AbstractInput, IntentAspect, RhythmicEscalationBranch,
};

fn main() {
    let input = AbstractInput {
        input_type: "rhythmic".to_string(),
        aspect: IntentAspect::Rhythmic,
        magnitude: 0.5,
        escalation_token: Some("more_groove".to_string()),
    };

    let result = intent_engine_process(&input, None);
    let esc = result.rhythmic_escalation();

    println!("rhythmic_escalation branches: {} (MICRO_TIMING_ONLY, pick one)", esc.branches.len());
    println!("Rule: timing only. No notes, density, pattern change, fills.");
    for (i, branch) in esc.branches.iter().enumerate() {
        let name = match branch {
            RhythmicEscalationBranch::ForwardPush => "FORWARD_PUSH",
            RhythmicEscalationBranch::BackwardLayback => "BACKWARD_LAYBACK",
            RhythmicEscalationBranch::Neutral => "NEUTRAL",
        };
        println!("  branch {}: {}", i, name);
    }
    println!("satisfies invariant: {}", esc.satisfies_invariant());
}
