use crate::musical_intent::MusicalIntent;
use crate::value_state::{HarmonicEscalationBranches, RhythmicEscalationBranches, ValueStateSet, ValueType};
use crate::constraints::ViolationList;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct IntentResult {
    intent: Arc<MusicalIntent>,
    value_states: ValueStateSet,
    violations: Arc<ViolationList>,
    harmonic_escalation: HarmonicEscalationBranches,
    rhythmic_escalation: RhythmicEscalationBranches,
    metadata: String,
}

impl IntentResult {
    pub fn new(
        intent: MusicalIntent,
        value_states: ValueStateSet,
        violations: ViolationList,
        metadata: impl Into<String>,
        harmonic_escalation: HarmonicEscalationBranches,
        rhythmic_escalation: RhythmicEscalationBranches,
    ) -> Self {
        Self {
            intent: Arc::new(intent),
            value_states,
            violations: Arc::new(violations),
            harmonic_escalation,
            rhythmic_escalation,
            metadata: metadata.into(),
        }
    }

    pub fn intent(&self) -> &MusicalIntent {
        &self.intent
    }

    pub fn value_states(&self) -> &ValueStateSet {
        &self.value_states
    }

    pub fn violations(&self) -> &ViolationList {
        &self.violations
    }

    pub fn metadata(&self) -> &str {
        &self.metadata
    }

    pub fn is_valid(&self) -> bool {
        for vt in [
            ValueType::TempoBias,
            ValueType::HarmonicTension,
            ValueType::RhythmicDensity,
            ValueType::DynamicRange,
        ] {
            if self.value_states.get(vt).is_invalid() {
                return false;
            }
        }
        true
    }

    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Mutually exclusive branches; pick one. At most one choice per branch.
    pub fn harmonic_escalation(&self) -> &HarmonicEscalationBranches {
        &self.harmonic_escalation
    }

    /// MICRO_TIMING_ONLY: timing only, no added content.
    pub fn rhythmic_escalation(&self) -> &RhythmicEscalationBranches {
        &self.rhythmic_escalation
    }
}
