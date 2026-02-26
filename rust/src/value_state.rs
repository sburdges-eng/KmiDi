use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueStateKind {
    Present,
    Missing,
    Invalid,
}

#[derive(Debug, Clone)]
pub enum ValueState {
    Present(f64),
    Missing { reason: String },
    Invalid { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    TempoBias,
    HarmonicTension,
    RhythmicDensity,
    DynamicRange,
}

/// HARMONIC ESCALATION RULE (codified):
///
/// One branch may choose:
///   - triad + inversion
///   OR
///   - triad + 7th
///
/// Never both.
/// Never more than one added harmonic degree.
///
/// Forbidden: 7ths+inversions, multiple extensions (9th, 11th), cascading "well since we can..."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicEscalationBranch {
    /// triad + 7th. One added degree.
    Sevenths,
    /// triad + inversion. One added degree.
    OneInversion,
}

pub const HARMONIC_ESCALATION_MAX_ADDED_DEGREES_PER_BRANCH: usize = 1;

/// Mutually exclusive branches; pick one. count is 0, 1, or 2.
#[derive(Debug, Clone)]
pub struct HarmonicEscalationBranches {
    pub branches: Vec<HarmonicEscalationBranch>,
}

impl HarmonicEscalationBranches {
    pub const MAX_BRANCHES: usize = 2;

    pub fn empty() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    pub fn more_harmonic_color() -> Self {
        Self {
            branches: vec![
                HarmonicEscalationBranch::Sevenths,
                HarmonicEscalationBranch::OneInversion,
            ],
        }
    }

    /// Invariant: at most one choice per branch; never more than one added degree per branch.
    pub fn satisfies_invariant(&self) -> bool {
        if self.branches.len() > Self::MAX_BRANCHES {
            return false;
        }
        for branch in &self.branches {
            if HarmonicEscalationBranch::added_degrees(*branch) > HARMONIC_ESCALATION_MAX_ADDED_DEGREES_PER_BRANCH
            {
                return false;
            }
        }
        true
    }
}

impl HarmonicEscalationBranch {
    /// Returns added harmonic degrees for a branch. Must be <= MAX_ADDED_DEGREES_PER_BRANCH (1).
    pub fn added_degrees(self) -> usize {
        match self {
            HarmonicEscalationBranch::Sevenths | HarmonicEscalationBranch::OneInversion => 1,
        }
    }
}

/// RHYTHMIC ESCALATION RULE (codified):
///
/// MICRO_TIMING_ONLY: push/pull, swing bias, humanization window.
///
/// Does NOT: add notes, change pattern, increase density, add fills,
/// change velocity curves beyond a small window.
///
/// Branches may differ by: forward push, backward layback, neutral.
/// No branch may be "busier".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhythmicEscalationBranch {
    /// timing only: push forward
    ForwardPush,
    /// timing only: lay back
    BackwardLayback,
    /// timing only: neutral
    Neutral,
}

#[derive(Debug, Clone)]
pub struct RhythmicEscalationBranches {
    pub branches: Vec<RhythmicEscalationBranch>,
}

impl RhythmicEscalationBranches {
    pub const MAX_BRANCHES: usize = 3;

    pub fn empty() -> Self {
        Self { branches: Vec::new() }
    }

    pub fn more_groove() -> Self {
        Self {
            branches: vec![
                RhythmicEscalationBranch::ForwardPush,
                RhythmicEscalationBranch::BackwardLayback,
                RhythmicEscalationBranch::Neutral,
            ],
        }
    }

    /// Invariant: timing only. No branch may be "busier".
    pub fn satisfies_invariant(&self) -> bool {
        self.branches.len() <= Self::MAX_BRANCHES
    }
}

impl ValueState {
    pub fn present(value: f64) -> Self {
        ValueState::Present(value)
    }

    pub fn missing(reason: impl Into<String>) -> Self {
        ValueState::Missing {
            reason: reason.into(),
        }
    }

    pub fn invalid(reason: impl Into<String>) -> Self {
        ValueState::Invalid {
            reason: reason.into(),
        }
    }

    pub fn is_present(&self) -> bool {
        matches!(self, ValueState::Present(_))
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, ValueState::Missing { .. })
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, ValueState::Invalid { .. })
    }

    pub fn get_value(&self) -> f64 {
        match self {
            ValueState::Present(v) => *v,
            _ => 0.0,
        }
    }

    pub fn get_reason(&self) -> &str {
        match self {
            ValueState::Missing { reason } | ValueState::Invalid { reason } => reason,
            _ => "",
        }
    }

    pub fn derive(_value_type: ValueType, intent_magnitude: f64, constrained: bool) -> Self {
        if constrained {
            ValueState::missing("constrained")
        } else if intent_magnitude < 0.0 || intent_magnitude > 1.0 {
            ValueState::invalid("magnitude_out_of_range")
        } else {
            ValueState::present(intent_magnitude)
        }
    }

    pub fn validate(&self, _value_type: ValueType) -> bool {
        match self {
            ValueState::Invalid { .. } => false,
            ValueState::Present(v) => *v >= 0.0 && *v <= 1.0,
            ValueState::Missing { .. } => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValueStateSet {
    states: HashMap<ValueType, ValueState>,
}

impl ValueStateSet {
    pub fn new() -> Self {
        let mut states = HashMap::new();
        for vt in [
            ValueType::TempoBias,
            ValueType::HarmonicTension,
            ValueType::RhythmicDensity,
            ValueType::DynamicRange,
        ] {
            states.insert(vt, ValueState::missing("not_derived"));
        }
        Self { states }
    }

    pub fn set(&mut self, value_type: ValueType, state: ValueState) {
        self.states.insert(value_type, state);
    }

    pub fn get(&self, value_type: ValueType) -> ValueState {
        self.states
            .get(&value_type)
            .cloned()
            .unwrap_or_else(|| ValueState::missing("invalid_type"))
    }

    pub fn preserve_absence(&mut self, other: &ValueStateSet) {
        for (vt, state) in &other.states {
            if state.is_missing() || state.is_invalid() {
                self.states.insert(*vt, state.clone());
            }
        }
    }
}

impl Default for ValueStateSet {
    fn default() -> Self {
        Self::new()
    }
}
