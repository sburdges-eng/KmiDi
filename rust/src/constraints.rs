use crate::value_state::{ValueState, ValueStateSet, ValueType};
use std::vec::Vec;

#[derive(Debug, Clone)]
pub enum ConstraintKind {
    FixValue,
    ForbidAspect,
    LimitRange,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub target_type: ValueType,
    pub min_val: f64,
    pub max_val: f64,
    pub fixed_val: f64,
    pub forbid: bool,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ConstraintSet {
    constraints: Vec<Constraint>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    pub fn add(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    pub fn clear(&mut self) {
        self.constraints.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter()
    }
}

impl Default for ConstraintSet {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub value_type: ValueType,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ViolationList {
    violations: Vec<Violation>,
}

impl ViolationList {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    pub fn add(&mut self, value_type: ValueType, message: impl Into<String>) {
        self.violations.push(Violation {
            value_type,
            message: message.into(),
        });
    }

    pub fn clear(&mut self) {
        self.violations.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn len(&self) -> usize {
        self.violations.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Violation> {
        self.violations.iter()
    }
}

impl Default for ViolationList {
    fn default() -> Self {
        Self::new()
    }
}

pub fn constraint_overrides_intent(c: &Constraint, value_type: ValueType) -> bool {
    c.target_type == value_type
        && matches!(c.kind, ConstraintKind::FixValue | ConstraintKind::ForbidAspect)
}

pub fn constraints_apply(
    constraints: &ConstraintSet,
    states: &mut ValueStateSet,
    violations: &mut ViolationList,
) {
    for c in constraints.iter() {
        if c.target_type != ValueType::TempoBias
            && c.target_type != ValueType::HarmonicTension
            && c.target_type != ValueType::RhythmicDensity
            && c.target_type != ValueType::DynamicRange
        {
            continue;
        }
        let state = states.get(c.target_type);
        match &c.kind {
            ConstraintKind::FixValue => {
                states.set(c.target_type, ValueState::present(c.fixed_val));
            }
            ConstraintKind::ForbidAspect => {
                if state.is_present() {
                    states.set(
                        c.target_type,
                        ValueState::invalid(if c.reason.is_empty() {
                            "forbidden"
                        } else {
                            &c.reason
                        }),
                    );
                    violations.add(c.target_type, "Constraint forbids this aspect");
                }
            }
            ConstraintKind::LimitRange => {
                if let ValueState::Present(v) = state {
                    if *v < c.min_val || *v > c.max_val {
                        let clamped = if *v < c.min_val {
                            c.min_val
                        } else {
                            c.max_val
                        };
                        states.set(c.target_type, ValueState::present(clamped));
                        violations.add(c.target_type, "Value clamped to constraint range");
                    }
                }
            }
        }
    }
}
