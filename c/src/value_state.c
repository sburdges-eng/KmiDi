#include "value_state.h"
#include <stdlib.h>
#include <string.h>

void harmonic_escalation_branches_clear(HarmonicEscalationBranches* out) {
    if (!out) return;
    out->count = 0;
}

void harmonic_escalation_branches_set_more_harmonic_color(HarmonicEscalationBranches* out) {
    if (!out) return;
    out->branches[0] = HARMONIC_BRANCH_SEVENTHS;
    out->branches[1] = HARMONIC_BRANCH_ONE_INVERSION;
    out->count = 2;
}

/* Rule: Never more than one added harmonic degree per branch. */
int harmonic_escalation_branch_added_degrees(HarmonicEscalationBranch branch) {
    switch (branch) {
        case HARMONIC_BRANCH_SEVENTHS:
        case HARMONIC_BRANCH_ONE_INVERSION:
            return 1;  /* Each valid branch = exactly one added degree. */
        default:
            return -1; /* Invalid/unknown branch. */
    }
}

/* Invariant: at most one choice per branch; never more than one added degree per branch. */
bool harmonic_escalation_satisfies_invariant(const HarmonicEscalationBranches* b) {
    if (!b) return true;
    if (b->count > HARMONIC_ESCALATION_MAX_BRANCHES) return false;
    for (size_t i = 0; i < b->count; i++) {
        int degrees = harmonic_escalation_branch_added_degrees(b->branches[i]);
        if (degrees < 0 || degrees > HARMONIC_ESCALATION_MAX_ADDED_DEGREES_PER_BRANCH)
            return false;
    }
    return true;
}

void rhythmic_escalation_branches_clear(RhythmicEscalationBranches* out) {
    if (!out) return;
    out->count = 0;
}

void rhythmic_escalation_branches_set_more_groove(RhythmicEscalationBranches* out) {
    if (!out) return;
    out->branches[0] = RHYTHMIC_BRANCH_FORWARD_PUSH;
    out->branches[1] = RHYTHMIC_BRANCH_BACKWARD_LAYBACK;
    out->branches[2] = RHYTHMIC_BRANCH_NEUTRAL;
    out->count = 3;
}

/* Invariant: timing only. No branch may be "busier". */
bool rhythmic_escalation_satisfies_invariant(const RhythmicEscalationBranches* b) {
    if (!b) return true;
    if (b->count > RHYTHMIC_ESCALATION_MAX_BRANCHES) return false;
    return true;
}

ValueState value_state_present(double value) {
    ValueState vs = {VALUE_STATE_PRESENT, value, {0}};
    return vs;
}

ValueState value_state_missing(const char* reason) {
    ValueState vs = {VALUE_STATE_MISSING, 0.0, {0}};
    if (reason) {
        strncpy(vs.reason, reason, sizeof(vs.reason) - 1);
        vs.reason[sizeof(vs.reason) - 1] = '\0';
    }
    return vs;
}

ValueState value_state_invalid(const char* reason) {
    ValueState vs = {VALUE_STATE_INVALID, 0.0, {0}};
    if (reason) {
        strncpy(vs.reason, reason, sizeof(vs.reason) - 1);
        vs.reason[sizeof(vs.reason) - 1] = '\0';
    }
    return vs;
}

ValueStateSet value_state_set_create(void) {
    ValueStateSet set = {0};
    for (int i = 0; i < VALUE_TYPE_COUNT; i++) {
        set.states[i] = value_state_missing("not_derived");
    }
    return set;
}

void value_state_set_set(ValueStateSet* set, ValueType type, ValueState state) {
    if (!set || type >= VALUE_TYPE_COUNT) return;
    set->states[type] = state;
}

ValueState value_state_set_get(const ValueStateSet* set, ValueType type) {
    if (!set || type >= VALUE_TYPE_COUNT) return value_state_missing("invalid_type");
    return set->states[type];
}

void value_state_set_preserve_absence(ValueStateSet* dest, const ValueStateSet* src) {
    if (!dest || !src) return;
    for (int i = 0; i < VALUE_TYPE_COUNT; i++) {
        if (value_state_is_missing(&src->states[i]) || value_state_is_invalid(&src->states[i])) {
            dest->states[i] = src->states[i];
        }
    }
}

bool value_state_is_present(const ValueState* vs) {
    return vs && vs->kind == VALUE_STATE_PRESENT;
}

bool value_state_is_missing(const ValueState* vs) {
    return vs && vs->kind == VALUE_STATE_MISSING;
}

bool value_state_is_invalid(const ValueState* vs) {
    return vs && vs->kind == VALUE_STATE_INVALID;
}

double value_state_get_value(const ValueState* vs) {
    return vs && vs->kind == VALUE_STATE_PRESENT ? vs->value : 0.0;
}

const char* value_state_get_reason(const ValueState* vs) {
    return vs ? vs->reason : "";
}

ValueState value_state_derive(ValueType type, double intent_magnitude, bool constrained) {
    (void)type;
    if (constrained) return value_state_missing("constrained");
    if (intent_magnitude < 0.0 || intent_magnitude > 1.0)
        return value_state_invalid("magnitude_out_of_range");
    return value_state_present(intent_magnitude);
}

bool value_state_validate(const ValueState* vs, ValueType type) {
    (void)type;
    if (!vs) return false;
    if (vs->kind == VALUE_STATE_INVALID) return false;
    if (vs->kind == VALUE_STATE_PRESENT && (vs->value < 0.0 || vs->value > 1.0)) return false;
    return true;
}
