#include "intent_engine.h"
#include <stdlib.h>
#include <string.h>

static IntentAspect aspect_from_input_type(const char* type) {
    if (!type) return INTENT_ASPECT_HARMONIC;
    if (strcmp(type, "chord_change") == 0 || strcmp(type, "harmonic") == 0) return INTENT_ASPECT_HARMONIC;
    if (strcmp(type, "rhythmic") == 0) return INTENT_ASPECT_RHYTHMIC;
    if (strcmp(type, "dynamic") == 0) return INTENT_ASPECT_DYNAMIC;
    if (strcmp(type, "tempo") == 0) return INTENT_ASPECT_TEMPO;
    return INTENT_ASPECT_HARMONIC;
}

MusicalIntent* infer_intent(const AbstractInput* input) {
    if (!input) return musical_intent_create();
    IntentAspect aspect = input->aspect;
    if (input->type && strlen(input->type) > 0) {
        aspect = aspect_from_input_type(input->type);
    }
    IntentComponent comp = {aspect, input->magnitude, "inferred"};
    return musical_intent_create_with_components(&comp, 1);
}

void derive_value_states(const MusicalIntent* intent, const ConstraintSet* constraints,
                         ValueStateSet* out_states, ViolationList* out_violations) {
    if (!intent || !out_states || !out_violations) return;
    *out_states = value_state_set_create();
    for (int i = 0; i < VALUE_TYPE_COUNT; i++) {
        ValueType vt = (ValueType)i;
        IntentAspect aspect = INTENT_ASPECT_HARMONIC;
        switch (vt) {
            case VALUE_TYPE_HARMONIC_TENSION: aspect = INTENT_ASPECT_HARMONIC; break;
            case VALUE_TYPE_RHYTHMIC_DENSITY: aspect = INTENT_ASPECT_RHYTHMIC; break;
            case VALUE_TYPE_DYNAMIC_RANGE: aspect = INTENT_ASPECT_DYNAMIC; break;
            case VALUE_TYPE_TEMPO_BIAS: aspect = INTENT_ASPECT_TEMPO; break;
            default: break;
        }
        bool has_intent = musical_intent_has_aspect(intent, aspect);
        bool constrained = false;
        if (constraints) {
            for (size_t j = 0; j < constraints->count; j++) {
                if (constraint_overrides_intent(&constraints->constraints[j], vt)) {
                    constrained = true;
                    break;
                }
            }
        }
        if (!has_intent) {
            value_state_set_set(out_states, vt, value_state_missing("no_intent"));
        } else {
            double mag = musical_intent_get_magnitude(intent, aspect);
            ValueState vs = value_state_derive(vt, mag, constrained);
            value_state_set_set(out_states, vt, vs);
        }
    }
    if (constraints) {
        constraints_apply(constraints, out_states, out_violations);
    }
}

bool validate_invariants(const IntentResult* result) {
    if (!result) return false;
    if (!result->intent) return false;
    if (!result->frozen) return false;
    if (!harmonic_escalation_satisfies_invariant(&result->harmonic_escalation)) return false;
    if (!rhythmic_escalation_satisfies_invariant(&result->rhythmic_escalation)) return false;
    for (int i = 0; i < VALUE_TYPE_COUNT; i++) {
        if (!value_state_validate(&result->value_states.states[i], (ValueType)i)) return false;
    }
    return true;
}

static bool is_empty_input(const AbstractInput* input) {
    if (!input) return true;
    if (input->type && input->type[0] != '\0') return false;
    return true;
}

/* Escalation v0: "more_harmonic_color" only. Two mutually exclusive branches: 7ths OR one inversion. */
static void compute_harmonic_escalation(const AbstractInput* input, const MusicalIntent* intent,
                                        HarmonicEscalationBranches* out) {
    harmonic_escalation_branches_clear(out);
    if (!input || !intent || !out) return;
    if (!input->escalation_token || strcmp(input->escalation_token, "more_harmonic_color") != 0)
        return;
    if (!musical_intent_has_aspect(intent, INTENT_ASPECT_HARMONIC))
        return;
    harmonic_escalation_branches_set_more_harmonic_color(out);
}

/* Escalation v1: "more_groove" only. Three branches: forward push, backward layback, neutral. */
static void compute_rhythmic_escalation(const AbstractInput* input, const MusicalIntent* intent,
                                        RhythmicEscalationBranches* out) {
    rhythmic_escalation_branches_clear(out);
    if (!input || !intent || !out) return;
    if (!input->escalation_token || strcmp(input->escalation_token, "more_groove") != 0)
        return;
    if (!musical_intent_has_aspect(intent, INTENT_ASPECT_RHYTHMIC))
        return;
    rhythmic_escalation_branches_set_more_groove(out);
}

IntentResult* intent_engine_process(const AbstractInput* input, const ConstraintSet* constraints) {
    if (!input || is_empty_input(input)) {
        MusicalIntent* empty = musical_intent_create();
        if (!empty) return NULL;
        ValueStateSet states = value_state_set_create();
        ViolationList* violations = violation_list_create();
        HarmonicEscalationBranches h_esc;
        RhythmicEscalationBranches r_esc;
        harmonic_escalation_branches_clear(&h_esc);
        rhythmic_escalation_branches_clear(&r_esc);
        IntentResult* result = intent_result_create(empty, &states, violations, "abstain", &h_esc, &r_esc);
        musical_intent_destroy(empty);
        if (violations) violation_list_destroy(violations);
        return result;
    }
    MusicalIntent* intent = infer_intent(input);
    if (!intent) return NULL;
    ValueStateSet states;
    ViolationList* violations = violation_list_create();
    if (!violations) { musical_intent_destroy(intent); return NULL; }
    derive_value_states(intent, constraints, &states, violations);
    HarmonicEscalationBranches h_esc;
    RhythmicEscalationBranches r_esc;
    compute_harmonic_escalation(input, intent, &h_esc);
    compute_rhythmic_escalation(input, intent, &r_esc);
    IntentResult* result = intent_result_create(intent, &states, violations, "processed", &h_esc, &r_esc);
    musical_intent_destroy(intent);
    violation_list_destroy(violations);
    if (result && !validate_invariants(result)) {
        return result;
    }
    return result;
}
