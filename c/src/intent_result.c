#include "intent_result.h"
#include <stdlib.h>
#include <string.h>

IntentResult* intent_result_create(const MusicalIntent* intent, const ValueStateSet* states,
                                   const ViolationList* violations, const char* metadata,
                                   const HarmonicEscalationBranches* harmonic_escalation,
                                   const RhythmicEscalationBranches* rhythmic_escalation) {
    if (!intent || !states) return NULL;
    IntentResult* result = (IntentResult*)calloc(1, sizeof(IntentResult));
    if (!result) return NULL;
    result->intent = musical_intent_create();
    if (!result->intent) { free(result); return NULL; }
    musical_intent_update(result->intent, intent);
    result->value_states = *states;
    result->violations = violation_list_create();
    if (violations) {
        for (size_t i = 0; i < violations->count; i++) {
            violation_list_add(result->violations, violations->violations[i].type,
                              violations->violations[i].message);
        }
    }
    if (metadata) {
        strncpy(result->metadata, metadata, sizeof(result->metadata) - 1);
        result->metadata[sizeof(result->metadata) - 1] = '\0';
    }
    if (harmonic_escalation) {
        result->harmonic_escalation = *harmonic_escalation;
        if (result->harmonic_escalation.count > HARMONIC_ESCALATION_MAX_BRANCHES)
            result->harmonic_escalation.count = HARMONIC_ESCALATION_MAX_BRANCHES;
    } else {
        harmonic_escalation_branches_clear(&result->harmonic_escalation);
    }
    if (rhythmic_escalation) {
        result->rhythmic_escalation = *rhythmic_escalation;
        if (result->rhythmic_escalation.count > RHYTHMIC_ESCALATION_MAX_BRANCHES)
            result->rhythmic_escalation.count = RHYTHMIC_ESCALATION_MAX_BRANCHES;
    } else {
        rhythmic_escalation_branches_clear(&result->rhythmic_escalation);
    }
    result->frozen = true;
    return result;
}

void intent_result_destroy(IntentResult* result) {
    if (!result) return;
    musical_intent_destroy(result->intent);
    violation_list_destroy(result->violations);
    free(result);
}

const MusicalIntent* intent_result_get_intent(const IntentResult* result) {
    return result ? result->intent : NULL;
}

const ValueStateSet* intent_result_get_value_states(const IntentResult* result) {
    return result ? &result->value_states : NULL;
}

const ViolationList* intent_result_get_violations(const IntentResult* result) {
    return result ? result->violations : NULL;
}

const char* intent_result_get_metadata(const IntentResult* result) {
    return result ? result->metadata : "";
}

bool intent_result_is_valid(const IntentResult* result) {
    if (!result) return false;
    for (int i = 0; i < VALUE_TYPE_COUNT; i++) {
        if (value_state_is_invalid(&result->value_states.states[i])) return false;
    }
    return true;
}

bool intent_result_has_violations(const IntentResult* result) {
    return result && result->violations && result->violations->count > 0;
}

static const HarmonicEscalationBranches k_empty_harmonic = {0};
static const RhythmicEscalationBranches k_empty_rhythmic = {0};

const HarmonicEscalationBranches* intent_result_get_harmonic_escalation(const IntentResult* result) {
    return result ? &result->harmonic_escalation : &k_empty_harmonic;
}

const RhythmicEscalationBranches* intent_result_get_rhythmic_escalation(const IntentResult* result) {
    return result ? &result->rhythmic_escalation : &k_empty_rhythmic;
}
