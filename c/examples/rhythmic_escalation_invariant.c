/*
 * Invariant test: "more_groove" without rhythmic intent -> NONE (no branches).
 * "more_groove" with rhythmic intent -> 3 branches.
 * Rhythmic escalation must not affect density.
 */
#include "../include/intent_engine.h"
#include <stdio.h>
#include <stdlib.h>

static int test_more_groove_without_rhythmic_intent(void) {
    AbstractInput input = {
        .type = "chord_change",
        .aspect = INTENT_ASPECT_HARMONIC,
        .magnitude = 0.7,
        .escalation_token = "more_groove"
    };
    IntentResult* result = intent_engine_process(&input, NULL);
    if (!result) return 1;
    const RhythmicEscalationBranches* esc = intent_result_get_rhythmic_escalation(result);
    if (esc->count != 0) {
        printf("FAIL: more_groove without rhythmic intent should yield 0 branches, got %zu\n", esc->count);
        intent_result_destroy(result);
        return 1;
    }
    intent_result_destroy(result);
    return 0;
}

static int test_more_groove_with_rhythmic_intent(void) {
    AbstractInput input = {
        .type = "rhythmic",
        .aspect = INTENT_ASPECT_RHYTHMIC,
        .magnitude = 0.5,
        .escalation_token = "more_groove"
    };
    IntentResult* result = intent_engine_process(&input, NULL);
    if (!result) return 1;
    const RhythmicEscalationBranches* esc = intent_result_get_rhythmic_escalation(result);
    if (esc->count != 3) {
        printf("FAIL: more_groove with rhythmic intent should yield 3 branches, got %zu\n", esc->count);
        intent_result_destroy(result);
        return 1;
    }
    if (!rhythmic_escalation_satisfies_invariant(esc)) {
        printf("FAIL: rhythmic escalation invariant violated\n");
        intent_result_destroy(result);
        return 1;
    }
    intent_result_destroy(result);
    return 0;
}

static int test_rhythmic_escalation_no_density_change(void) {
    AbstractInput input = {
        .type = "rhythmic",
        .aspect = INTENT_ASPECT_RHYTHMIC,
        .magnitude = 0.5,
        .escalation_token = "more_groove"
    };
    IntentResult* result = intent_engine_process(&input, NULL);
    if (!result) return 1;
    const ValueStateSet* states = intent_result_get_value_states(result);
    ValueState density = value_state_set_get(states, VALUE_TYPE_RHYTHMIC_DENSITY);
    if (!value_state_is_present(&density) && !value_state_is_missing(&density)) {
        printf("FAIL: rhythmic_density should be PRESENT or MISSING, not INVALID\n");
        intent_result_destroy(result);
        return 1;
    }
    intent_result_destroy(result);
    return 0;
}

int main(void) {
    int r = 0;
    r |= test_more_groove_without_rhythmic_intent();
    r |= test_more_groove_with_rhythmic_intent();
    r |= test_rhythmic_escalation_no_density_change();
    if (r == 0) printf("rhythmic escalation invariant tests: PASS\n");
    return r;
}
