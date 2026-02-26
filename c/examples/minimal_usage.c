#include "../include/intent_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    AbstractInput input = {
        .type = "chord_change",
        .aspect = INTENT_ASPECT_HARMONIC,
        .magnitude = 0.7
    };

    ConstraintSet* constraints = constraint_set_create();

    IntentResult* result = intent_engine_process(&input, constraints);

    if (!result) {
        fprintf(stderr, "invariant violation: result is NULL\n");
        constraint_set_destroy(constraints);
        return 1;
    }

    const ValueStateSet* states = intent_result_get_value_states(result);
    ValueState harmonic = value_state_set_get(states, VALUE_TYPE_HARMONIC_TENSION);
    ValueState density = value_state_set_get(states, VALUE_TYPE_RHYTHMIC_DENSITY);

    printf("chord_change input -> harmonic_tension: ");
    if (value_state_is_present(&harmonic)) {
        printf("PRESENT(%.2f)\n", harmonic.value);
    } else {
        printf("MISSING/INVALID\n");
    }

    printf("chord_change input -> rhythmic_density: ");
    if (value_state_is_missing(&density)) {
        printf("MISSING(no_intent) - preserved absence, no added density\n");
    } else if (value_state_is_present(&density)) {
        printf("PRESENT - ERROR: should not add density\n");
    }

    intent_result_destroy(result);
    constraint_set_destroy(constraints);
    return 0;
}
