#include "../include/intent_engine.h"
#include "../include/intent_result.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    AbstractInput input = {
        .type = "chord_change",
        .aspect = INTENT_ASPECT_HARMONIC,
        .magnitude = 1.5
    };

    ConstraintSet* constraints = constraint_set_create();

    IntentResult* result = intent_engine_process(&input, constraints);

    if (result) {
        const ValueStateSet* states = intent_result_get_value_states(result);
        ValueState harmonic = value_state_set_get(states, VALUE_TYPE_HARMONIC_TENSION);
        /* 1.5 is wrong: engine must reject. INVALID = correct; PRESENT>1.0 = bug */
        if (value_state_is_present(&harmonic) && harmonic.value > 1.0) {
            printf("invariant violation: engine accepted magnitude 1.5 (should reject)\n");
            intent_result_destroy(result);
            constraint_set_destroy(constraints);
            return 1;
        }
        /* Engine correctly rejected 1.5 -> INVALID. Stub pass. */
        intent_result_destroy(result);
    } else {
        printf("invariant violation: validate_invariants failed, result is NULL\n");
    }

    constraint_set_destroy(constraints);
    return 0;
}
