/*
 * Escalation v1: "more_groove" yields three mutually exclusive branches.
 * MICRO_TIMING_ONLY: forward push, backward layback, neutral.
 * Does NOT: add notes, change pattern, increase density, add fills.
 */
#include "../include/intent_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    AbstractInput input = {
        .type = "rhythmic",
        .aspect = INTENT_ASPECT_RHYTHMIC,
        .magnitude = 0.5,
        .escalation_token = "more_groove"
    };

    IntentResult* result = intent_engine_process(&input, NULL);
    if (!result) return 1;

    const RhythmicEscalationBranches* esc = intent_result_get_rhythmic_escalation(result);
    printf("rhythmic_escalation branches: %zu (MICRO_TIMING_ONLY, pick one)\n", esc->count);
    printf("Rule: timing only. No notes, density, pattern change, fills.\n");
    for (size_t i = 0; i < esc->count; i++) {
        const char* name = "?";
        switch (esc->branches[i]) {
            case RHYTHMIC_BRANCH_FORWARD_PUSH: name = "FORWARD_PUSH"; break;
            case RHYTHMIC_BRANCH_BACKWARD_LAYBACK: name = "BACKWARD_LAYBACK"; break;
            case RHYTHMIC_BRANCH_NEUTRAL: name = "NEUTRAL"; break;
        }
        printf("  branch %zu: %s\n", i, name);
    }
    printf("satisfies invariant: %s\n", rhythmic_escalation_satisfies_invariant(esc) ? "yes" : "no");

    intent_result_destroy(result);
    return 0;
}
