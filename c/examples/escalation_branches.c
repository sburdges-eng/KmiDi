/*
 * Escalation: "more_harmonic_color" yields two mutually exclusive branches.
 * Downstream must pick ONE branch. No combining: not 7ths+inversions.
 */
#include "../include/intent_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    AbstractInput input = {
        .type = "chord_change",
        .aspect = INTENT_ASPECT_HARMONIC,
        .magnitude = 0.7,
        .escalation_token = "more_harmonic_color"
    };

    IntentResult* result = intent_engine_process(&input, NULL);
    if (!result) return 1;

    const HarmonicEscalationBranches* esc = intent_result_get_harmonic_escalation(result);
    printf("harmonic_escalation branches: %zu (mutually exclusive, pick one)\n", esc->count);
    printf("Rule: triad+inversion OR triad+7th. Never both. Never >1 added degree.\n");
    for (size_t i = 0; i < esc->count; i++) {
        int deg = harmonic_escalation_branch_added_degrees(esc->branches[i]);
        printf("  branch %zu: %s (added_degrees=%d)\n", i,
               esc->branches[i] == HARMONIC_BRANCH_SEVENTHS ? "SEVENTHS" : "ONE_INVERSION", deg);
    }
    printf("satisfies invariant: %s\n", harmonic_escalation_satisfies_invariant(esc) ? "yes" : "no");

    intent_result_destroy(result);
    return 0;
}
