#ifndef INTENT_RESULT_H
#define INTENT_RESULT_H

#include <stddef.h>
#include <stdbool.h>
#include "musical_intent.h"
#include "value_state.h"
#include "constraints.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    MusicalIntent* intent;
    ValueStateSet value_states;
    ViolationList* violations;
    HarmonicEscalationBranches harmonic_escalation;
    RhythmicEscalationBranches rhythmic_escalation;  /* MICRO_TIMING_ONLY: timing only, no added content */
    char metadata[512];
    bool frozen;
} IntentResult;

IntentResult* intent_result_create(const MusicalIntent* intent, const ValueStateSet* states,
                                   const ViolationList* violations, const char* metadata,
                                   const HarmonicEscalationBranches* harmonic_escalation,
                                   const RhythmicEscalationBranches* rhythmic_escalation);
void intent_result_destroy(IntentResult* result);

const MusicalIntent* intent_result_get_intent(const IntentResult* result);
const ValueStateSet* intent_result_get_value_states(const IntentResult* result);
const ViolationList* intent_result_get_violations(const IntentResult* result);
const char* intent_result_get_metadata(const IntentResult* result);

bool intent_result_is_valid(const IntentResult* result);
bool intent_result_has_violations(const IntentResult* result);
const HarmonicEscalationBranches* intent_result_get_harmonic_escalation(const IntentResult* result);
const RhythmicEscalationBranches* intent_result_get_rhythmic_escalation(const IntentResult* result);

#ifdef __cplusplus
}
#endif

#endif /* INTENT_RESULT_H */
