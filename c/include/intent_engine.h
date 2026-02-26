#ifndef INTENT_ENGINE_H
#define INTENT_ENGINE_H

#include "musical_intent.h"
#include "value_state.h"
#include "constraints.h"
#include "intent_result.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* type;
    IntentAspect aspect;
    double magnitude;
    const char* escalation_token;  /* e.g. "more_harmonic_color" — only affects harmony */
} AbstractInput;

MusicalIntent* infer_intent(const AbstractInput* input);
void derive_value_states(const MusicalIntent* intent, const ConstraintSet* constraints,
                         ValueStateSet* out_states, ViolationList* out_violations);
bool validate_invariants(const IntentResult* result);

IntentResult* intent_engine_process(const AbstractInput* input, const ConstraintSet* constraints);

#ifdef __cplusplus
}
#endif

#endif /* INTENT_ENGINE_H */
