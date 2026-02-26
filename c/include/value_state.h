#ifndef VALUE_STATE_H
#define VALUE_STATE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VALUE_STATE_PRESENT,
    VALUE_STATE_MISSING,
    VALUE_STATE_INVALID
} ValueStateKind;

typedef struct {
    ValueStateKind kind;
    double value;
    char reason[256];
} ValueState;

typedef enum {
    VALUE_TYPE_TEMPO_BIAS,
    VALUE_TYPE_HARMONIC_TENSION,
    VALUE_TYPE_RHYTHMIC_DENSITY,
    VALUE_TYPE_DYNAMIC_RANGE,
    VALUE_TYPE_COUNT
} ValueType;

/*
 * HARMONIC ESCALATION RULE (codified):
 *
 * One branch may choose:
 *   - triad + inversion
 *   OR
 *   - triad + 7th
 *
 * Never both.
 * Never more than one added harmonic degree.
 *
 * Forbidden: 7ths+inversions, multiple extensions (9th, 11th), cascading "well since we can..."
 */
typedef enum {
    HARMONIC_BRANCH_SEVENTHS,      /* triad + 7th. One added degree. */
    HARMONIC_BRANCH_ONE_INVERSION  /* triad + inversion. One added degree. */
} HarmonicEscalationBranch;

#define HARMONIC_ESCALATION_MAX_BRANCHES 2
#define HARMONIC_ESCALATION_MAX_ADDED_DEGREES_PER_BRANCH 1
typedef struct {
    HarmonicEscalationBranch branches[HARMONIC_ESCALATION_MAX_BRANCHES];
    size_t count;  /* 0 = no escalation. 1-2 = mutually exclusive branches, pick one. */
} HarmonicEscalationBranches;

void harmonic_escalation_branches_clear(HarmonicEscalationBranches* out);
void harmonic_escalation_branches_set_more_harmonic_color(HarmonicEscalationBranches* out);
bool harmonic_escalation_satisfies_invariant(const HarmonicEscalationBranches* b);

/* Returns added harmonic degrees for a branch. Must be <= MAX_ADDED_DEGREES_PER_BRANCH (1). */
int harmonic_escalation_branch_added_degrees(HarmonicEscalationBranch branch);

/*
 * RHYTHMIC ESCALATION RULE (codified):
 *
 * MICRO_TIMING_ONLY: push/pull, swing bias, humanization window.
 *
 * Does NOT: add notes, change pattern, increase density, add fills,
 * change velocity curves beyond a small window.
 *
 * Branches may differ by: forward push, backward layback, neutral.
 * No branch may be "busier".
 */
typedef enum {
    RHYTHMIC_BRANCH_FORWARD_PUSH,   /* timing only: push forward */
    RHYTHMIC_BRANCH_BACKWARD_LAYBACK, /* timing only: lay back */
    RHYTHMIC_BRANCH_NEUTRAL         /* timing only: neutral */
} RhythmicEscalationBranch;

#define RHYTHMIC_ESCALATION_MAX_BRANCHES 3
typedef struct {
    RhythmicEscalationBranch branches[RHYTHMIC_ESCALATION_MAX_BRANCHES];
    size_t count;  /* 0 = no escalation. 1-3 = mutually exclusive branches, pick one. */
} RhythmicEscalationBranches;

void rhythmic_escalation_branches_clear(RhythmicEscalationBranches* out);
void rhythmic_escalation_branches_set_more_groove(RhythmicEscalationBranches* out);
bool rhythmic_escalation_satisfies_invariant(const RhythmicEscalationBranches* b);

typedef struct {
    ValueState states[VALUE_TYPE_COUNT];
} ValueStateSet;

ValueState value_state_present(double value);
ValueState value_state_missing(const char* reason);
ValueState value_state_invalid(const char* reason);

ValueStateSet value_state_set_create(void);
void value_state_set_set(ValueStateSet* set, ValueType type, ValueState state);
ValueState value_state_set_get(const ValueStateSet* set, ValueType type);
void value_state_set_preserve_absence(ValueStateSet* dest, const ValueStateSet* src);

bool value_state_is_present(const ValueState* vs);
bool value_state_is_missing(const ValueState* vs);
bool value_state_is_invalid(const ValueState* vs);
double value_state_get_value(const ValueState* vs);
const char* value_state_get_reason(const ValueState* vs);

ValueState value_state_derive(ValueType type, double intent_magnitude, bool constrained);
bool value_state_validate(const ValueState* vs, ValueType type);

#ifdef __cplusplus
}
#endif

#endif /* VALUE_STATE_H */
