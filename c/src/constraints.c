#include "constraints.h"
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 8

ConstraintSet* constraint_set_create(void) {
    ConstraintSet* set = (ConstraintSet*)calloc(1, sizeof(ConstraintSet));
    if (!set) return NULL;
    set->capacity = INITIAL_CAPACITY;
    set->constraints = (Constraint*)calloc(set->capacity, sizeof(Constraint));
    return set;
}

void constraint_set_destroy(ConstraintSet* set) {
    if (!set) return;
    free(set->constraints);
    free(set);
}

void constraint_set_add(ConstraintSet* set, const Constraint* c) {
    if (!set || !c) return;
    if (set->count >= set->capacity) {
        size_t new_cap = set->capacity * 2;
        Constraint* new_arr = (Constraint*)realloc(set->constraints, new_cap * sizeof(Constraint));
        if (!new_arr) return;
        set->constraints = new_arr;
        set->capacity = new_cap;
    }
    set->constraints[set->count++] = *c;
}

void constraint_set_clear(ConstraintSet* set) {
    if (set) set->count = 0;
}

ViolationList* violation_list_create(void) {
    ViolationList* list = (ViolationList*)calloc(1, sizeof(ViolationList));
    if (!list) return NULL;
    list->capacity = INITIAL_CAPACITY;
    list->violations = (Violation*)calloc(list->capacity, sizeof(Violation));
    return list;
}

void violation_list_destroy(ViolationList* list) {
    if (!list) return;
    free(list->violations);
    free(list);
}

void violation_list_add(ViolationList* list, ValueType type, const char* message) {
    if (!list || !message) return;
    if (list->count >= list->capacity) {
        size_t new_cap = list->capacity * 2;
        Violation* new_arr = (Violation*)realloc(list->violations, new_cap * sizeof(Violation));
        if (!new_arr) return;
        list->violations = new_arr;
        list->capacity = new_cap;
    }
    list->violations[list->count].type = type;
    strncpy(list->violations[list->count].message, message, sizeof(list->violations[0].message) - 1);
    list->violations[list->count].message[sizeof(list->violations[0].message) - 1] = '\0';
    list->count++;
}

void violation_list_clear(ViolationList* list) {
    if (list) list->count = 0;
}

bool constraint_overrides_intent(const Constraint* c, ValueType type) {
    if (!c) return false;
    return c->target_type == type && (c->kind == CONSTRAINT_FIX_VALUE || c->kind == CONSTRAINT_FORBID_ASPECT);
}

void constraints_apply(const ConstraintSet* constraints, ValueStateSet* states, ViolationList* violations) {
    if (!constraints || !states || !violations) return;
    for (size_t i = 0; i < constraints->count; i++) {
        const Constraint* c = &constraints->constraints[i];
        if (c->target_type >= VALUE_TYPE_COUNT) continue;
        ValueState* vs = &states->states[c->target_type];
        switch (c->kind) {
            case CONSTRAINT_FIX_VALUE:
                *vs = value_state_present(c->fixed_val);
                break;
            case CONSTRAINT_FORBID_ASPECT:
                if (value_state_is_present(vs)) {
                    *vs = value_state_invalid(c->reason[0] ? c->reason : "forbidden");
                    violation_list_add(violations, c->target_type, "Constraint forbids this aspect");
                }
                break;
            case CONSTRAINT_LIMIT_RANGE:
                if (value_state_is_present(vs) && (vs->value < c->min_val || vs->value > c->max_val)) {
                    double clamped = vs->value < c->min_val ? c->min_val : c->max_val;
                    vs->value = clamped;
                    violation_list_add(violations, c->target_type, "Value clamped to constraint range");
                }
                break;
        }
    }
}
