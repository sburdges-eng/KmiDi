#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include <stddef.h>
#include <stdbool.h>
#include "value_state.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CONSTRAINT_FIX_VALUE,
    CONSTRAINT_FORBID_ASPECT,
    CONSTRAINT_LIMIT_RANGE
} ConstraintKind;

typedef struct {
    ConstraintKind kind;
    ValueType target_type;
    double min_val;
    double max_val;
    double fixed_val;
    bool forbid;
    char reason[256];
} Constraint;

typedef struct {
    Constraint* constraints;
    size_t count;
    size_t capacity;
} ConstraintSet;

ConstraintSet* constraint_set_create(void);
void constraint_set_destroy(ConstraintSet* set);
void constraint_set_add(ConstraintSet* set, const Constraint* c);
void constraint_set_clear(ConstraintSet* set);

typedef struct {
    ValueType type;
    char message[512];
} Violation;

typedef struct {
    Violation* violations;
    size_t count;
    size_t capacity;
} ViolationList;

ViolationList* violation_list_create(void);
void violation_list_destroy(ViolationList* list);
void violation_list_add(ViolationList* list, ValueType type, const char* message);
void violation_list_clear(ViolationList* list);

void constraints_apply(const ConstraintSet* constraints, ValueStateSet* states, ViolationList* violations);
bool constraint_overrides_intent(const Constraint* c, ValueType type);

#ifdef __cplusplus
}
#endif

#endif /* CONSTRAINTS_H */
