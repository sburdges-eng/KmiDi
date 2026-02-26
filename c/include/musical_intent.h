#ifndef MUSICAL_INTENT_H
#define MUSICAL_INTENT_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MusicalIntent MusicalIntent;

typedef enum {
    INTENT_ASPECT_HARMONIC,
    INTENT_ASPECT_RHYTHMIC,
    INTENT_ASPECT_DYNAMIC,
    INTENT_ASPECT_TEMPO,
    INTENT_ASPECT_COUNT
} IntentAspect;

typedef struct {
    IntentAspect aspect;
    double magnitude;
    const char* reason;
} IntentComponent;

MusicalIntent* musical_intent_create(void);
MusicalIntent* musical_intent_create_with_components(const IntentComponent* components, size_t count);
void musical_intent_destroy(MusicalIntent* intent);

void musical_intent_update(MusicalIntent* intent, const MusicalIntent* other);
void musical_intent_merge(MusicalIntent* dest, const MusicalIntent* src);

char* musical_intent_serialize(const MusicalIntent* intent);
MusicalIntent* musical_intent_deserialize(const char* json);

double musical_intent_distance(const MusicalIntent* a, const MusicalIntent* b);

bool musical_intent_has_aspect(const MusicalIntent* intent, IntentAspect aspect);
double musical_intent_get_magnitude(const MusicalIntent* intent, IntentAspect aspect);

#ifdef __cplusplus
}
#endif

#endif /* MUSICAL_INTENT_H */
