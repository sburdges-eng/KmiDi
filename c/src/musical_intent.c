#include "musical_intent.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_COMPONENTS 16

struct MusicalIntent {
    IntentComponent components[MAX_COMPONENTS];
    size_t component_count;
};

MusicalIntent* musical_intent_create(void) {
    MusicalIntent* intent = (MusicalIntent*)calloc(1, sizeof(MusicalIntent));
    return intent;
}

MusicalIntent* musical_intent_create_with_components(const IntentComponent* components, size_t count) {
    MusicalIntent* intent = musical_intent_create();
    if (!intent) return NULL;
    size_t to_copy = count < MAX_COMPONENTS ? count : MAX_COMPONENTS;
    for (size_t i = 0; i < to_copy; i++) {
        intent->components[i] = components[i];
        if (components[i].reason) {
            intent->components[i].reason = strdup(components[i].reason);
        }
    }
    intent->component_count = to_copy;
    return intent;
}

void musical_intent_destroy(MusicalIntent* intent) {
    if (!intent) return;
    for (size_t i = 0; i < intent->component_count; i++) {
        if (intent->components[i].reason) {
            free((void*)intent->components[i].reason);
        }
    }
    free(intent);
}

void musical_intent_update(MusicalIntent* intent, const MusicalIntent* other) {
    if (!intent || !other) return;
    for (size_t i = 0; i < other->component_count && intent->component_count < MAX_COMPONENTS; i++) {
        bool found = false;
        for (size_t j = 0; j < intent->component_count; j++) {
            if (intent->components[j].aspect == other->components[i].aspect) {
                intent->components[j].magnitude = other->components[i].magnitude;
                if (intent->components[j].reason) free((void*)intent->components[j].reason);
                intent->components[j].reason = other->components[i].reason ? strdup(other->components[i].reason) : NULL;
                found = true;
                break;
            }
        }
        if (!found) {
            intent->components[intent->component_count] = other->components[i];
            if (other->components[i].reason) {
                intent->components[intent->component_count].reason = strdup(other->components[i].reason);
            }
            intent->component_count++;
        }
    }
}

void musical_intent_merge(MusicalIntent* dest, const MusicalIntent* src) {
    musical_intent_update(dest, src);
}

char* musical_intent_serialize(const MusicalIntent* intent) {
    if (!intent) return NULL;
    size_t len = 64;
    char* buf = (char*)malloc(len);
    if (!buf) return NULL;
    buf[0] = '{';
    size_t pos = 1;
    for (size_t i = 0; i < intent->component_count; i++) {
        const IntentComponent* c = &intent->components[i];
        const char* aspect_str[] = {"harmonic", "rhythmic", "dynamic", "tempo"};
        int aspect_idx = (int)c->aspect;
        if (aspect_idx >= 0 && aspect_idx < 4) {
            int n = snprintf(buf + pos, len - pos, "\"%s\":%.4f", aspect_str[aspect_idx], c->magnitude);
            if (n > 0 && (size_t)n < len - pos) pos += (size_t)n;
            if (i < intent->component_count - 1) { buf[pos++] = ','; buf[pos] = '\0'; }
        }
    }
    buf[pos++] = '}';
    buf[pos] = '\0';
    return buf;
}

MusicalIntent* musical_intent_deserialize(const char* json) {
    (void)json;
    return musical_intent_create();
}

double musical_intent_distance(const MusicalIntent* a, const MusicalIntent* b) {
    if (!a || !b) return 1.0;
    double dist = 0.0;
    for (size_t i = 0; i < a->component_count; i++) {
        double mag_b = 0.0;
        for (size_t j = 0; j < b->component_count; j++) {
            if (b->components[j].aspect == a->components[i].aspect) {
                mag_b = b->components[j].magnitude;
                break;
            }
        }
        dist += fabs(a->components[i].magnitude - mag_b);
    }
    return dist;
}

bool musical_intent_has_aspect(const MusicalIntent* intent, IntentAspect aspect) {
    if (!intent) return false;
    for (size_t i = 0; i < intent->component_count; i++) {
        if (intent->components[i].aspect == aspect) return true;
    }
    return false;
}

double musical_intent_get_magnitude(const MusicalIntent* intent, IntentAspect aspect) {
    if (!intent) return 0.0;
    for (size_t i = 0; i < intent->component_count; i++) {
        if (intent->components[i].aspect == aspect) return intent->components[i].magnitude;
    }
    return 0.0;
}
