#pragma once

/**
 * Intent IR JSON Serialization
 *
 * Lightweight JSON serialization for IntentFrame.
 * Uses cJSON library for portability.
 */

#include "IntentIR.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration - cJSON is included in implementation
struct cJSON;

/**
 * Serialize IntentFrame to JSON string.
 * Caller must free the returned string.
 * Returns NULL on error.
 */
char* intent_frame_to_json(const IntentFrame* frame);

/**
 * Deserialize IntentFrame from JSON string.
 * Returns true on success, false on error.
 */
bool intent_frame_from_json(const char* json_str, IntentFrame* frame);

/**
 * Serialize IntentFrame to cJSON object.
 * Caller must call cJSON_Delete on returned object.
 * Returns NULL on error.
 */
struct cJSON* intent_frame_to_cjson(const IntentFrame* frame);

/**
 * Deserialize IntentFrame from cJSON object.
 * Returns true on success, false on error.
 */
bool intent_frame_from_cjson(const struct cJSON* json, IntentFrame* frame);

#ifdef __cplusplus
}
#endif
