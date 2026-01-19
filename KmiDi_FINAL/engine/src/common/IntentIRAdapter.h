#pragma once

/**
 * IntentIRAdapter - Convert between IntentFrame (IR v1) and IntentResult
 *
 * Provides compatibility layer during migration from IntentResult to IntentFrame.
 * Engines can gradually migrate to consume IntentFrame directly.
 */

#include "kmidi/IntentIR.h"
#include "KellyTypes.h"
#include <optional>

namespace kelly {

/**
 * Convert IntentFrame (IR v1) to IntentResult for engine compatibility.
 *
 * Maps IR biases to concrete musical parameters that engines expect.
 * This is a lossy conversion - IR contains only biases, IntentResult has concrete values.
 */
IntentResult convertIntentIRToIntentResult(const IntentFrame& frame);

/**
 * Convert IntentResult to IntentFrame (IR v1).
 *
 * Reverse mapping - extracts biases from concrete parameters.
 * This is approximate since IntentResult doesn't preserve all bias information.
 */
IntentFrame convertIntentResultToIntentIR(const IntentResult& result);

/**
 * Check if IntentFrame version is supported by this adapter.
 */
bool isIntentIRVersionSupported(uint16_t version);

/**
 * Clamp and validate IntentFrame before use.
 * Modifies frame in-place.
 */
void prepareIntentFrame(IntentFrame& frame);

} // namespace kelly
