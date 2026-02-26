#pragma once

#include "common/Types.h"
#include <string>

namespace kelly {

/**
 * Bridge between plugin IntentResult and minimal_listening_core C library.
 * Delegates validation and escalation to core as single source of truth.
 *
 * Define KELLY_USE_CORE_BRIDGE when building with liblistening.a linked.
 * Without it, validateViaCore() returns the input unchanged.
 */
class CoreBridge {
public:
    /**
     * Validate plugin result via core. If core returns abstain or invalid,
     * overrides plugin result accordingly.
     *
     * @param pluginResult Result from IntentPipeline (emotion, mode, tempo, escalationTokenPresent)
     * @return Plugin IntentResult, possibly overridden to ABSTAIN/INVALID by core
     */
    static IntentResult validateViaCore(const IntentResult& pluginResult);
};

} // namespace kelly
