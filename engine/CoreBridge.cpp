#include "engine/CoreBridge.h"
#include <cstring>
#include <algorithm>

#ifdef KELLY_USE_CORE_BRIDGE
extern "C" {
#include "intent_engine.h"
#include "intent_result.h"
#include "value_state.h"
}
// Core uses ::IntentResult (C global); plugin uses kelly::IntentResult
using CoreIntentResult = ::IntentResult;
#endif

namespace kelly {

#ifdef KELLY_USE_CORE_BRIDGE
namespace {

// Map plugin emotion to core AbstractInput for validation
AbstractInput buildAbstractInput(const IntentResult& pluginResult) {
    AbstractInput in = {};
    in.type = "chord_change";
    in.aspect = INTENT_ASPECT_HARMONIC;
    // valence in [-1,1] -> magnitude in [0,1]
    float valence = pluginResult.emotion.valence;
    in.magnitude = 0.5f + (valence * 0.5f);
    in.magnitude = std::clamp(static_cast<double>(in.magnitude), 0.0, 1.0);
    if (pluginResult.escalationTokenPresent) {
        in.escalation_token = "more_harmonic_color";
    } else {
        in.escalation_token = nullptr;
    }
    return in;
}

} // namespace

IntentResult CoreBridge::validateViaCore(const IntentResult& pluginResult) {
    // Already abstain or invalid — pass through
    if (pluginResult.isAbstain() || pluginResult.isInvalid()) {
        return pluginResult;
    }

    AbstractInput in = buildAbstractInput(pluginResult);
    CoreIntentResult* coreResult = intent_engine_process(&in, nullptr);
    if (!coreResult) {
        return pluginResult; // Core failed; keep plugin result
    }

    IntentResult out = pluginResult;

    // Core abstain
    const char* meta = intent_result_get_metadata(coreResult);
    if (meta && std::strstr(meta, "abstain") != nullptr) {
        intent_result_destroy(coreResult);
        return IntentResult::abstain();
    }

    // Core invalid (violations or value state invalid)
    if (!intent_result_is_valid(coreResult) || intent_result_has_violations(coreResult)) {
        intent_result_destroy(coreResult);
        return IntentResult::invalid();
    }

    // Use core escalation for gate — richer than plugin bool
    const HarmonicEscalationBranches* h = intent_result_get_harmonic_escalation(coreResult);
    const RhythmicEscalationBranches* r = intent_result_get_rhythmic_escalation(coreResult);
    out.escalationTokenPresent = (h && h->count > 0) || (r && r->count > 0);

    intent_result_destroy(coreResult);
    return out;
}

#else
// No CoreBridge: pass through unchanged
IntentResult CoreBridge::validateViaCore(const IntentResult& pluginResult) {
    return pluginResult;
}
#endif

} // namespace kelly
