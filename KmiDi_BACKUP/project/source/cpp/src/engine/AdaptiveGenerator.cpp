#include "AdaptiveGenerator.h"
#include "KellyBrain.h"
#include <algorithm>
#include <cmath>

namespace kelly {

AdaptiveGenerator::AdaptiveGenerator(KellyBrain& brain, PreferenceTracker& preferenceTracker)
    : brain_(brain), preferenceTracker_(preferenceTracker) {
}

GeneratedMidi AdaptiveGenerator::generateMidi(const IntentResult& intent, int bars) {
    IntentResult adaptedIntent = intent;

    if (adaptiveEnabled_.load() && preferenceTracker_.isEnabled()) {
        adaptedIntent = adaptIntent(intent);
    }

    return brain_.generateMidi(adaptedIntent, bars);
}

GeneratedMidi AdaptiveGenerator::generateMidiFromWound(const Wound& wound, int bars) {
    // Process wound normally first
    IntentResult baseIntent = brain_.fromWound(wound);

    // Adapt if enabled
    if (adaptiveEnabled_.load() && preferenceTracker_.isEnabled()) {
        baseIntent = adaptIntent(baseIntent);
    }

    return brain_.generateMidi(baseIntent, bars);
}

IntentResult AdaptiveGenerator::adaptIntent(const IntentResult& baseIntent) {
    IntentResult adapted = baseIntent;

    if (!adaptiveEnabled_.load()) return adapted;

    // Get preferred adjustments from preference tracker
    auto adjustments = getPreferredAdjustments();

    // Apply adjustments to intent parameters
    // Note: Parameter names should match those used in recordParameterAdjustment
    if (adjustments.count("tempoBpm")) adapted.tempoBpm += static_cast<int>(adjustments["tempoBpm"]);
    if (adjustments.count("melodicRange")) adapted.melodicRange = std::clamp(adapted.melodicRange + adjustments["melodicRange"], 0.0f, 1.0f);
    if (adjustments.count("leapProbability")) adapted.leapProbability = std::clamp(adapted.leapProbability + adjustments["leapProbability"], 0.0f, 1.0f);
    if (adjustments.count("humanization")) adapted.humanization = std::clamp(adapted.humanization + adjustments["humanization"], 0.0f, 1.0f);
    if (adjustments.count("baseVelocity")) adapted.baseVelocity = std::clamp(adapted.baseVelocity + adjustments["baseVelocity"], 0.0f, 1.0f);
    if (adjustments.count("dynamicRange")) adapted.dynamicRange = std::clamp(adapted.dynamicRange + adjustments["dynamicRange"], 0.0f, 1.0f);

    return adapted;
}

std::map<std::string, float> AdaptiveGenerator::getPreferredAdjustments() const {
    return preferenceTracker_.getPreferredAdjustments();
}

} // namespace kelly
