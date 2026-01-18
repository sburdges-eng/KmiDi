#include "AdaptiveGenerator.h"
#include <algorithm>
#include <cmath>
#include <cctype>

namespace kelly {

AdaptiveGenerator::AdaptiveGenerator(MidiKompanionBrain& brain, PreferenceTracker& preferenceTracker)
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

    const auto adjustments = preferenceTracker_.getAverageParameterAdjustments();
    auto clamp01 = [](float value) {
        return std::max(0.0f, std::min(1.0f, value));
    };

    for (const auto& entry : adjustments) {
        std::string name = entry.first;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const float delta = entry.second;

        if (name == "tempo" || name == "bpm" || name == "tempo_bpm") {
            adapted.tempoBpm = std::clamp(adapted.tempoBpm + static_cast<int>(std::round(delta)), 40, 240);
        } else if (name == "swing" || name == "swing_amount") {
            adapted.swingAmount = clamp01(adapted.swingAmount + delta);
        } else if (name == "syncopation" || name == "syncopation_level") {
            adapted.syncopationLevel = clamp01(adapted.syncopationLevel + delta);
        } else if (name == "humanization" || name == "humanize") {
            adapted.humanization = clamp01(adapted.humanization + delta);
        } else if (name == "melodic_range") {
            adapted.melodicRange = clamp01(adapted.melodicRange + delta);
        } else if (name == "leap_probability" || name == "leap_prob") {
            adapted.leapProbability = clamp01(adapted.leapProbability + delta);
        } else if (name == "base_velocity" || name == "velocity") {
            adapted.baseVelocity = clamp01(adapted.baseVelocity + delta);
        } else if (name == "dynamic_range") {
            adapted.dynamicRange = clamp01(adapted.dynamicRange + delta);
        } else if (name == "chromaticism" || name == "allow_chromaticism") {
            if (delta > 0.1f) {
                adapted.allowChromaticism = true;
                adapted.allowDissonance = true;
            }
        }
    }

    return adapted;
}

std::map<std::string, float> AdaptiveGenerator::getPreferredAdjustments() const {
    return preferenceTracker_.getAverageParameterAdjustments();
}

} // namespace kelly
