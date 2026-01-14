#include "KellyMLPipeline.h"

#include <array>
#include <cstddef>

// This file illustrates how an audio/MIDI engine would call into the Kelly ML pipeline.
// No UI or AppKit/JUCE UI dependencies are present here.

namespace kelly::ml {

struct AudioEngineExample {
    KellyMLPipeline pipeline;

    AudioEngineExample() {
        // Load models; empty paths are allowed (fallbacks will be used).
        pipeline.loadModels("models/emotion.json",
                            "models/melody.json",
                            "models/harmony.json",
                            "models/dynamics.json",
                            "models/groove.json");
    }

    // Called on a non-audio thread (e.g., ML thread) to prepare suggestions.
    KellyMLOutput processFrame(const float* feature128) {
        EmotionState emotion = pipeline.processAudioFeatures(feature128, 128);
        // The audio engine can prepare an audio context buffer if desired; pass nullptr to use embedding-derived context.
        return pipeline.generateMusicalSuggestions(emotion, nullptr, 0, nullptr, 0);
    }
};

} // namespace kelly::ml
