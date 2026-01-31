#include "engine/EmotionThesaurus.h"
#include "engine/EmotionThesaurusLoader.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <juce_core/juce_core.h>

namespace kelly {

EmotionThesaurus::EmotionThesaurus() {
    // Try to load from JSON files first, fall back to hardcoded if not found
    juce::File dataDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
        .getParentDirectory()
        .getChildFile("Kelly_MIDI_Project")
        .getChildFile("kellymidicompanion")
        .getChildFile("kellymidicompanion_data");
    
    int loaded = EmotionThesaurusLoader::loadFromJsonFiles(dataDir, *this);
    
    if (loaded == 0) {
        // Fall back to hardcoded emotions if JSON loading failed
    initializeThesaurus();
    }
}

void EmotionThesaurus::addNode(EmotionNode node) {
    // Build lowercase name index
    std::string lowerName = node.name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    nameIndex_[lowerName] = node.id;
    
    nodes_[node.id] = std::move(node);
}

void EmotionThesaurus::initializeThesaurus() {
    // =========================================================================
    // SADNESS CLUSTER (Negative valence, varying arousal)
    // =========================================================================
    addNode({1, "Grief", EmotionCategory::Sadness, 1.0f, -0.9f, 0.3f, {2, 3, 4}});
    addNode({2, "Melancholy", EmotionCategory::Sadness, 0.5f, -0.6f, 0.2f, {1, 5, 6}});
    addNode({3, "Despair", EmotionCategory::Sadness, 0.9f, -1.0f, 0.4f, {1, 4}});
    addNode({4, "Sorrow", EmotionCategory::Sadness, 0.7f, -0.8f, 0.3f, {1, 3, 5}});
    addNode({5, "Wistfulness", EmotionCategory::Sadness, 0.3f, -0.3f, 0.2f, {2, 6}});
    addNode({6, "Longing", EmotionCategory::Sadness, 0.6f, -0.4f, 0.5f, {2, 5, 7}});
    addNode({7, "Heartache", EmotionCategory::Sadness, 0.8f, -0.7f, 0.4f, {1, 4, 6}});
    addNode({8, "Emptiness", EmotionCategory::Sadness, 0.6f, -0.5f, 0.1f, {2, 9}});
    addNode({9, "Numbness", EmotionCategory::Sadness, 0.4f, -0.3f, 0.05f, {8, 10}});
    addNode({10, "Resignation", EmotionCategory::Sadness, 0.5f, -0.4f, 0.15f, {9, 2}});
    
    // =========================================================================
    // ANGER CLUSTER (Negative valence, high arousal)
    // =========================================================================
    addNode({20, "Rage", EmotionCategory::Anger, 1.0f, -0.8f, 1.0f, {21, 22}});
    addNode({21, "Fury", EmotionCategory::Anger, 0.95f, -0.9f, 0.95f, {20, 22}});
    addNode({22, "Frustration", EmotionCategory::Anger, 0.6f, -0.5f, 0.7f, {20, 23, 24}});
    addNode({23, "Irritation", EmotionCategory::Anger, 0.4f, -0.3f, 0.5f, {22, 24}});
    addNode({24, "Annoyance", EmotionCategory::Anger, 0.3f, -0.2f, 0.4f, {22, 23}});
    addNode({25, "Resentment", EmotionCategory::Anger, 0.7f, -0.6f, 0.5f, {22, 26}});
    addNode({26, "Bitterness", EmotionCategory::Anger, 0.6f, -0.7f, 0.4f, {25, 10}});
    addNode({27, "Hostility", EmotionCategory::Anger, 0.8f, -0.7f, 0.8f, {20, 21}});
    
    // =========================================================================
    // FEAR CLUSTER (Negative valence, high arousal, defensive)
    // =========================================================================
    addNode({40, "Terror", EmotionCategory::Fear, 1.0f, -0.9f, 1.0f, {41, 42}});
    addNode({41, "Panic", EmotionCategory::Fear, 0.95f, -0.8f, 0.95f, {40, 42}});
    addNode({42, "Anxiety", EmotionCategory::Fear, 0.6f, -0.5f, 0.7f, {41, 43, 44}});
    addNode({43, "Worry", EmotionCategory::Fear, 0.4f, -0.3f, 0.5f, {42, 44}});
    addNode({44, "Unease", EmotionCategory::Fear, 0.3f, -0.2f, 0.4f, {42, 43}});
    addNode({45, "Dread", EmotionCategory::Fear, 0.8f, -0.7f, 0.6f, {40, 46}});
    addNode({46, "Apprehension", EmotionCategory::Fear, 0.5f, -0.4f, 0.5f, {45, 43}});
    addNode({47, "Vulnerability", EmotionCategory::Fear, 0.6f, -0.4f, 0.3f, {44, 8}});
    
    // =========================================================================
    // JOY CLUSTER (Positive valence, varying arousal)
    // =========================================================================
    addNode({60, "Ecstasy", EmotionCategory::Joy, 1.0f, 1.0f, 1.0f, {61, 62}});
    addNode({61, "Elation", EmotionCategory::Joy, 0.9f, 0.9f, 0.9f, {60, 62}});
    addNode({62, "Happiness", EmotionCategory::Joy, 0.7f, 0.7f, 0.7f, {61, 63, 64}});
    addNode({63, "Contentment", EmotionCategory::Joy, 0.5f, 0.6f, 0.3f, {62, 64}});
    addNode({64, "Serenity", EmotionCategory::Joy, 0.4f, 0.5f, 0.2f, {63, 65}});
    addNode({65, "Peace", EmotionCategory::Joy, 0.3f, 0.4f, 0.15f, {64, 66}});
    addNode({66, "Calm", EmotionCategory::Joy, 0.2f, 0.3f, 0.1f, {65}});
    addNode({67, "Relief", EmotionCategory::Joy, 0.6f, 0.5f, 0.5f, {62, 63}});
    addNode({68, "Gratitude", EmotionCategory::Joy, 0.6f, 0.6f, 0.4f, {62, 80}});
    addNode({69, "Hope", EmotionCategory::Joy, 0.7f, 0.6f, 0.6f, {62, 90}});
    
    // =========================================================================
    // TRUST CLUSTER (Positive valence, moderate arousal)
    // =========================================================================
    addNode({80, "Love", EmotionCategory::Trust, 0.9f, 0.8f, 0.7f, {81, 82, 68}});
    addNode({81, "Adoration", EmotionCategory::Trust, 0.85f, 0.85f, 0.75f, {80, 82}});
    addNode({82, "Affection", EmotionCategory::Trust, 0.6f, 0.6f, 0.5f, {80, 83}});
    addNode({83, "Tenderness", EmotionCategory::Trust, 0.5f, 0.5f, 0.4f, {82, 84}});
    addNode({84, "Warmth", EmotionCategory::Trust, 0.4f, 0.5f, 0.35f, {83, 63}});
    addNode({85, "Acceptance", EmotionCategory::Trust, 0.5f, 0.4f, 0.3f, {84, 63}});
    addNode({86, "Belonging", EmotionCategory::Trust, 0.6f, 0.5f, 0.4f, {80, 85}});
    
    // =========================================================================
    // ANTICIPATION CLUSTER (Positive valence, high arousal, forward-looking)
    // =========================================================================
    addNode({90, "Excitement", EmotionCategory::Anticipation, 0.9f, 0.7f, 0.9f, {91, 69}});
    addNode({91, "Eagerness", EmotionCategory::Anticipation, 0.7f, 0.6f, 0.75f, {90, 92}});
    addNode({92, "Curiosity", EmotionCategory::Anticipation, 0.5f, 0.4f, 0.6f, {91, 93}});
    addNode({93, "Interest", EmotionCategory::Anticipation, 0.4f, 0.3f, 0.5f, {92}});
    addNode({94, "Vigilance", EmotionCategory::Anticipation, 0.6f, 0.2f, 0.7f, {46, 93}});
    
    // =========================================================================
    // SURPRISE CLUSTER (Neutral valence, high arousal spike)
    // =========================================================================
    addNode({100, "Amazement", EmotionCategory::Surprise, 0.9f, 0.5f, 0.95f, {101}});
    addNode({101, "Astonishment", EmotionCategory::Surprise, 0.85f, 0.4f, 0.9f, {100, 102}});
    addNode({102, "Shock", EmotionCategory::Surprise, 0.8f, -0.1f, 0.85f, {101, 40}});
    addNode({103, "Startle", EmotionCategory::Surprise, 0.7f, 0.0f, 0.8f, {102}});
    addNode({104, "Wonder", EmotionCategory::Surprise, 0.6f, 0.6f, 0.6f, {100, 92}});
    
    // =========================================================================
    // DISGUST CLUSTER (Negative valence, rejection)
    // =========================================================================
    addNode({110, "Revulsion", EmotionCategory::Disgust, 0.9f, -0.8f, 0.7f, {111}});
    addNode({111, "Loathing", EmotionCategory::Disgust, 0.85f, -0.85f, 0.6f, {110, 112}});
    addNode({112, "Contempt", EmotionCategory::Disgust, 0.7f, -0.6f, 0.5f, {111, 26}});
    addNode({113, "Distaste", EmotionCategory::Disgust, 0.4f, -0.3f, 0.4f, {112}});
    addNode({114, "Disapproval", EmotionCategory::Disgust, 0.5f, -0.4f, 0.45f, {112, 113}});
    
    // =========================================================================
    // COMPOUND / COMPLEX EMOTIONS (Kelly's specialty)
    // =========================================================================
    addNode({200, "Bittersweetness", EmotionCategory::Sadness, 0.6f, 0.1f, 0.4f, {2, 62, 6}});
    addNode({201, "Nostalgia", EmotionCategory::Sadness, 0.5f, 0.2f, 0.3f, {5, 6, 200}});
    addNode({202, "Catharsis", EmotionCategory::Joy, 0.8f, 0.3f, 0.7f, {1, 67}});
    addNode({203, "Yearning", EmotionCategory::Anticipation, 0.7f, -0.2f, 0.6f, {6, 91}});
    addNode({204, "Ambivalence", EmotionCategory::Surprise, 0.5f, 0.0f, 0.5f, {42, 69}});
    addNode({205, "Melancholic Hope", EmotionCategory::Sadness, 0.6f, 0.1f, 0.5f, {2, 69}});
    addNode({206, "Tender Grief", EmotionCategory::Sadness, 0.75f, -0.3f, 0.35f, {1, 83}});
    addNode({207, "Defiant Joy", EmotionCategory::Joy, 0.8f, 0.4f, 0.8f, {62, 20}});
    addNode({208, "Quiet Rage", EmotionCategory::Anger, 0.7f, -0.6f, 0.3f, {20, 9}});
    addNode({209, "Anxious Hope", EmotionCategory::Fear, 0.6f, 0.1f, 0.65f, {42, 69}});
}

std::optional<EmotionNode> EmotionThesaurus::findById(int id) const {
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<EmotionNode> EmotionThesaurus::findByName(const std::string& name) const {
    std::string lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    
    auto it = nameIndex_.find(lowerName);
    if (it != nameIndex_.end()) {
        return findById(it->second);
    }
    return std::nullopt;
}

std::vector<EmotionNode> EmotionThesaurus::findByCategory(EmotionCategory category) const {
    std::vector<EmotionNode> results;
    for (const auto& [id, node] : nodes_) {
        if (node.category == category) {
            results.push_back(node);
        }
    }
    return results;
}

float EmotionThesaurus::distance(const EmotionNode& node, float v, float ar, float i) {
    float dv = node.valence - v;
    float da = node.arousal - ar;
    float di = node.intensity - i;
    return std::sqrt(dv * dv + da * da + di * di);
}

EmotionNode EmotionThesaurus::findNearest(float valence, float arousal, float intensity) const {
    EmotionNode nearest = nodes_.begin()->second;
    float minDist = distance(nearest, valence, arousal, intensity);
    
    for (const auto& [id, node] : nodes_) {
        float d = distance(node, valence, arousal, intensity);
        if (d < minDist) {
            minDist = d;
            nearest = node;
        }
    }
    return nearest;
}

std::vector<EmotionNode> EmotionThesaurus::findRelated(int emotionId) const {
    std::vector<EmotionNode> results;
    auto it = nodes_.find(emotionId);
    if (it == nodes_.end()) return results;
    
    for (int relatedId : it->second.relatedEmotions) {
        auto related = findById(relatedId);
        if (related) {
            results.push_back(*related);
        }
    }
    return results;
}

std::string EmotionThesaurus::suggestMode(const EmotionNode& emotion) const {
    // Valence-based mode selection with nuance
    if (emotion.valence < -0.7f) {
        return emotion.arousal > 0.6f ? "phrygian" : "minor";
    } else if (emotion.valence < -0.3f) {
        return emotion.arousal > 0.5f ? "dorian" : "aeolian";
    } else if (emotion.valence < 0.3f) {
        // Ambivalent emotions get modal mixture
        return emotion.arousal > 0.5f ? "mixolydian" : "dorian";
    } else if (emotion.valence < 0.7f) {
        return emotion.arousal > 0.6f ? "lydian" : "major";
    } else {
        return emotion.arousal > 0.7f ? "lydian" : "ionian";
    }
}

float EmotionThesaurus::suggestTempoModifier(const EmotionNode& emotion) const {
    // Base tempo modifier on arousal and intensity
    // Range: 0.5 (half speed) to 2.0 (double speed)
    float arousalFactor = 0.5f + (emotion.arousal * 1.0f);  // 0.5 to 1.5
    float intensityBoost = emotion.intensity * 0.3f;         // 0.0 to 0.3
    
    return std::clamp(arousalFactor + intensityBoost, 0.5f, 2.0f);
}

float EmotionThesaurus::suggestDynamicRange(const EmotionNode& emotion) const {
    // High intensity = wide dynamic range
    // Negative valence + high arousal = extreme contrasts
    float base = emotion.intensity;
    
    if (emotion.valence < -0.5f && emotion.arousal > 0.7f) {
        return std::min(1.0f, base + 0.3f);  // Boost for angry/fearful
    }
    
    if (emotion.valence > 0.5f && emotion.arousal < 0.3f) {
        return std::max(0.2f, base - 0.2f);  // Compress for peaceful
    }
    
    return base;
}

} // namespace kelly
