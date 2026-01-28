#pragma once
/*
 * KellyBrain.h - Core Intent Processing Engine
 * =============================================
 * Processes Wounds → Musical Parameters via EmotionThesaurus
 * 
 * Re-enabled bindings using unified KellyTypes.h
 * Works with IntentPipeline for Wound → EmotionNode → IntentResult flow
 */

#include "KellyTypes.h"
#include <memory>
#include <functional>
#include <unordered_map>

namespace kelly {

// =============================================================================
// Emotion Thesaurus (216-node system)
// =============================================================================

class EmotionThesaurus {
public:
    EmotionThesaurus();
    ~EmotionThesaurus() = default;
    
    // Core lookups
    const EmotionNode* findByName(const std::string& name) const;
    const EmotionNode* findById(int id) const;
    const EmotionNode* findByPosition(int layer, int sub, int subSub) const;
    
    // Dimensional queries
    std::vector<const EmotionNode*> findByValence(float minVal, float maxVal) const;
    std::vector<const EmotionNode*> findByArousal(float minArousal, float maxArousal) const;
    std::vector<const EmotionNode*> findNearby(const EmotionNode& node, float threshold = 0.3f) const;
    
    // Category queries
    std::vector<const EmotionNode*> getCategory(EmotionCategory category) const;
    std::vector<const EmotionNode*> getSubEmotions(int layerIndex) const;
    
    // Synonym resolution
    const EmotionNode* resolveVernacular(const std::string& vernacular) const;
    
    // Stats
    size_t size() const { return nodes_.size(); }
    
    // Load from JSON
    bool loadFromFile(const std::string& path);
    bool loadFromJson(const std::string& json);

private:
    void initializeDefault();
    float distance(const EmotionNode& a, const EmotionNode& b) const;
    
    std::unordered_map<int, EmotionNode> nodes_;
    std::unordered_map<std::string, int> nameIndex_;
    std::unordered_map<std::string, int> synonymIndex_;
};

// =============================================================================
// Intent Pipeline (Wound → IntentResult)
// =============================================================================

class IntentPipeline {
public:
    IntentPipeline();
    explicit IntentPipeline(std::shared_ptr<EmotionThesaurus> thesaurus);
    ~IntentPipeline() = default;
    
    // Main processing
    IntentResult processWound(const Wound& wound);
    IntentResult processEmotion(const EmotionNode& emotion, float intensity = 0.7f);
    IntentResult processText(const std::string& description);
    
    // Access thesaurus
    EmotionThesaurus& thesaurus() { return *thesaurus_; }
    const EmotionThesaurus& thesaurus() const { return *thesaurus_; }
    
    // Configuration
    void setDefaultKey(const std::string& key) { defaultKey_ = key; }
    void setDefaultTempo(int bpm) { defaultTempo_ = bpm; }
    void setGenre(const std::string& genre) { genre_ = genre; }
    
    // Rule-break preferences
    void enableRuleBreak(RuleBreakType type, bool enabled);
    void setRuleBreakIntensity(RuleBreakType type, float intensity);

private:
    // Internal processing stages
    EmotionNode resolveEmotion(const Wound& wound);
    std::vector<RuleBreak> determineRuleBreaks(const EmotionNode& emotion, const Wound& wound);
    std::vector<std::string> generateProgression(const EmotionNode& emotion, const std::vector<RuleBreak>& breaks);
    MusicalAttributes deriveAttributes(const EmotionNode& emotion);
    
    std::shared_ptr<EmotionThesaurus> thesaurus_;
    std::string defaultKey_ = "C";
    int defaultTempo_ = 120;
    std::string genre_ = "lo-fi";
    
    std::array<bool, static_cast<size_t>(RuleBreakType::COUNT)> enabledRuleBreaks_;
    std::array<float, static_cast<size_t>(RuleBreakType::COUNT)> ruleBreakIntensities_;
};

// =============================================================================
// KellyBrain - High-Level Interface
// =============================================================================

class KellyBrain {
public:
    KellyBrain();
    ~KellyBrain() = default;
    
    // Initialize from data directory
    bool initialize(const std::string& dataPath);
    
    // Main generation entry points
    IntentResult fromWound(const Wound& wound);
    IntentResult fromText(const std::string& description);
    IntentResult fromEmotion(const std::string& emotionName, float intensity = 0.7f);
    
    // Generate MIDI from intent
    GeneratedMidi generateMidi(const IntentResult& intent, int bars = 8);
    
    // Direct access
    IntentPipeline& pipeline() { return *pipeline_; }
    EmotionThesaurus& thesaurus() { return pipeline_->thesaurus(); }
    
    // Convenience
    static std::string woundToDescription(const Wound& wound);
    static Wound descriptionToWound(const std::string& description);

private:
    std::unique_ptr<IntentPipeline> pipeline_;
    bool initialized_ = false;
};

// =============================================================================
// Factory Functions
// =============================================================================

inline std::unique_ptr<KellyBrain> createKellyBrain() {
    return std::make_unique<KellyBrain>();
}

inline std::unique_ptr<IntentPipeline> createIntentPipeline() {
    return std::make_unique<IntentPipeline>();
}

inline std::unique_ptr<EmotionThesaurus> createEmotionThesaurus() {
    return std::make_unique<EmotionThesaurus>();
}

} // namespace kelly
