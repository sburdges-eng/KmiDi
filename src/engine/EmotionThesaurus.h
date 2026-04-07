#pragma once
/*
 * EmotionThesaurus.h - 216-Node Emotion Lexicon
 * ==============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: WoundProcessor, VADCalculator, IntentPipeline
 * - Engine Layer: EmotionThesaurusLoader (JSON hydration)
 * - Common Layer: Types.h (EmotionNode, IDs, VAD fields)
 *
 * Purpose: Canonical lookup graph for affect labels, synonyms, and VAD coords.
 *
 * Features:
 * - findById, findByName, findNearest, findNearestVAD
 * - Thread-safe accessors where noted in implementation
 */

#include "common/Types.h"
#include <unordered_map>
#include <optional>
#include <mutex>

namespace kelly {

class EmotionThesaurus {
public:
    EmotionThesaurus();
    
    //==========================================================================
    // Lookup Methods
    //==========================================================================
    
    /** Find an emotion by its unique ID */
    std::optional<EmotionNode> findById(int id) const;
    
    /** Find an emotion by name (case-insensitive) */
    std::optional<EmotionNode> findByName(const std::string& name) const;
    
    /** Find emotions in a specific category */
    std::vector<EmotionNode> findByCategory(EmotionCategory category) const;
    
    /** Find the closest emotion to given valence/arousal/intensity coordinates */
    EmotionNode findNearest(float valence, float arousal, float intensity) const;
    
    /** Find the closest emotion to given VAD (Valence-Arousal-Dominance) coordinates */
    EmotionNode findNearestVAD(float valence, float arousal, float dominance) const;
    
    /** Find emotions related to a given emotion */
    std::vector<EmotionNode> findRelated(int emotionId) const;
    
    //==========================================================================
    // Traversal
    //==========================================================================
    
    /** Get all emotions (thread-safe copy) */
    std::unordered_map<int, EmotionNode> all() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_;
    }
    
    /** Get emotion count */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_.size();
    }
    
    //==========================================================================
    // Musical Mapping
    //==========================================================================
    
    /** Get suggested mode for an emotion */
    std::string suggestMode(const EmotionNode& emotion) const;
    
    /** Get tempo modifier (0.5 to 2.0, where 1.0 is neutral) */
    float suggestTempoModifier(const EmotionNode& emotion) const;
    
    /** Get dynamic range (0.0 to 1.0) */
    float suggestDynamicRange(const EmotionNode& emotion) const;
    
    //==========================================================================
    // Loading
    //==========================================================================
    
    /** Add a node to the thesaurus (public for loader) */
    void addNode(EmotionNode node);
    
    /** Add a synonym/alias for an existing emotion (for findByName lookup) */
    void addSynonym(int emotionId, const std::string& synonym);
    
private:
    std::unordered_map<int, EmotionNode> nodes_;
    std::unordered_map<std::string, int> nameIndex_;  // lowercase name -> id
    mutable std::mutex mutex_;  // Protects nodes_ and nameIndex_ for thread safety
    
    void initializeThesaurus();
    
    // Distance calculations
    static float distance(const EmotionNode& a, float v, float ar, float i);
    static float distanceVAD(const EmotionNode& a, float v, float ar, float d);
};

} // namespace kelly
