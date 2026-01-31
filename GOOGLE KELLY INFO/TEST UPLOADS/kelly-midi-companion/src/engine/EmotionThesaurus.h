#pragma once

#include "common/Types.h"
#include <unordered_map>
#include <optional>

namespace kelly {

/**
 * The 216-node emotion thesaurus.
 * 
 * Organized around three dimensions:
 * - Valence: Negative (-1.0) to Positive (+1.0)
 * - Arousal: Calm (0.0) to Excited (1.0)  
 * - Intensity: Subtle (0.0) to Extreme (1.0)
 *
 * Each emotion maps to musical attributes that drive generation.
 */
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
    
    /** Find emotions related to a given emotion */
    std::vector<EmotionNode> findRelated(int emotionId) const;
    
    //==========================================================================
    // Traversal
    //==========================================================================
    
    /** Get all emotions */
    const std::unordered_map<int, EmotionNode>& all() const { return nodes_; }
    
    /** Get emotion count */
    size_t size() const { return nodes_.size(); }
    
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
    
private:
    std::unordered_map<int, EmotionNode> nodes_;
    std::unordered_map<std::string, int> nameIndex_;  // lowercase name -> id
    
    void initializeThesaurus();
    
    static float distance(const EmotionNode& a, float v, float ar, float i);
};

} // namespace kelly
