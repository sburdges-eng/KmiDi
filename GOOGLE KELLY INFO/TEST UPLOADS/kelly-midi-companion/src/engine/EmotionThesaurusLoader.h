#pragma once

#include "engine/EmotionThesaurus.h"
#include <juce_core/juce_core.h>
#include <string>
#include <vector>

namespace kelly {

/**
 * Loads the 216-node emotion thesaurus from JSON files.
 * 
 * Expected JSON structure:
 * {
 *   "category": "sad",
 *   "sub_emotions": {
 *     "grief": {
 *       "sub_sub_emotions": {
 *         "bereaved": {
 *           "intensity_tiers": {
 *             "1_subtle": ["touched", "moved"],
 *             "2_mild": ["bereaved", "mourning"],
 *             ...
 *           }
 *         }
 *       }
 *     }
 *   }
 * }
 */
class EmotionThesaurusLoader {
public:
    /**
     * Load all emotion JSON files and populate the thesaurus.
     * 
     * @param dataDirectory Path to directory containing emotion JSON files
     * @param thesaurus The thesaurus to populate
     * @return Number of emotions loaded
     */
    static int loadFromJsonFiles(const juce::File& dataDirectory, EmotionThesaurus& thesaurus);
    
    /**
     * Load a single emotion JSON file.
     * 
     * @param jsonFile Path to JSON file
     * @param thesaurus The thesaurus to populate
     * @return Number of emotions loaded from this file
     */
    static int loadFromJsonFile(const juce::File& jsonFile, EmotionThesaurus& thesaurus);
    
private:
    static EmotionCategory categoryFromString(const std::string& categoryStr);
    static float valenceFromString(const std::string& valenceStr);
    static float intensityFromTier(const std::string& tierStr);
    static float arousalFromIntensity(float intensity, EmotionCategory category);
    
    static void processEmotionNode(
        const juce::var& nodeData,
        const std::string& categoryName,
        const std::string& subEmotionName,
        const std::string& subSubEmotionName,
        const std::string& tierName,
        const std::string& emotionWord,
        EmotionThesaurus& thesaurus,
        int& nextId
    );
    
    static void processIntensityTier(
        const juce::var& tierData,
        const std::string& categoryName,
        const std::string& subEmotionName,
        const std::string& subSubEmotionName,
        const std::string& tierName,
        EmotionThesaurus& thesaurus,
        int& nextId
    );
    
    static void processSubSubEmotion(
        const juce::var& subSubData,
        const std::string& categoryName,
        const std::string& subEmotionName,
        const std::string& subSubEmotionName,
        EmotionThesaurus& thesaurus,
        int& nextId
    );
    
    static void processSubEmotion(
        const juce::var& subData,
        const std::string& categoryName,
        const std::string& subEmotionName,
        EmotionThesaurus& thesaurus,
        int& nextId
    );
};

} // namespace kelly

