#pragma once

#include "common/Types.h"
#include "engine/EmotionThesaurus.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

namespace kelly {

/**
 * Wound Processor - Processes emotional wounds into structured emotion data.
 * 
 * Analyzes wound descriptions using keyword matching and emotion thesaurus
 * to identify the most appropriate emotional state.
 */
class WoundProcessor {
public:
    explicit WoundProcessor(EmotionThesaurus& thesaurus) : thesaurus_(thesaurus) {}
    ~WoundProcessor() = default;
    
    /**
     * Process a wound description into an emotion node.
     * Uses keyword matching and thesaurus lookup to find the best match.
     * @param wound The wound to process
     * @return The identified emotion node
     */
    EmotionNode processWound(const Wound& wound);
    
    /**
     * Find emotion by keywords in text.
     * @param text The text to analyze
     * @param intensity The intensity level
     * @return The best matching emotion node
     */
    EmotionNode findEmotionByKeywords(const std::string& text, float intensity = 0.5f);

private:
    EmotionThesaurus& thesaurus_;
    
    struct EmotionClue {
        std::vector<std::string> keywords;
        int primaryEmotionId;
        float confidence;
    };
    
    std::vector<EmotionClue> buildEmotionClues();
};

} // namespace kelly

