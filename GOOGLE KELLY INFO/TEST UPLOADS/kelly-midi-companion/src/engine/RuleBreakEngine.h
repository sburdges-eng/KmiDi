#pragma once

#include "common/Types.h"
#include <vector>
#include <algorithm>

namespace kelly {

/**
 * Rule Break Engine - Generates intentional music theory violations.
 * 
 * Creates rule breaks based on emotional intensity and type.
 * Intense negative emotions trigger more dissonance and structural breaks.
 */
class RuleBreakEngine {
public:
    RuleBreakEngine() = default;
    ~RuleBreakEngine() = default;
    
    /**
     * Generate rule breaks based on emotion intensity and type.
     * @param emotion The emotion node
     * @return Vector of rule breaks to apply
     */
    std::vector<RuleBreak> generateRuleBreaks(const EmotionNode& emotion);
    
    /**
     * Generate rule breaks for a journey between two emotions.
     * @param emotionA Starting emotion
     * @param emotionB Target emotion
     * @return Vector of rule breaks for the transition
     */
    std::vector<RuleBreak> generateJourneyRuleBreaks(
        const EmotionNode& emotionA,
        const EmotionNode& emotionB
    );

private:
    /**
     * Calculate rule break severity based on emotion intensity.
     */
    float calculateSeverity(const EmotionNode& emotion, float baseSeverity = 0.5f) const;
};

} // namespace kelly

