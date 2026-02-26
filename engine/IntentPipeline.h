#pragma once

#include "common/Types.h"
#include <string>

namespace kelly {

/**
 * Processes emotional descriptions and converts them into musical intent.
 * Analyzes wounds, journeys, and generates IntentResults that guide MIDI generation.
 */
class IntentPipeline {
public:
    IntentPipeline();
    ~IntentPipeline() = default;
    
    /** Process a wound description into musical intent */
    IntentResult process(const Wound& wound);
    
    /** Process a Side A to Side B journey into musical intent */
    IntentResult processJourney(const SideA& current, const SideB& desired);
    
private:
    /** Analyze text to extract emotional content */
    Emotion analyzeEmotion(const std::string& text, float intensity);
    
    /** Determine musical mode from emotion */
    std::string determineMode(const Emotion& emotion);
    
    /** Calculate tempo multiplier from emotion */
    float calculateTempo(const Emotion& emotion);
    
    /** Check for rule breaks in the intent */
    std::vector<RuleBreak> checkRuleBreaks(const Emotion& emotion, const std::string& mode);
    
    /** Simple keyword-based emotion detection */
    float detectEmotionalValence(const std::string& text);
    float detectEmotionalArousal(const std::string& text);

    /** LISTENING: Detect explicit escalation tokens ("More", "Richer", etc.) */
    bool detectEscalationToken(const std::string& text);
};

} // namespace kelly

