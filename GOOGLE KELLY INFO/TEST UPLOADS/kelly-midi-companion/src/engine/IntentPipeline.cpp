#include "engine/IntentPipeline.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace kelly {

IntentPipeline::IntentPipeline() : woundProcessor_(thesaurus_) {
}

IntentResult IntentPipeline::process(const Wound& wound) {
    // Phase 1: Process wound to emotion (using WoundProcessor)
    EmotionNode emotion = woundProcessor_.processWound(wound);
    
    // Phase 2: Generate rule breaks (using RuleBreakEngine)
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateRuleBreaks(emotion);
    
    // Phase 3: Compile everything into musical parameters
    return compileMusicalParams(wound, emotion, ruleBreaks);
}

IntentResult IntentPipeline::processJourney(const SideA& current, const SideB& desired) {
    // Get emotions for both sides (using WoundProcessor)
    EmotionNode emotionA = current.emotionId 
        ? thesaurus_.findById(*current.emotionId).value_or(
            woundProcessor_.processWound({current.description, current.intensity, "sideA"}))
        : woundProcessor_.processWound({current.description, current.intensity, "sideA"});
    
    EmotionNode emotionB = desired.emotionId
        ? thesaurus_.findById(*desired.emotionId).value_or(
            woundProcessor_.processWound({desired.description, desired.intensity, "sideB"}))
        : woundProcessor_.processWound({desired.description, desired.intensity, "sideB"});
    
    // The journey emotion is a blend - starting from A, leaning toward B
    // This creates musical tension that resolves
    float blendedValence = emotionA.valence * 0.4f + emotionB.valence * 0.6f;
    float blendedArousal = emotionA.arousal * 0.4f + emotionB.arousal * 0.6f;
    float blendedIntensity = std::max(emotionA.intensity, emotionB.intensity);
    
    EmotionNode journeyEmotion = thesaurus_.findNearest(
        blendedValence, blendedArousal, blendedIntensity);
    
    // Create wound representing the journey
    Wound journeyWound{
        "Journey from " + emotionA.name + " toward " + emotionB.name,
        blendedIntensity,
        "cassette_journey"
    };
    
    // Generate rule breaks that serve the transition (using RuleBreakEngine)
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateJourneyRuleBreaks(emotionA, emotionB);
    
    return compileMusicalParams(journeyWound, journeyEmotion, ruleBreaks);
}

// processWound and generateRuleBreaks have been moved to WoundProcessor and RuleBreakEngine

IntentResult IntentPipeline::compileMusicalParams(
    const Wound& wound,
    const EmotionNode& emotion,
    const std::vector<RuleBreak>& ruleBreaks
) {
    IntentResult result;
    result.wound = wound;
    result.emotion = emotion;
    result.ruleBreaks = ruleBreaks;
    
    // Base parameters from emotion
    result.mode = thesaurus_.suggestMode(emotion);
    result.tempo = thesaurus_.suggestTempoModifier(emotion);
    result.dynamicRange = thesaurus_.suggestDynamicRange(emotion);
    
    // Default safe values
    result.allowDissonance = false;
    result.syncopationLevel = 0.3f;
    result.humanization = 0.4f;
    
    // Apply rule breaks to override parameters
    for (const auto& rb : ruleBreaks) {
        switch (rb.type) {
            case RuleBreakType::Harmony:
                result.allowDissonance = true;
                break;
                
            case RuleBreakType::Rhythm:
                result.syncopationLevel = std::max(result.syncopationLevel, rb.severity);
                result.humanization = std::max(result.humanization, rb.severity * 0.8f);
                break;
                
            case RuleBreakType::Dynamics:
                result.dynamicRange = std::max(result.dynamicRange, rb.severity);
                break;
                
            case RuleBreakType::Melody:
            case RuleBreakType::Form:
                // These affect generation algorithms, not direct parameters
                break;
        }
    }
    
    return result;
}

} // namespace kelly
