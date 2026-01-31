#include "engine/RuleBreakEngine.h"

namespace kelly {

float RuleBreakEngine::calculateSeverity(const EmotionNode& emotion, float baseSeverity) const {
    // Severity increases with intensity
    float severity = baseSeverity * (0.5f + emotion.intensity * 0.5f);
    
    // Extreme emotions get more severe rule breaks
    if (emotion.intensity > 0.8f) {
        severity = std::min(1.0f, severity * 1.3f);
    }
    
    return std::clamp(severity, 0.0f, 1.0f);
}

std::vector<RuleBreak> RuleBreakEngine::generateRuleBreaks(const EmotionNode& emotion) {
    std::vector<RuleBreak> ruleBreaks;
    
    // Negative valence -> harmonic rule breaks (dissonance, unresolved tensions)
    if (emotion.valence < -0.3f) {
        float severity = calculateSeverity(emotion, 0.6f);
        ruleBreaks.push_back({
            RuleBreakType::Harmony,
            severity,
            "Intentional dissonance",
            "Negative emotions require harmonic tension to express truth"
        });
    }
    
    // High arousal -> rhythmic rule breaks (syncopation, displaced accents)
    if (emotion.arousal > 0.7f) {
        float severity = calculateSeverity(emotion, 0.5f);
        ruleBreaks.push_back({
            RuleBreakType::Rhythm,
            severity,
            "Syncopated rhythms",
            "High arousal demands rhythmic complexity and displacement"
        });
    }
    
    // High intensity -> dynamic rule breaks (extreme contrasts)
    if (emotion.intensity > 0.7f) {
        float severity = calculateSeverity(emotion, 0.7f);
        ruleBreaks.push_back({
            RuleBreakType::Dynamics,
            severity,
            "Extreme dynamic range",
            "Intense emotions require dramatic dynamic shifts"
        });
    }
    
    // Very negative + high intensity -> melodic rule breaks (wide leaps, chromaticism)
    if (emotion.valence < -0.5f && emotion.intensity > 0.6f) {
        float severity = calculateSeverity(emotion, 0.6f);
        ruleBreaks.push_back({
            RuleBreakType::Melody,
            severity,
            "Wide intervallic leaps",
            "Extreme negative emotions break melodic conventions"
        });
    }
    
    // Very intense emotions -> form rule breaks (structural disruption)
    if (emotion.intensity > 0.85f) {
        float severity = calculateSeverity(emotion, 0.5f);
        ruleBreaks.push_back({
            RuleBreakType::Form,
            severity,
            "Structural disruption",
            "Overwhelming emotions break conventional song structure"
        });
    }
    
    return ruleBreaks;
}

std::vector<RuleBreak> RuleBreakEngine::generateJourneyRuleBreaks(
    const EmotionNode& emotionA,
    const EmotionNode& emotionB)
{
    std::vector<RuleBreak> ruleBreaks = generateRuleBreaks(emotionB);  // Start with target emotion breaks
    
    // Calculate emotional distance
    float emotionalDistance = std::abs(emotionA.valence - emotionB.valence) +
                              std::abs(emotionA.arousal - emotionB.arousal);
    
    // If emotions are far apart, add journey-specific form break
    if (emotionalDistance > 1.0f) {
        float severity = std::min(1.0f, emotionalDistance / 2.0f);
        ruleBreaks.push_back({
            RuleBreakType::Form,
            severity,
            "Dramatic structural shift",
            "The emotional journey requires breaking conventional song structure"
        });
    }
    
    // If transitioning from negative to positive (or vice versa), add harmony break
    if ((emotionA.valence < 0 && emotionB.valence > 0) ||
        (emotionA.valence > 0 && emotionB.valence < 0)) {
        ruleBreaks.push_back({
            RuleBreakType::Harmony,
            emotionalDistance * 0.5f,
            "Modal interchange for transition",
            "Crossing emotional boundaries requires harmonic exploration"
        });
    }
    
    return ruleBreaks;
}

} // namespace kelly
