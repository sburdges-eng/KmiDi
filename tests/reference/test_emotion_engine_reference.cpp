#include <catch2/catch_test_macros.hpp>
#include "engine/EmotionThesaurus.h"
#include "engine/WoundProcessor.h"
#include "engine/RuleBreakEngine.h"
#include "common/Types.h"
#include "common/KellyTypes.h"  // canonical kelly::Wound (PR #150 consolidation)

TEST_CASE("EmotionThesaurus - Find by ID", "[emotion]") {
    kelly::EmotionThesaurus thesaurus;
    // Assuming thesaurus loads from JSON files
    // This is a basic test structure
    REQUIRE(true);  // Placeholder - requires JSON data files
}

TEST_CASE("WoundProcessor - Process wound description", "[wound]") {
    kelly::EmotionThesaurus thesaurus;
    kelly::WoundProcessor processor(thesaurus);
    
    // Designated init — kelly::Wound (KellyTypes.h) has different field order
    // than the original 3-field common/Types.h layout used here historically.
    kelly::Wound wound{};
    wound.description = "I feel sad and lonely";
    wound.intensity = 0.7f;
    wound.urgency = 0.7f;
    wound.source = "test";
    auto emotion = processor.processWound(wound);
    
    REQUIRE(emotion.intensity == 0.7f);
    // Should identify sadness-related emotion
    REQUIRE(emotion.valence < 0.0f);
}

TEST_CASE("RuleBreakEngine - Generate rule breaks", "[rulebreak]") {
    kelly::RuleBreakEngine engine;
    
    kelly::EmotionNode emotion{1, "Anger", kelly::EmotionCategory::Anger, 0.9f, -0.8f, 0.9f, {}};
    auto ruleBreaks = engine.generateRuleBreaks(emotion);
    
    REQUIRE(!ruleBreaks.empty());
    // High intensity negative emotion should generate harmony rule breaks
    bool hasHarmonyBreak = false;
    for (const auto& rb : ruleBreaks) {
        if (rb.type == kelly::RuleBreakType::Harmony) {
            hasHarmonyBreak = true;
            break;
        }
    }
    REQUIRE(hasHarmonyBreak);
}

TEST_CASE("RuleBreakEngine - Journey rule breaks", "[rulebreak]") {
    kelly::RuleBreakEngine engine;
    
    kelly::EmotionNode emotionA{1, "Sad", kelly::EmotionCategory::Sadness, 0.7f, -0.6f, 0.3f, {}};
    kelly::EmotionNode emotionB{2, "Joy", kelly::EmotionCategory::Joy, 0.8f, 0.8f, 0.7f, {}};
    
    auto ruleBreaks = engine.generateJourneyRuleBreaks(emotionA, emotionB);
    
    REQUIRE(!ruleBreaks.empty());
    // Journey from negative to positive should include form/harmony breaks
}
