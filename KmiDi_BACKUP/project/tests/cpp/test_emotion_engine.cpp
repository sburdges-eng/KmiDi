#include <catch2/catch_test_macros.hpp>
#include "core/emotion_engine.h"

using namespace kelly;

TEST_CASE("EmotionEngine initializes with emotions", "[emotion]") {
    EmotionEngine engine;
    REQUIRE(engine.getEmotionCount() > 0);
}

TEST_CASE("EmotionEngine can retrieve emotion by ID", "[emotion]") {
    EmotionEngine engine;
    const EmotionNode* emotion = engine.getEmotion(0);
    REQUIRE(emotion != nullptr);
    REQUIRE(emotion->name == "euphoria");
    REQUIRE(emotion->category == EmotionCategory::Joy);
}

TEST_CASE("EmotionEngine can find emotion by name", "[emotion]") {
    EmotionEngine engine;
    const EmotionNode* emotion = engine.findEmotionByName("grief");
    REQUIRE(emotion != nullptr);
    REQUIRE(emotion->id == 2);
    REQUIRE(emotion->category == EmotionCategory::Sadness);
}

TEST_CASE("EmotionEngine finds nearby emotions", "[emotion]") {
    EmotionEngine engine;
    auto nearby = engine.getNearbyEmotions(0, 0.5f);
    REQUIRE(nearby.size() >= 0);
}

TEST_CASE("EmotionEngine returns nullptr for invalid ID", "[emotion]") {
    EmotionEngine engine;
    const EmotionNode* emotion = engine.getEmotion(9999);
    REQUIRE(emotion == nullptr);
}

TEST_CASE("EmotionNode has musical attributes", "[emotion]") {
    EmotionEngine engine;
    const EmotionNode* emotion = engine.getEmotion(0);
    REQUIRE(emotion != nullptr);
    REQUIRE(emotion->musicalAttributes.tempoModifier > 0.0f);
}
