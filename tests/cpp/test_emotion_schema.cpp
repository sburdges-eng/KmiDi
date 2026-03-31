#include <catch2/catch_test_macros.hpp>

#include "penta/ml/AudioEmotionRunner.h"
#include "penta/common/RTState.h"

using namespace penta::ml;

TEST_CASE("EmotionResult has valence in [-1, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.valence >= -1.0f);
    REQUIRE(e.valence <= 1.0f);
    e.valence = -1.0f;
    REQUIRE(e.valence == -1.0f);
    e.valence = 1.0f;
    REQUIRE(e.valence == 1.0f);
}

TEST_CASE("EmotionResult has arousal in [0, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.arousal >= 0.0f);
    REQUIRE(e.arousal <= 1.0f);
}

TEST_CASE("EmotionResult has dominance in [0, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.dominance >= 0.0f);
    REQUIRE(e.dominance <= 1.0f);
}

TEST_CASE("EmotionResult has confidence in [0, 1]", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.confidence >= 0.0f);
    REQUIRE(e.confidence <= 1.0f);
}

TEST_CASE("RTState emotion fields match schema", "[EmotionSchema][parity]") {
    penta::RTState state;
    float v = state.valence.load();
    float a = state.arousal.load();
    float d = state.dominance.load();
    float c = state.emotionConfidence.load();
    REQUIRE(v >= -1.0f);
    REQUIRE(v <= 1.0f);
    REQUIRE(a >= 0.0f);
    REQUIRE(a <= 1.0f);
    REQUIRE(d >= 0.0f);
    REQUIRE(d <= 1.0f);
    REQUIRE(c >= 0.0f);
    REQUIRE(c <= 1.0f);
}

TEST_CASE("EmotionResult defaults match schema defaults", "[EmotionSchema][parity]") {
    EmotionResult e;
    REQUIRE(e.valence == 0.0f);
    REQUIRE(e.arousal == 0.5f);
    REQUIRE(e.dominance == 0.5f);
    REQUIRE(e.confidence == 0.0f);
}
