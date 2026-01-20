/**
 * Intent IR v1 C++ Integration Test
 *
 * Tests the full integration: KellyBrain → IntentFrame → MidiGenerator → MIDI
 */

#include <gtest/gtest.h>
#include "engine/KellyBrain.h"
#include "engine/AdaptiveGenerator.h"
#include "midi/MidiGenerator.h"
#include "common/IntentIRAdapter.h"
#include "shared/include/kmidi/IntentIR.h"
#include "shared/include/kmidi/IntentIR_JSON.h"  // JSON serialization
#include "intent_ir_ffi.h"  // Rust FFI functions
#include <cstring>

using namespace kelly;

class IntentIRIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        brain_ = std::make_unique<KellyBrain>();
        // Try multiple possible data paths
        const char* data_paths[] = {
            "./data",
            "../data",
            "../../data",
            "${CMAKE_SOURCE_DIR}/data",  // Will be replaced at build time if needed
            nullptr
        };

        bool initialized = false;
        for (int i = 0; data_paths[i] != nullptr; ++i) {
            if (brain_->initialize(data_paths[i])) {
                initialized = true;
                break;
            }
        }

        // If initialization fails, skip tests that require it
        if (!initialized) {
            GTEST_SKIP() << "Could not initialize KellyBrain - data directory not found";
        }
    }

    std::unique_ptr<KellyBrain> brain_;
};

TEST_F(IntentIRIntegrationTest, KellyBrainFromTextToIntentFrame) {
    IntentFrame frame = brain_->fromTextToIntentFrame("I feel lost and alone");

    EXPECT_EQ(frame.meta.ir_version, INTENT_IR_VERSION);
    EXPECT_LE(frame.emotion.valence, 0.0f);  // Negative for "lost"
    EXPECT_GT(frame.meta.intent_id, 0);

    // Validate frame
    IntentFrame validated = frame;
    prepareIntentFrame(validated);
    // Use Rust FFI validation function
    int validation_result = validate_intent_frame_ffi(&validated);
    EXPECT_EQ(validation_result, 0);  // 0 means valid
}

TEST_F(IntentIRIntegrationTest, KellyBrainFromEmotionToIntentFrame) {
    IntentFrame frame = brain_->fromEmotionToIntentFrame("grief", 0.8f);

    EXPECT_EQ(frame.meta.ir_version, INTENT_IR_VERSION);
    EXPECT_LT(frame.emotion.valence, 0.0f);  // Grief is negative valence
    EXPECT_GT(frame.emotion.arousal, 0.0f);
}

TEST_F(IntentIRIntegrationTest, KellyBrainGenerateMidiFromIntentFrame) {
    IntentFrame frame = brain_->fromTextToIntentFrame("I feel joyful");
    prepareIntentFrame(frame);

    GeneratedMidi midi = brain_->generateMidiFromIntentFrame(frame, 4);

    EXPECT_GT(midi.bpm, 0.0f);
    EXPECT_GT(midi.lengthInBeats, 0.0);
    // Should have some generated content
    EXPECT_FALSE(midi.chords.empty() || midi.melody.empty() || midi.bass.empty());
}

TEST_F(IntentIRIntegrationTest, MidiGeneratorWithIntentFrame) {
    MidiGenerator generator;

    IntentFrame frame = brain_->fromTextToIntentFrame("I feel energetic");
    prepareIntentFrame(frame);

    GeneratedMidi midi = generator.generate(frame, 8, 0.5f, 0.4f, 0.0f, 0.75f);

    EXPECT_GT(midi.bpm, 0.0f);
    EXPECT_EQ(midi.bars, 8);
    EXPECT_GT(midi.lengthInBeats, 0.0);
}

TEST_F(IntentIRIntegrationTest, IntentFrameRoundTripConversion) {
    // Create IntentFrame from text
    IntentFrame frame1 = brain_->fromTextToIntentFrame("I feel peaceful");
    prepareIntentFrame(frame1);

    // Convert to IntentResult
    IntentResult result = convertIntentIRToIntentResult(frame1);
    EXPECT_GT(result.tempoBpm, 0);
    EXPECT_FALSE(result.emotion.name.empty());

    // Convert back to IntentFrame
    IntentFrame frame2 = convertIntentResultToIntentIR(result);
    EXPECT_EQ(frame2.meta.ir_version, INTENT_IR_VERSION);
    EXPECT_FLOAT_EQ(frame2.emotion.valence, frame1.emotion.valence);
    EXPECT_FLOAT_EQ(frame2.emotion.arousal, frame1.emotion.arousal);
}

TEST_F(IntentIRIntegrationTest, IntentFrameValidation) {
    IntentFrame frame = brain_->fromTextToIntentFrame("test");

    // Should be valid after preparation
    prepareIntentFrame(frame);
    int validation_result = validate_intent_frame_ffi(&frame);
    EXPECT_EQ(validation_result, 0);  // 0 means valid

    // Invalid version should fail
    frame.meta.ir_version = 999;
    validation_result = validate_intent_frame_ffi(&frame);
    EXPECT_NE(validation_result, 0);  // Non-zero means invalid
}

TEST_F(IntentIRIntegrationTest, IntentFrameClamping) {
    IntentFrame frame = brain_->fromTextToIntentFrame("test");

    // Set out-of-range values
    frame.emotion.valence = 5.0f;
    frame.music.tempo_bias = -10.0f;

    // Clamp should fix them (using Rust FFI)
    clamp_intent_frame_ffi(&frame);
    EXPECT_LE(frame.emotion.valence, 1.0f);
    EXPECT_GE(frame.emotion.valence, -1.0f);
    EXPECT_LE(frame.music.tempo_bias, 1.0f);
    EXPECT_GE(frame.music.tempo_bias, -1.0f);
}

TEST_F(IntentIRIntegrationTest, FullPipelineIntentFrame) {
    // Full pipeline: Text → IntentFrame → MIDI
    IntentFrame frame = brain_->fromTextToIntentFrame("I feel creative and inspired");
    prepareIntentFrame(frame);

    GeneratedMidi midi = brain_->generateMidiFromIntentFrame(frame, 8);

    EXPECT_GT(midi.bpm, 0.0f);
    EXPECT_GT(midi.lengthInBeats, 0.0);
    EXPECT_FALSE(midi.chords.empty());
}

TEST_F(IntentIRIntegrationTest, IntentFrameJSONSerialization) {
    IntentFrame frame = brain_->fromTextToIntentFrame("I feel happy");
    prepareIntentFrame(frame);

    // Serialize to JSON
    char* json = intent_frame_to_json(&frame);
    ASSERT_NE(json, nullptr);
    EXPECT_NE(strlen(json), 0);

    // Deserialize from JSON
    IntentFrame frame2;
    bool success = intent_frame_from_json(json, &frame2);
    EXPECT_TRUE(success);
    EXPECT_EQ(frame2.meta.ir_version, INTENT_IR_VERSION);
    EXPECT_FLOAT_EQ(frame2.emotion.valence, frame.emotion.valence);

    free(json);
}

TEST_F(IntentIRIntegrationTest, IntentFrameJourney) {
    SideA current;
    current.description = "I feel sad";
    current.intensity = 0.7f;

    SideB desired;
    desired.description = "I want to feel hopeful";
    desired.intensity = 0.8f;

    IntentFrame frame = brain_->fromJourneyToIntentFrame(current, desired);
    prepareIntentFrame(frame);

    EXPECT_EQ(frame.meta.ir_version, INTENT_IR_VERSION);
    // Journey should show transition from negative to positive
    EXPECT_GT(frame.emotion.valence, -1.0f);  // Not completely negative
}
