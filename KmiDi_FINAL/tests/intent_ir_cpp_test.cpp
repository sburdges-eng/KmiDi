#include <gtest/gtest.h>
#include "common/IntentIRAdapter.h"
#include "common/IntentIRExtractor.h"
#include "engine/IntentPipeline.h"
#include "engines/MelodyEngine.h"
#include "engines/DrumGrooveEngine.h"
#include "engines/BassEngine.h"
#include "engines/DynamicsEngine.h"
#include "engines/PadEngine.h"
#include <cstring>

using namespace kelly;

class IntentIRTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test fixtures if needed
    }
};

TEST_F(IntentIRTest, IntentFrameDefaultConstruction) {
    IntentFrame frame;
    EXPECT_EQ(frame.meta.ir_version, INTENT_IR_VERSION);
    EXPECT_EQ(frame.emotion.valence, 0.0f);
    EXPECT_EQ(frame.music.tempo_bias, 0.0f);
}

TEST_F(IntentIRTest, IntentFrameValidation) {
    IntentFrame frame;
    frame.meta.ir_version = INTENT_IR_VERSION;

    // Valid frame should pass
    EXPECT_TRUE(intent_frame_validate(&frame));

    // Invalid version should fail
    frame.meta.ir_version = 999;
    EXPECT_FALSE(intent_frame_validate(&frame));
}

TEST_F(IntentIRTest, IntentFrameClamping) {
    IntentFrame frame;
    frame.emotion.valence = 5.0f;  // Out of range
    frame.music.tempo_bias = -10.0f;  // Out of range

    intent_frame_clamp(&frame);

    EXPECT_LE(frame.emotion.valence, 1.0f);
    EXPECT_GE(frame.emotion.valence, -1.0f);
    EXPECT_LE(frame.music.tempo_bias, 1.0f);
    EXPECT_GE(frame.music.tempo_bias, -1.0f);
}

TEST_F(IntentIRTest, IntentPipelineProducesIntentFrame) {
    IntentPipeline pipeline;
    Wound wound{"I feel lost", 0.8f, "test"};

    IntentFrame frame = pipeline.processToIntentFrame(wound, 12345);

    EXPECT_EQ(frame.meta.ir_version, INTENT_IR_VERSION);
    EXPECT_EQ(frame.meta.session_id, 12345);
    EXPECT_GE(frame.emotion.valence, -1.0f);
    EXPECT_LE(frame.emotion.valence, 1.0f);
}

TEST_F(IntentIRTest, MelodyEngineConsumesIntentFrame) {
    IntentFrame frame;
    frame.emotion.valence = 0.5f;
    frame.emotion.arousal = 0.7f;
    frame.music.melodic_activity = 0.8f;
    frame.music.contour_variance = 0.6f;
    frame.music.mode_preference = 1;  // Major

    MelodyEngine engine;
    MelodyOutput output = engine.generateFromIntentFrame(frame, "C", 4, 120);

    EXPECT_FALSE(output.notes.empty());
    EXPECT_GT(output.totalTicks, 0);
}

TEST_F(IntentIRTest, DrumGrooveEngineConsumesIntentFrame) {
    IntentFrame frame;
    frame.music.rhythmic_density = 0.7f;
    frame.music.groove_strength = 0.8f;
    frame.music.tempo_bias = 0.3f;

    DrumGrooveEngine engine;
    GrooveOutput output = engine.generateFromIntentFrame(frame, "pop", 4, 120);

    EXPECT_FALSE(output.hits.empty());
    EXPECT_GT(output.totalTicks, 0);
}

TEST_F(IntentIRTest, BassEngineConsumesIntentFrame) {
    IntentFrame frame;
    frame.music.rhythmic_density = 0.6f;
    frame.music.groove_strength = 0.5f;

    std::vector<std::string> chords = {"C", "Am", "F", "G"};

    BassEngine engine;
    BassOutput output = engine.generateFromIntentFrame(frame, chords, "C", 4, 120);

    EXPECT_FALSE(output.notes.empty());
    EXPECT_GT(output.totalTicks, 0);
}

TEST_F(IntentIRTest, DynamicsEngineConsumesIntentFrame) {
    IntentFrame frame;
    frame.music.dynamic_range = 0.7f;
    frame.music.texture_density = 0.5f;

    std::vector<MidiNote> notes;
    notes.push_back({60, 0, 480, 80});  // C4, 1 beat

    DynamicsEngine engine;
    DynamicsOutput output = engine.applyFromIntentFrame(frame, notes, 0.6f);

    EXPECT_FALSE(output.notes.empty());
    EXPECT_GE(output.notes[0].velocity, 0);
    EXPECT_LE(output.notes[0].velocity, 127);
}

TEST_F(IntentIRTest, PadEngineConsumesIntentFrame) {
    IntentFrame frame;
    frame.music.texture_density = 0.8f;
    frame.music.harmonic_tension = 0.3f;
    frame.music.dynamic_range = 0.5f;

    std::vector<std::string> chords = {"C", "Am", "F", "G"};

    PadEngine engine;
    PadOutput output = engine.generateFromIntentFrame(frame, chords, "C", 4, 120);

    EXPECT_FALSE(output.notes.empty());
    EXPECT_GT(output.totalTicks, 0);
}

TEST_F(IntentIRTest, IntentIRExtractorHelpers) {
    IntentFrame frame;
    frame.music.melodic_activity = 0.7f;
    frame.music.contour_variance = 0.6f;
    frame.music.rhythmic_density = 0.8f;
    frame.music.groove_strength = 0.5f;

    MelodyParams melody = MelodyParams::fromIntentFrame(frame);
    EXPECT_FLOAT_EQ(melody.melodic_activity, 0.7f);
    EXPECT_FLOAT_EQ(melody.contour_variance, 0.6f);

    RhythmParams rhythm = RhythmParams::fromIntentFrame(frame);
    EXPECT_FLOAT_EQ(rhythm.rhythmic_density, 0.8f);
    EXPECT_FLOAT_EQ(rhythm.groove_strength, 0.5f);
}

TEST_F(IntentIRTest, IntentIRAdapterRoundTrip) {
    IntentFrame frame;
    frame.emotion.valence = 0.5f;
    frame.emotion.arousal = 0.7f;
    frame.music.tempo_bias = 0.3f;
    frame.music.mode_preference = 1;

    // Convert to IntentResult
    IntentResult result = convertIntentIRToIntentResult(frame);
    EXPECT_GT(result.tempoBpm, 0);
    EXPECT_FALSE(result.mode.empty());

    // Convert back to IntentFrame
    IntentFrame frame2 = convertIntentResultToIntentIR(result);
    EXPECT_EQ(frame2.meta.ir_version, INTENT_IR_VERSION);
    // Note: Round-trip may not be exact due to lossy conversion
}

TEST_F(IntentIRTest, VersionNegotiation) {
    EXPECT_TRUE(isIntentIRVersionSupported(1));
    EXPECT_FALSE(isIntentIRVersionSupported(0));
    EXPECT_FALSE(isIntentIRVersionSupported(2));
    EXPECT_FALSE(isIntentIRVersionSupported(999));
}

TEST_F(IntentIRTest, JSONSerialization) {
    IntentFrame frame;
    frame.emotion.valence = 0.5f;
    frame.music.tempo_bias = 0.3f;
    frame.meta.intent_id = 12345;

    // Serialize to JSON
    char* json = intent_frame_to_json(&frame);
    ASSERT_NE(json, nullptr);

    // Deserialize from JSON
    IntentFrame frame2;
    bool success = intent_frame_from_json(json, &frame2);
    EXPECT_TRUE(success);
    EXPECT_FLOAT_EQ(frame2.emotion.valence, 0.5f);
    EXPECT_FLOAT_EQ(frame2.music.tempo_bias, 0.3f);
    EXPECT_EQ(frame2.meta.intent_id, 12345);

    free(json);
}
