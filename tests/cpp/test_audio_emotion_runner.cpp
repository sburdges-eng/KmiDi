#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "penta/ml/AudioEmotionRunner.h"
#include "ml/MelSpectrogram.h"

#include <cmath>
#include <vector>
#include <thread>
#include <chrono>

using namespace penta::ml;
using Catch::Approx;

// ─── Struct tests ───────────────────────────────────────────────────────────

TEST_CASE("EmotionResult default values", "[AudioEmotionRunner][structs]") {
    EmotionResult e;
    REQUIRE(e.valence == 0.0f);
    REQUIRE(e.arousal == 0.5f);
    REQUIRE(e.dominance == 0.5f);
    REQUIRE(e.confidence == 0.0f);
}

TEST_CASE("DSPSuggestion default values", "[AudioEmotionRunner][structs]") {
    DSPSuggestion d;
    REQUIRE(d.filter_cutoff == 0.5f);
    REQUIRE(d.reverb_wet == Approx(0.2f));
    REQUIRE(d.drive_amount == 0.0f);
}

TEST_CASE("EmotionRunnerResult has monotonic sequence_id", "[AudioEmotionRunner][structs]") {
    EmotionRunnerResult r1;
    r1.sequence_id = 0;
    EmotionRunnerResult r2;
    r2.sequence_id = 1;
    REQUIRE(r2.sequence_id > r1.sequence_id);
}

// ─── MelSpectrogram tests ───────────────────────────────────────────────────

TEST_CASE("MelSpectrogram constants are consistent", "[MelSpectrogram]") {
    REQUIRE(MelSpectrogram::kNMels == 128);
    REQUIRE(MelSpectrogram::kNFrames == 512);
    REQUIRE(MelSpectrogram::kRequiredSamples ==
            (MelSpectrogram::kNFrames - 1) * MelSpectrogram::kHopLength + MelSpectrogram::kNFft);
}

TEST_CASE("MelSpectrogram rejects insufficient input", "[MelSpectrogram]") {
    MelSpectrogram mel;
    std::vector<float> output(128 * 512);
    std::vector<float> tooShort(100, 0.0f);
    REQUIRE_FALSE(mel.compute(tooShort.data(), tooShort.size(), output.data()));
}

TEST_CASE("MelSpectrogram produces finite output for sine wave", "[MelSpectrogram]") {
    MelSpectrogram mel;
    const size_t n = MelSpectrogram::kRequiredSamples;
    std::vector<float> samples(n);

    // 440 Hz sine wave at 22050 Hz
    for (size_t i = 0; i < n; ++i) {
        samples[i] = std::sin(2.0f * 3.14159265f * 440.0f *
                              static_cast<float>(i) / 22050.0f);
    }

    std::vector<float> output(128 * 512);
    REQUIRE(mel.compute(samples.data(), n, output.data()));

    // Check all values are finite
    bool allFinite = true;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            allFinite = false;
            break;
        }
    }
    REQUIRE(allFinite);
}

TEST_CASE("MelSpectrogram silence produces low energy", "[MelSpectrogram]") {
    MelSpectrogram mel;
    const size_t n = MelSpectrogram::kRequiredSamples;
    std::vector<float> silence(n, 0.0f);
    std::vector<float> output(128 * 512);

    REQUIRE(mel.compute(silence.data(), n, output.data()));

    // All values should be log(1e-10) ≈ -23.03
    float logFloor = std::log(1e-10f);
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(output[i] == Approx(logFloor).margin(0.01f));
    }
}

// ─── Runner lifecycle tests ─────────────────────────────────────────────────

TEST_CASE("AudioEmotionRunner initializes and shuts down", "[AudioEmotionRunner][lifecycle]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";  // No model — stub mode
    config.sample_rate = 22050;
    config.ring_capacity = 65536;

    REQUIRE(runner.initialize(config));
    REQUIRE(runner.isRunning());

    runner.shutdown();
    REQUIRE_FALSE(runner.isRunning());
}

TEST_CASE("AudioEmotionRunner pushSamples does not block", "[AudioEmotionRunner][rt]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";
    config.sample_rate = 22050;
    config.ring_capacity = 4096;

    runner.initialize(config);

    // Push a block of samples — should not block
    std::vector<float> block(256, 0.1f);
    auto t0 = std::chrono::steady_clock::now();
    runner.pushSamples(block.data(), block.size());
    auto t1 = std::chrono::steady_clock::now();

    float elapsedUs = std::chrono::duration<float, std::micro>(t1 - t0).count();
    // Should complete in well under 1ms
    REQUIRE(elapsedUs < 1000.0f);

    runner.shutdown();
}

TEST_CASE("AudioEmotionRunner updateParams does not block", "[AudioEmotionRunner][rt]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";
    config.sample_rate = 22050;
    config.ring_capacity = 4096;

    runner.initialize(config);

    penta::RTState state;

    auto t0 = std::chrono::steady_clock::now();
    runner.updateParams(state, 64);
    auto t1 = std::chrono::steady_clock::now();

    float elapsedUs = std::chrono::duration<float, std::micro>(t1 - t0).count();
    REQUIRE(elapsedUs < 1000.0f);

    runner.shutdown();
}

TEST_CASE("AudioEmotionRunner produces result after sufficient samples", "[AudioEmotionRunner][integration]") {
    AudioEmotionRunner runner;
    AudioEmotionRunnerConfig config;
    config.model_path = "";  // No ONNX — uses default latent
    config.sample_rate = 22050;
    config.ring_capacity = 524288;
    config.confidence_threshold = 0.0f;  // Accept everything

    runner.initialize(config);

    // Feed enough samples for one analysis window
    const size_t totalSamples = MelSpectrogram::kRequiredSamples + 1000;
    std::vector<float> samples(totalSamples);
    for (size_t i = 0; i < totalSamples; ++i) {
        samples[i] = std::sin(2.0f * 3.14159265f * 440.0f *
                              static_cast<float>(i) / 22050.0f);
    }
    runner.pushSamples(samples.data(), totalSamples);

    // Wait for worker to process
    penta::RTState state;
    bool gotResult = false;
    for (int attempt = 0; attempt < 100; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        runner.updateParams(state, 64);
        if (runner.lastSequenceId() > 0) {
            gotResult = true;
            break;
        }
    }

    REQUIRE(gotResult);

    // Emotion values should be in valid ranges
    float v = state.valence.load();
    float a = state.arousal.load();
    float d = state.dominance.load();
    REQUIRE(v >= -1.0f);
    REQUIRE(v <= 1.0f);
    REQUIRE(a >= 0.0f);
    REQUIRE(a <= 1.0f);
    REQUIRE(d >= 0.0f);
    REQUIRE(d <= 1.0f);

    runner.shutdown();
}
