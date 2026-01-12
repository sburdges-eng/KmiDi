#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "penta/harmony/HarmonyEngine.h"
#include "penta/harmony/ChordAnalyzer.h"
#include "penta/common/RTTypes.h"

// A mock audio buffer for testing
static std::vector<float> createSineWave(double frequency, double sampleRate, double duration, float amplitude = 0.5f) {
    std::vector<float> buffer(static_cast<size_t>(sampleRate * duration));
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<float>(amplitude * std::sin(2.0 * M_PI * frequency * i / sampleRate));
    }
    return buffer;
}

TEST_CASE("HarmonyEngine latency benchmark", "[harmony][performance][benchmark]") {
    penta::harmony::HarmonyEngine::Config config;
    config.sampleRate = kDefaultSampleRate; // From penta/common/RTTypes.h
    config.hopSize = 512;
    penta::harmony::HarmonyEngine harmonyEngine(config);

    // Create a dummy audio buffer
    std::vector<float> audioBuffer = createSineWave(440.0, config.sampleRate, 1.0); // 1 second of 440Hz sine wave

    BENCHMARK("HarmonyEngine processAudio") {
        return harmonyEngine.processAudio(audioBuffer.data(), audioBuffer.size());
    };

    // Ensure the engine is actually processing and not just returning defaults
    auto analysis = harmonyEngine.getAnalysis();
    INFO("Dominant Chord: " << analysis.dominantChord);
    REQUIRE(analysis.numDetectedChords > 0);

    // Target: <100μs @ 48kHz/512 samples. Check against typical block size.
    // JUCE audio block sizes are often 512, 1024, 2048 samples.
    // A single benchmark run processes the whole 1-second buffer, so we need to adjust.
    // Estimated latency per block:
    // Total benchmark time / (total samples / block size) * 1000 microseconds/ms
    // We are benchmarking the entire buffer, so compare it to the total processing time expected

    // This benchmark will run multiple times by Catch2. The important part is the raw time.
}

TEST_CASE("ChordAnalyzer latency benchmark", "[harmony][performance][benchmark]") {
    penta::harmony::ChordAnalyzer analyzer;
    std::vector<float> buffer(penta::harmony::ChordAnalyzer::kFFTSize); // Process one FFT block
    std::iota(buffer.begin(), buffer.end(), 0.0f); // Fill with dummy data

    BENCHMARK("ChordAnalyzer analyze (scalar)") {
        return analyzer.analyze(buffer.data());
    };

    // The SIMD version is automatically used if AVX2 is available, so this benchmark will reflect that.
    // No specific assertion here, just to get timing.
}
