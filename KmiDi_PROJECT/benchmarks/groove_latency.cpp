#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "penta/groove/GrooveEngine.h"
#include "penta/groove/OnsetDetector.h"
#include "penta/groove/TempoEstimator.h"
#include "penta/groove/RhythmQuantizer.h"
#include "penta/common/RTTypes.h"

// A mock audio buffer for testing
static std::vector<float> createSineWave(double frequency, double sampleRate, double duration, float amplitude = 0.5f) {
    std::vector<float> buffer(static_cast<size_t>(sampleRate * duration));
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<float>(amplitude * std::sin(2.0 * M_PI * frequency * i / sampleRate));
    }
    return buffer;
}

TEST_CASE("GrooveEngine latency benchmark", "[groove][performance][benchmark]") {
    penta::groove::GrooveEngine::Config config;
    config.sampleRate = kDefaultSampleRate; // From penta/common/RTTypes.h
    config.hopSize = 512;
    penta::groove::GrooveEngine grooveEngine(config);

    // Create a dummy audio buffer
    std::vector<float> audioBuffer = createSineWave(120.0, config.sampleRate, 1.0); // 1 second of sine wave

    BENCHMARK("GrooveEngine processAudio") {
        return grooveEngine.processAudio(audioBuffer.data(), audioBuffer.size());
    };

    // Ensure the engine is actually processing
    auto analysis = grooveEngine.getAnalysis();
    INFO("Current Tempo: " << analysis.currentTempo);
    REQUIRE(analysis.currentTempo > 0.0f);
    REQUIRE(analysis.onsetPositions.size() > 0);

    // Target: <200μs @ 48kHz/512 samples. This is for a single audio block.
    // The benchmark runs the whole 1-second buffer. The result will give an overall time.
}

TEST_CASE("OnsetDetector latency benchmark", "[groove][performance][benchmark]") {
    penta::groove::OnsetDetector detector;
    std::vector<float> buffer(512); // Process one audio block
    std::iota(buffer.begin(), buffer.end(), 0.0f); // Fill with dummy data

    BENCHMARK("OnsetDetector process") {
        return detector.process(buffer.data(), buffer.size());
    };
}

TEST_CASE("TempoEstimator latency benchmark", "[groove][performance][benchmark]") {
    penta::groove::TempoEstimator estimator;
    // Add some dummy onsets
    for (int i = 0; i < 100; ++i) {
        estimator.addOnset(static_cast<uint64_t>(i * 22050)); // ~2 onsets per second
    }

    BENCHMARK("TempoEstimator estimate") {
        return estimator.estimateTempo();
    };
}

TEST_CASE("RhythmQuantizer latency benchmark", "[groove][performance][benchmark]") {
    penta::groove::RhythmQuantizer quantizer;
    // Dummy values
    uint64_t timestamp = 10000;
    uint64_t samplesPerBeat = 22050; // 120 BPM at 44.1kHz
    uint64_t barStartPosition = 0;

    BENCHMARK("RhythmQuantizer quantize") {
        return quantizer.quantize(timestamp, samplesPerBeat, barStartPosition);
    };

    BENCHMARK("RhythmQuantizer applySwing") {
        return quantizer.applySwing(timestamp, samplesPerBeat, barStartPosition);
    };
}
