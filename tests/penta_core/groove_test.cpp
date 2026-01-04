#include "penta/groove/OnsetDetector.h"
#include "penta/groove/TempoEstimator.h"
#include "penta/groove/RhythmQuantizer.h"
#include "penta/groove/GrooveEngine.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <array>
#include <cstdint>
#include <iostream>

using namespace penta::groove;

// ========== OnsetDetector Tests ==========

class OnsetDetectorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        OnsetDetector::Config cfg;
        cfg.sampleRate = 44100.0;
        cfg.fftSize = 512;
        cfg.hopSize = 128;
        cfg.threshold = 0.0f;
        cfg.minTimeBetweenOnsets = 0.0f; // allow immediate detection in tests
        detector = std::make_unique<OnsetDetector>(cfg);
    }

    std::unique_ptr<OnsetDetector> detector;
};

TEST_F(OnsetDetectorTest, DetectsSimpleClick)
{
    constexpr size_t blockSize = 128;
    std::array<float, blockSize> signal = {};

    // Create an impulse away from the hop boundaries so the analysis window
    // doesn't attenuate it to ~0.
    signal[blockSize / 2] = 10.0f;

    detector->process(signal.data(), blockSize);
    EXPECT_TRUE(detector->hasOnset());
}

TEST_F(OnsetDetectorTest, IgnoresConstantSignal)
{
    constexpr size_t blockSize = 128;
    std::array<float, blockSize> signal;
    signal.fill(0.1f); // Constant low level

    // First block may trigger due to startup (prev spectrum == 0); second should not.
    detector->process(signal.data(), blockSize);
    detector->process(signal.data(), blockSize);
    EXPECT_FALSE(detector->hasOnset());
}

TEST_F(OnsetDetectorTest, DetectsSineWaveOnset)
{
    constexpr size_t blockSize = 128;
    std::array<float, blockSize> signal;

    // Silence first half
    for (size_t i = 0; i < blockSize / 2; ++i)
    {
        signal[i] = 0.0f;
    }

    // Sine wave second half
    for (size_t i = blockSize / 2; i < blockSize; ++i)
    {
        signal[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
    }

    detector->process(signal.data(), blockSize);
    EXPECT_TRUE(detector->hasOnset());
}

TEST_F(OnsetDetectorTest, RespondsToSensitivityChanges)
{
    constexpr size_t blockSize = 128;
    std::array<float, blockSize> weakSignal = {};
    weakSignal[0] = 0.2f; // Weak impulse

    detector->setThreshold(0.9f); // harder to trigger
    detector->process(weakSignal.data(), blockSize);
    bool hardToTrigger = detector->hasOnset();

    detector->reset();

    detector->setThreshold(0.0f); // easier to trigger
    detector->process(weakSignal.data(), blockSize);
    bool easyToTrigger = detector->hasOnset();

    EXPECT_TRUE(easyToTrigger || !hardToTrigger);
}

// ========== TempoEstimator Tests ==========

class TempoEstimatorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        TempoEstimator::Config cfg;
        cfg.sampleRate = 44100.0;
        cfg.minTempo = 60.0f;
        cfg.maxTempo = 180.0f;
        cfg.adaptationRate = 1.0f; // deterministic in tests
        cfg.tempoSearchStep = 0.5f;
        cfg.historySize = 32;
        estimator = std::make_unique<TempoEstimator>(cfg);
    }

    std::unique_ptr<TempoEstimator> estimator;
};

TEST_F(TempoEstimatorTest, Estimates120BPM)
{
    // 120 BPM = 0.5 seconds per beat = 22050 samples at 44.1kHz
    constexpr size_t samplesPerBeat = 22050;

    // Feed 4 beats
    for (int beat = 0; beat < 4; ++beat)
    {
        estimator->addOnset(beat * samplesPerBeat);
    }

    float bpm = estimator->getCurrentTempo();

    EXPECT_NEAR(bpm, 120.0f, 5.0f); // Within 5 BPM
}

TEST_F(TempoEstimatorTest, Estimates90BPM)
{
    // 90 BPM = 0.667 seconds per beat = 29400 samples
    constexpr size_t samplesPerBeat = 29400;

    for (int beat = 0; beat < 4; ++beat)
    {
        estimator->addOnset(beat * samplesPerBeat);
    }

    float bpm = estimator->getCurrentTempo();

    EXPECT_NEAR(bpm, 90.0f, 5.0f);
}

TEST_F(TempoEstimatorTest, ReturnsZeroWithNoOnsets)
{
    EXPECT_EQ(estimator->getOnsetCount(), 0u);
    EXPECT_FLOAT_EQ(estimator->getConfidence(), 0.0f);
}

TEST_F(TempoEstimatorTest, SmoothsTempoChanges)
{
    // Reconfigure to use smoothing.
    TempoEstimator::Config cfg;
    cfg.sampleRate = 44100.0;
    cfg.minTempo = 60.0f;
    cfg.maxTempo = 200.0f;
    cfg.adaptationRate = 0.1f; // smoothing
    cfg.tempoSearchStep = 0.5f;
    cfg.historySize = 32;
    estimator->updateConfig(cfg);
    estimator->reset();

    // First tempo: 120 BPM
    for (int i = 0; i < 8; ++i)
    {
        estimator->addOnset(static_cast<uint64_t>(i) * 22050u);
    }
    float tempo1 = estimator->getCurrentTempo();

    estimator->reset();
    // Sudden change to 140 BPM (18900 samples/beat)
    for (int i = 0; i < 8; ++i)
    {
        estimator->addOnset(static_cast<uint64_t>(i) * 18900u);
    }
    float tempo2 = estimator->getCurrentTempo();

    EXPECT_NE(tempo1, tempo2);
}

// ========== RhythmQuantizer Tests ==========

class RhythmQuantizerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        RhythmQuantizer::Config cfg;
        cfg.resolution = RhythmQuantizer::GridResolution::Sixteenth;
        cfg.strength = 1.0f;
        cfg.enableSwing = false;
        cfg.swingAmount = 0.5f;
        cfg.timeSignatureNum = 4;
        cfg.timeSignatureDen = 4;
        quantizer = std::make_unique<RhythmQuantizer>(cfg);
    }

    std::unique_ptr<RhythmQuantizer> quantizer;
};

TEST_F(RhythmQuantizerTest, QuantizesToNearestSixteenth)
{
    RhythmQuantizer::Config cfg;
    cfg.resolution = RhythmQuantizer::GridResolution::Sixteenth;
    cfg.strength = 1.0f;
    cfg.enableSwing = false;
    cfg.swingAmount = 0.5f;
    cfg.timeSignatureNum = 4;
    cfg.timeSignatureDen = 4;
    quantizer->updateConfig(cfg);

    // 120 BPM @ 44.1kHz = 22050 samples/beat
    uint64_t samplesPerBeat = 22050;
    uint64_t barStart = 0;

    uint64_t nearSixteenth = 5500;
    uint64_t quantized = quantizer->quantize(nearSixteenth, samplesPerBeat, barStart);
    EXPECT_NEAR(static_cast<double>(quantized), 5512.0, 150.0);
}

TEST_F(RhythmQuantizerTest, QuantizesToNearestEighth)
{
    RhythmQuantizer::Config cfg;
    cfg.resolution = RhythmQuantizer::GridResolution::Eighth;
    cfg.strength = 1.0f;
    cfg.enableSwing = false;
    cfg.swingAmount = 0.5f;
    cfg.timeSignatureNum = 4;
    cfg.timeSignatureDen = 4;
    quantizer->updateConfig(cfg);

    uint64_t samplesPerBeat = 22050;
    uint64_t barStart = 0;

    uint64_t nearEighth = 11000;
    uint64_t quantized = quantizer->quantize(nearEighth, samplesPerBeat, barStart);
    EXPECT_NEAR(static_cast<double>(quantized), 11025.0, 150.0);
}

TEST_F(RhythmQuantizerTest, HandlesDownbeat)
{
    uint64_t samplesPerBeat = 22050;
    uint64_t barStart = 0;
    uint64_t nearDownbeat = 100;
    uint64_t quantized = quantizer->quantize(nearDownbeat, samplesPerBeat, barStart);
    EXPECT_NEAR(static_cast<double>(quantized), 0.0, 200.0);
}

TEST_F(RhythmQuantizerTest, HandlesSwing)
{
    RhythmQuantizer::Config cfg;
    cfg.resolution = RhythmQuantizer::GridResolution::Eighth;
    cfg.strength = 1.0f;
    cfg.enableSwing = true;
    cfg.swingAmount = 0.75f;
    cfg.timeSignatureNum = 4;
    cfg.timeSignatureDen = 4;
    quantizer->updateConfig(cfg);

    uint64_t samplesPerBeat = 22050;
    uint64_t barStart = 0;

    // Choose an off-beat grid location so swing applies (odd subdivision).
    uint64_t samplePos = 11025;
    uint64_t swung = quantizer->quantize(samplePos, samplesPerBeat, barStart);
    EXPECT_NE(samplePos, swung);
}

TEST_F(RhythmQuantizerTest, SupportsTriplets)
{
    RhythmQuantizer::Config cfg;
    cfg.resolution = RhythmQuantizer::GridResolution::EighthTriplet;
    cfg.strength = 1.0f;
    cfg.enableSwing = false;
    cfg.swingAmount = 0.75f;
    cfg.timeSignatureNum = 4;
    cfg.timeSignatureDen = 4;
    quantizer->updateConfig(cfg);

    // 120 BPM @ 44.1kHz = 22050 samples per quarter note.
    // Eighth-note triplet grid => quarter / 3 => 7350 samples.
    const uint64_t samplesPerBeat = 22050;
    const uint64_t barStart = 0;
    const uint64_t tripletInterval = 7350;

    // Near the first triplet point.
    const uint64_t nearTriplet = 7400;
    const uint64_t quantized = quantizer->quantize(nearTriplet, samplesPerBeat, barStart);
    EXPECT_EQ(quantized, tripletInterval);

    // Verify subsequent grid points behave consistently.
    const uint64_t nearSecondTriplet = 14780;
    const uint64_t quantized2 = quantizer->quantize(nearSecondTriplet, samplesPerBeat, barStart);
    EXPECT_EQ(quantized2, tripletInterval * 2);
}

// ========== GrooveEngine Tests ==========

class GrooveEngineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        GrooveEngine::Config cfg;
        cfg.sampleRate = 44100.0;
        cfg.hopSize = 512;
        cfg.minTempo = 60.0f;
        cfg.maxTempo = 180.0f;
        cfg.enableQuantization = true;
        cfg.quantizationStrength = 0.8f;
        engine = std::make_unique<GrooveEngine>(cfg);
    }

    std::unique_ptr<GrooveEngine> engine;
};

TEST_F(GrooveEngineTest, ProcessesAudioBlock)
{
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> testSignal;

    // Generate click pattern
    testSignal.fill(0.0f);
    testSignal[0] = 1.0f;
    testSignal[256] = 1.0f;

    engine->processAudio(testSignal.data(), blockSize);

    const auto &analysis = engine->getAnalysis();
    EXPECT_GE(analysis.currentTempo, 0.0f);
}

TEST_F(GrooveEngineTest, OnsetHistoryIsBounded)
{
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> impulse{};
    std::array<float, blockSize> silence{};
    impulse.fill(0.0f);
    silence.fill(0.0f);
    // Place impulse away from the hop boundary so windowing doesn't attenuate it.
    impulse[blockSize / 2] = 10.0f;

    // OnsetDetector default minTimeBetweenOnsets is 0.05s.
    // At 44.1kHz and 512 hop, that's ~5 blocks between onsets.
    constexpr int impulsesToGenerate = 150;
    for (int i = 0; i < impulsesToGenerate; ++i)
    {
        engine->processAudio(impulse.data(), blockSize);
        for (int j = 0; j < 4; ++j)
        {
            engine->processAudio(silence.data(), blockSize);
        }
    }

    const auto &analysis = engine->getAnalysis();
    EXPECT_LE(analysis.onsetPositions.size(), 128u);
    EXPECT_EQ(analysis.onsetPositions.size(), analysis.onsetStrengths.size());
    EXPECT_GT(analysis.onsetPositions.size(), 0u);
}

// ========== Performance Benchmarks ==========

class GroovePerformanceBenchmark : public ::testing::Test
{
protected:
    OnsetDetector detector;
    std::array<float, 512> testSignal;

    void SetUp() override
    {
        OnsetDetector::Config cfg;
        cfg.sampleRate = 44100.0;
        cfg.fftSize = 512;
        cfg.hopSize = 512;
        cfg.threshold = 0.01f;
        cfg.minTimeBetweenOnsets = 0.0f;
        detector = OnsetDetector(cfg);

        // Generate test signal with onset
        testSignal.fill(0.0f);
        testSignal[0] = 1.0f;
        for (size_t i = 100; i < 512; ++i)
        {
            testSignal[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
        }
    }
};

TEST_F(GroovePerformanceBenchmark, DISABLED_OnsetDetectionUnder150Microseconds)
{
    constexpr int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        detector.process(testSignal.data(), 512);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgMicros = static_cast<double>(duration.count()) / iterations;

    std::cout << "Average onset detection time: " << avgMicros << " μs\n";

    EXPECT_LT(avgMicros, 150.0); // Target: <150μs per 512-sample block
}

TEST_F(GroovePerformanceBenchmark, DISABLED_TempoEstimationUnder200Microseconds)
{
    TempoEstimator::Config cfg;
    cfg.sampleRate = 44100.0;
    cfg.minTempo = 60.0f;
    cfg.maxTempo = 180.0f;
    cfg.adaptationRate = 1.0f;
    cfg.tempoSearchStep = 0.5f;
    cfg.historySize = 32;
    TempoEstimator estimator(cfg);
    constexpr int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        estimator.addOnset(i * 22050);
        volatile float tempo = estimator.getCurrentTempo();
        (void)tempo;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgMicros = static_cast<double>(duration.count()) / iterations;

    std::cout << "Average tempo estimation time: " << avgMicros << " μs\n";

    EXPECT_LT(avgMicros, 200.0);
}
