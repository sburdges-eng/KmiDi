/**
 * Tests for Penta-Core Diagnostics Engine
 *
 * Tests DiagnosticsEngine, PerformanceMonitor, and AudioAnalyzer
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "penta/diagnostics/DiagnosticsEngine.h"
#include "penta/diagnostics/PerformanceMonitor.h"
#include "penta/diagnostics/AudioAnalyzer.h"

using namespace penta::diagnostics;

// ========== PerformanceMonitor Tests ==========

class PerformanceMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        monitor = std::make_unique<PerformanceMonitor>();
    }

    std::unique_ptr<PerformanceMonitor> monitor;
};

TEST_F(PerformanceMonitorTest, InitializesToZero) {
    EXPECT_EQ(monitor->getXrunCount(), 0);
    EXPECT_EQ(monitor->getPeakLatencyUs(), 0.0f);
}

TEST_F(PerformanceMonitorTest, MeasuresLatency) {
    monitor->beginMeasurement();

    // Simulate some work (1ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    monitor->endMeasurement();

    float latency = monitor->getAverageLatencyUs();

    // Should be at least 500 microseconds (may vary due to OS scheduling)
    EXPECT_GT(latency, 500.0f);
}

TEST_F(PerformanceMonitorTest, TracksPeakLatency) {
    // Do multiple measurements
    for (int i = 0; i < 5; ++i) {
        monitor->beginMeasurement();
        std::this_thread::sleep_for(std::chrono::microseconds(100 * (i + 1)));
        monitor->endMeasurement();
    }

    float peak = monitor->getPeakLatencyUs();
    float avg = monitor->getAverageLatencyUs();

    // Peak should be >= average
    EXPECT_GE(peak, avg);
}

TEST_F(PerformanceMonitorTest, RecordsXruns) {
    EXPECT_EQ(monitor->getXrunCount(), 0);

    monitor->recordXrun();
    EXPECT_EQ(monitor->getXrunCount(), 1);

    monitor->recordXrun();
    monitor->recordXrun();
    EXPECT_EQ(monitor->getXrunCount(), 3);
}

TEST_F(PerformanceMonitorTest, CalculatesCpuUsage) {
    // Simulate processing a buffer
    monitor->beginMeasurement();
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    monitor->endMeasurement();

    // Calculate CPU usage for 256 sample buffer at 44100 Hz
    // 256 samples / 44100 Hz = 5.8ms available time
    float cpuUsage = monitor->getCpuUsagePercent(256, 44100.0);

    // Should be a reasonable percentage
    EXPECT_GE(cpuUsage, 0.0f);
    EXPECT_LE(cpuUsage, 100.0f);
}

TEST_F(PerformanceMonitorTest, ResetClearsStats) {
    monitor->recordXrun();
    monitor->recordXrun();

    monitor->beginMeasurement();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    monitor->endMeasurement();

    monitor->reset();

    EXPECT_EQ(monitor->getXrunCount(), 0);
    EXPECT_EQ(monitor->getPeakLatencyUs(), 0.0f);
}

TEST_F(PerformanceMonitorTest, MultipleMeasurementsAccumulate) {
    constexpr int numMeasurements = 100;

    for (int i = 0; i < numMeasurements; ++i) {
        monitor->beginMeasurement();
        // Minimal work
        volatile int x = 0;
        for (int j = 0; j < 100; ++j) { x += j; }
        monitor->endMeasurement();
    }

    // Average should be calculable
    float avg = monitor->getAverageLatencyUs();
    EXPECT_GT(avg, 0.0f);
}

// ========== AudioAnalyzer Tests ==========

class AudioAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        analyzer = std::make_unique<AudioAnalyzer>();
    }

    std::unique_ptr<AudioAnalyzer> analyzer;

    // Helper to create a sine wave buffer
    std::vector<float> createSineWave(size_t frames, float amplitude = 0.5f) {
        std::vector<float> buffer(frames);
        for (size_t i = 0; i < frames; ++i) {
            buffer[i] = amplitude * std::sin(2.0f * M_PI * i / frames);
        }
        return buffer;
    }

    // Helper to create silence
    std::vector<float> createSilence(size_t frames) {
        return std::vector<float>(frames, 0.0f);
    }

    // Helper to create clipping signal
    std::vector<float> createClipping(size_t frames) {
        std::vector<float> buffer(frames);
        for (size_t i = 0; i < frames; ++i) {
            buffer[i] = 1.5f * std::sin(2.0f * M_PI * i / frames);  // Over 1.0
        }
        return buffer;
    }
};

TEST_F(AudioAnalyzerTest, InitializesToZero) {
    EXPECT_EQ(analyzer->getRmsLevel(), 0.0f);
    EXPECT_EQ(analyzer->getPeakLevel(), 0.0f);
    EXPECT_FALSE(analyzer->isClipping());
}

TEST_F(AudioAnalyzerTest, AnalyzesSilence) {
    auto buffer = createSilence(512);

    analyzer->analyze(buffer.data(), buffer.size(), 1);

    EXPECT_NEAR(analyzer->getRmsLevel(), 0.0f, 0.01f);
    EXPECT_NEAR(analyzer->getPeakLevel(), 0.0f, 0.01f);
    EXPECT_FALSE(analyzer->isClipping());
}

TEST_F(AudioAnalyzerTest, AnalyzesSineWave) {
    auto buffer = createSineWave(512, 0.5f);

    analyzer->analyze(buffer.data(), buffer.size(), 1);

    // RMS of sine wave with amplitude A is A/sqrt(2) ≈ 0.707*A
    float expectedRms = 0.5f / std::sqrt(2.0f);
    EXPECT_NEAR(analyzer->getRmsLevel(), expectedRms, 0.1f);

    // Peak should be close to amplitude
    EXPECT_NEAR(analyzer->getPeakLevel(), 0.5f, 0.05f);

    EXPECT_FALSE(analyzer->isClipping());
}

TEST_F(AudioAnalyzerTest, DetectsClipping) {
    auto buffer = createClipping(512);

    analyzer->analyze(buffer.data(), buffer.size(), 1);

    EXPECT_TRUE(analyzer->isClipping());
    EXPECT_GT(analyzer->getPeakLevel(), 1.0f);
}

TEST_F(AudioAnalyzerTest, CalculatesDynamicRange) {
    // Analyze loud signal
    auto loudBuffer = createSineWave(512, 0.9f);
    analyzer->analyze(loudBuffer.data(), loudBuffer.size(), 1);

    // Then quiet signal
    auto quietBuffer = createSineWave(512, 0.1f);
    analyzer->analyze(quietBuffer.data(), quietBuffer.size(), 1);

    float dynamicRange = analyzer->getDynamicRange();

    // Should report some dynamic range
    EXPECT_GT(dynamicRange, 0.0f);
}

TEST_F(AudioAnalyzerTest, HandlesStereoInput) {
    // Interleaved stereo buffer
    std::vector<float> buffer(1024);  // 512 frames x 2 channels
    for (size_t i = 0; i < 512; ++i) {
        float sample = 0.5f * std::sin(2.0f * M_PI * i / 512);
        buffer[i * 2] = sample;      // Left
        buffer[i * 2 + 1] = sample;  // Right
    }

    analyzer->analyze(buffer.data(), 512, 2);

    // Should process both channels
    EXPECT_GT(analyzer->getRmsLevel(), 0.0f);
}

TEST_F(AudioAnalyzerTest, ResetClearsState) {
    auto buffer = createSineWave(512, 0.8f);
    analyzer->analyze(buffer.data(), buffer.size(), 1);

    EXPECT_GT(analyzer->getRmsLevel(), 0.0f);

    analyzer->reset();

    EXPECT_EQ(analyzer->getRmsLevel(), 0.0f);
    EXPECT_EQ(analyzer->getPeakLevel(), 0.0f);
}

TEST_F(AudioAnalyzerTest, ConfigurableClippingThreshold) {
    // Set lower threshold
    analyzer->setClippingThreshold(0.8f);

    auto buffer = createSineWave(512, 0.85f);
    analyzer->analyze(buffer.data(), buffer.size(), 1);

    // Should detect clipping at lower level
    EXPECT_TRUE(analyzer->isClipping());
}

TEST_F(AudioAnalyzerTest, PeakDecay) {
    // Analyze loud signal
    auto loudBuffer = createSineWave(512, 0.9f);
    analyzer->analyze(loudBuffer.data(), loudBuffer.size(), 1);
    float initialPeak = analyzer->getPeakLevel();

    // Set decay rate
    analyzer->setDecayRate(0.5f);

    // Analyze quiet signal
    auto quietBuffer = createSineWave(512, 0.1f);
    for (int i = 0; i < 10; ++i) {
        analyzer->analyze(quietBuffer.data(), quietBuffer.size(), 1);
    }

    // Peak should have decayed
    float finalPeak = analyzer->getPeakLevel();
    EXPECT_LT(finalPeak, initialPeak);
}

// ========== DiagnosticsEngine Tests ==========

class DiagnosticsEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        DiagnosticsEngine::Config config;
        config.enablePerformanceMonitoring = true;
        config.enableAudioAnalysis = true;
        config.updateIntervalMs = 10;

        engine = std::make_unique<DiagnosticsEngine>(config);
    }

    std::unique_ptr<DiagnosticsEngine> engine;
};

TEST_F(DiagnosticsEngineTest, CreatesWithDefaultConfig) {
    DiagnosticsEngine defaultEngine;

    // Should not throw
    auto stats = defaultEngine.getStats();
    EXPECT_EQ(stats.xrunCount, 0);
}

TEST_F(DiagnosticsEngineTest, MeasuresPerformance) {
    engine->beginMeasurement();

    // Simulate work
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    engine->endMeasurement();

    auto stats = engine->getStats();
    EXPECT_GT(stats.averageLatencyMs, 0.0f);
}

TEST_F(DiagnosticsEngineTest, AnalyzesAudio) {
    // Create test audio
    std::vector<float> buffer(512);
    for (size_t i = 0; i < 512; ++i) {
        buffer[i] = 0.5f * std::sin(2.0f * M_PI * i / 512);
    }

    engine->analyzeAudio(buffer.data(), buffer.size(), 1);

    auto stats = engine->getStats();
    EXPECT_GT(stats.rmsLevel, 0.0f);
    EXPECT_GT(stats.peakLevel, 0.0f);
}

TEST_F(DiagnosticsEngineTest, DetectsClipping) {
    // Create clipping signal
    std::vector<float> buffer(512);
    for (size_t i = 0; i < 512; ++i) {
        buffer[i] = 1.5f * std::sin(2.0f * M_PI * i / 512);
    }

    engine->analyzeAudio(buffer.data(), buffer.size(), 1);

    auto stats = engine->getStats();
    EXPECT_TRUE(stats.clipping);
}

TEST_F(DiagnosticsEngineTest, GeneratesPerformanceReport) {
    engine->beginMeasurement();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    engine->endMeasurement();

    std::string report = engine->getPerformanceReport();

    // Report should contain some content
    EXPECT_FALSE(report.empty());
    // Should contain latency information
    EXPECT_TRUE(report.find("latency") != std::string::npos ||
                report.find("Latency") != std::string::npos ||
                report.find("ms") != std::string::npos);
}

TEST_F(DiagnosticsEngineTest, GeneratesAudioReport) {
    std::vector<float> buffer(512, 0.5f);
    engine->analyzeAudio(buffer.data(), buffer.size(), 1);

    std::string report = engine->getAudioReport();

    EXPECT_FALSE(report.empty());
}

TEST_F(DiagnosticsEngineTest, ResetClearsAllStats) {
    // Generate some stats
    engine->beginMeasurement();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    engine->endMeasurement();

    std::vector<float> buffer(512, 0.5f);
    engine->analyzeAudio(buffer.data(), buffer.size(), 1);

    // Verify stats exist
    auto statsBefore = engine->getStats();
    EXPECT_GT(statsBefore.averageLatencyMs, 0.0f);

    // Reset
    engine->reset();

    auto statsAfter = engine->getStats();
    EXPECT_EQ(statsAfter.xrunCount, 0);
}

TEST_F(DiagnosticsEngineTest, UpdatesConfig) {
    DiagnosticsEngine::Config newConfig;
    newConfig.enablePerformanceMonitoring = false;
    newConfig.enableAudioAnalysis = true;
    newConfig.updateIntervalMs = 50;

    engine->updateConfig(newConfig);

    // Engine should continue to work
    std::vector<float> buffer(512, 0.3f);
    engine->analyzeAudio(buffer.data(), buffer.size(), 1);

    auto stats = engine->getStats();
    EXPECT_GT(stats.rmsLevel, 0.0f);
}

TEST_F(DiagnosticsEngineTest, MoveConstruction) {
    engine->beginMeasurement();
    engine->endMeasurement();

    DiagnosticsEngine movedEngine = std::move(*engine);

    // Moved engine should work
    auto stats = movedEngine.getStats();
    EXPECT_GE(stats.averageLatencyMs, 0.0f);
}

// ========== RT-Safety Tests ==========

class RTSafetyTest : public ::testing::Test {
protected:
    PerformanceMonitor monitor;
    AudioAnalyzer analyzer;
};

TEST_F(RTSafetyTest, PerformanceMonitorNoExceptions) {
    // These should be noexcept
    EXPECT_NO_THROW({
        monitor.beginMeasurement();
        monitor.endMeasurement();
        monitor.recordXrun();
    });
}

TEST_F(RTSafetyTest, AudioAnalyzerNoExceptions) {
    std::vector<float> buffer(512, 0.5f);

    EXPECT_NO_THROW({
        analyzer.analyze(buffer.data(), buffer.size(), 1);
        analyzer.getRmsLevel();
        analyzer.getPeakLevel();
        analyzer.isClipping();
        analyzer.getDynamicRange();
    });
}

TEST_F(RTSafetyTest, DiagnosticsEngineNoExceptions) {
    DiagnosticsEngine engine;
    std::vector<float> buffer(512, 0.3f);

    EXPECT_NO_THROW({
        engine.beginMeasurement();
        engine.endMeasurement();
        engine.analyzeAudio(buffer.data(), buffer.size(), 1);
    });
}

// ========== Edge Cases ==========

class EdgeCaseTest : public ::testing::Test {
protected:
    AudioAnalyzer analyzer;
};

TEST_F(EdgeCaseTest, EmptyBuffer) {
    // Analyze empty buffer - should not crash
    float* nullBuffer = nullptr;
    analyzer.analyze(nullBuffer, 0, 1);

    EXPECT_EQ(analyzer.getRmsLevel(), 0.0f);
}

TEST_F(EdgeCaseTest, SingleSample) {
    float sample = 0.5f;
    analyzer.analyze(&sample, 1, 1);

    EXPECT_NEAR(analyzer.getRmsLevel(), 0.5f, 0.01f);
}

TEST_F(EdgeCaseTest, VeryLargeBuffer) {
    std::vector<float> buffer(44100 * 10, 0.3f);  // 10 seconds at 44.1kHz

    analyzer.analyze(buffer.data(), buffer.size(), 1);

    EXPECT_GT(analyzer.getRmsLevel(), 0.0f);
}

TEST_F(EdgeCaseTest, NegativeValues) {
    std::vector<float> buffer(512, -0.5f);

    analyzer.analyze(buffer.data(), buffer.size(), 1);

    // RMS should still be positive
    EXPECT_GT(analyzer.getRmsLevel(), 0.0f);
}

TEST_F(EdgeCaseTest, ManyChannels) {
    // 8-channel surround
    std::vector<float> buffer(512 * 8, 0.3f);

    analyzer.analyze(buffer.data(), 512, 8);

    EXPECT_GT(analyzer.getRmsLevel(), 0.0f);
}

// ========== Performance Benchmarks ==========

class DiagnosticsBenchmark : public ::testing::Test {
protected:
    AudioAnalyzer analyzer;
    PerformanceMonitor monitor;

    std::vector<float> createBuffer(size_t frames) {
        std::vector<float> buffer(frames);
        for (size_t i = 0; i < frames; ++i) {
            buffer[i] = 0.5f * std::sin(2.0f * M_PI * i / frames);
        }
        return buffer;
    }
};

TEST_F(DiagnosticsBenchmark, AudioAnalysisUnder100Microseconds) {
    auto buffer = createBuffer(512);
    constexpr int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        analyzer.analyze(buffer.data(), buffer.size(), 1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgMicros = static_cast<double>(duration.count()) / iterations;

    std::cout << "Average audio analysis time: " << avgMicros << " μs\n";

    EXPECT_LT(avgMicros, 100.0);  // Target: <100μs per analysis
}

TEST_F(DiagnosticsBenchmark, MeasurementOverheadMinimal) {
    constexpr int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        monitor.beginMeasurement();
        monitor.endMeasurement();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double avgNanos = static_cast<double>(duration.count()) / iterations;

    std::cout << "Average measurement overhead: " << avgNanos << " ns\n";

    EXPECT_LT(avgNanos, 1000.0);  // Target: <1μs overhead
}

TEST_F(DiagnosticsBenchmark, StereoAnalysisPerformance) {
    // Stereo buffer (common case)
    std::vector<float> buffer(1024);  // 512 frames x 2 channels
    for (size_t i = 0; i < 512; ++i) {
        float sample = 0.5f * std::sin(2.0f * M_PI * i / 512);
        buffer[i * 2] = sample;
        buffer[i * 2 + 1] = sample;
    }

    constexpr int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        analyzer.analyze(buffer.data(), 512, 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgMicros = static_cast<double>(duration.count()) / iterations;

    std::cout << "Average stereo analysis time: " << avgMicros << " μs\n";

    EXPECT_LT(avgMicros, 150.0);  // Target: <150μs for stereo
}
