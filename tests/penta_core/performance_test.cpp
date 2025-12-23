#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>

#include "penta/common/SIMDKernels.h"
#include "penta/common/RTTypes.h"
#include "penta/harmony/ChordAnalyzer.h"
#include "penta/groove/OnsetDetector.h"
#include "penta/groove/TempoEstimator.h"

using namespace penta;
using namespace std::chrono;

// =============================================================================
// Performance Test Fixtures
// =============================================================================

class PerformanceTest : public ::testing::Test {
protected:
    static constexpr size_t kBufferSize = 512;
    static constexpr double kSampleRate = 48000.0;
    static constexpr size_t kIterations = 1000;

    std::vector<float> testBuffer;
    std::vector<float> window;

    void SetUp() override {
        // Generate test audio data
        testBuffer.resize(kBufferSize);
        window.resize(kBufferSize);

        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < kBufferSize; ++i) {
            testBuffer[i] = dist(rng);
            // Hann window
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (kBufferSize - 1)));
        }
    }

    // Helper to measure execution time in microseconds
    template<typename Func>
    double measureMicroseconds(Func&& func, size_t iterations = kIterations) {
        auto start = high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            func();
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start).count();
        return static_cast<double>(duration) / 1000.0 / iterations;  // microseconds per call
    }
};

// =============================================================================
// SIMD Kernel Performance Tests
// =============================================================================

TEST_F(PerformanceTest, RMSCalculation_Performance) {
    // Target: < 10μs per 512-sample block
    double avgMicroseconds = measureMicroseconds([&]() {
        volatile float result = SIMDKernels::calculateRMS(testBuffer.data(), kBufferSize);
        (void)result;
    });

    EXPECT_LT(avgMicroseconds, 10.0) << "RMS calculation took " << avgMicroseconds << "μs (target: <10μs)";
    std::cout << "RMS calculation: " << avgMicroseconds << "μs per block" << std::endl;
}

TEST_F(PerformanceTest, SumOfSquares_Performance) {
    // Target: < 5μs per 512-sample block
    double avgMicroseconds = measureMicroseconds([&]() {
        volatile float result = SIMDKernels::sumOfSquares(testBuffer.data(), kBufferSize);
        (void)result;
    });

    EXPECT_LT(avgMicroseconds, 5.0) << "Sum of squares took " << avgMicroseconds << "μs (target: <5μs)";
    std::cout << "Sum of squares: " << avgMicroseconds << "μs per block" << std::endl;
}

TEST_F(PerformanceTest, SpectralFlux_Performance) {
    std::vector<float> prevBuffer(kBufferSize);
    std::copy(testBuffer.begin(), testBuffer.end(), prevBuffer.begin());

    // Target: < 10μs per 512-sample block
    double avgMicroseconds = measureMicroseconds([&]() {
        volatile float result = SIMDKernels::spectralFlux(
            testBuffer.data(), prevBuffer.data(), kBufferSize
        );
        (void)result;
    });

    EXPECT_LT(avgMicroseconds, 10.0) << "Spectral flux took " << avgMicroseconds << "μs (target: <10μs)";
    std::cout << "Spectral flux: " << avgMicroseconds << "μs per block" << std::endl;
}

TEST_F(PerformanceTest, ApplyWindow_Performance) {
    std::vector<float> workBuffer(kBufferSize);

    // Target: < 5μs per 512-sample block
    double avgMicroseconds = measureMicroseconds([&]() {
        std::copy(testBuffer.begin(), testBuffer.end(), workBuffer.begin());
        SIMDKernels::applyWindow(workBuffer.data(), window.data(), kBufferSize);
    });

    EXPECT_LT(avgMicroseconds, 10.0) << "Apply window took " << avgMicroseconds << "μs (target: <10μs)";
    std::cout << "Apply window: " << avgMicroseconds << "μs per block" << std::endl;
}

TEST_F(PerformanceTest, DotProduct_Performance) {
    // Target: < 5μs per 512-sample block
    double avgMicroseconds = measureMicroseconds([&]() {
        volatile float result = SIMDKernels::dotProduct(
            testBuffer.data(), window.data(), kBufferSize
        );
        (void)result;
    });

    EXPECT_LT(avgMicroseconds, 5.0) << "Dot product took " << avgMicroseconds << "μs (target: <5μs)";
    std::cout << "Dot product: " << avgMicroseconds << "μs per block" << std::endl;
}

TEST_F(PerformanceTest, Autocorrelation_Performance) {
    // Test autocorrelation at various lags
    // Target: < 20μs per call for lag analysis
    double avgMicroseconds = measureMicroseconds([&]() {
        volatile float result = SIMDKernels::autocorrelationAtLag(
            testBuffer.data(), kBufferSize, kBufferSize / 4
        );
        (void)result;
    });

    EXPECT_LT(avgMicroseconds, 20.0) << "Autocorrelation took " << avgMicroseconds << "μs (target: <20μs)";
    std::cout << "Autocorrelation: " << avgMicroseconds << "μs per lag" << std::endl;
}

// =============================================================================
// Harmony Engine Performance Tests
// =============================================================================

TEST_F(PerformanceTest, HarmonyEngine_ChordAnalysis_Performance) {
    harmony::ChordAnalyzer analyzer;
    std::array<bool, 12> pitchClassSet = {true, false, false, false, true, false, false, true, false, false, false, false};

    // Target: < 100μs per analysis (as per spec)
    double avgMicroseconds = measureMicroseconds([&]() {
        volatile auto result = analyzer.analyze(pitchClassSet);
        (void)result;
    });

    EXPECT_LT(avgMicroseconds, 100.0) << "Chord analysis took " << avgMicroseconds << "μs (target: <100μs)";
    std::cout << "Chord analysis: " << avgMicroseconds << "μs per call" << std::endl;
}

TEST_F(PerformanceTest, HarmonyEngine_ChordAnalysisSIMD_Performance) {
    harmony::ChordAnalyzer analyzer;
    std::array<bool, 12> pitchClassSet = {true, false, false, false, true, false, false, true, false, false, false, false};

    // SIMD should be faster than scalar
    double avgMicroseconds = measureMicroseconds([&]() {
        volatile auto result = analyzer.analyzeSIMD(pitchClassSet);
        (void)result;
    });

    EXPECT_LT(avgMicroseconds, 50.0) << "SIMD chord analysis took " << avgMicroseconds << "μs (target: <50μs)";
    std::cout << "SIMD chord analysis: " << avgMicroseconds << "μs per call" << std::endl;
}

// =============================================================================
// Groove Engine Performance Tests
// =============================================================================

TEST_F(PerformanceTest, GrooveEngine_OnsetDetection_Performance) {
    groove::OnsetDetector::Config config;
    config.sampleRate = kSampleRate;
    config.fftSize = kBufferSize;
    config.hopSize = kBufferSize / 4;
    groove::OnsetDetector detector(config);

    // Target: < 200μs per 512-sample block (as per spec)
    double avgMicroseconds = measureMicroseconds([&]() {
        detector.process(testBuffer.data(), kBufferSize);
    });

    EXPECT_LT(avgMicroseconds, 200.0) << "Onset detection took " << avgMicroseconds << "μs (target: <200μs)";
    std::cout << "Onset detection: " << avgMicroseconds << "μs per block" << std::endl;
}

TEST_F(PerformanceTest, GrooveEngine_TempoEstimation_Performance) {
    groove::TempoEstimator::Config config;
    config.sampleRate = kSampleRate;
    groove::TempoEstimator estimator(config);

    // Pre-fill with some onset data
    for (size_t i = 0; i < 16; ++i) {
        estimator.addOnset(i * static_cast<uint64_t>(kSampleRate * 0.5));  // 120 BPM
    }

    // Target: < 50μs per onset processing
    double avgMicroseconds = measureMicroseconds([&]() {
        estimator.addOnset(estimator.getSamplesPerBeat() * 20);
    });

    EXPECT_LT(avgMicroseconds, 50.0) << "Tempo estimation took " << avgMicroseconds << "μs (target: <50μs)";
    std::cout << "Tempo estimation: " << avgMicroseconds << "μs per onset" << std::endl;
}

// =============================================================================
// Latency Verification Tests
// =============================================================================

TEST_F(PerformanceTest, LatencyVerification_HarmonyUnder100us) {
    harmony::ChordAnalyzer analyzer;
    std::array<bool, 12> pitchClassSet = {true, false, false, false, true, false, false, true, false, false, false, false};

    // Worst-case latency test: measure maximum time
    double maxMicroseconds = 0.0;
    for (size_t i = 0; i < 100; ++i) {
        auto start = high_resolution_clock::now();
        volatile auto result = analyzer.analyze(pitchClassSet);
        auto end = high_resolution_clock::now();
        (void)result;

        double elapsed = static_cast<double>(
            duration_cast<nanoseconds>(end - start).count()
        ) / 1000.0;

        maxMicroseconds = std::max(maxMicroseconds, elapsed);
    }

    EXPECT_LT(maxMicroseconds, 100.0) << "Worst-case harmony latency: " << maxMicroseconds << "μs (target: <100μs)";
    std::cout << "Worst-case harmony latency: " << maxMicroseconds << "μs" << std::endl;
}

TEST_F(PerformanceTest, LatencyVerification_GrooveUnder200us) {
    groove::OnsetDetector::Config config;
    config.sampleRate = kSampleRate;
    config.fftSize = kBufferSize;
    config.hopSize = kBufferSize / 4;
    groove::OnsetDetector detector(config);

    // Worst-case latency test
    double maxMicroseconds = 0.0;
    for (size_t i = 0; i < 100; ++i) {
        auto start = high_resolution_clock::now();
        detector.process(testBuffer.data(), kBufferSize);
        auto end = high_resolution_clock::now();

        double elapsed = static_cast<double>(
            duration_cast<nanoseconds>(end - start).count()
        ) / 1000.0;

        maxMicroseconds = std::max(maxMicroseconds, elapsed);
    }

    EXPECT_LT(maxMicroseconds, 200.0) << "Worst-case groove latency: " << maxMicroseconds << "μs (target: <200μs)";
    std::cout << "Worst-case groove latency: " << maxMicroseconds << "μs" << std::endl;
}

// =============================================================================
// Memory Allocation Tests (RT-Safety)
// =============================================================================

TEST_F(PerformanceTest, RTSafety_NoAllocationsDuringProcess) {
    harmony::ChordAnalyzer analyzer;
    groove::OnsetDetector::Config config;
    config.sampleRate = kSampleRate;
    config.fftSize = kBufferSize;
    groove::OnsetDetector detector(config);

    std::array<bool, 12> pitchClassSet = {true, false, false, false, true, false, false, true, false, false, false, false};

    // These operations should not allocate memory after initial setup
    // We can't easily test for allocations in a portable way, but we can
    // verify the operations complete without exceptions

    EXPECT_NO_THROW({
        for (size_t i = 0; i < 1000; ++i) {
            auto chord = analyzer.analyze(pitchClassSet);
            detector.process(testBuffer.data(), kBufferSize);
            (void)chord;
        }
    });
}

// =============================================================================
// SIMD Correctness Verification
// =============================================================================

TEST_F(PerformanceTest, SIMD_RMS_Correctness) {
    // Verify SIMD produces correct results by comparing with known values
    std::vector<float> testData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float rms = SIMDKernels::calculateRMS(testData.data(), testData.size());

    // RMS = sqrt((1+4+9+16+25+36+49+64)/8) = sqrt(204/8) = sqrt(25.5) ≈ 5.05
    float expected = std::sqrt((1 + 4 + 9 + 16 + 25 + 36 + 49 + 64) / 8.0f);
    EXPECT_NEAR(rms, expected, 0.001f);
}

TEST_F(PerformanceTest, SIMD_SpectralFlux_Correctness) {
    std::vector<float> current = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> previous = {0.5f, 3.0f, 2.0f, 3.5f};

    float flux = SIMDKernels::spectralFlux(current.data(), previous.data(), 4);

    // Positive differences: (1-0.5)=0.5, (3-2)=1.0, (4-3.5)=0.5
    // Sum: 0.5 + 0 + 1.0 + 0.5 = 2.0
    EXPECT_NEAR(flux, 2.0f, 0.001f);
}

TEST_F(PerformanceTest, SIMD_DotProduct_Correctness) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};

    float dot = SIMDKernels::dotProduct(a.data(), b.data(), 4);

    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_NEAR(dot, 70.0f, 0.001f);
}
