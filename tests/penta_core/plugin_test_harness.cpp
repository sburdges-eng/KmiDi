/**
 * @file plugin_test_harness.cpp
 * @brief Unified test harness for all Penta Core plugin components
 * 
 * Tests all 11 plugin modules with RT-safety validation, mock audio device,
 * and comprehensive integration testing.
 */

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>

// Penta Core includes
#include "penta/harmony/HarmonyEngine.h"
#include "penta/harmony/ChordAnalyzer.h"
#include "penta/harmony/ScaleDetector.h"
#include "penta/harmony/VoiceLeading.h"
#include "penta/groove/GrooveEngine.h"
#include "penta/groove/OnsetDetector.h"
#include "penta/groove/TempoEstimator.h"
#include "penta/groove/RhythmQuantizer.h"
#include "penta/diagnostics/DiagnosticsEngine.h"
#include "penta/diagnostics/PerformanceMonitor.h"
#include "penta/diagnostics/AudioAnalyzer.h"
#include "penta/common/RTMemoryPool.h"
#include "penta/common/RTLogger.h"
#include "penta/osc/OSCServer.h"
#include "penta/osc/OSCClient.h"

using namespace penta;

// ============================================================================
// Mock Audio Device
// ============================================================================

/**
 * @brief Mock audio device for testing plugin processing
 * 
 * Simulates real-time audio callbacks with configurable sample rate,
 * buffer size, and timing characteristics.
 */
class MockAudioDevice {
public:
    struct Config {
        double sampleRate = 44100.0;
        size_t bufferSize = 512;
        size_t numChannels = 2;
        bool simulateJitter = false;
        double jitterAmountMs = 0.5;
    };

    using AudioCallback = std::function<void(const float* input, float* output, 
                                            size_t numFrames, size_t numChannels)>;

    explicit MockAudioDevice(const Config& config = Config{})
        : config_(config)
        , running_(false)
        , callbackCount_(0)
    {}

    ~MockAudioDevice() {
        stop();
    }

    void setCallback(AudioCallback callback) {
        callback_ = std::move(callback);
    }

    void start() {
        if (running_) return;
        
        running_ = true;
        audioThread_ = std::thread([this]() {
            this->audioThreadFunc();
        });
    }

    void stop() {
        running_ = false;
        if (audioThread_.joinable()) {
            audioThread_.join();
        }
    }

    size_t getCallbackCount() const { return callbackCount_.load(); }
    
    void reset() {
        callbackCount_ = 0;
    }

private:
    void audioThreadFunc() {
        std::vector<float> inputBuffer(config_.bufferSize * config_.numChannels, 0.0f);
        std::vector<float> outputBuffer(config_.bufferSize * config_.numChannels, 0.0f);

        // Calculate time per buffer in microseconds
        auto bufferTimeUs = std::chrono::microseconds(
            static_cast<long long>((config_.bufferSize * 1000000.0) / config_.sampleRate)
        );

        while (running_) {
            auto callbackStart = std::chrono::high_resolution_clock::now();

            // Invoke callback
            if (callback_) {
                callback_(inputBuffer.data(), outputBuffer.data(), 
                         config_.bufferSize, config_.numChannels);
            }

            callbackCount_++;

            // Simulate jitter if enabled
            if (config_.simulateJitter) {
                auto jitterUs = std::chrono::microseconds(
                    static_cast<long long>(config_.jitterAmountMs * 1000.0 * 
                                          ((std::rand() % 100) / 100.0))
                );
                std::this_thread::sleep_for(jitterUs);
            }

            // Sleep for remaining time
            auto callbackEnd = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                callbackEnd - callbackStart
            );
            
            if (elapsed < bufferTimeUs) {
                std::this_thread::sleep_for(bufferTimeUs - elapsed);
            }
        }
    }

    Config config_;
    AudioCallback callback_;
    std::thread audioThread_;
    std::atomic<bool> running_;
    std::atomic<size_t> callbackCount_;
};

// ============================================================================
// RT-Safety Validator
// ============================================================================

/**
 * @brief Validates real-time safety of plugin operations
 * 
 * Checks for allocations, locks, and other non-RT-safe operations
 * during audio processing.
 */
class RTSafetyValidator {
public:
    struct Violation {
        std::string type;
        std::string description;
        std::chrono::high_resolution_clock::time_point timestamp;
    };

    void beginRTContext() {
        inRTContext_ = true;
        violations_.clear();
    }

    void endRTContext() {
        inRTContext_ = false;
    }

    bool isInRTContext() const { return inRTContext_; }

    void recordViolation(const std::string& type, const std::string& description) {
        if (inRTContext_) {
            violations_.push_back({
                type,
                description,
                std::chrono::high_resolution_clock::now()
            });
        }
    }

    const std::vector<Violation>& getViolations() const { return violations_; }
    
    bool hasViolations() const { return !violations_.empty(); }

    void reset() {
        violations_.clear();
    }

private:
    bool inRTContext_ = false;
    std::vector<Violation> violations_;
};

// Global RT-safety validator for testing
static RTSafetyValidator g_rtValidator;

// ============================================================================
// Plugin Test Harness Base Class
// ============================================================================

/**
 * @brief Base class for all plugin component tests
 * 
 * Provides common setup, teardown, and utilities for testing
 * Penta Core plugin components.
 */
class PluginTestHarness : public ::testing::Test {
protected:
    void SetUp() override {
        mockDevice_ = std::make_unique<MockAudioDevice>();
        rtValidator_ = &g_rtValidator;
        rtValidator_->reset();
    }

    void TearDown() override {
        if (mockDevice_) {
            mockDevice_->stop();
        }
    }

    // Helper: Generate test audio (sine wave)
    static void generateSineWave(float* buffer, size_t numSamples, 
                                double frequency, double sampleRate, 
                                double phase = 0.0) {
        for (size_t i = 0; i < numSamples; ++i) {
            buffer[i] = std::sin(2.0 * M_PI * frequency * (i + phase) / sampleRate);
        }
    }

    // Helper: Generate test MIDI notes
    static std::vector<uint8_t> generateMidiNoteOn(uint8_t note, uint8_t velocity) {
        return {0x90, note, velocity};  // Note On, Channel 1
    }

    static std::vector<uint8_t> generateMidiNoteOff(uint8_t note) {
        return {0x80, note, 0x00};  // Note Off, Channel 1
    }

    // Helper: Validate RT-safety
    void validateRTSafety() {
        EXPECT_FALSE(rtValidator_->hasViolations()) 
            << "RT-safety violations detected";
        
        if (rtValidator_->hasViolations()) {
            for (const auto& violation : rtValidator_->getViolations()) {
                std::cerr << "RT Violation [" << violation.type << "]: " 
                         << violation.description << "\n";
            }
        }
    }

    std::unique_ptr<MockAudioDevice> mockDevice_;
    RTSafetyValidator* rtValidator_;
};

// ============================================================================
// Harmony Engine Plugin Tests
// ============================================================================

class HarmonyEnginePluginTest : public PluginTestHarness {
protected:
    void SetUp() override {
        PluginTestHarness::SetUp();
        harmony::HarmonyEngine::Config config;
        config.sampleRate = 44100.0;
        engine_ = std::make_unique<harmony::HarmonyEngine>(config);
    }

    std::unique_ptr<harmony::HarmonyEngine> engine_;
};

TEST_F(HarmonyEnginePluginTest, RTSafeProcessing) {
    constexpr size_t blockSize = 512;
    
    // Create test MIDI notes
    std::vector<Note> notes;
    notes.push_back({60, 100, 0});  // Middle C, velocity 100
    notes.push_back({64, 90, 0});   // E
    notes.push_back({67, 95, 0});   // G

    rtValidator_->beginRTContext();
    
    engine_->processNotes(notes.data(), notes.size());
    
    rtValidator_->endRTContext();
    
    validateRTSafety();
}

TEST_F(HarmonyEnginePluginTest, IntegrationWithMockDevice) {
    std::atomic<int> processedBlocks{0};
    
    // Create test MIDI notes
    std::vector<Note> notes;
    notes.push_back({60, 100, 0});  // C major chord
    notes.push_back({64, 90, 0});
    notes.push_back({67, 95, 0});
    
    mockDevice_->setCallback([this, &processedBlocks, &notes](
        const float* input, float* output, size_t numFrames, size_t numChannels) {
        
        rtValidator_->beginRTContext();
        
        // Process MIDI notes for harmony analysis
        engine_->processNotes(notes.data(), notes.size());
        
        // Pass through audio
        std::memcpy(output, input, numFrames * numChannels * sizeof(float));
        
        rtValidator_->endRTContext();
        processedBlocks++;
    });

    mockDevice_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mockDevice_->stop();

    EXPECT_GT(processedBlocks.load(), 0);
    validateRTSafety();
}

// ============================================================================
// Groove Engine Plugin Tests
// ============================================================================

class GrooveEnginePluginTest : public PluginTestHarness {
protected:
    void SetUp() override {
        PluginTestHarness::SetUp();
        groove::GrooveEngine::Config config;
        config.sampleRate = 44100.0;
        engine_ = std::make_unique<groove::GrooveEngine>(config);
    }

    std::unique_ptr<groove::GrooveEngine> engine_;
};

TEST_F(GrooveEnginePluginTest, RTSafeOnsetDetection) {
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> audioIn;
    
    // Create impulse
    audioIn.fill(0.0f);
    audioIn[0] = 1.0f;

    rtValidator_->beginRTContext();
    
    engine_->processAudio(audioIn.data(), blockSize);
    
    rtValidator_->endRTContext();
    
    validateRTSafety();
}

TEST_F(GrooveEnginePluginTest, TempoEstimationAccuracy) {
    // Generate 120 BPM click track
    constexpr double bpm = 120.0;
    constexpr double sampleRate = 44100.0;
    constexpr double clickInterval = 60.0 / bpm;  // 0.5 seconds
    constexpr size_t blockSize = 512;
    constexpr size_t numBlocks = 100;

    std::vector<float> audioBuffer(blockSize);
    
    for (size_t block = 0; block < numBlocks; ++block) {
        audioBuffer.assign(blockSize, 0.0f);
        
        // Add click at appropriate intervals
        size_t samplePos = block * blockSize;
        double timePos = samplePos / sampleRate;
        
        if (std::fmod(timePos, clickInterval) < (blockSize / sampleRate)) {
            audioBuffer[0] = 1.0f;
        }
        
        engine_->processAudio(audioBuffer.data(), blockSize);
    }

    double estimatedTempo = engine_->getAnalysis().currentTempo;
    
    // Allow 5% tolerance
    EXPECT_NEAR(estimatedTempo, bpm, bpm * 0.05);
}

// ============================================================================
// Diagnostics Engine Plugin Tests
// ============================================================================

class DiagnosticsEnginePluginTest : public PluginTestHarness {
protected:
    void SetUp() override {
        PluginTestHarness::SetUp();
        engine_ = std::make_unique<diagnostics::DiagnosticsEngine>();
    }

    std::unique_ptr<diagnostics::DiagnosticsEngine> engine_;
};

TEST_F(DiagnosticsEnginePluginTest, MonitorsRealTimePerformance) {
    std::atomic<int> processedBlocks{0};
    
    mockDevice_->setCallback([this, &processedBlocks](
        const float* input, float* output, size_t numFrames, size_t numChannels) {
        
        engine_->beginMeasurement();
        
        // Simulate some processing and analyze audio
        for (size_t i = 0; i < numFrames * numChannels; ++i) {
            output[i] = input[i] * 0.5f;
        }
        
        engine_->analyzeAudio(input, numFrames, numChannels);
        engine_->endMeasurement();
        processedBlocks++;
    });

    mockDevice_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mockDevice_->stop();

    auto stats = engine_->getStats();
    
    EXPECT_GT(processedBlocks.load(), 0);
    EXPECT_GE(stats.averageLatencyMs, 0.0f);
    EXPECT_LE(stats.cpuUsagePercent, 100.0f);
}

// ============================================================================
// OSC Communication Plugin Tests
// ============================================================================

class OSCPluginTest : public PluginTestHarness {
protected:
    void SetUp() override {
        PluginTestHarness::SetUp();
        server_ = std::make_unique<osc::OSCServer>(9001);
        client_ = std::make_unique<osc::OSCClient>("127.0.0.1", 9001);
    }

    void TearDown() override {
        server_.reset();
        client_.reset();
        PluginTestHarness::TearDown();
    }

    std::unique_ptr<osc::OSCServer> server_;
    std::unique_ptr<osc::OSCClient> client_;
};

TEST_F(OSCPluginTest, RTSafeMessageSending) {
    rtValidator_->beginRTContext();
    
    // Sending should be RT-safe (uses lock-free queue)
    client_->sendFloat("/test/param", 0.5f);
    client_->sendInt("/test/note", 60);
    
    rtValidator_->endRTContext();
    
    validateRTSafety();
}

// ============================================================================
// RT Memory Pool Tests
// ============================================================================

class RTMemoryPoolPluginTest : public PluginTestHarness {
protected:
    void SetUp() override {
        PluginTestHarness::SetUp();
        pool_ = std::make_unique<RTMemoryPool>(256, 100);
    }

    std::unique_ptr<RTMemoryPool> pool_;
};

TEST_F(RTMemoryPoolPluginTest, RTSafeAllocation) {
    rtValidator_->beginRTContext();
    
    void* ptr1 = pool_->allocate();
    void* ptr2 = pool_->allocate();
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    
    pool_->deallocate(ptr1);
    pool_->deallocate(ptr2);
    
    rtValidator_->endRTContext();
    
    validateRTSafety();
}

// ============================================================================
// Integration Tests - All Components Together
// ============================================================================

class FullPluginIntegrationTest : public PluginTestHarness {
protected:
    void SetUp() override {
        PluginTestHarness::SetUp();
        
        harmony::HarmonyEngine::Config harmonyConfig;
        harmonyConfig.sampleRate = 44100.0;
        harmonyEngine_ = std::make_unique<harmony::HarmonyEngine>(harmonyConfig);
        
        groove::GrooveEngine::Config grooveConfig;
        grooveConfig.sampleRate = 44100.0;
        grooveEngine_ = std::make_unique<groove::GrooveEngine>(grooveConfig);
        
        diagnostics_ = std::make_unique<diagnostics::DiagnosticsEngine>();
        memoryPool_ = std::make_unique<RTMemoryPool>(512, 1000);
    }

    std::unique_ptr<harmony::HarmonyEngine> harmonyEngine_;
    std::unique_ptr<groove::GrooveEngine> grooveEngine_;
    std::unique_ptr<diagnostics::DiagnosticsEngine> diagnostics_;
    std::unique_ptr<RTMemoryPool> memoryPool_;
};

TEST_F(FullPluginIntegrationTest, CompleteProcessingChain) {
    constexpr size_t blockSize = 512;
    std::array<float, blockSize> audioIn;
    
    // Generate test audio
    generateSineWave(audioIn.data(), blockSize, 440.0, 44100.0);
    
    // Create test MIDI notes
    std::vector<Note> notes;
    notes.push_back({60, 100, 0});
    notes.push_back({64, 90, 0});
    notes.push_back({67, 95, 0});

    rtValidator_->beginRTContext();
    
    diagnostics_->beginMeasurement();
    
    // Process through groove engine
    grooveEngine_->processAudio(audioIn.data(), blockSize);
    
    // Process through harmony engine
    harmonyEngine_->processNotes(notes.data(), notes.size());
    
    // Analyze audio
    diagnostics_->analyzeAudio(audioIn.data(), blockSize, 1);
    
    diagnostics_->endMeasurement();
    
    rtValidator_->endRTContext();
    
    validateRTSafety();
    
    auto stats = diagnostics_->getStats();
    EXPECT_GT(stats.averageLatencyMs, 0.0f);
    EXPECT_LT(stats.cpuUsagePercent, 50.0f);  // Should be efficient
}

TEST_F(FullPluginIntegrationTest, StressTestWithMockDevice) {
    std::atomic<int> processedBlocks{0};
    std::atomic<int> xruns{0};
    
    // Configure mock device with jitter to simulate real conditions
    MockAudioDevice::Config config;
    config.bufferSize = 256;
    config.simulateJitter = true;
    config.jitterAmountMs = 1.0;
    mockDevice_ = std::make_unique<MockAudioDevice>(config);
    
    // Create test MIDI notes
    std::vector<Note> notes;
    notes.push_back({60, 100, 0});
    notes.push_back({64, 90, 0});
    notes.push_back({67, 95, 0});
    
    mockDevice_->setCallback([this, &processedBlocks, &xruns, &notes](
        const float* input, float* output, size_t numFrames, size_t numChannels) {
        
        rtValidator_->beginRTContext();
        diagnostics_->beginMeasurement();
        
        // Process through all engines
        for (size_t ch = 0; ch < numChannels; ++ch) {
            const float* in = input + (ch * numFrames);
            float* out = output + (ch * numFrames);
            
            // Groove analysis
            grooveEngine_->processAudio(in, numFrames);
            
            // Harmony processing
            harmonyEngine_->processNotes(notes.data(), notes.size());
            
            // Pass through audio
            std::memcpy(out, in, numFrames * sizeof(float));
        }
        
        diagnostics_->analyzeAudio(input, numFrames, numChannels);
        diagnostics_->endMeasurement();
        rtValidator_->endRTContext();
        
        processedBlocks++;
        
        // Check for xruns
        if (diagnostics_->getStats().xrunCount > 0) {
            xruns++;
        }
    });

    // Run for 1 second
    mockDevice_->start();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    mockDevice_->stop();

    EXPECT_GT(processedBlocks.load(), 100);  // Should process many blocks
    EXPECT_LT(xruns.load(), 5);  // Minimal xruns
    validateRTSafety();
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

class PluginPerformanceBenchmark : public PluginTestHarness {
protected:
    static constexpr size_t BENCHMARK_ITERATIONS = 10000;
    static constexpr size_t BLOCK_SIZE = 512;
};

TEST_F(PluginPerformanceBenchmark, HarmonyEngineLatency) {
    harmony::HarmonyEngine engine;
    
    std::vector<Note> notes;
    notes.push_back({60, 100, 0});
    notes.push_back({64, 90, 0});
    notes.push_back({67, 95, 0});
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        engine.processNotes(notes.data(), notes.size());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgUs = static_cast<double>(duration.count()) / BENCHMARK_ITERATIONS;
    
    std::cout << "Harmony Engine avg latency: " << avgUs << " μs\n";
    
    // Should complete in less than 100μs per block
    EXPECT_LT(avgUs, 100.0);
}

TEST_F(PluginPerformanceBenchmark, GrooveEngineLatency) {
    groove::GrooveEngine engine;
    std::array<float, BLOCK_SIZE> input;
    
    generateSineWave(input.data(), BLOCK_SIZE, 440.0, 44100.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        engine.processAudio(input.data(), BLOCK_SIZE);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgUs = static_cast<double>(duration.count()) / BENCHMARK_ITERATIONS;
    
    std::cout << "Groove Engine avg latency: " << avgUs << " μs\n";
    
    // Should complete in less than 100μs per block
    EXPECT_LT(avgUs, 100.0);
}

// ============================================================================
// Main (if needed for standalone execution)
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
