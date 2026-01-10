/**
 * test_ml_pipeline.cpp - Tests for ML inference pipeline
 * 
 * Tests:
 * - MultiModelProcessor initialization and fallback
 * - Feature extraction
 * - Lock-free ring buffer
 * - Async inference pipeline
 * - Model configuration
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "ml/MultiModelProcessor.h"
#include "ml/MLFeatureExtractor.h"
#include "ml/LockFreeRingBuffer.h"
#include "ml/InferenceRequest.h"
#include "ml/ModelConfig.h"

#include <array>
#include <thread>
#include <chrono>

using namespace kelly;
using namespace Kelly::ML;

// =============================================================================
// LockFreeRingBuffer Tests
// =============================================================================

TEST_CASE("LockFreeRingBuffer basic operations", "[ml][ringbuffer]") {
    LockFreeRingBuffer<int, 16> buffer;
    
    SECTION("Empty buffer") {
        REQUIRE(buffer.isEmpty());
        REQUIRE(buffer.availableToRead() == 0);
        REQUIRE(buffer.availableToWrite() == 16);
    }
    
    SECTION("Push and pop single element") {
        int value = 42;
        REQUIRE(buffer.push(&value, 1));
        REQUIRE(buffer.availableToRead() == 1);
        
        int result = 0;
        REQUIRE(buffer.pop(&result, 1));
        REQUIRE(result == 42);
        REQUIRE(buffer.isEmpty());
    }
    
    SECTION("Push multiple elements") {
        std::array<int, 4> values = {1, 2, 3, 4};
        REQUIRE(buffer.push(values.data(), 4));
        REQUIRE(buffer.availableToRead() == 4);
        
        std::array<int, 4> results{};
        REQUIRE(buffer.pop(results.data(), 4));
        REQUIRE(results == values);
    }
    
    SECTION("Buffer overflow") {
        std::array<int, 16> values{};
        REQUIRE(buffer.push(values.data(), 16));
        REQUIRE(buffer.isFull());
        
        int extra = 0;
        REQUIRE_FALSE(buffer.push(&extra, 1));  // Should fail
    }
    
    SECTION("Wraparound") {
        // Fill half
        std::array<int, 8> first{};
        std::fill(first.begin(), first.end(), 1);
        REQUIRE(buffer.push(first.data(), 8));
        
        // Pop half
        std::array<int, 8> pop1{};
        REQUIRE(buffer.pop(pop1.data(), 8));
        
        // Push full (wraps around)
        std::array<int, 16> second{};
        std::fill(second.begin(), second.end(), 2);
        REQUIRE(buffer.push(second.data(), 16));
        
        // Pop all
        std::array<int, 16> pop2{};
        REQUIRE(buffer.pop(pop2.data(), 16));
        for (int v : pop2) {
            REQUIRE(v == 2);
        }
    }
}

TEST_CASE("LockFreeRingBuffer thread safety", "[ml][ringbuffer][threading]") {
    LockFreeRingBuffer<InferenceRequest, 256> buffer;
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::atomic<bool> running{true};
    
    // Producer thread
    std::thread producer([&]() {
        while (produced < 1000) {
            InferenceRequest req;
            req.timestamp = produced.load();
            if (buffer.push(&req, 1)) {
                produced++;
            }
            std::this_thread::yield();
        }
    });
    
    // Consumer thread
    std::thread consumer([&]() {
        while (consumed < 1000) {
            InferenceRequest req;
            if (buffer.pop(&req, 1)) {
                consumed++;
            }
            std::this_thread::yield();
        }
    });
    
    producer.join();
    consumer.join();
    
    REQUIRE(produced == 1000);
    REQUIRE(consumed == 1000);
}

// =============================================================================
// InferenceRequest/Result Tests
// =============================================================================

TEST_CASE("InferenceRequest initialization", "[ml][request]") {
    InferenceRequest req;
    
    REQUIRE(req.timestamp == 0);
    for (float f : req.features) {
        REQUIRE(f == 0.0f);
    }
}

TEST_CASE("InferenceResult initialization", "[ml][result]") {
    InferenceResult res;
    
    REQUIRE(res.timestamp == 0);
    for (float f : res.emotionVector) {
        REQUIRE(f == 0.0f);
    }
}

// =============================================================================
// MultiModelProcessor Tests
// =============================================================================

TEST_CASE("MultiModelProcessor initialization", "[ml][processor]") {
    MultiModelProcessor processor;
    
    SECTION("Before initialization") {
        REQUIRE_FALSE(processor.isInitialized());
    }
    
    SECTION("Initialize with non-existent directory") {
        juce::File nonExistent("/nonexistent/path/models");
        // Should still initialize (fallback mode)
        REQUIRE(processor.initialize(nonExistent));
        REQUIRE(processor.isInitialized());
    }
    
    SECTION("Model memory estimates") {
        REQUIRE(processor.getTotalParams() > 0);
        REQUIRE(processor.getTotalMemoryKB() > 0);
    }
}

TEST_CASE("MultiModelProcessor fallback inference", "[ml][processor][inference]") {
    MultiModelProcessor processor;
    juce::File tempDir = juce::File::getSpecialLocation(
        juce::File::tempDirectory).getChildFile("kelly_test_models");
    
    processor.initialize(tempDir);
    
    SECTION("EmotionRecognizer fallback") {
        std::array<float, 128> input{};
        // Set some test values
        for (size_t i = 0; i < 64; ++i) {
            input[i] = static_cast<float>(i) / 128.0f;
        }
        
        auto result = processor.runFullPipeline(input);
        REQUIRE(result.valid);
        
        // Check emotion embedding has non-zero values
        bool hasNonZero = false;
        for (float v : result.emotionEmbedding) {
            if (v != 0.0f) hasNonZero = true;
        }
        REQUIRE(hasNonZero);
    }
    
    SECTION("Full pipeline") {
        std::array<float, 128> input{};
        for (size_t i = 0; i < 128; ++i) {
            input[i] = std::sin(static_cast<float>(i) * 0.1f) * 0.5f;
        }
        
        auto result = processor.runFullPipeline(input);
        
        REQUIRE(result.valid);
        REQUIRE(result.emotionEmbedding.size() == 64);
        REQUIRE(result.melodyProbabilities.size() == 128);
        REQUIRE(result.harmonyPrediction.size() == 64);
        REQUIRE(result.dynamicsOutput.size() == 16);
        REQUIRE(result.grooveParameters.size() == 32);
    }
    
    // Cleanup
    tempDir.deleteRecursively();
}

TEST_CASE("MultiModelProcessor model enable/disable", "[ml][processor]") {
    MultiModelProcessor processor;
    juce::File tempDir = juce::File::getSpecialLocation(
        juce::File::tempDirectory).getChildFile("kelly_test_models2");
    processor.initialize(tempDir);
    
    SECTION("Disable emotion recognizer") {
        processor.setModelEnabled(ModelType::EmotionRecognizer, false);
        REQUIRE_FALSE(processor.isModelEnabled(ModelType::EmotionRecognizer));
        
        std::array<float, 128> input{};
        auto result = processor.runFullPipeline(input);
        
        // Emotion should be all zeros when disabled
        bool allZero = true;
        for (float v : result.emotionEmbedding) {
            if (v != 0.0f) allZero = false;
        }
        REQUIRE(allZero);
    }
    
    tempDir.deleteRecursively();
}

// =============================================================================
// AsyncMLPipeline Tests
// =============================================================================

TEST_CASE("AsyncMLPipeline basic operation", "[ml][async]") {
    MultiModelProcessor processor;
    juce::File tempDir = juce::File::getSpecialLocation(
        juce::File::tempDirectory).getChildFile("kelly_test_models3");
    processor.initialize(tempDir);
    
    AsyncMLPipeline asyncPipeline(processor);
    
    SECTION("Start and stop") {
        asyncPipeline.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        asyncPipeline.stop();
    }
    
    SECTION("Submit and receive result") {
        asyncPipeline.start();
        
        std::array<float, 128> features{};
        for (size_t i = 0; i < 128; ++i) {
            features[i] = static_cast<float>(i) / 128.0f;
        }
        
        auto requestId = asyncPipeline.submitFeatures(features);
        REQUIRE(requestId > 0);
        
        // Wait for result (with timeout)
        auto startTime = std::chrono::steady_clock::now();
        while (!asyncPipeline.hasResult(requestId)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            auto elapsed = std::chrono::steady_clock::now() - startTime;
            if (elapsed > std::chrono::seconds(1)) {
                FAIL("Timeout waiting for inference result");
            }
        }
        
        auto result = asyncPipeline.getResult(requestId);
        REQUIRE(result.valid);
        
        asyncPipeline.stop();
    }
    
    tempDir.deleteRecursively();
}

// =============================================================================
// ModelConfig Tests
// =============================================================================

TEST_CASE("ModelConfig default directory", "[ml][config]") {
    using namespace kelly::ml;
    
    auto dir = getDefaultModelsDirectory();
    // Should return a valid path (may or may not exist)
    REQUIRE(dir.getFullPathName().isNotEmpty());
}

TEST_CASE("ModelConfig model paths", "[ml][config]") {
    using namespace kelly::ml;
    
    juce::File tempDir = juce::File::getSpecialLocation(
        juce::File::tempDirectory).getChildFile("kelly_config_test");
    tempDir.createDirectory();
    
    SECTION("Non-existent model returns JSON path") {
        auto path = getModelPath(tempDir, ModelFiles::EmotionRecognizer);
        REQUIRE(path.getFileName() == "emotionrecognizer.json");
    }
    
    SECTION("Existing JSON model found") {
        auto jsonFile = tempDir.getChildFile("emotionrecognizer.json");
        jsonFile.replaceWithText("{}");
        
        auto path = getModelPath(tempDir, ModelFiles::EmotionRecognizer);
        REQUIRE(path.existsAsFile());
        REQUIRE(path.getFileExtension() == ".json");
    }
    
    SECTION("ONNX fallback when JSON missing") {
        auto onnxFile = tempDir.getChildFile("emotionrecognizer.onnx");
        onnxFile.replaceWithText("");  // Fake ONNX file
        
        auto path = getModelPath(tempDir, ModelFiles::EmotionRecognizer);
        REQUIRE(path.existsAsFile());
        REQUIRE(path.getFileExtension() == ".onnx");
    }
    
    tempDir.deleteRecursively();
}

TEST_CASE("InferenceConfig defaults", "[ml][config]") {
    using namespace kelly::ml;
    
    InferenceConfig config;
    
    REQUIRE(config.maxInferenceTimeMs == 10.0f);
    REQUIRE(config.lookaheadBufferMs == 20);
    REQUIRE(config.enableEmotionRecognizer);
    REQUIRE(config.enableMelodyTransformer);
    REQUIRE(config.useFallbackOnError);
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST_CASE("Inference latency benchmark", "[ml][performance][!benchmark]") {
    MultiModelProcessor processor;
    juce::File tempDir = juce::File::getSpecialLocation(
        juce::File::tempDirectory).getChildFile("kelly_bench");
    processor.initialize(tempDir);
    
    std::array<float, 128> input{};
    for (size_t i = 0; i < 128; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        processor.runFullPipeline(input);
    }
    
    // Benchmark
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        processor.runFullPipeline(input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avgMs = static_cast<float>(duration.count()) / iterations / 1000.0f;
    
    INFO("Average inference time: " << avgMs << " ms");
    
    // Target: <10ms
    REQUIRE(avgMs < 10.0f);
    
    tempDir.deleteRecursively();
}

