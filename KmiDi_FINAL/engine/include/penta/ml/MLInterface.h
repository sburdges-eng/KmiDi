#pragma once

#include "penta/common/Platform.h"
/**
 * @file MLInterface.h
 * @brief Real-time safe ML inference interface for Penta-Core
 *
 * This module provides a lock-free interface between audio processing
 * and ML inference running on a separate thread.
 */

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace penta::ml {

// Forward declarations
class InferenceEngine;
class ModelCache;

/**
 * @brief Model types supported by the ML interface
 */
enum class ModelType {
    ChordPredictor,      ///< Predicts next chord in progression
    GrooveTransfer,      ///< Transfers groove/timing style
    KeyDetector,         ///< ML-enhanced key detection
    IntentMapper,        ///< Maps emotional intent to parameters
    // Extended types used by DAiW model registry
    EmotionRecognizer,   ///< Audio features -> emotion embedding
    MelodyTransformer,   ///< Emotion embedding -> note probabilities
    HarmonyPredictor,    ///< Context -> chord/harmony predictions
    DynamicsEngine,      ///< Emotion -> expression parameters
    GroovePredictor,     ///< Emotion -> groove/timing parameters
    Custom               ///< User-defined model
};

/**
 * @brief Inference request structure (RT-safe, no allocations)
 */
constexpr size_t MAX_INPUT_SIZE = 128;

struct InferenceRequest {
    ModelType model_type;
    std::array<float, MAX_INPUT_SIZE> input_data;  ///< Fixed-size input buffer
    size_t input_size;                             ///< Actual input size
    uint64_t request_id;                           ///< Unique request identifier
    uint64_t timestamp;                            ///< Audio sample timestamp
};

/**
 * @brief Inference result structure (RT-safe, no allocations)
 */
struct InferenceResult {
    ModelType model_type;
    std::array<float, 128> output_data;  ///< Fixed-size output buffer
    size_t output_size;                   ///< Actual output size
    uint64_t request_id;                  ///< Matching request ID
    float confidence;                     ///< Model confidence [0, 1]
    float latency_ms;                     ///< Inference latency
    bool success;                         ///< Inference succeeded
};

/**
 * @brief Configuration for ML inference engine
 */
struct MLConfig {
    std::string model_directory;          ///< Path to model files
    size_t max_concurrent_requests = 4;   ///< Queue depth
    bool use_gpu = false;                 ///< Enable GPU acceleration
    bool use_coreml = true;               ///< Use CoreML on Apple
    size_t inference_thread_priority = 0; ///< Thread priority (0 = default)
    float timeout_ms = 100.0f;            ///< Request timeout
};

/**
 * @brief Lock-free queue for RT-safe communication
 * @tparam T Element type
 * @tparam Capacity Maximum queue size
 */
template <typename T, size_t Capacity = 16>
class LockFreeQueue {
public:
    bool try_push(const T& item) noexcept {
        size_t write = write_pos_.load(std::memory_order_relaxed);
        size_t next = (write + 1) % Capacity;
        if (next == read_pos_.load(std::memory_order_acquire)) {
            return false;  // Full
        }
        buffer_[write] = item;
        write_pos_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& item) noexcept {
        size_t read = read_pos_.load(std::memory_order_relaxed);
        if (read == write_pos_.load(std::memory_order_acquire)) {
            return false;  // Empty
        }
        item = buffer_[read];
        read_pos_.store((read + 1) % Capacity, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return read_pos_.load(std::memory_order_acquire) ==
               write_pos_.load(std::memory_order_acquire);
    }

private:
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> read_pos_{0};
    std::atomic<size_t> write_pos_{0};
};

/**
 * @brief ML Inference Interface
 *
 * Provides RT-safe communication between audio thread and ML inference.
 * All audio-thread methods are lock-free and allocation-free.
 */
class MLInterface {
public:
    explicit MLInterface(const MLConfig& config);
    ~MLInterface();

    // Non-copyable, non-movable
    MLInterface(const MLInterface&) = delete;
    MLInterface& operator=(const MLInterface&) = delete;

    /**
     * @brief Start inference thread
     * @return true if started successfully
     */
    bool start();

    /**
     * @brief Stop inference thread
     */
    void stop();

    /**
     * @brief Check if inference engine is running
     */
    bool isRunning() const noexcept;

    // =========================================================================
    // Audio Thread API (RT-Safe, lock-free, no allocations)
    // =========================================================================

    /**
     * @brief Submit inference request from audio thread
     * @param request The inference request
     * @return true if request was queued, false if queue full
     * @note RT-safe: no allocations, lock-free
     */
    bool submitRequest(const InferenceRequest& request) noexcept;

    /**
     * @brief Poll for inference result
     * @param result Output parameter for result
     * @return true if result available, false otherwise
     * @note RT-safe: no allocations, lock-free
     */
    bool pollResult(InferenceResult& result) noexcept;

    /**
     * @brief Get next request ID (atomic increment)
     * @return Unique request identifier
     */
    uint64_t getNextRequestId() noexcept;

    // =========================================================================
    // Non-RT API (for initialization and configuration)
    // =========================================================================

    /**
     * @brief Load a model file
     * @param type Model type
     * @param path Path to model file (ONNX format)
     * @return true if loaded successfully
     */
    bool loadModel(ModelType type, const std::string& path);

    /**
     * @brief Load all models defined in a registry JSON.
     * @param registry_path Path to registry.json (see /models/registry.json)
     * @return true if all models loaded (partial failures return false)
     */
    bool loadRegistry(const std::string& registry_path);

    /**
     * @brief Unload a model
     * @param type Model type to unload
     */
    void unloadModel(ModelType type);

    /**
     * @brief Check if model is loaded
     * @param type Model type
     * @return true if model is loaded and ready
     */
    bool isModelLoaded(ModelType type) const;

    /**
     * @brief Get inference statistics
     */
    struct Stats {
        uint64_t total_requests;
        uint64_t completed_requests;
        uint64_t failed_requests;
        uint64_t queue_overflows;
        float avg_latency_ms;
        float max_latency_ms;
    };
    Stats getStats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Convenience functions for common inference patterns
// ============================================================================

/**
 * @brief Create chord prediction request
 * @param pitch_classes 12-element pitch class histogram
 * @param request_id Unique request ID
 * @return Configured inference request
 */
inline InferenceRequest createChordRequest(
    const std::array<float, 12>& pitch_classes,
    uint64_t request_id) noexcept {
    InferenceRequest req{};
    req.model_type = ModelType::ChordPredictor;
    std::copy(pitch_classes.begin(), pitch_classes.end(), req.input_data.begin());
    req.input_size = 12;
    req.request_id = request_id;
    return req;
}

/**
 * @brief Create groove transfer request
 * @param timing_deviations 32-element timing deviation vector
 * @param style_embedding 16-element style embedding
 * @param request_id Unique request ID
 * @return Configured inference request
 */
inline InferenceRequest createGrooveRequest(
    const std::array<float, 32>& timing_deviations,
    const std::array<float, 16>& style_embedding,
    uint64_t request_id) noexcept {
    InferenceRequest req{};
    req.model_type = ModelType::GrooveTransfer;
    std::copy(timing_deviations.begin(), timing_deviations.end(),
              req.input_data.begin());
    std::copy(style_embedding.begin(), style_embedding.end(),
              req.input_data.begin() + 32);
    req.input_size = 48;
    req.request_id = request_id;
    return req;
}

}  // namespace penta::ml
