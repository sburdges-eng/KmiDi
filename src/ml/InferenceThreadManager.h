#pragma once

#include "ml/LockFreeRingBuffer.h"
#include "ml/RTNeuralProcessor.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <semaphore>    // std::counting_semaphore (C++20)
#include <juce_core/juce_core.h>

namespace kelly {

/**
 * InferenceRequest - Request for ML inference.
 */
struct InferenceRequest {
    std::array<float, 128> features;
    int64_t timestamp;

    InferenceRequest() : timestamp(0) {
        features.fill(0.0f);
    }
};

/**
 * InferenceResult - Result from ML inference.
 */
struct InferenceResult {
    std::array<float, 64> emotionVector;
    int64_t timestamp;

    InferenceResult() : timestamp(0) {
        emotionVector.fill(0.0f);
    }
};

/**
 * InferenceThreadManager - Manages ML inference in separate thread.
 *
 * Provides non-blocking inference for real-time audio processing.
 * Uses lock-free ring buffers for thread-safe communication.
 */
class InferenceThreadManager {
public:
    static constexpr size_t BUFFER_SIZE = 256;

    InferenceThreadManager() : running_(false) {}

    ~InferenceThreadManager() {
        stop();
    }

    /**
     * Start inference thread and load model.
     * @param modelPath Path to model file
     */
    void start(const juce::File& modelPath) {
        if (running_.load()) {
            stop();
        }

        if (!processor_.loadModel(modelPath)) {
            juce::Logger::writeToLog("Failed to load ML model: " + modelPath.getFullPathName());
            return;
        }

        running_.store(true);
        inferenceThread_ = std::thread(&InferenceThreadManager::inferenceLoop, this);
    }

    /**
     * Stop inference thread.
     */
    void stop() {
        running_.store(false);
        if (inferenceThread_.joinable()) {
            inferenceThread_.join();
        }
    }

    /**
     * Submit inference request (called from audio thread - never blocks).
     * T6.6: After pushing the request, release the semaphore so the inference
     * thread wakes immediately rather than sleeping 100µs.
     * @param request Inference request
     * @return true if submitted successfully
     */
    bool submitRequest(const InferenceRequest& request) noexcept {
        if (requestBuffer_.push(&request, 1)) {
            requestSem_.release();  // T6.6: wake inference thread
            return true;
        }
        return false;
    }

    /**
     * Get inference result (called from audio thread - never blocks).
     * @param result Output result
     * @return true if result available
     */
    bool getResult(InferenceResult& result) noexcept {
        return resultBuffer_.pop(&result, 1);
    }

    /**
     * Check if inference thread is running.
     */
    bool isRunning() const {
        return running_.load();
    }

    /**
     * Get number of pending requests.
     */
    size_t getPendingRequests() const {
        return requestBuffer_.availableToRead();
    }

    /**
     * Get number of available results.
     */
    size_t getAvailableResults() const {
        return resultBuffer_.availableToRead();
    }

private:
    /**
     * Inference loop running in separate thread.
     * T6.6: Blocks on counting_semaphore rather than spinning with 100µs sleep.
     * try_acquire_for(10ms) bounds latency if a release is missed at shutdown.
     */
    void inferenceLoop() {
        InferenceRequest request;

        while (running_.load()) {
            // T6.6: Wait for a request to be submitted (or timeout after 10ms to
            // re-check running_ and avoid indefinite blocking on shutdown).
            requestSem_.try_acquire_for(std::chrono::milliseconds(10));

            // Drain all pending requests — multiple blocks may have piled up
            // during a heavy inference pass.
            while (requestBuffer_.pop(&request, 1)) {
                // Perform inference
                InferenceResult result;
                result.emotionVector = processor_.inferEmotion(request.features);
                result.timestamp = request.timestamp;

                // T6.5: Drop-oldest policy — audio thread always sees latest
                // result; stale results are evicted instead of new ones
                // being silently discarded.
                resultBuffer_.pushOverwrite(&result, 1);
            }
        }
    }

    RTNeuralProcessor processor_;
    LockFreeRingBuffer<InferenceRequest, BUFFER_SIZE> requestBuffer_;
    LockFreeRingBuffer<InferenceResult, BUFFER_SIZE> resultBuffer_;
    std::thread inferenceThread_;
    std::atomic<bool> running_;
    // T6.6: Counting semaphore — released by submitRequest(), acquired by
    // inferenceLoop().  Replaces the 100µs spin-sleep.  Max count matches
    // BUFFER_SIZE so the semaphore counter never overflows.
    std::counting_semaphore<BUFFER_SIZE> requestSem_{0};
};

} // namespace kelly
