#pragma once

#include "ml/LockFreeRingBuffer.h"
#include "ml/RTNeuralProcessor.h"
#include <atomic>
#include <chrono>
#include <juce_core/juce_core.h>
#include <semaphore> // std::binary_semaphore (C++20)
#include <thread>
#include <cstddef>

namespace kelly {

/**
 * InferenceRequest - Request for ML inference.
 */
struct InferenceRequest {
  std::array<float, 128> features;
  int64_t timestamp;

  InferenceRequest() : timestamp(0) { features.fill(0.0f); }
};

/**
 * InferenceResult - Result from ML inference.
 */
struct InferenceResult {
  std::array<float, 64> emotionVector;
  int64_t timestamp;

  InferenceResult() : timestamp(0) { emotionVector.fill(0.0f); }
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

  ~InferenceThreadManager() { stop(); }

  /**
   * Start inference thread and load model.
   * @param modelPath Path to model file
   */
  void start(const juce::File &modelPath) {
    stop();

    if (!processor_.loadModel(modelPath)) {
      juce::Logger::writeToLog("Failed to load ML model: " +
                               modelPath.getFullPathName());
      return;
    }

    running_.store(true, std::memory_order_release);
    inferenceThread_ =
        std::thread(&InferenceThreadManager::inferenceLoop, this);
  }

  /**
   * Stop inference thread.
   */
  void stop() {
    const bool wasRunning = running_.exchange(false, std::memory_order_acq_rel);
    if (!wasRunning && !inferenceThread_.joinable()) {
      return;
    }

    if (inferenceThread_.joinable()) {
      inferenceThread_.join();
    }
  }

  /**
   * Submit inference request (called from audio thread - never blocks).
   * T6.6: Signal the inference thread that work is available.
   *
   * Uses binary-semaphore wake semantics: try_acquire resets the counter to
   * 0 before release sets it to 1, ensuring release() never exceeds max (1).
   * ([thread.sema.cnt]/6: release(n) requires n <= max() - counter; for
   * binary_semaphore max()==1 so calling release() with counter==1 is UB.)
   *
   * @param request Inference request
   * @return true if submitted successfully
   */
  bool submitRequest(const InferenceRequest &request) noexcept {
    if (requestBuffer_.push(&request, 1)) {
      // Drain the semaphore counter back to 0 before setting it to 1.
      // try_acquire is non-blocking; if the counter was already 0 this
      // is a no-op. This ensures release() never violates the max==1
      // precondition of binary_semaphore.
      (void)requestWake_.try_acquire();
      requestWake_.release(); // T6.6: wake inference thread
      return true;
    }
    return false;
  }

  /**
   * Get the latest inference result (called from audio thread - never blocks).
   * Drains all stale results from the buffer and returns only the freshest one.
   * This provides the drop-oldest semantics previously attempted by
   * pushOverwrite, but correctly confined to the CONSUMER thread.
   * @param result Output result (populated only when return value is true)
   * @return true if at least one result was available
   */
  bool getResult(InferenceResult &result) noexcept {
    return resultBuffer_.popLatest(&result);
  }

  /**
   * Check if inference thread is running.
   */
  bool isRunning() const { return running_.load(); }

  /**
   * Get number of pending requests.
   */
  size_t getPendingRequests() const { return requestBuffer_.availableToRead(); }

  /**
   * Get number of available results.
   */
  size_t getAvailableResults() const { return resultBuffer_.availableToRead(); }

private:
  /**
   * Inference loop running in separate thread.
   * T6.6: Blocks on binary_semaphore rather than spinning with 100µs sleep.
   * try_acquire_for(10ms) bounds latency if a wake is missed at shutdown.
   *
   * One wake covers N pending requests: the inner while loop drains everything
   * that accumulated while the worker was busy, so every submitted request is
   * processed even if only one wake arrived for N submits.
   */
  void inferenceLoop() {
    InferenceRequest request;

    while (running_.load()) {
      // T6.6: Wait for a request to be submitted (or timeout after 10ms to
      // re-check running_ and avoid indefinite blocking on shutdown).
      requestWake_.try_acquire_for(std::chrono::milliseconds(10));

      // Drain all pending requests — multiple blocks may have piled up
      // during a heavy inference pass.
      while (requestBuffer_.pop(&request, 1)) {
        // Perform inference
        InferenceResult result;
        result.emotionVector = processor_.inferEmotion(request.features);
        result.timestamp = request.timestamp;

        // T6.5 (revised): plain push with drop-newest on the producer
        // side.  The audio thread uses resultBuffer_.popLatest() to
        // drain stale results and obtain only the freshest one.
        // pushOverwrite was removed because it mutated readPos_ from
        // the PRODUCER thread, violating SPSC ring invariants.
        resultBuffer_.push(&result, 1);
      }
    }
  }

  RTNeuralProcessor processor_;
  LockFreeRingBuffer<InferenceRequest, BUFFER_SIZE> requestBuffer_;
  LockFreeRingBuffer<InferenceResult, BUFFER_SIZE> resultBuffer_;
  std::thread inferenceThread_;
  std::atomic<bool> running_;
  // T6.6: Binary semaphore — signalled by submitRequest(), acquired by
  // inferenceLoop().  Replaces the 100µs spin-sleep.  Binary semantics
  // (max == 1) prevent the counter from exceeding its maximum regardless
  // of how many requests arrive between two consecutive wakes; the inner
  // drain loop in inferenceLoop() handles burst coalescing.
  std::binary_semaphore requestWake_{0};
};

} // namespace kelly
