#pragma once

#include "penta/common/Platform.h"
#include <array>
#include <atomic>

namespace penta::groove {

/**
 * Real-time tempo estimation using autocorrelation
 * Tracks tempo changes with adaptive filtering
 *
 * RT-Safe Design:
 * - Fixed-size circular buffer for onset history (no allocations)
 * - Lock-free atomic operations for tempo/confidence updates
 * - Optimized correlation algorithm with early exits
 */
class TempoEstimator {
public:
    struct Config {
        double sampleRate;
        float minTempo;
        float maxTempo;
        float adaptationRate;   // How quickly to adapt to changes (0.0-1.0)
        float tempoSearchStep;  // Step size for tempo search (BPM)
        size_t historySize;     // Number of onsets to consider (max 64)

        Config()
            : sampleRate(48000.0)
            , minTempo(60.0f)
            , maxTempo(180.0f)
            , adaptationRate(0.1f)
            , tempoSearchStep(0.5f)
            , historySize(32)
        {}
    };

    explicit TempoEstimator(const Config& config = Config{});
    ~TempoEstimator() = default;

    // Prevent copying (contains atomics)
    TempoEstimator(const TempoEstimator&) = delete;
    TempoEstimator& operator=(const TempoEstimator&) = delete;

    // RT-safe: Add onset time for tempo calculation
    void addOnset(uint64_t samplePosition) noexcept;

    // RT-safe: Get current tempo estimate
    float getCurrentTempo() const noexcept {
        return currentTempo_.load(std::memory_order_relaxed);
    }

    // RT-safe: Get confidence of tempo estimate (0.0-1.0)
    float getConfidence() const noexcept {
        return confidence_.load(std::memory_order_relaxed);
    }

    // RT-safe: Get samples per beat
    uint64_t getSamplesPerBeat() const noexcept;

    // RT-safe: Check if we have enough data for reliable estimation
    bool hasReliableEstimate() const noexcept {
        return onsetCount_ >= 4 && getConfidence() > 0.3f;
    }

    // RT-safe: Get number of onsets recorded
    size_t getOnsetCount() const noexcept { return onsetCount_; }

    // Configuration (call from non-RT thread)
    void updateConfig(const Config& config) noexcept;
    void reset() noexcept;

private:
    void estimateTempo() noexcept;
    float findBestInterval(const float* intervals, size_t count) const noexcept;
    float computeCorrelation(const float* intervals, size_t count, float testInterval) const noexcept;

    Config config_;

    // Fixed-size circular buffer for onset history (RT-safe)
    static constexpr size_t kMaxHistorySize = 64;
    std::array<uint64_t, kMaxHistorySize> onsetHistory_{};
    size_t onsetWriteIndex_{0};
    size_t onsetCount_{0};

    // Atomic tempo/confidence for lock-free reads
    std::atomic<float> currentTempo_{120.0f};
    std::atomic<float> confidence_{0.0f};

    uint64_t lastOnsetPosition_{0};
};

} // namespace penta::groove
