#include "penta/groove/TempoEstimator.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace penta::groove {

// NOTE: This is the legacy implementation maintained for backward compatibility.
// The main implementation is in src_penta-core/groove/TempoEstimator.cpp
// which uses RT-safe circular buffers and lock-free atomics.

TempoEstimator::TempoEstimator(const Config& config)
    : config_(config)
    , lastOnsetPosition_(0)
{
    // Validate history size
    if (config_.historySize > kMaxHistorySize) {
        config_.historySize = kMaxHistorySize;
    }
}

void TempoEstimator::addOnset(uint64_t samplePosition) noexcept {
    // Add to circular buffer
    onsetHistory_[onsetWriteIndex_] = samplePosition;
    onsetWriteIndex_ = (onsetWriteIndex_ + 1) % config_.historySize;

    // Track count up to history size
    if (onsetCount_ < config_.historySize) {
        ++onsetCount_;
    }

    lastOnsetPosition_ = samplePosition;

    // Estimate tempo if we have enough onsets
    if (onsetCount_ >= 4) {
        estimateTempo();
    }
}

uint64_t TempoEstimator::getSamplesPerBeat() const noexcept {
    float tempo = getCurrentTempo();
    if (tempo <= 0.0f) return 0;
    return static_cast<uint64_t>((60.0 * config_.sampleRate) / tempo);
}

void TempoEstimator::updateConfig(const Config& config) noexcept {
    // Validate and update config
    config_ = config;
    if (config_.historySize > kMaxHistorySize) {
        config_.historySize = kMaxHistorySize;
    }

    // Reset if history size changed significantly
    if (onsetCount_ > config_.historySize) {
        reset();
    }
}

void TempoEstimator::reset() noexcept {
    onsetWriteIndex_ = 0;
    onsetCount_ = 0;
    currentTempo_.store(120.0f, std::memory_order_relaxed);
    confidence_.store(0.0f, std::memory_order_relaxed);
    lastOnsetPosition_ = 0;
    onsetHistory_.fill(0);
}

void TempoEstimator::estimateTempo() noexcept {
    if (onsetCount_ < 4) {
        return;  // Need at least 4 onsets
    }

    // Calculate inter-onset intervals using stack-allocated array
    float intervals[kMaxHistorySize - 1];
    size_t intervalCount = 0;

    // Build interval list from circular buffer
    size_t readIndex = (onsetWriteIndex_ + config_.historySize - onsetCount_) % config_.historySize;
    uint64_t prevOnset = onsetHistory_[readIndex];

    for (size_t i = 1; i < onsetCount_; ++i) {
        readIndex = (readIndex + 1) % config_.historySize;
        uint64_t currentOnset = onsetHistory_[readIndex];

        uint64_t ioi = currentOnset - prevOnset;
        if (ioi > 0) {
            intervals[intervalCount++] = static_cast<float>(ioi) / static_cast<float>(config_.sampleRate);
        }
        prevOnset = currentOnset;
    }

    if (intervalCount < 3) {
        return;
    }

    // Find best tempo
    float bestInterval = findBestInterval(intervals, intervalCount);
    if (bestInterval <= 0.0f) {
        return;
    }

    // Convert to BPM
    float estimatedTempo = 60.0f / bestInterval;
    estimatedTempo = std::clamp(estimatedTempo, config_.minTempo, config_.maxTempo);

    // Smooth tempo updates
    float oldTempo = currentTempo_.load(std::memory_order_relaxed);
    float newTempo = oldTempo * (1.0f - config_.adaptationRate) +
                     estimatedTempo * config_.adaptationRate;
    currentTempo_.store(newTempo, std::memory_order_relaxed);

    // Update confidence
    float conf = std::clamp(computeCorrelation(intervals, intervalCount, bestInterval), 0.0f, 1.0f);
    confidence_.store(conf, std::memory_order_relaxed);
}

float TempoEstimator::findBestInterval(const float* intervals, size_t count) const noexcept {
    if (count == 0) {
        return 0.0f;
    }

    float bestInterval = 0.0f;
    float bestScore = -std::numeric_limits<float>::infinity();

    for (float testTempo = config_.minTempo; testTempo <= config_.maxTempo; testTempo += config_.tempoSearchStep) {
        float testInterval = 60.0f / testTempo;
        float score = computeCorrelation(intervals, count, testInterval);
        if (score > bestScore) {
            bestScore = score;
            bestInterval = testInterval;
        }
    }

    return bestInterval;
}

float TempoEstimator::computeCorrelation(
    const float* intervals,
    size_t count,
    float testInterval
) const noexcept {
    if (count == 0 || testInterval <= 0.0f) {
        return 0.0f;
    }

    constexpr float kToleranceRatio = 0.12f;
    float tolerance = testInterval * kToleranceRatio;
    float invTwoSigmaSq = 1.0f / (2.0f * tolerance * tolerance);

    float score = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float interval = intervals[i];
        float multiple = std::max(1.0f, std::round(interval / testInterval));
        float expected = multiple * testInterval;
        float error = std::abs(interval - expected);
        score += std::exp(-(error * error) * invTwoSigmaSq);
    }

    return score / static_cast<float>(count);
}

} // namespace penta::groove
