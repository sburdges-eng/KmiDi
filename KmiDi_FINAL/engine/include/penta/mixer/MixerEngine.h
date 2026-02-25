#pragma once

#include "penta/common/Platform.h"
/*
 * MixerEngine.h - Real-time safe audio mixer for Penta-Core
 * ===========================================================
 *
 * RT-safe multi-channel mixer with channel strips, sends/returns, and master bus.
 * Designed for Side A (Work State) - no allocations during audio processing.
 *
 * Architecture:
 * - Pre-allocated channel strips with gain, pan, mute, solo
 * - Send/return buses for effects
 * - Master bus with limiting
 * - Lock-free parameter updates from UI (Side B)
 * - SIMD-optimized mixing operations
 *
 * Thread Safety:
 * - processAudio() is RT-safe and lock-free
 * - Parameter updates via atomic operations
 * - No memory allocation in audio thread
 *
 * Integration:
 * - Works with OSCHub for parameter automation
 * - Connects to MixerConsolePanel (JUCE UI)
 * - Driven by Python mixer_params via PythonBridge
 */

#include "../common/RTTypes.h"
#include "../common/RTMemoryPool.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace penta {
namespace mixer {

// =============================================================================
// Constants
// =============================================================================

constexpr size_t kMaxChannels = 64;
constexpr size_t kMaxSendBuses = 8;
constexpr size_t kMaxFramesPerBlock = 8192;

// =============================================================================
// Channel Strip
// =============================================================================

/**
 * Single mixer channel with gain, pan, mute, solo, and sends.
 * All controls are RT-safe via atomic operations.
 */
class ChannelStrip {
public:
    ChannelStrip();
    ~ChannelStrip() = default;

    // RT-safe audio processing
    void processAudio(
        const float* input,
        float* outputL,
        float* outputR,
        size_t numFrames
    ) noexcept;

    // RT-safe parameter setters (atomic)
    void setGain(float gainDb) noexcept;
    void setPan(float pan) noexcept;           // -1.0 (left) to +1.0 (right)
    void setMute(bool muted) noexcept;
    void setSolo(bool soloed) noexcept;
    void setSendLevel(size_t sendBus, float level) noexcept;  // 0.0-1.0

    // Non-RT getters
    float getGain() const { return gainDb_.load(std::memory_order_relaxed); }
    float getPan() const { return pan_.load(std::memory_order_relaxed); }
    bool isMuted() const { return muted_.load(std::memory_order_relaxed); }
    bool isSoloed() const { return soloed_.load(std::memory_order_relaxed); }
    float getSendLevel(size_t sendBus) const;

    // Metering (updated by processAudio)
    float getPeakLevel() const { return peakLevel_.load(std::memory_order_relaxed); }
    float getRMSLevel() const { return rmsLevel_.load(std::memory_order_relaxed); }

    // Reset meters
    void resetMeters() noexcept;

private:
    // Parameters (atomic for lock-free updates)
    std::atomic<float> gainDb_{0.0f};
    std::atomic<float> pan_{0.0f};
    std::atomic<bool> muted_{false};
    std::atomic<bool> soloed_{false};
    std::array<std::atomic<float>, kMaxSendBuses> sendLevels_{};

    // Metering
    std::atomic<float> peakLevel_{0.0f};
    std::atomic<float> rmsLevel_{0.0f};

    // Internal processing state (not atomic - only accessed in audio thread)
    float currentGainLinear_{1.0f};
    float currentPanL_{0.707f};  // -3dB pan law
    float currentPanR_{0.707f};

    // Helper functions
    static float dbToLinear(float db) noexcept {
        return std::pow(10.0f, db / 20.0f);
    }

    void updatePanCoefficients(float pan) noexcept;
};

// =============================================================================
// Send Bus
// =============================================================================

/**
 * Auxiliary send bus for effects.
 * Receives sends from channels, processes effect, returns to master.
 */
class SendBus {
public:
    SendBus();
    ~SendBus() = default;

    // RT-safe processing
    void processAudio(
        float* bufferL,
        float* bufferR,
        size_t numFrames
    ) noexcept;

    // Accumulate send from channel
    void addSend(
        const float* sourceL,
        const float* sourceR,
        float level,
        size_t numFrames
    ) noexcept;

    // Clear send buffer (call before each process cycle)
    void clearBuffer() noexcept;

    // RT-safe controls
    void setReturnLevel(float level) noexcept;
    void setMute(bool muted) noexcept;

    float getReturnLevel() const { return returnLevel_.load(std::memory_order_relaxed); }
    bool isMuted() const { return muted_.load(std::memory_order_relaxed); }

private:
    // Send accumulation buffer
    std::array<float, kMaxFramesPerBlock> sendBufferL_{};
    std::array<float, kMaxFramesPerBlock> sendBufferR_{};

    // Parameters
    std::atomic<float> returnLevel_{1.0f};
    std::atomic<bool> muted_{false};
};

// =============================================================================
// Master Bus
// =============================================================================

/**
 * Master output bus with optional limiting.
 */
class MasterBus {
public:
    MasterBus();
    ~MasterBus() = default;

    // RT-safe processing
    void processAudio(
        float* bufferL,
        float* bufferR,
        size_t numFrames
    ) noexcept;

    // RT-safe controls
    void setGain(float gainDb) noexcept;
    void setLimiterThreshold(float thresholdDb) noexcept;
    void setLimiterEnabled(bool enabled) noexcept;

    float getGain() const { return gainDb_.load(std::memory_order_relaxed); }
    float getLimiterThreshold() const { return limiterThreshold_.load(std::memory_order_relaxed); }
    bool isLimiterEnabled() const { return limiterEnabled_.load(std::memory_order_relaxed); }

    // Metering
    float getPeakLevelL() const { return peakLevelL_.load(std::memory_order_relaxed); }
    float getPeakLevelR() const { return peakLevelR_.load(std::memory_order_relaxed); }

private:
    std::atomic<float> gainDb_{0.0f};
    std::atomic<float> limiterThreshold_{-1.0f};
    std::atomic<bool> limiterEnabled_{true};

    std::atomic<float> peakLevelL_{0.0f};
    std::atomic<float> peakLevelR_{0.0f};

    // Limiter state
    float limiterGainReduction_{1.0f};

    static float dbToLinear(float db) noexcept {
        return std::pow(10.0f, db / 20.0f);
    }

    void applyLimiter(float* bufferL, float* bufferR, size_t numFrames) noexcept;
};

// =============================================================================
// Mixer Engine
// =============================================================================

/**
 * Main mixer engine coordinating all channels, sends, and master bus.
 *
 * Usage:
 *   MixerEngine mixer(48000.0);
 *   mixer.setNumChannels(8);
 *
 *   // In audio callback:
 *   mixer.processAudio(inputs, outputs, numFrames);
 *
 *   // From UI thread:
 *   mixer.setChannelGain(0, -6.0f);
 *   mixer.setChannelPan(0, -0.5f);
 */
class MixerEngine {
public:
    explicit MixerEngine(double sampleRate);
    ~MixerEngine() = default;

    // Configuration (non-RT)
    void setNumChannels(size_t numChannels);
    void setNumSendBuses(size_t numBuses);
    size_t getNumChannels() const { return numChannels_; }
    size_t getNumSendBuses() const { return numSendBuses_; }

    // RT-safe audio processing
    void processAudio(
        const float** inputs,      // [numChannels][numFrames]
        float** outputs,           // [2][numFrames] (stereo out)
        size_t numFrames
    ) noexcept;

    // Channel controls (RT-safe via atomic operations)
    void setChannelGain(size_t channel, float gainDb) noexcept;
    void setChannelPan(size_t channel, float pan) noexcept;
    void setChannelMute(size_t channel, bool muted) noexcept;
    void setChannelSolo(size_t channel, bool soloed) noexcept;
    void setChannelSend(size_t channel, size_t sendBus, float level) noexcept;

    // Send bus controls
    void setSendReturnLevel(size_t sendBus, float level) noexcept;
    void setSendMute(size_t sendBus, bool muted) noexcept;

    // Master controls
    void setMasterGain(float gainDb) noexcept;
    void setMasterLimiter(bool enabled, float thresholdDb) noexcept;

    // Solo logic
    bool isAnySoloed() const;
    void clearAllSolo() noexcept;

    // Metering
    float getChannelPeak(size_t channel) const;
    float getChannelRMS(size_t channel) const;
    float getMasterPeakL() const { return master_.getPeakLevelL(); }
    float getMasterPeakR() const { return master_.getPeakLevelR(); }

    // Reset
    void resetAllMeters() noexcept;

private:
    double sampleRate_;
    size_t numChannels_;
    size_t numSendBuses_;

    // Channel strips
    std::array<ChannelStrip, kMaxChannels> channels_;

    // Send buses
    std::array<SendBus, kMaxSendBuses> sendBuses_;

    // Master bus
    MasterBus master_;

    // Mixing buffers (pre-allocated)
    std::array<float, kMaxFramesPerBlock> mixBufferL_{};
    std::array<float, kMaxFramesPerBlock> mixBufferR_{};

    // Solo state tracking
    std::atomic<size_t> soloCount_{0};

    // Validation
    bool isValidChannel(size_t channel) const noexcept {
        return channel < numChannels_;
    }

    bool isValidSendBus(size_t sendBus) const noexcept {
        return sendBus < numSendBuses_;
    }
};

} // namespace mixer
} // namespace penta
