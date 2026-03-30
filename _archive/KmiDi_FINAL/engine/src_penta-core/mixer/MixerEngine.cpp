#include "penta/mixer/MixerEngine.h"
#include <algorithm>
#include <cstring>

namespace penta {
namespace mixer {

// =============================================================================
// ChannelStrip Implementation
// =============================================================================

ChannelStrip::ChannelStrip() {
    // Initialize send levels to 0
    for (auto& sendLevel : sendLevels_) {
        sendLevel.store(0.0f, std::memory_order_relaxed);
    }
    updatePanCoefficients(0.0f);
}

void ChannelStrip::processAudio(
    const float* input,
    float* outputL,
    float* outputR,
    size_t numFrames
) noexcept {
    // Load current parameters (relaxed ordering OK for audio params)
    const bool muted = muted_.load(std::memory_order_relaxed);
    if (muted) {
        // Muted - zero output
        std::memset(outputL, 0, numFrames * sizeof(float));
        std::memset(outputR, 0, numFrames * sizeof(float));
        peakLevel_.store(0.0f, std::memory_order_relaxed);
        rmsLevel_.store(0.0f, std::memory_order_relaxed);
        return;
    }

    // Update gain coefficient
    const float targetGainLinear = dbToLinear(gainDb_.load(std::memory_order_relaxed));

    // Update pan coefficients
    const float pan = pan_.load(std::memory_order_relaxed);
    updatePanCoefficients(pan);

    // Process audio with gain and pan
    float peakSquared = 0.0f;
    float sumSquared = 0.0f;

    for (size_t i = 0; i < numFrames; ++i) {
        const float sample = input[i];
        const float gained = sample * targetGainLinear;

        outputL[i] = gained * currentPanL_;
        outputR[i] = gained * currentPanR_;

        // Metering
        const float absSample = std::abs(gained);
        peakSquared = std::max(peakSquared, absSample * absSample);
        sumSquared += gained * gained;
    }

    // Update meters
    peakLevel_.store(std::sqrt(peakSquared), std::memory_order_relaxed);
    const float rms = std::sqrt(sumSquared / static_cast<float>(numFrames));
    rmsLevel_.store(rms, std::memory_order_relaxed);

    // Smooth gain transitions (prevents clicks)
    currentGainLinear_ = targetGainLinear;
}

void ChannelStrip::setGain(float gainDb) noexcept {
    gainDb_.store(gainDb, std::memory_order_relaxed);
}

void ChannelStrip::setPan(float pan) noexcept {
    pan_.store(std::clamp(pan, -1.0f, 1.0f), std::memory_order_relaxed);
}

void ChannelStrip::setMute(bool muted) noexcept {
    muted_.store(muted, std::memory_order_relaxed);
}

void ChannelStrip::setSolo(bool soloed) noexcept {
    soloed_.store(soloed, std::memory_order_relaxed);
}

void ChannelStrip::setSendLevel(size_t sendBus, float level) noexcept {
    if (sendBus < kMaxSendBuses) {
        sendLevels_[sendBus].store(std::clamp(level, 0.0f, 1.0f), std::memory_order_relaxed);
    }
}

float ChannelStrip::getSendLevel(size_t sendBus) const {
    if (sendBus < kMaxSendBuses) {
        return sendLevels_[sendBus].load(std::memory_order_relaxed);
    }
    return 0.0f;
}

void ChannelStrip::resetMeters() noexcept {
    peakLevel_.store(0.0f, std::memory_order_relaxed);
    rmsLevel_.store(0.0f, std::memory_order_relaxed);
}

void ChannelStrip::updatePanCoefficients(float pan) noexcept {
    // Constant power pan law (-3dB center)
    // pan = -1.0 (hard left), 0.0 (center), +1.0 (hard right)
    const float angle = (pan + 1.0f) * 0.25f * 3.14159265359f;  // 0 to π/2
    currentPanL_ = std::cos(angle);
    currentPanR_ = std::sin(angle);
}

// =============================================================================
// SendBus Implementation
// =============================================================================

SendBus::SendBus() {
    std::memset(sendBufferL_.data(), 0, kMaxFramesPerBlock * sizeof(float));
    std::memset(sendBufferR_.data(), 0, kMaxFramesPerBlock * sizeof(float));
}

void SendBus::processAudio(
    float* bufferL,
    float* bufferR,
    size_t numFrames
) noexcept {
    const bool muted = muted_.load(std::memory_order_relaxed);
    if (muted) {
        return;  // Don't add anything if muted
    }

    const float returnLevel = returnLevel_.load(std::memory_order_relaxed);

    // Add send buffer to output with return level
    for (size_t i = 0; i < numFrames; ++i) {
        bufferL[i] += sendBufferL_[i] * returnLevel;
        bufferR[i] += sendBufferR_[i] * returnLevel;
    }
}

void SendBus::addSend(
    const float* sourceL,
    const float* sourceR,
    float level,
    size_t numFrames
) noexcept {
    // Accumulate send with level
    for (size_t i = 0; i < numFrames; ++i) {
        sendBufferL_[i] += sourceL[i] * level;
        sendBufferR_[i] += sourceR[i] * level;
    }
}

void SendBus::clearBuffer() noexcept {
    std::memset(sendBufferL_.data(), 0, kMaxFramesPerBlock * sizeof(float));
    std::memset(sendBufferR_.data(), 0, kMaxFramesPerBlock * sizeof(float));
}

void SendBus::setReturnLevel(float level) noexcept {
    returnLevel_.store(std::clamp(level, 0.0f, 2.0f), std::memory_order_relaxed);
}

void SendBus::setMute(bool muted) noexcept {
    muted_.store(muted, std::memory_order_relaxed);
}

// =============================================================================
// MasterBus Implementation
// =============================================================================

MasterBus::MasterBus() = default;

void MasterBus::processAudio(
    float* bufferL,
    float* bufferR,
    size_t numFrames
) noexcept {
    const float gainLinear = dbToLinear(gainDb_.load(std::memory_order_relaxed));

    // Apply gain
    float peakL = 0.0f;
    float peakR = 0.0f;

    for (size_t i = 0; i < numFrames; ++i) {
        bufferL[i] *= gainLinear;
        bufferR[i] *= gainLinear;

        peakL = std::max(peakL, std::abs(bufferL[i]));
        peakR = std::max(peakR, std::abs(bufferR[i]));
    }

    // Apply limiter if enabled
    if (limiterEnabled_.load(std::memory_order_relaxed)) {
        applyLimiter(bufferL, bufferR, numFrames);
    }

    // Update meters
    peakLevelL_.store(peakL, std::memory_order_relaxed);
    peakLevelR_.store(peakR, std::memory_order_relaxed);
}

void MasterBus::setGain(float gainDb) noexcept {
    gainDb_.store(gainDb, std::memory_order_relaxed);
}

void MasterBus::setLimiterThreshold(float thresholdDb) noexcept {
    limiterThreshold_.store(thresholdDb, std::memory_order_relaxed);
}

void MasterBus::setLimiterEnabled(bool enabled) noexcept {
    limiterEnabled_.store(enabled, std::memory_order_relaxed);
}

void MasterBus::applyLimiter(float* bufferL, float* bufferR, size_t numFrames) noexcept {
    const float threshold = dbToLinear(limiterThreshold_.load(std::memory_order_relaxed));
    constexpr float attackCoeff = 0.999f;   // Fast attack
    constexpr float releaseCoeff = 0.9999f; // Slow release

    for (size_t i = 0; i < numFrames; ++i) {
        // Calculate peak level
        const float peak = std::max(std::abs(bufferL[i]), std::abs(bufferR[i]));

        // Calculate required gain reduction
        float targetGainReduction = 1.0f;
        if (peak > threshold) {
            targetGainReduction = threshold / peak;
        }

        // Smooth gain reduction (envelope follower)
        if (targetGainReduction < limiterGainReduction_) {
            // Attack
            limiterGainReduction_ = limiterGainReduction_ * attackCoeff +
                                   targetGainReduction * (1.0f - attackCoeff);
        } else {
            // Release
            limiterGainReduction_ = limiterGainReduction_ * releaseCoeff +
                                   targetGainReduction * (1.0f - releaseCoeff);
        }

        // Apply gain reduction
        bufferL[i] *= limiterGainReduction_;
        bufferR[i] *= limiterGainReduction_;
    }
}

// =============================================================================
// MixerEngine Implementation
// =============================================================================

MixerEngine::MixerEngine(double sampleRate)
    : sampleRate_(sampleRate)
    , numChannels_(0)
    , numSendBuses_(0) {
    std::memset(mixBufferL_.data(), 0, kMaxFramesPerBlock * sizeof(float));
    std::memset(mixBufferR_.data(), 0, kMaxFramesPerBlock * sizeof(float));
}

void MixerEngine::setNumChannels(size_t numChannels) {
    numChannels_ = std::min(numChannels, kMaxChannels);
}

void MixerEngine::setNumSendBuses(size_t numBuses) {
    numSendBuses_ = std::min(numBuses, kMaxSendBuses);
}

void MixerEngine::processAudio(
    const float** inputs,
    float** outputs,
    size_t numFrames
) noexcept {
    if (numFrames > kMaxFramesPerBlock) {
        return;  // Safety check
    }

    // Clear mix buffers
    std::memset(mixBufferL_.data(), 0, numFrames * sizeof(float));
    std::memset(mixBufferR_.data(), 0, numFrames * sizeof(float));

    // Clear send buses
    for (size_t i = 0; i < numSendBuses_; ++i) {
        sendBuses_[i].clearBuffer();
    }

    // Check if any channels are soloed
    const bool anySoloed = isAnySoloed();

    // Process each channel
    std::array<float, kMaxFramesPerBlock> channelL{};
    std::array<float, kMaxFramesPerBlock> channelR{};

    for (size_t ch = 0; ch < numChannels_; ++ch) {
        const bool soloed = channels_[ch].isSoloed();
        const bool muted = channels_[ch].isMuted();

        // Skip if muted OR if another channel is soloed and this one isn't
        if (muted || (anySoloed && !soloed)) {
            continue;
        }

        // Process channel
        channels_[ch].processAudio(inputs[ch], channelL.data(), channelR.data(), numFrames);

        // Mix into main buffers
        for (size_t i = 0; i < numFrames; ++i) {
            mixBufferL_[i] += channelL[i];
            mixBufferR_[i] += channelR[i];
        }

        // Send to buses
        for (size_t bus = 0; bus < numSendBuses_; ++bus) {
            const float sendLevel = channels_[ch].getSendLevel(bus);
            if (sendLevel > 0.0f) {
                sendBuses_[bus].addSend(channelL.data(), channelR.data(), sendLevel, numFrames);
            }
        }
    }

    // Process send returns
    for (size_t bus = 0; bus < numSendBuses_; ++bus) {
        sendBuses_[bus].processAudio(mixBufferL_.data(), mixBufferR_.data(), numFrames);
    }

    // Copy to master bus buffers
    std::memcpy(outputs[0], mixBufferL_.data(), numFrames * sizeof(float));
    std::memcpy(outputs[1], mixBufferR_.data(), numFrames * sizeof(float));

    // Process master bus
    master_.processAudio(outputs[0], outputs[1], numFrames);
}

void MixerEngine::setChannelGain(size_t channel, float gainDb) noexcept {
    if (isValidChannel(channel)) {
        channels_[channel].setGain(gainDb);
    }
}

void MixerEngine::setChannelPan(size_t channel, float pan) noexcept {
    if (isValidChannel(channel)) {
        channels_[channel].setPan(pan);
    }
}

void MixerEngine::setChannelMute(size_t channel, bool muted) noexcept {
    if (isValidChannel(channel)) {
        channels_[channel].setMute(muted);
    }
}

void MixerEngine::setChannelSolo(size_t channel, bool soloed) noexcept {
    if (isValidChannel(channel)) {
        const bool wasSoloed = channels_[channel].isSoloed();
        channels_[channel].setSolo(soloed);

        // Update solo count
        if (soloed && !wasSoloed) {
            soloCount_.fetch_add(1, std::memory_order_relaxed);
        } else if (!soloed && wasSoloed) {
            soloCount_.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

void MixerEngine::setChannelSend(size_t channel, size_t sendBus, float level) noexcept {
    if (isValidChannel(channel) && isValidSendBus(sendBus)) {
        channels_[channel].setSendLevel(sendBus, level);
    }
}

void MixerEngine::setSendReturnLevel(size_t sendBus, float level) noexcept {
    if (isValidSendBus(sendBus)) {
        sendBuses_[sendBus].setReturnLevel(level);
    }
}

void MixerEngine::setSendMute(size_t sendBus, bool muted) noexcept {
    if (isValidSendBus(sendBus)) {
        sendBuses_[sendBus].setMute(muted);
    }
}

void MixerEngine::setMasterGain(float gainDb) noexcept {
    master_.setGain(gainDb);
}

void MixerEngine::setMasterLimiter(bool enabled, float thresholdDb) noexcept {
    master_.setLimiterEnabled(enabled);
    master_.setLimiterThreshold(thresholdDb);
}

bool MixerEngine::isAnySoloed() const {
    return soloCount_.load(std::memory_order_relaxed) > 0;
}

void MixerEngine::clearAllSolo() noexcept {
    for (size_t i = 0; i < numChannels_; ++i) {
        channels_[i].setSolo(false);
    }
    soloCount_.store(0, std::memory_order_relaxed);
}

float MixerEngine::getChannelPeak(size_t channel) const {
    if (isValidChannel(channel)) {
        return channels_[channel].getPeakLevel();
    }
    return 0.0f;
}

float MixerEngine::getChannelRMS(size_t channel) const {
    if (isValidChannel(channel)) {
        return channels_[channel].getRMSLevel();
    }
    return 0.0f;
}

void MixerEngine::resetAllMeters() noexcept {
    for (size_t i = 0; i < numChannels_; ++i) {
        channels_[i].resetMeters();
    }
}

} // namespace mixer
} // namespace penta
