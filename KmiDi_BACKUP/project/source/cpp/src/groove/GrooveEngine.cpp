#include "penta/groove/GrooveEngine.h"

#include <algorithm>

namespace penta::groove
{

    GrooveEngine::GrooveEngine(const Config &config)
        : config_(config), analysis_{}, onsetDetector_(std::make_unique<OnsetDetector>()), tempoEstimator_(std::make_unique<TempoEstimator>()), quantizer_(std::make_unique<RhythmQuantizer>()), samplePosition_(0)
    {
        constexpr size_t kMaxOnsetHistory = 128;

        analysis_.currentTempo = 120.0f;
        analysis_.tempoConfidence = 0.0f;
        analysis_.timeSignatureNum = 4;
        analysis_.timeSignatureDen = 4;
        analysis_.swing = 0.0f;

        // Preallocate analysis/history buffers so processAudio can remain allocation-free.
        analysis_.onsetPositions.reserve(kMaxOnsetHistory);
        analysis_.onsetStrengths.reserve(kMaxOnsetHistory);
        onsetHistory_.reserve(kMaxOnsetHistory);
        // TODO: Week 3-4 implementation
    }

    GrooveEngine::~GrooveEngine() = default;

    void GrooveEngine::processAudio(const float *buffer, size_t frames) noexcept
    {
        if (onsetDetector_)
        {
            onsetDetector_->process(buffer, frames);

            if (onsetDetector_->hasOnset())
            {
                uint64_t onsetPos = onsetDetector_->getOnsetPosition();
                float onsetStrength = onsetDetector_->getOnsetStrength();

                constexpr size_t kMaxOnsetHistory = 128;
                auto pushBounded = [](auto &vec, const auto &value)
                {
                    if (vec.size() < kMaxOnsetHistory)
                    {
                        vec.push_back(value);
                        return;
                    }
                    std::move(vec.begin() + 1, vec.end(), vec.begin());
                    vec.back() = value;
                };

                pushBounded(analysis_.onsetPositions, onsetPos);
                pushBounded(analysis_.onsetStrengths, onsetStrength);
            }
        }

        samplePosition_ += frames;
    }

    uint64_t GrooveEngine::quantizeToGrid(uint64_t timestamp) const noexcept
    {
        // Stub implementation
        return timestamp;
    }

    uint64_t GrooveEngine::applySwing(uint64_t position) const noexcept
    {
        // Stub implementation
        return position;
    }

    void GrooveEngine::updateConfig(const Config &config)
    {
        config_ = config;
    }

    void GrooveEngine::reset()
    {
        if (onsetDetector_)
            onsetDetector_->reset();
        if (tempoEstimator_)
            tempoEstimator_->reset();
        samplePosition_ = 0;
        onsetHistory_.clear();
        analysis_ = GrooveAnalysis{};
    }

    void GrooveEngine::updateTempoEstimate() noexcept
    {
        // Stub implementation - TODO Week 3
    }

    void GrooveEngine::detectTimeSignature() noexcept
    {
        // Stub implementation - TODO Week 3
    }

    void GrooveEngine::analyzeSwing() noexcept
    {
        // Stub implementation - TODO Week 4
    }

} // namespace penta::groove
