#include "penta/groove/GrooveEngine.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace penta::groove
{

    GrooveEngine::GrooveEngine(const Config &config)
        : config_(config), analysis_{}, samplePosition_(0)
    {
        // Keep analysis/history buffers bounded and preallocated so the audio thread
        // never performs dynamic allocations.
        constexpr size_t kMaxOnsetHistory = 128;

        analysis_.currentTempo = 120.0f;
        analysis_.tempoConfidence = 0.0f;
        analysis_.timeSignatureNum = 4;
        analysis_.timeSignatureDen = 4;
        analysis_.swing = 0.0f;

        analysis_.onsetPositions.reserve(kMaxOnsetHistory);
        analysis_.onsetStrengths.reserve(kMaxOnsetHistory);
        onsetHistory_.reserve(kMaxOnsetHistory);

        // Configure onset detector
        OnsetDetector::Config onsetConfig;
        onsetConfig.sampleRate = config_.sampleRate;
        onsetConfig.hopSize = config_.hopSize;
        onsetDetector_ = std::make_unique<OnsetDetector>(onsetConfig);

        // Configure tempo estimator
        TempoEstimator::Config tempoConfig;
        tempoConfig.sampleRate = config_.sampleRate;
        tempoConfig.minTempo = config_.minTempo;
        tempoConfig.maxTempo = config_.maxTempo;
        tempoEstimator_ = std::make_unique<TempoEstimator>(tempoConfig);

        // Configure quantizer
        RhythmQuantizer::Config quantConfig;
        quantConfig.strength = config_.quantizationStrength;
        quantConfig.timeSignatureNum = analysis_.timeSignatureNum;
        quantConfig.timeSignatureDen = analysis_.timeSignatureDen;
        quantizer_ = std::make_unique<RhythmQuantizer>(quantConfig);
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

                // Keep a bounded history for tempo/time signature estimation.
                // IMPORTANT: avoid push_back growth and avoid erase(begin()) on the
                // audio thread.
                constexpr size_t kMaxOnsetHistory = 128;
                auto pushBounded = [](auto &vec, const auto &value)
                {
                    if (vec.size() < kMaxOnsetHistory)
                    {
                        vec.push_back(value);
                        return;
                    }
                    // Shift left by one and append at end (fixed O(N), no alloc).
                    std::move(vec.begin() + 1, vec.end(), vec.begin());
                    vec.back() = value;
                };

                pushBounded(analysis_.onsetPositions, onsetPos);
                pushBounded(analysis_.onsetStrengths, onsetStrength);
                pushBounded(onsetHistory_, onsetPos);

                // Update tempo estimate in real-time
                if (tempoEstimator_)
                {
                    tempoEstimator_->addOnset(onsetPos);
                    analysis_.currentTempo = tempoEstimator_->getCurrentTempo();
                    analysis_.tempoConfidence = tempoEstimator_->getConfidence();
                }

                // Update time signature and swing analyses once we have data
                detectTimeSignature();
                analyzeSwing();

                // Keep quantizer in sync with latest analysis
                if (quantizer_)
                {
                    RhythmQuantizer::Config qConfig;
                    qConfig.strength = config_.quantizationStrength;
                    qConfig.enableSwing = true;
                    qConfig.swingAmount = std::clamp(0.5f + (analysis_.swing * 0.25f), 0.0f, 1.0f);
                    qConfig.timeSignatureNum = analysis_.timeSignatureNum;
                    qConfig.timeSignatureDen = analysis_.timeSignatureDen;
                    quantizer_->updateConfig(qConfig);
                }
            }
        }

        samplePosition_ += frames;
    }

    uint64_t GrooveEngine::quantizeToGrid(uint64_t timestamp) const noexcept
    {
        if (!quantizer_ || !config_.enableQuantization || analysis_.currentTempo <= 0.0f)
        {
            return timestamp;
        }

        // Calculate samples per beat based on current tempo
        uint64_t samplesPerBeat = static_cast<uint64_t>(
            (60.0 * config_.sampleRate) / analysis_.currentTempo);

        if (samplesPerBeat == 0)
        {
            return timestamp;
        }

        // Calculate bar start position (assume 4 beats per bar for now)
        uint64_t samplesPerBar = samplesPerBeat * analysis_.timeSignatureNum;
        uint64_t barStartPosition = (timestamp / samplesPerBar) * samplesPerBar;

        // Use the RhythmQuantizer to quantize to grid
        return quantizer_->quantize(timestamp, samplesPerBeat, barStartPosition);
    }

    uint64_t GrooveEngine::applySwing(uint64_t position) const noexcept
    {
        if (!quantizer_ || analysis_.currentTempo <= 0.0f)
        {
            return position;
        }

        // Calculate samples per beat based on current tempo
        uint64_t samplesPerBeat = static_cast<uint64_t>(
            (60.0 * config_.sampleRate) / analysis_.currentTempo);

        if (samplesPerBeat == 0)
        {
            return position;
        }

        // Calculate bar start position
        uint64_t samplesPerBar = samplesPerBeat * analysis_.timeSignatureNum;
        uint64_t barStartPosition = (position / samplesPerBar) * samplesPerBar;

        // Use the RhythmQuantizer to apply swing timing
        return quantizer_->applySwing(position, samplesPerBeat, barStartPosition);
    }

    void GrooveEngine::updateConfig(const Config &config)
    {
        config_ = config;

        OnsetDetector::Config onsetConfig;
        onsetConfig.sampleRate = config_.sampleRate;
        onsetConfig.hopSize = config_.hopSize;
        onsetDetector_ = std::make_unique<OnsetDetector>(onsetConfig);

        TempoEstimator::Config tempoConfig;
        tempoConfig.sampleRate = config_.sampleRate;
        tempoConfig.minTempo = config_.minTempo;
        tempoConfig.maxTempo = config_.maxTempo;
        tempoEstimator_ = std::make_unique<TempoEstimator>(tempoConfig);

        if (quantizer_)
        {
            RhythmQuantizer::Config qConfig;
            qConfig.strength = config_.quantizationStrength;
            qConfig.enableSwing = true;
            qConfig.timeSignatureNum = analysis_.timeSignatureNum;
            qConfig.timeSignatureDen = analysis_.timeSignatureDen;
            qConfig.swingAmount = std::clamp(0.5f + (analysis_.swing * 0.25f), 0.0f, 1.0f);
            quantizer_->updateConfig(qConfig);
        }
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
        analysis_.currentTempo = 120.0f;
        analysis_.tempoConfidence = 0.0f;
        analysis_.timeSignatureNum = 4;
        analysis_.timeSignatureDen = 4;
        analysis_.swing = 0.0f;
    }

    void GrooveEngine::updateTempoEstimate() noexcept
    {
        if (!tempoEstimator_)
        {
            return;
        }

        tempoEstimator_->reset();
        for (uint64_t onset : onsetHistory_)
        {
            tempoEstimator_->addOnset(onset);
        }

        analysis_.currentTempo = tempoEstimator_->getCurrentTempo();
        analysis_.tempoConfidence = tempoEstimator_->getConfidence();
    }

    void GrooveEngine::detectTimeSignature() noexcept
    {
        if (analysis_.onsetPositions.size() < 8 || analysis_.currentTempo <= 0.0f)
        {
            return; // Need enough onsets for pattern detection
        }

        double samplesPerBeat = (60.0 * config_.sampleRate) / analysis_.currentTempo;
        if (samplesPerBeat <= 0.0)
        {
            return;
        }

        // Build a simple beat-strength histogram relative to the first onset
        constexpr size_t kMaxBeats = 64;
        std::array<float, kMaxBeats> beatStrengths{};
        uint64_t firstOnset = analysis_.onsetPositions.front();

        for (size_t i = 0; i < analysis_.onsetPositions.size(); ++i)
        {
            uint64_t onset = analysis_.onsetPositions[i];
            float strength = (i < analysis_.onsetStrengths.size()) ? analysis_.onsetStrengths[i] : 1.0f;

            auto beatIndex = static_cast<size_t>(std::llround(
                static_cast<double>(onset - firstOnset) / samplesPerBeat));

            if (beatIndex < kMaxBeats)
            {
                beatStrengths[beatIndex] += strength;
            }
        }

        float totalStrength = 0.0f;
        for (float v : beatStrengths)
        {
            totalStrength += v;
        }
        if (totalStrength <= 0.0f)
        {
            return;
        }

        // Score common signatures by how regularly strong beats repeat
        const int candidates[] = {2, 3, 4, 6};
        float bestScore = -1.0f;
        int bestNum = 4;

        for (int candidate : candidates)
        {
            float downbeatStrength = 0.0f;
            for (size_t i = 0; i < beatStrengths.size(); ++i)
            {
                if (i % candidate == 0)
                {
                    downbeatStrength += beatStrengths[i];
                }
            }

            float score = downbeatStrength / totalStrength;
            if (score > bestScore)
            {
                bestScore = score;
                bestNum = candidate;
            }
        }

        analysis_.timeSignatureNum = static_cast<uint32_t>(bestNum);
        analysis_.timeSignatureDen = 4; // Focus on simple meters for now
    }

    void GrooveEngine::analyzeSwing() noexcept
    {
        if (analysis_.onsetPositions.size() < 4 || analysis_.currentTempo <= 0.0f)
        {
            analysis_.swing = 0.0f;
            return; // Need more data
        }

        const double beatIntervalSamples = (60.0 / analysis_.currentTempo) * config_.sampleRate;
        if (beatIntervalSamples <= 0.0)
        {
            return;
        }

        double sumOffBeat = 0.0;
        size_t countOffBeat = 0;

        for (uint64_t onset : analysis_.onsetPositions)
        {
            double beatPos = std::fmod(static_cast<double>(onset), beatIntervalSamples) / beatIntervalSamples;

            // Focus on eighth-note upbeats (around 0.5 position in the beat)
            if (beatPos > 0.3 && beatPos < 0.7)
            {
                sumOffBeat += beatPos;
                ++countOffBeat;
            }
        }

        if (countOffBeat == 0)
        {
            analysis_.swing = 0.0f;
            return;
        }

        float avgOffBeat = static_cast<float>(sumOffBeat / static_cast<double>(countOffBeat));

        // Swing ratio: 0.5 = straight; >0.5 = delayed upbeat
        float swing = (avgOffBeat - 0.5f) * 2.0f; // Map to -1 .. 1
        analysis_.swing = std::clamp(swing, -1.0f, 1.0f);
    }

} // namespace penta::groove
