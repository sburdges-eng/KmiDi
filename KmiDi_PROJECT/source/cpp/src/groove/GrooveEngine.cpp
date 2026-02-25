#include "penta/groove/GrooveEngine.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace penta::groove
{

    GrooveEngine::GrooveEngine(const Config &config)
        : config_(config), analysis_{}, onsetDetector_(std::make_unique<OnsetDetector>()), tempoEstimator_(std::make_unique<TempoEstimator>()), quantizer_(std::make_unique<RhythmQuantizer>()), samplePosition_(0), lastAnalysisPosition_(0)
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
                pushBounded(onsetHistory_, onsetPos);

                // Feed onset to tempo estimator
                if (tempoEstimator_)
                {
                    tempoEstimator_->addOnset(onsetPos);
                }
            }
        }

        samplePosition_ += frames;

        // Periodically update tempo estimate and analysis
        // Run analysis every ~100ms worth of samples to reduce CPU load
        constexpr uint64_t kAnalysisInterval = 4410; // ~100ms at 44.1kHz

        if (samplePosition_ - lastAnalysisPosition_ >= kAnalysisInterval)
        {
            updateTempoEstimate();
            detectTimeSignature();
            analyzeSwing();
            lastAnalysisPosition_ = samplePosition_;
        }
    }

    uint64_t GrooveEngine::quantizeToGrid(uint64_t timestamp) const noexcept
    {
        if (!config_.enableQuantization || !quantizer_)
        {
            return timestamp;
        }

        // Get samples per beat from current tempo
        uint64_t samplesPerBeat = 0;
        if (analysis_.currentTempo > 0.0f)
        {
            samplesPerBeat = static_cast<uint64_t>(
                (60.0 * config_.sampleRate) / analysis_.currentTempo);
        }

        if (samplesPerBeat == 0)
        {
            return timestamp;
        }

        // Calculate bar start position (assume we're counting from 0)
        // samplesPerBar = samplesPerBeat * timeSignatureNum
        uint64_t samplesPerBar = samplesPerBeat * analysis_.timeSignatureNum;
        uint64_t barStartPosition = (timestamp / samplesPerBar) * samplesPerBar;

        return quantizer_->quantize(timestamp, samplesPerBeat, barStartPosition);
    }

    uint64_t GrooveEngine::applySwing(uint64_t position) const noexcept
    {
        if (analysis_.swing <= 0.0f || !quantizer_)
        {
            return position;
        }

        // Get samples per beat from current tempo
        uint64_t samplesPerBeat = 0;
        if (analysis_.currentTempo > 0.0f)
        {
            samplesPerBeat = static_cast<uint64_t>(
                (60.0 * config_.sampleRate) / analysis_.currentTempo);
        }

        if (samplesPerBeat == 0)
        {
            return position;
        }

        // Calculate bar start position
        uint64_t samplesPerBar = samplesPerBeat * analysis_.timeSignatureNum;
        uint64_t barStartPosition = (position / samplesPerBar) * samplesPerBar;

        return quantizer_->applySwing(position, samplesPerBeat, barStartPosition);
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
        lastAnalysisPosition_ = 0;  // Reset analysis timing to allow immediate analysis after reset
        onsetHistory_.clear();
        analysis_ = GrooveAnalysis{};
    }

    void GrooveEngine::updateTempoEstimate() noexcept
    {
        if (!tempoEstimator_)
        {
            return;
        }

        // Get tempo from estimator and update analysis
        if (tempoEstimator_->hasReliableEstimate())
        {
            analysis_.currentTempo = tempoEstimator_->getCurrentTempo();
            analysis_.tempoConfidence = tempoEstimator_->getConfidence();
        }
    }

    void GrooveEngine::detectTimeSignature() noexcept
    {
        // Need at least 8 onsets to detect time signature patterns
        if (onsetHistory_.size() < 8)
        {
            return;
        }

        // Calculate inter-onset intervals (IOI)
        std::vector<uint64_t> intervals;
        intervals.reserve(onsetHistory_.size() - 1);

        for (size_t i = 1; i < onsetHistory_.size(); ++i)
        {
            if (onsetHistory_[i] > onsetHistory_[i - 1])
            {
                intervals.push_back(onsetHistory_[i] - onsetHistory_[i - 1]);
            }
        }

        if (intervals.size() < 4)
        {
            return;
        }

        // Find the median interval as the reference beat
        std::vector<uint64_t> sortedIntervals = intervals;
        std::sort(sortedIntervals.begin(), sortedIntervals.end());
        uint64_t medianInterval = sortedIntervals[sortedIntervals.size() / 2];

        if (medianInterval == 0)
        {
            return;
        }

        // Count how many intervals are approximately 3x or 4x the median
        // This helps detect 3/4 vs 4/4 time signatures
        int count3 = 0;
        int count4 = 0;
        constexpr float kTolerance = 0.15f;

        for (uint64_t interval : intervals)
        {
            float ratio = static_cast<float>(interval) / static_cast<float>(medianInterval);

            float dist3 = std::abs(ratio - 3.0f);
            float dist4 = std::abs(ratio - 4.0f);

            // Check for 3-beat or 4-beat grouping (mutually exclusive)
            // Use else-if to prevent counting intervals in overlapping tolerance ranges twice
            if (dist3 < 3.0f * kTolerance && dist3 <= dist4)
            {
                // Closer to 3-beat grouping (or equidistant, prefer 3/4 detection)
                count3++;
            }
            else if (dist4 < 4.0f * kTolerance)
            {
                // Closer to 4-beat grouping
                count4++;
            }
        }

        // Determine time signature based on grouping patterns
        // Also look at accents in onset strengths
        if (count3 > count4 && count3 >= 2)
        {
            analysis_.timeSignatureNum = 3;
            analysis_.timeSignatureDen = 4;
        }
        else
        {
            // Default to 4/4 - most common
            analysis_.timeSignatureNum = 4;
            analysis_.timeSignatureDen = 4;
        }
    }

    void GrooveEngine::analyzeSwing() noexcept
    {
        // Need at least 4 onsets to analyze swing
        if (onsetHistory_.size() < 4)
        {
            return;
        }

        // Get samples per beat for grid reference
        uint64_t samplesPerBeat = 0;
        if (analysis_.currentTempo > 0.0f)
        {
            samplesPerBeat = static_cast<uint64_t>(
                (60.0 * config_.sampleRate) / analysis_.currentTempo);
        }

        if (samplesPerBeat == 0)
        {
            return;
        }

        // Swing is measured by how much off-beat notes are delayed
        // In a swung rhythm, the off-beats (8th notes on "and") are pushed later
        // Perfect swing ratio is around 2:1 (triplet feel)

        uint64_t eighthNote = samplesPerBeat / 2;
        if (eighthNote == 0)
        {
            return;
        }

        // Analyze timing deviations from straight 8th note grid
        std::vector<float> offbeatDeviations;
        offbeatDeviations.reserve(onsetHistory_.size());

        for (size_t i = 0; i < onsetHistory_.size(); ++i)
        {
            uint64_t position = onsetHistory_[i];

            // Find position within the beat
            uint64_t positionInBeat = position % samplesPerBeat;

            // Check if this is close to an off-beat position (halfway through beat)
            int64_t deviationFromOffbeat = static_cast<int64_t>(positionInBeat) -
                                           static_cast<int64_t>(eighthNote);

            // Only consider onsets that are roughly on the off-beat
            if (std::abs(deviationFromOffbeat) < static_cast<int64_t>(eighthNote / 2))
            {
                // Normalize deviation: positive = pushed back (swung)
                float normalizedDeviation = static_cast<float>(deviationFromOffbeat) /
                                            static_cast<float>(eighthNote);
                offbeatDeviations.push_back(normalizedDeviation);
            }
        }

        if (offbeatDeviations.empty())
        {
            analysis_.swing = 0.0f;
            return;
        }

        // Calculate average deviation
        float totalDeviation = 0.0f;
        for (float dev : offbeatDeviations)
        {
            totalDeviation += dev;
        }
        float avgDeviation = totalDeviation / static_cast<float>(offbeatDeviations.size());

        // Convert to swing amount (0.0 = straight, 1.0 = full triplet swing)
        // Full triplet swing means off-beat is at 2/3 of the beat instead of 1/2
        // That's a deviation of +0.33 (from 0.5 to 0.67)
        constexpr float kMaxSwingDeviation = 0.33f;

        // Clamp to valid range and normalize
        avgDeviation = std::clamp(avgDeviation, 0.0f, kMaxSwingDeviation);
        analysis_.swing = avgDeviation / kMaxSwingDeviation;

        // Update quantizer config with detected swing
        if (quantizer_ && analysis_.swing > 0.1f)
        {
            RhythmQuantizer::Config qConfig;
            qConfig.enableSwing = true;
            qConfig.swingAmount = analysis_.swing;
            qConfig.timeSignatureNum = analysis_.timeSignatureNum;
            qConfig.timeSignatureDen = analysis_.timeSignatureDen;
            quantizer_->updateConfig(qConfig);
        }
    }

} // namespace penta::groove
