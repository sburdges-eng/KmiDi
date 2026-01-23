#include "penta/harmony/HarmonyEngine.h"

namespace penta::harmony {

HarmonyEngine::HarmonyEngine(const Config& config)
    : config_(config)
{
    chordAnalyzer_ = std::make_unique<ChordAnalyzer>();
    scaleDetector_ = std::make_unique<ScaleDetector>();
    voiceLeading_ = std::make_unique<VoiceLeading>();

    activeNotes_.fill(0);
    pitchClassSet_.fill(false);

    chordHistory_.resize(kHistoryCapacity);
    scaleHistory_.resize(kHistoryCapacity);
}

HarmonyEngine::~HarmonyEngine() = default;

void HarmonyEngine::processNotes(const Note* notes, size_t count) noexcept {
    // Update active notes and pitch class set
    for (size_t i = 0; i < count; ++i) {
        const auto& note = notes[i];

        if (note.velocity > 0) {
            activeNotes_[note.pitch] = note.velocity;
            pitchClassSet_[note.pitch % 12] = true;
        } else {
            activeNotes_[note.pitch] = 0;
            // Check if this was the last note of this pitch class
            bool hasNote = false;
            for (int j = note.pitch % 12; j < 128; j += 12) {
                if (activeNotes_[j] > 0) {
                    hasNote = true;
                    break;
                }
            }
            if (!hasNote) {
                pitchClassSet_[note.pitch % 12] = false;
            }
        }
    }

    updateChordAnalysis();

    if (config_.enableScaleDetection) {
        updateScaleDetection();
    }
}

void HarmonyEngine::updateChordAnalysis() noexcept {
    chordAnalyzer_->update(pitchClassSet_);
    currentChord_ = chordAnalyzer_->getCurrentChord();

    const bool hasHistory = chordHistoryCount_ > 0;
    if (hasHistory) {
        const size_t lastIndex = (chordHistoryWriteIndex_ + kHistoryCapacity - 1) % kHistoryCapacity;
        const auto& last = chordHistory_[lastIndex];
        if (last.root == currentChord_.root &&
            last.quality == currentChord_.quality &&
            last.pitchClass == currentChord_.pitchClass) {
            return;
        }
    }

    chordHistory_[chordHistoryWriteIndex_] = currentChord_;
    chordHistoryWriteIndex_ = (chordHistoryWriteIndex_ + 1) % kHistoryCapacity;
    chordHistoryCount_ = std::min(chordHistoryCount_ + 1, kHistoryCapacity);
}

void HarmonyEngine::updateScaleDetection() noexcept {
    // Build weighted histogram from active notes
    std::array<float, 12> histogram{};
    for (size_t i = 0; i < 128; ++i) {
        if (activeNotes_[i] > 0) {
            histogram[i % 12] += activeNotes_[i] / 127.0f;
        }
    }

    scaleDetector_->update(histogram);
    currentScale_ = scaleDetector_->getCurrentScale();

    const bool hasHistory = scaleHistoryCount_ > 0;
    if (hasHistory) {
        const size_t lastIndex = (scaleHistoryWriteIndex_ + kHistoryCapacity - 1) % kHistoryCapacity;
        const auto& last = scaleHistory_[lastIndex];
        if (last.tonic == currentScale_.tonic &&
            last.mode == currentScale_.mode &&
            last.degrees == currentScale_.degrees) {
            return;
        }
    }

    scaleHistory_[scaleHistoryWriteIndex_] = currentScale_;
    scaleHistoryWriteIndex_ = (scaleHistoryWriteIndex_ + 1) % kHistoryCapacity;
    scaleHistoryCount_ = std::min(scaleHistoryCount_ + 1, kHistoryCapacity);
}

std::vector<Note> HarmonyEngine::suggestVoiceLeading(
    const Chord& targetChord,
    const std::vector<Note>& currentVoices
) const noexcept {
    if (!config_.enableVoiceLeading) {
        return {};
    }

    return voiceLeading_->findOptimalVoicing(targetChord, currentVoices);
}

void HarmonyEngine::updateConfig(const Config& config) {
    config_ = config;

    if (chordAnalyzer_) {
        chordAnalyzer_->setConfidenceThreshold(config.confidenceThreshold);
    }

    if (scaleDetector_) {
        scaleDetector_->setConfidenceThreshold(config.confidenceThreshold);
    }
}

std::vector<Chord> HarmonyEngine::getChordHistory(size_t maxCount) const {
    const size_t count = std::min(maxCount, chordHistoryCount_);
    std::vector<Chord> history;
    history.reserve(count);
    if (count == 0) {
        return history;
    }
    size_t start = (chordHistoryWriteIndex_ + kHistoryCapacity - count) % kHistoryCapacity;
    for (size_t i = 0; i < count; ++i) {
        history.push_back(chordHistory_[(start + i) % kHistoryCapacity]);
    }
    return history;
}

std::vector<Scale> HarmonyEngine::getScaleHistory(size_t maxCount) const {
    const size_t count = std::min(maxCount, scaleHistoryCount_);
    std::vector<Scale> history;
    history.reserve(count);
    if (count == 0) {
        return history;
    }
    size_t start = (scaleHistoryWriteIndex_ + kHistoryCapacity - count) % kHistoryCapacity;
    for (size_t i = 0; i < count; ++i) {
        history.push_back(scaleHistory_[(start + i) % kHistoryCapacity]);
    }
    return history;
}

} // namespace penta::harmony
