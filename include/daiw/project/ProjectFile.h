/**
 * @file ProjectFile.h
 * @brief Project/session file serialization
 *
 * Provides serialization for DAW project state using JSON format.
 * Stores:
 * - Track configuration
 * - Mixer state (stub)
 * - Plugin settings (stub)
 * - MIDI sequences
 * - Audio file references
 */

#pragma once

#include "daiw/types.hpp"
#include "daiw/midi/MidiSequence.h"
#include <string>
#include <vector>
#include <memory>

namespace daiw {
namespace project {

/**
 * @brief Track type enumeration
 */
enum class TrackType {
    MIDI,
    Audio,
    Aux
};

/**
 * @brief Track information
 */
struct Track {
    std::string name;
    TrackType type = TrackType::MIDI;
    int index = 0;
    bool muted = false;
    bool soloed = false;
    float volume = 1.0f;    // 0.0 to 1.0
    float pan = 0.0f;       // -1.0 (left) to +1.0 (right)
    
    // MIDI-specific
    midi::MidiSequence midiSequence;
    
    // Audio-specific
    std::string audioFilePath;
    
    // Future: Plugin chain, automation, etc.
};

/**
 * @brief Mixer state (stub)
 */
struct MixerState {
    float masterVolume = 1.0f;
    bool masterMuted = false;
    
    // Future: Bus routing, sends, etc.
};

/**
 * @brief Project metadata
 */
struct ProjectMetadata {
    std::string name = "Untitled";
    std::string author;
    std::string createdDate;
    std::string modifiedDate;
    int versionMajor = 1;
    int versionMinor = 0;
};

/**
 * @brief Complete project state
 */
class ProjectFile {
public:
    /**
     * @brief Default constructor
     */
    ProjectFile();

    /**
     * @brief Load project from JSON file
     * @param filepath Path to .idaw.json file
     * @return true if successful
     */
    bool load(const std::string& filepath);

    /**
     * @brief Save project to JSON file
     * @param filepath Path to .idaw.json file
     * @return true if successful
     */
    bool save(const std::string& filepath) const;

    /**
     * @brief Get project metadata
     */
    [[nodiscard]] const ProjectMetadata& getMetadata() const {
        return metadata_;
    }

    /**
     * @brief Set project metadata
     */
    void setMetadata(const ProjectMetadata& metadata) {
        metadata_ = metadata;
    }

    /**
     * @brief Get all tracks
     */
    [[nodiscard]] const std::vector<Track>& getTracks() const {
        return tracks_;
    }

    /**
     * @brief Add a track
     */
    void addTrack(const Track& track) {
        tracks_.push_back(track);
    }

    /**
     * @brief Remove a track
     */
    void removeTrack(size_t index) {
        if (index < tracks_.size()) {
            tracks_.erase(tracks_.begin() + index);
        }
    }

    /**
     * @brief Get mixer state
     */
    [[nodiscard]] const MixerState& getMixerState() const {
        return mixer_;
    }

    /**
     * @brief Set mixer state
     */
    void setMixerState(const MixerState& mixer) {
        mixer_ = mixer;
    }

    /**
     * @brief Get project tempo
     */
    [[nodiscard]] float getTempo() const {
        return tempo_.bpm;
    }

    /**
     * @brief Set project tempo
     */
    void setTempo(float bpm) {
        tempo_.bpm = bpm;
    }

    /**
     * @brief Get time signature
     */
    [[nodiscard]] const TimeSignature& getTimeSignature() const {
        return timeSignature_;
    }

    /**
     * @brief Set time signature
     */
    void setTimeSignature(const TimeSignature& sig) {
        timeSignature_ = sig;
    }

    /**
     * @brief Get sample rate
     */
    [[nodiscard]] SampleRate getSampleRate() const {
        return sampleRate_;
    }

    /**
     * @brief Set sample rate
     */
    void setSampleRate(SampleRate rate) {
        sampleRate_ = rate;
    }

    /**
     * @brief Convert to JSON string
     */
    [[nodiscard]] std::string toJSON() const;

    /**
     * @brief Load from JSON string
     */
    bool fromJSON(const std::string& json);

private:
    ProjectMetadata metadata_;
    std::vector<Track> tracks_;
    MixerState mixer_;
    Tempo tempo_;
    TimeSignature timeSignature_;
    SampleRate sampleRate_ = DEFAULT_SAMPLE_RATE;
};

} // namespace project
} // namespace daiw
