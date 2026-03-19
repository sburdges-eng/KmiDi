/**
 * @file ProjectFile.cpp
 * @brief Project file implementation with JSON serialization
 *
 * Uses simple JSON formatting for now.
 * For production: consider using a JSON library (nlohmann/json, rapidjson, etc.)
 */

#include "daiw/project/ProjectFile.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <juce_core/juce_core.h>

namespace daiw {
namespace project {

ProjectFile::ProjectFile() {
    tempo_.bpm = 120.0f;
    timeSignature_.numerator = 4;
    timeSignature_.denominator = 4;
}

bool ProjectFile::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    if (!fromJSON(buffer.str())) {
        return false;
    }

    // After loading, ensure all audio paths are absolute (resolved relative to the project file)
    juce::File projectFile(filepath);
    for (auto& track : tracks_) {
        if (track.type == TrackType::Audio && !track.audioFilePath.empty()) {
            juce::File audioFile = projectFile.getSiblingFile(track.audioFilePath);
            track.audioFilePath = audioFile.getFullPathName().toStdString();
        }
    }

    return verifyInvariants();
}

bool ProjectFile::save(const std::string& filepath) const {
    // Before saving, temporarily relativize audio paths for the JSON output
    juce::File projectFile(filepath);
    
    juce::DynamicObject::Ptr root = new juce::DynamicObject();

    juce::DynamicObject::Ptr metadata = new juce::DynamicObject();
    metadata->setProperty("name", juce::String(metadata_.name));
    metadata->setProperty("author", juce::String(metadata_.author));
    metadata->setProperty("created", juce::String(metadata_.createdDate));
    metadata->setProperty("modified", juce::String(metadata_.modifiedDate));
    metadata->setProperty(
        "version",
        juce::String(metadata_.versionMajor) + "." + juce::String(metadata_.versionMinor)
    );
    metadata->setProperty("schemaVersion", metadata_.schemaVersion);
    root->setProperty("metadata", juce::var(metadata));

    juce::DynamicObject::Ptr settings = new juce::DynamicObject();
    settings->setProperty("tempo", tempo_.bpm);
    settings->setProperty(
        "timeSignature",
        juce::String(static_cast<int>(timeSignature_.numerator)) + "/" +
            juce::String(static_cast<int>(timeSignature_.denominator))
    );
    settings->setProperty("sampleRate", static_cast<int>(sampleRate_));
    root->setProperty("settings", juce::var(settings));

    juce::DynamicObject::Ptr mixer = new juce::DynamicObject();
    mixer->setProperty("masterVolume", mixer_.masterVolume);
    mixer->setProperty("masterMuted", mixer_.masterMuted);
    root->setProperty("mixer", juce::var(mixer));

    juce::Array<juce::var> tracks;
    for (const auto& track : tracks_) {
        juce::DynamicObject::Ptr trackObj = new juce::DynamicObject();
        trackObj->setProperty("name", juce::String(track.name));
        trackObj->setProperty(
            "type",
            track.type == TrackType::MIDI ? "midi" :
                track.type == TrackType::Audio ? "audio" : "aux"
        );
        trackObj->setProperty("index", track.index);
        trackObj->setProperty("muted", track.muted);
        trackObj->setProperty("soloed", track.soloed);
        trackObj->setProperty("volume", track.volume);
        trackObj->setProperty("pan", track.pan);

        if (track.type == TrackType::MIDI) {
            trackObj->setProperty("midiEvents", static_cast<int>(track.midiSequence.size()));
        } else if (track.type == TrackType::Audio) {
            // Relativize path for freeze-safe portability
            juce::File absPath(track.audioFilePath);
            juce::String relPath = absPath.getRelativePathFrom(projectFile.getParentDirectory());
            trackObj->setProperty("audioFile", relPath);
        }

        tracks.add(juce::var(trackObj));
    }
    root->setProperty("tracks", tracks);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    juce::var rootVar(root);
    file << juce::JSON::toString(rootVar, true).toStdString();
    return true;
}

bool ProjectFile::verifyInvariants() const {
    // 1. Schema version must be current (or compatible)
    if (metadata_.schemaVersion < ProjectMetadata::SCHEMA_VERSION) {
        return false;
    }

    // 2. Tempo must be reasonable
    if (tempo_.bpm < 20.0f || tempo_.bpm > 999.0f) {
        return false;
    }

    // 3. Each track must have a name
    for (const auto& track : tracks_) {
        if (track.name.empty()) {
            return false;
        }
    }

    // 4. Volume levels must be normalized (0..2 for headroom)
    if (mixer_.masterVolume < 0.0f || mixer_.masterVolume > 2.0f) {
        return false;
    }

    return true;
}

std::string ProjectFile::toJSON() const {
    // Note: Use save() for relative path logic; toJSON is for raw string export
    juce::DynamicObject::Ptr root = new juce::DynamicObject();
    // ... rest of implementation (matching save() metadata) ...
    return ""; // Should be implemented if needed outside save()
}

bool ProjectFile::fromJSON(const std::string& json) {
    if (json.empty()) {
        return false;
    }

    juce::var rootVar = juce::JSON::parse(juce::String(json));
    if (rootVar.isVoid() || !rootVar.isObject()) {
        return false;
    }

    auto* rootObj = rootVar.getDynamicObject();
    auto metadataVar = rootObj->getProperty("metadata");
    if (auto* metadata = metadataVar.getDynamicObject()) {
        metadata_.name = metadata->getProperty("name").toString().toStdString();
        metadata_.author = metadata->getProperty("author").toString().toStdString();
        metadata_.createdDate = metadata->getProperty("created").toString().toStdString();
        metadata_.modifiedDate = metadata->getProperty("modified").toString().toStdString();
        metadata_.schemaVersion = static_cast<int>(metadata->getProperty("schemaVersion"));
    }


    auto settingsVar = rootObj->getProperty("settings");
    if (auto* settings = settingsVar.getDynamicObject()) {
        tempo_.bpm = static_cast<float>(settings->getProperty("tempo"));
        auto timeSig = settings->getProperty("timeSignature").toString().toStdString();
        auto slashPos = timeSig.find('/');
        if (slashPos != std::string::npos) {
            try {
                timeSignature_.numerator =
                    static_cast<uint8_t>(std::stoi(timeSig.substr(0, slashPos)));
                timeSignature_.denominator =
                    static_cast<uint8_t>(std::stoi(timeSig.substr(slashPos + 1)));
            } catch (...) {
                // Keep defaults
            }
        }
        sampleRate_ = static_cast<SampleRate>(static_cast<int>(settings->getProperty("sampleRate")));
    }

    auto mixerVar = rootObj->getProperty("mixer");
    if (auto* mixer = mixerVar.getDynamicObject()) {
        mixer_.masterVolume = static_cast<float>(mixer->getProperty("masterVolume"));
        mixer_.masterMuted = static_cast<bool>(mixer->getProperty("masterMuted"));
    }

    tracks_.clear();
    auto tracksVar = rootObj->getProperty("tracks");
    if (tracksVar.isArray()) {
        const auto* trackArray = tracksVar.getArray();
        for (const auto& trackVar : *trackArray) {
            if (!trackVar.isObject()) {
                continue;
            }
            Track track;
            auto* trackObj = trackVar.getDynamicObject();
            track.name = trackObj->getProperty("name").toString().toStdString();
            auto typeStr = trackObj->getProperty("type").toString();
            if (typeStr == "audio") {
                track.type = TrackType::Audio;
            } else if (typeStr == "aux") {
                track.type = TrackType::Aux;
            } else {
                track.type = TrackType::MIDI;
            }
            track.index = static_cast<int>(trackObj->getProperty("index"));
            track.muted = static_cast<bool>(trackObj->getProperty("muted"));
            track.soloed = static_cast<bool>(trackObj->getProperty("soloed"));
            track.volume = static_cast<float>(trackObj->getProperty("volume"));
            track.pan = static_cast<float>(trackObj->getProperty("pan"));
            track.audioFilePath =
                trackObj->getProperty("audioFile").toString().toStdString();

            tracks_.push_back(track);
        }
    }

    return true;
}

} // namespace project
} // namespace daiw
