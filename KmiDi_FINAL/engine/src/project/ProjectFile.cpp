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
    return fromJSON(buffer.str());
}

bool ProjectFile::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    file << toJSON();
    return true;
}

std::string ProjectFile::toJSON() const {
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
            trackObj->setProperty("audioFile", juce::String(track.audioFilePath));
        }

        tracks.add(juce::var(trackObj));
    }
    root->setProperty("tracks", tracks);

    juce::var rootVar(root);
    return juce::JSON::toString(rootVar, true).toStdString();
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
