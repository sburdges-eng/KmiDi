#include "plugin/PluginProcessor.h"
#include "plugin/PluginEditor.h"
#include "midi/ChordGenerator.h"
#include "midi/MidiBuilder.h"
#include "engine/StateMachineConductor.h"
#include "common/GuardrailValidator.h"
#include "common/DebugLog.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <sstream>

namespace kelly {

namespace {

StateMachineConductor& conductor() {
    static StateMachineConductor c;
    return c;
}

OutputLayerAuthorization permissiveAuthorization(const GeneratedMidi& midi) {
    OutputLayerAuthorization auth;
    auth.chordsAuthorized = true;
    auth.melodyAuthorized = true;
    auth.bassAuthorized = true;
    auth.velocitiesAuthorized = true;
    auth.rhythmDensityIncreaseAuthorized = true;
    auth.maxVoiceCount = 8;
    auth.maxChordCount = midi.chords.size();
    auth.maxRhythmicDensity = 0.0;
    return auth;
}

std::optional<ValidatedOutputPlan> validatedBoundaryPlan(const GeneratedMidi& midi) {
    auto validation = OutputPlanValidator::validate(GeneratedMidi(midi), permissiveAuthorization(midi));
    if (!validation.validated.has_value()) {
        return std::nullopt;
    }
    return std::move(*validation.validated);
}

std::string toLowerCopy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string trimCopy(const std::string& text) {
    const size_t start = text.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return {};
    }
    const size_t end = text.find_last_not_of(" \t\n\r");
    return text.substr(start, end - start + 1);
}

bool containsAnyToken(const std::string& lower, const std::initializer_list<const char*> tokens) {
    for (const auto* token : tokens) {
        if (lower.find(token) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void applyPitchVariation(GeneratedMidi& result, float intensity) {
    if (result.chords.empty()) {
        return;
    }
    const int semitoneShift = std::clamp(static_cast<int>(std::round(1.0f + intensity * 3.0f)), 1, 4);
    for (size_t i = 0; i < result.chords.size(); ++i) {
        auto& chord = result.chords[i];
        if (chord.pitches.empty()) {
            continue;
        }
        const int dir = (i % 2 == 0) ? 1 : -1;
        const size_t voice = chord.pitches.size() - 1;
        chord.pitches[voice] = std::clamp(chord.pitches[voice] + (dir * semitoneShift), 0, 127);
        std::sort(chord.pitches.begin(), chord.pitches.end());
    }
}

void applyTimingVariation(GeneratedMidi& result, float humanizeAmount) {
    if (result.chords.empty() || humanizeAmount <= 0.0f) {
        return;
    }
    const double maxShift = static_cast<double>(humanizeAmount) * 0.2;
    for (size_t i = 0; i < result.chords.size(); ++i) {
        auto& chord = result.chords[i];
        const double phase = static_cast<double>(i % 4);
        const double shift = (phase == 0.0) ? maxShift
                          : (phase == 1.0) ? (-maxShift * 0.55)
                          : (phase == 2.0) ? (maxShift * 0.35)
                                            : (-maxShift * 0.2);
        chord.startBeat = std::max(0.0, chord.startBeat + shift);
        const double durationScale = 1.0 + (phase == 0.0 ? 0.05 : -0.03) * humanizeAmount;
        chord.duration = std::max(0.25, chord.duration * durationScale);
    }
}

std::vector<int> buildDynamicsContour(const GeneratedMidi& result, float intensity, float humanizeAmount) {
    std::vector<int> velocities;
    velocities.reserve(result.chords.size());
    const int base = std::clamp(static_cast<int>(std::round(56.0f + intensity * 44.0f)), 1, 127);
    const int shape = std::clamp(static_cast<int>(std::round(6.0f + humanizeAmount * 8.0f)), 4, 14);
    for (size_t i = 0; i < result.chords.size(); ++i) {
        const int offset = (i % 2 == 0) ? shape : -shape / 2;
        velocities.push_back(std::clamp(base + offset, 1, 127));
    }
    return velocities;
}

void recomputeLengthInBeats(GeneratedMidi& result) {
    result.lengthInBeats = 0.0;
    for (const auto& chord : result.chords) {
        result.lengthInBeats = std::max(result.lengthInBeats, chord.startBeat + chord.duration);
    }
    if (result.melody.has_value()) {
        for (const auto& note : *result.melody) {
            result.lengthInBeats = std::max(result.lengthInBeats, note.startBeat + note.duration);
        }
    }
    if (result.bass.has_value()) {
        for (const auto& note : *result.bass) {
            result.lengthInBeats = std::max(result.lengthInBeats, note.startBeat + note.duration);
        }
    }
    if (result.lengthInBeats <= 0.0) {
        result.lengthInBeats = 16.0;
    }
}

float resolveOutputBpm(double hostBpm, bool tempoLockEnabled, float tempoMultiplier) {
    const float baseBpm = static_cast<float>(hostBpm > 0.0 ? hostBpm : 120.0);
    if (tempoLockEnabled) {
        return baseBpm;
    }
    const float scaled = baseBpm * std::clamp(tempoMultiplier, 0.5f, 1.8f);
    return std::clamp(scaled, 40.0f, 220.0f);
}

uint64_t fnv1a64(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char c : value) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string toHex64(uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << value;
    return oss.str();
}

struct CanonicalIntentPayload {
    std::string seed;
    std::string text;
    std::array<float, 4> magnitudes{};
};

enum class CanonicalIntentParseStatus : uint8_t {
    NotJsonInput = 0,
    Valid,
    Invalid
};

struct CanonicalIntentParseResult {
    CanonicalIntentParseStatus status = CanonicalIntentParseStatus::NotJsonInput;
    CanonicalIntentPayload payload{};
    std::string reason;
};

bool isCanonicalIntentJsonInput(const std::string& text) {
    const std::string trimmed = trimCopy(text);
    return !trimmed.empty() && trimmed.front() == '{';
}

bool isAllowedKey(const juce::String& key, std::initializer_list<const char*> allowed) {
    for (const auto* candidate : allowed) {
        if (key == candidate) {
            return true;
        }
    }
    return false;
}

bool parseMagnitude(const juce::var& value, float& outMagnitude) {
    double asDouble = 0.0;
    if (value.isDouble()) {
        asDouble = static_cast<double>(value);
    } else if (value.isInt() || value.isInt64()) {
        asDouble = static_cast<double>(static_cast<int64_t>(value));
    } else {
        return false;
    }

    if (!std::isfinite(asDouble) || asDouble < 0.0 || asDouble > 1.0) {
        return false;
    }

    outMagnitude = static_cast<float>(asDouble);
    return true;
}

std::string canonicalIntentHash(const CanonicalIntentPayload& payload) {
    std::ostringstream normalized;
    normalized << "schemaVersion=1.0"
               << "|seed=" << payload.seed
               << "|text=" << payload.text
               << "|harmonic=" << std::fixed << std::setprecision(6) << payload.magnitudes[0]
               << "|rhythmic=" << std::fixed << std::setprecision(6) << payload.magnitudes[1]
               << "|dynamic=" << std::fixed << std::setprecision(6) << payload.magnitudes[2]
               << "|tempo=" << std::fixed << std::setprecision(6) << payload.magnitudes[3];
    return toHex64(fnv1a64(normalized.str()));
}

CanonicalIntentParseResult parseCanonicalIntentPayload(const std::string& rawText) {
    CanonicalIntentParseResult result;
    if (!isCanonicalIntentJsonInput(rawText)) {
        return result;
    }

    const juce::var parsed = juce::JSON::parse(juce::String(rawText));
    if (parsed.isVoid()) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "malformed_json";
        return result;
    }

    auto* root = parsed.getDynamicObject();
    if (root == nullptr) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "intent_root_not_object";
        return result;
    }

    const auto topAllowed = {"schemaVersion", "seed", "text", "intent"};
    for (int i = 0; i < root->getProperties().size(); ++i) {
        const juce::Identifier key = root->getProperties().getName(i);
        if (!isAllowedKey(key.toString(), topAllowed)) {
            result.status = CanonicalIntentParseStatus::Invalid;
            result.reason = "unknown_top_level_field:" + key.toString().toStdString();
            return result;
        }
    }

    if (!root->hasProperty("schemaVersion")
        || !root->hasProperty("seed")
        || !root->hasProperty("text")
        || !root->hasProperty("intent")) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "missing_required_top_level_field";
        return result;
    }

    const auto schemaVersion = root->getProperty("schemaVersion");
    if (!schemaVersion.isString() || schemaVersion.toString() != "1.0") {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "unsupported_schema_version";
        return result;
    }

    const auto seedValue = root->getProperty("seed");
    if (!seedValue.isString()) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "seed_not_string";
        return result;
    }

    result.payload.seed = trimCopy(seedValue.toString().toStdString());
    if (result.payload.seed.empty()) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "missing_seed";
        return result;
    }

    const auto textValue = root->getProperty("text");
    if (!textValue.isString()) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "text_not_string";
        return result;
    }

    result.payload.text = trimCopy(textValue.toString().toStdString());
    if (result.payload.text.empty()) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "missing_text";
        return result;
    }

    auto* intentObject = root->getProperty("intent").getDynamicObject();
    if (intentObject == nullptr) {
        result.status = CanonicalIntentParseStatus::Invalid;
        result.reason = "intent_not_object";
        return result;
    }

    const auto intentAllowed = {"harmonic", "rhythmic", "dynamic", "tempo"};
    for (int i = 0; i < intentObject->getProperties().size(); ++i) {
        const juce::Identifier key = intentObject->getProperties().getName(i);
        if (!isAllowedKey(key.toString(), intentAllowed)) {
            result.status = CanonicalIntentParseStatus::Invalid;
            result.reason = "unknown_intent_field:" + key.toString().toStdString();
            return result;
        }
    }

    static constexpr std::array<const char*, 4> kOrderedAspectKeys = {"harmonic", "rhythmic", "dynamic", "tempo"};
    for (size_t idx = 0; idx < kOrderedAspectKeys.size(); ++idx) {
        const auto* key = kOrderedAspectKeys[idx];
        if (!intentObject->hasProperty(key)) {
            result.status = CanonicalIntentParseStatus::Invalid;
            result.reason = std::string("missing_intent_field:") + key;
            return result;
        }
        float magnitude = 0.0f;
        if (!parseMagnitude(intentObject->getProperty(key), magnitude)) {
            result.status = CanonicalIntentParseStatus::Invalid;
            result.reason = std::string("invalid_intent_magnitude:") + key;
            return result;
        }
        result.payload.magnitudes[idx] = magnitude;
    }

    result.status = CanonicalIntentParseStatus::Valid;
    return result;
}

std::string canonicalMidiHash(const GeneratedMidi& midi) {
    std::ostringstream normalized;
    normalized << std::fixed << std::setprecision(6);
    normalized << "bpm=" << midi.bpm << "|len=" << midi.lengthInBeats;
    normalized << "|chords=" << midi.chords.size();
    for (const auto& chord : midi.chords) {
        normalized << "|c@" << chord.startBeat << "," << chord.duration << ":";
        for (const int pitch : chord.pitches) {
            normalized << pitch << ",";
        }
    }

    normalized << "|velocities=";
    if (midi.chordVelocities.has_value()) {
        for (const int velocity : *midi.chordVelocities) {
            normalized << velocity << ",";
        }
    } else {
        normalized << "none";
    }

    normalized << "|melody=";
    if (midi.melody.has_value()) {
        for (const auto& note : *midi.melody) {
            normalized << note.pitch << "," << note.velocity << "," << note.startBeat << "," << note.duration << ";";
        }
    } else {
        normalized << "none";
    }

    normalized << "|bass=";
    if (midi.bass.has_value()) {
        for (const auto& note : *midi.bass) {
            normalized << note.pitch << "," << note.velocity << "," << note.startBeat << "," << note.duration << ";";
        }
    } else {
        normalized << "none";
    }

    return toHex64(fnv1a64(normalized.str()));
}

} // namespace

PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "KellyParams", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_INTENSITY, 1},
        "Intensity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.7f
    ));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_HUMANIZE, 1},
        "Humanize",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        0.0f
    ));
    
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_TEMPO_LOCK, 1},
        "Lock to DAW Tempo",
        true
    ));
    
    return {params.begin(), params.end()};
}

void PluginProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    currentSampleRate_ = sampleRate;
    
    // Get tempo from host if available
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto bpm = position->getBpm()) {
                currentBpm_ = *bpm;
            }
        }
    }
}

void PluginProcessor::releaseResources() {
    stopPlayback();
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    
    // Clear audio (we're MIDI-only)
    buffer.clear();
    
    // Get host transport info
    double hostBpm = 120.0;
    double hostBeat = 0.0;
    bool hostPlaying = false;
    
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto bpm = position->getBpm()) {
                hostBpm = *bpm;
                currentBpm_ = hostBpm;
            }
            if (auto ppq = position->getPpqPosition()) {
                hostBeat = *ppq;
            }
            hostPlaying = position->getIsPlaying();
        }
    }
    
    // If host just started playing, reset our playback position
    if (hostPlaying && !wasHostPlaying_) {
        playbackPositionBeats_ = hostBeat;
        
        // Reset note states when transport restarts
        std::lock_guard<std::mutex> lock(notesMutex_);
        for (auto& note : scheduledNotes_) {
            note.noteOnSent = false;
            note.noteOffSent = false;
        }
    }
    wasHostPlaying_ = hostPlaying;
    
    // Only output MIDI if we're playing and host is playing
    if (!isPlaying_ || !hostPlaying) {
        return;
    }
    
    // Calculate beat range for this buffer
    double samplesPerBeat = (currentSampleRate_ * 60.0) / hostBpm;
    double beatsPerSample = 1.0 / samplesPerBeat;
    double bufferBeats = buffer.getNumSamples() * beatsPerSample;
    
    double blockStartBeat = hostBeat;
    double blockEndBeat = hostBeat + bufferBeats;
    
    // Process scheduled notes
    std::lock_guard<std::mutex> lock(notesMutex_);
    
    for (auto& note : scheduledNotes_) {
        // Note On
        if (!note.noteOnSent && note.startBeat >= blockStartBeat && note.startBeat < blockEndBeat) {
            int sampleOffset = static_cast<int>((note.startBeat - blockStartBeat) * samplesPerBeat);
            sampleOffset = std::clamp(sampleOffset, 0, buffer.getNumSamples() - 1);
            
            midiMessages.addEvent(
                juce::MidiMessage::noteOn(1, note.pitch, static_cast<juce::uint8>(note.velocity)),
                sampleOffset
            );
            note.noteOnSent = true;
        }
        
        // Note Off
        if (note.noteOnSent && !note.noteOffSent && note.endBeat >= blockStartBeat && note.endBeat < blockEndBeat) {
            int sampleOffset = static_cast<int>((note.endBeat - blockStartBeat) * samplesPerBeat);
            sampleOffset = std::clamp(sampleOffset, 0, buffer.getNumSamples() - 1);
            
            midiMessages.addEvent(
                juce::MidiMessage::noteOff(1, note.pitch),
                sampleOffset
            );
            note.noteOffSent = true;
        }
    }
    
    // Check if all notes have been played (loop or stop)
    bool allDone = true;
    for (const auto& note : scheduledNotes_) {
        if (!note.noteOffSent) {
            allDone = false;
            break;
        }
    }
    
    // Loop playback - reset all notes when done
    if (allDone && !scheduledNotes_.empty()) {
        for (auto& note : scheduledNotes_) {
            note.noteOnSent = false;
            note.noteOffSent = false;
        }
    }
}

void PluginProcessor::scheduleNotesForPlayback() {
    const auto plan = validatedBoundaryPlan(lastGenerated_);
    if (!plan.has_value()) {
        return;
    }
    const auto& midi = plan->midi();

    std::lock_guard<std::mutex> lock(notesMutex_);
    scheduledNotes_.clear();
    
    // Get current host position as our start point
    double startOffset = 0.0;
    if (auto* playHead = getPlayHead()) {
        if (auto position = playHead->getPosition()) {
            if (auto ppq = position->getPpqPosition()) {
                startOffset = *ppq;
            }
        }
    }
    
    // Schedule all chord notes
    for (size_t chordIndex = 0; chordIndex < midi.chords.size(); ++chordIndex) {
        const auto& chord = midi.chords[chordIndex];
        int chordVelocity = 80;
        if (midi.chordVelocities.has_value() && chordIndex < midi.chordVelocities->size()) {
            chordVelocity = std::clamp((*midi.chordVelocities)[chordIndex], 1, 127);
        }
        for (int pitch : chord.pitches) {
            ScheduledNote note;
            note.pitch = pitch;
            note.velocity = chordVelocity;
            note.startBeat = startOffset + chord.startBeat;
            note.endBeat = startOffset + chord.startBeat + chord.duration * 0.9; // Slight gap
            note.noteOnSent = false;
            note.noteOffSent = false;
            scheduledNotes_.push_back(note);
        }
    }
    
    // LISTENING_GUARDRAIL: Only schedule melody/bass when explicitly set (I-1)
    if (midi.melody.has_value()) {
        for (const auto& midiNote : *midi.melody) {
            ScheduledNote note;
            note.pitch = midiNote.pitch;
            note.velocity = midiNote.velocity;
            note.startBeat = startOffset + midiNote.startBeat;
            note.endBeat = startOffset + midiNote.startBeat + midiNote.duration * 0.9;
            note.noteOnSent = false;
            note.noteOffSent = false;
            scheduledNotes_.push_back(note);
        }
    }
    if (midi.bass.has_value()) {
        for (const auto& midiNote : *midi.bass) {
            ScheduledNote note;
            note.pitch = midiNote.pitch;
            note.velocity = midiNote.velocity;
            note.startBeat = startOffset + midiNote.startBeat;
            note.endBeat = startOffset + midiNote.startBeat + midiNote.duration * 0.9;
            note.noteOnSent = false;
            note.noteOffSent = false;
            scheduledNotes_.push_back(note);
        }
    }
}

void PluginProcessor::startPlayback(bool fromUserAction) {
    // #region agent log
    debugLogB("PluginProcessor.cpp:startPlayback", "startPlayback called", "fromUserAction", fromUserAction, "B");
    // #endregion
    // LISTENING_GUARDRAIL: P-1 — playback only from explicit user action
    jassert(fromUserAction);
    if (!fromUserAction) {
        return;
    }

    const auto sessionId = conductor().currentSessionId();
    const auto gate = conductor().requestPlaybackStart(sessionId, fromUserAction);
    if (gate.decision != ConductorDecision::VALID || !gate.playbackAuthorized) {
        return;
    }

    const auto validatedPlan = validatedBoundaryPlan(lastGenerated_);
    if (!validatedPlan.has_value()) {
        return;
    }

    auto token = UserActionAuthority::authorizePlayback(fromUserAction);
    if (!token.isAuthorized()) {
        return;
    }

    lastGenerated_ = validatedPlan->midi();
    scheduleNotesForPlayback();
    isPlaying_ = true;
}

void PluginProcessor::stopPlayback() {
    const auto sessionId = conductor().currentSessionId();
    const auto result = conductor().requestPlaybackStop(sessionId);
    if (result.decision != ConductorDecision::VALID) {
        return;
    }
    isPlaying_ = false;
    
    // Send note-offs for any hanging notes would require access to midiMessages
    // which we don't have here - the next processBlock will handle cleanup
}

// TRUST MVP: Choke point. If this function doesn't exist, nothing else counts.
GeneratedMidi PluginProcessor::runTrustMVP(const UserInput& input) {
    // #region agent log
    debugLog("PluginProcessor.cpp:runTrustMVP", "runTrustMVP entry", "input_text", input.text, "A");
    // #endregion
    pendingSuggestions_.clear();
    lastIntentSeed_.clear();
    lastIntentHash_.clear();
    lastOutputMidiHash_.clear();

    UserInput normalizedInput = input;
    const auto canonicalIntent = parseCanonicalIntentPayload(input.text);
    if (canonicalIntent.status == CanonicalIntentParseStatus::Invalid) {
        // #region agent log
        debugLog("PluginProcessor.cpp:runTrustMVP", "INVALID", "reason", canonicalIntent.reason, "E");
        // #endregion
        lastIntent_ = IntentResult::invalid();
        return {};
    }
    if (canonicalIntent.status == CanonicalIntentParseStatus::Valid) {
        normalizedInput.text = canonicalIntent.payload.text;
        lastIntentSeed_ = canonicalIntent.payload.seed;
        lastIntentHash_ = canonicalIntentHash(canonicalIntent.payload);
        // #region agent log
        debugLog("PluginProcessor.cpp:runTrustMVP", "canonical_intent_seed", "seed", lastIntentSeed_, "A");
        debugLog("PluginProcessor.cpp:runTrustMVP", "canonical_intent_hash", "intent_hash", lastIntentHash_, "A");
        // #endregion
    }

    const bool hasTextDelta = normalizedInput.hasTextDelta();
    const bool hasPerformanceDelta = normalizedInput.hasPerformanceDelta();
    const bool hasParameterDelta = normalizedInput.hasParameterDelta();
    // #region agent log
    debugLogB("PluginProcessor.cpp:runTrustMVP", "delta detection", "has_text_delta", hasTextDelta, "E");
    debugLogB("PluginProcessor.cpp:runTrustMVP", "delta detection", "has_performance_delta", hasPerformanceDelta, "E");
    debugLogB("PluginProcessor.cpp:runTrustMVP", "delta detection", "has_parameter_delta", hasParameterDelta, "E");
    // #endregion

    // Step 3: No meaningful delta -> ABSTAIN.
    if (!normalizedInput.hasAnyDelta()) {
        // #region agent log
        debugLog("PluginProcessor.cpp:runTrustMVP", "ABSTAIN", "reason", "no_meaningful_delta", "E");
        // #endregion
        lastIntent_ = IntentResult::abstain(AbstainReason::EMPTY_INPUT);
        (void)conductor().submitInput(normalizedInput);
        return {};
    }

    float constrainedIntensity = 0.7f;
    if (auto* raw = parameters.getRawParameterValue(PARAM_INTENSITY)) {
        constrainedIntensity = raw->load();
    }
    float constrainedHumanize = 0.0f;
    if (auto* raw = parameters.getRawParameterValue(PARAM_HUMANIZE)) {
        constrainedHumanize = std::clamp(raw->load(), 0.0f, 1.0f);
    }
    bool tempoLockEnabled = true;
    if (auto* raw = parameters.getRawParameterValue(PARAM_TEMPO_LOCK)) {
        tempoLockEnabled = raw->load() >= 0.5f;
    }
    bool humanizeConstraintApplied = false;

    // R-3: parameter deltas are hard constraints; reject invalid values.
    if (hasParameterDelta) {
        const auto& param = *normalizedInput.parameterDelta;
        if (param.id == PARAM_INTENSITY || param.id == "intensity") {
            if (param.value < 0.0f || param.value > 1.0f) {
                // #region agent log
                debugLog("PluginProcessor.cpp:runTrustMVP", "INVALID", "reason", "intensity_out_of_range", "E");
                // #endregion
                lastIntent_ = IntentResult::invalid();
                (void)conductor().submitInput(normalizedInput);
                return {};
            }
            constrainedIntensity = param.value;
        } else if (param.id == PARAM_HUMANIZE || param.id == "humanize") {
            if (param.value < 0.0f || param.value > 1.0f) {
                // #region agent log
                debugLog("PluginProcessor.cpp:runTrustMVP", "INVALID", "reason", "humanize_out_of_range", "E");
                // #endregion
                lastIntent_ = IntentResult::invalid();
                (void)conductor().submitInput(normalizedInput);
                return {};
            }
            constrainedHumanize = param.value;
            humanizeConstraintApplied = true;
        } else if (param.id == PARAM_TEMPO_LOCK || param.id == "tempoLock") {
            if (!(param.value == 0.0f || param.value == 1.0f)) {
                // #region agent log
                debugLog("PluginProcessor.cpp:runTrustMVP", "INVALID", "reason", "tempo_lock_invalid", "E");
                // #endregion
                lastIntent_ = IntentResult::invalid();
                (void)conductor().submitInput(normalizedInput);
                return {};
            }
            tempoLockEnabled = (param.value >= 0.5f);
        } else {
            // #region agent log
            debugLog("PluginProcessor.cpp:runTrustMVP", "INVALID", "reason", "unknown_parameter_constraint", "E");
            // #endregion
            lastIntent_ = IntentResult::invalid();
            (void)conductor().submitInput(normalizedInput);
            return {};
        }
    }

    // Text carries intent bias in current architecture; performance/parameter-only inputs abstain.
    if (!hasTextDelta) {
        // #region agent log
        debugLog("PluginProcessor.cpp:runTrustMVP", "ABSTAIN", "reason", "no_text_intent", "E");
        // #endregion
        lastIntent_ = IntentResult::abstain(AbstainReason::UNDERSPECIFIED);
        (void)conductor().submitInput(normalizedInput);
        return {};
    }

    std::string text = trimCopy(normalizedInput.text);
    const std::string lowerText = toLowerCopy(text);
    if (lastIntentHash_.empty()) {
        lastIntentHash_ = toHex64(fnv1a64(lowerText));
    }
    const bool varyPitchToken = containsAnyToken(lowerText, {"vary_pitch", "vary pitch"});
    const bool varyTimingToken = containsAnyToken(lowerText, {"vary_timing", "vary timing"});
    const bool varyDynamicsToken = containsAnyToken(lowerText, {"vary_dynamics", "vary dynamics"});

    // TRUST MVP choke routing: generation authorization comes from conductor state-machine transitions.
    auto submit = conductor().submitInput(normalizedInput);
    if (submit.decision != ConductorDecision::VALID || !submit.midi.has_value()) {
        if (submit.decision == ConductorDecision::INVALID) {
            lastIntent_ = IntentResult::invalid();
        } else {
            lastIntent_ = IntentResult::abstain(AbstainReason::UNDERSPECIFIED);
        }
        return {};
    }

    IntentResult outputIntent;
    outputIntent.status = IntentResultStatus::VALID;
    outputIntent.escalationTokenPresent = containsAnyToken(lowerText, {"more", "richer", "push it", "go further"});
    outputIntent.tempo = tempoLockEnabled ? 1.0f : 1.1f;
    outputIntent.emotion.valence = constrainedIntensity - 0.5f;
    outputIntent.emotion.arousal = constrainedHumanize;
    lastIntent_ = outputIntent;

    // LISTENING_GUARDRAIL: Never produce output without passing gate.
    jassert(canProduceOutput(outputIntent));

    GeneratedMidi result = *submit.midi;
    if (varyPitchToken && result.chords.empty()) {
        lastIntent_ = IntentResult::abstain(AbstainReason::GENERATOR_EMPTY);
        return {};
    }
    if (!varyDynamicsToken) {
        result.chordVelocities = std::nullopt;
    }
    if ((varyTimingToken || humanizeConstraintApplied) && result.chords.empty()) {
        lastIntent_ = IntentResult::abstain(AbstainReason::GENERATOR_EMPTY);
        return {};
    }
    result.bpm = resolveOutputBpm(currentBpm_, tempoLockEnabled, outputIntent.tempo);
    recomputeLengthInBeats(result);
    lastOutputMidiHash_ = canonicalMidiHash(result);

    lastGenerated_ = result;
    pendingSuggestions_.clear();
    // #region agent log
    debugLogI("PluginProcessor.cpp:runTrustMVP", "VALID output", "chord_count", (int)result.chords.size(), "A");
    debugLog("PluginProcessor.cpp:runTrustMVP", "intent_hash", "intent_hash", lastIntentHash_, "A");
    debugLog("PluginProcessor.cpp:runTrustMVP", "output_midi_hash", "midi_hash", lastOutputMidiHash_, "A");
    if (!lastIntentSeed_.empty()) {
        debugLog("PluginProcessor.cpp:runTrustMVP", "seed", "seed", lastIntentSeed_, "A");
    }
    // #endregion
    return result; // NO playback
}

// Bypassed: TRUST MVP choke point only.
GeneratedMidi PluginProcessor::generateFromWound(const std::string& description, float intensity) {
    UserInput input;
    input.text = description;
    ParameterDelta intensityConstraint;
    intensityConstraint.id = PARAM_INTENSITY;
    intensityConstraint.value = intensity;
    input.parameterDelta = intensityConstraint;
    return runTrustMVP(input);
}

// Bypassed: TRUST MVP choke point only. Multiple inputs → ABSTAIN at call site.
GeneratedMidi PluginProcessor::generateFromJourney(const SideA& current, const SideB& desired) {
    pendingSuggestions_.clear();
    auto trim = [](std::string s) {
        const size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) {
            return std::string{};
        }
        const size_t end = s.find_last_not_of(" \t\n\r");
        return s.substr(start, end - start + 1);
    };
    const std::string sideA = trim(current.description);
    const std::string sideB = trim(desired.description);
    if (sideA.empty() || sideB.empty()) {
        lastIntent_ = IntentResult::abstain(AbstainReason::EMPTY_INPUT);
        return {};
    }

    UserInput input;
    input.text = sideA + " " + sideB;
    ParameterDelta intensityConstraint;
    intensityConstraint.id = PARAM_INTENSITY;
    intensityConstraint.value = std::clamp((current.intensity + desired.intensity) * 0.5f, 0.0f, 1.0f);
    input.parameterDelta = intensityConstraint;
    return runTrustMVP(input);
}

std::vector<GeneratedMidi> PluginProcessor::requestSuggestions() {
    pendingSuggestions_.clear();
    if (lastGenerated_.chords.empty())
        return pendingSuggestions_;
    auto variations = chordGenerator_.suggestVariations(lastGenerated_.chords);
    for (auto& chords : variations) {
        GeneratedMidi midi = lastGenerated_;
        midi.chords = std::move(chords);
        recomputeLengthInBeats(midi);
        pendingSuggestions_.push_back(std::move(midi));
    }
    return pendingSuggestions_;
}

std::vector<PluginProcessor::SuggestionComparison> PluginProcessor::getPendingSuggestionComparisons() const {
    std::vector<SuggestionComparison> comparisons;
    if (lastGenerated_.chords.empty() || pendingSuggestions_.empty()) {
        return comparisons;
    }

    for (size_t idx = 0; idx < pendingSuggestions_.size(); ++idx) {
        const auto& suggested = pendingSuggestions_[idx];
        SuggestionComparison meta;
        meta.index = idx;
        const size_t sharedChordCount = std::min(lastGenerated_.chords.size(), suggested.chords.size());
        meta.changedChords = static_cast<int>(
            std::max(lastGenerated_.chords.size(), suggested.chords.size()) - sharedChordCount);
        double startShiftSum = 0.0;
        double durationShiftSum = 0.0;

        for (size_t chordIdx = 0; chordIdx < sharedChordCount; ++chordIdx) {
            const auto& baseChord = lastGenerated_.chords[chordIdx];
            const auto& suggestedChord = suggested.chords[chordIdx];
            auto base = lastGenerated_.chords[chordIdx].pitches;
            auto branch = suggested.chords[chordIdx].pitches;

            if (base != branch) {
                ++meta.changedChords;
            }
            if (base.size() != branch.size()) {
                meta.voiceCountPreserved = false;
            }

            std::sort(base.begin(), base.end());
            std::sort(branch.begin(), branch.end());
            const size_t sharedVoiceCount = std::min(base.size(), branch.size());
            for (size_t voiceIdx = 0; voiceIdx < sharedVoiceCount; ++voiceIdx) {
                meta.maxSemitoneShift = std::max(meta.maxSemitoneShift, std::abs(branch[voiceIdx] - base[voiceIdx]));
            }

            const double startShift = std::abs(suggestedChord.startBeat - baseChord.startBeat);
            const double durationShift = std::abs(suggestedChord.duration - baseChord.duration);
            meta.maxStartBeatShift = std::max(meta.maxStartBeatShift, startShift);
            startShiftSum += startShift;
            durationShiftSum += durationShift;
        }
        if (sharedChordCount > 0) {
            meta.meanStartBeatShift = startShiftSum / static_cast<double>(sharedChordCount);
            meta.meanDurationShift = durationShiftSum / static_cast<double>(sharedChordCount);
        }
        comparisons.push_back(meta);
    }
    return comparisons;
}

bool PluginProcessor::applySuggestion(size_t index) {
    if (index >= pendingSuggestions_.size())
        return false;
    lastGenerated_ = pendingSuggestions_[index];
    pendingSuggestions_.clear();
    return true;
}

std::vector<float> PluginProcessor::getRenderedAudioForDisplay() const {
    if (lastGenerated_.chords.empty())
        return {};
    const auto plan = validatedBoundaryPlan(lastGenerated_);
    if (!plan.has_value()) {
        return {};
    }
    return renderMidiToPcm(*plan, currentSampleRate_ > 0 ? currentSampleRate_ : 44100.0);
}

bool PluginProcessor::exportMidiToFile(const juce::File& file) {
    if (lastGenerated_.chords.empty()) {
        return false;
    }
    
    const auto plan = validatedBoundaryPlan(lastGenerated_);
    if (!plan.has_value()) {
        return false;
    }

    MidiBuilder builder;
    auto midiFile = builder.buildMidiFile(*plan);
    
    juce::FileOutputStream stream(file);
    if (stream.openedOk()) {
        midiFile.writeTo(stream);
        return true;
    }
    return false;
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    return new PluginEditor(*this);
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(parameters.state.getType())) {
        parameters.replaceState(juce::ValueTree::fromXml(*xml));
    }
}

} // namespace kelly

// Plugin entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new kelly::PluginProcessor();
}
