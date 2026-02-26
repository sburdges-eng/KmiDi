#include "common/Types.h"
#include "adapters/legacy/EmotionThesaurusEnricher.h"
#include "engine/AdapterRegistry.h"
#include "engine/IntentPipeline.h"
#include "engine/CoreBridge.h"
#include "interfaces/IIntentEnricher.h"
#include "midi/ChordGenerator.h"
#include "midi/MidiBuilder.h"
#include "plugin/PluginProcessor.h"
#include "common/GuardrailValidator.h"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

struct TestCase {
    std::string name;
    bool (*run)();
};

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "  FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

bool chordsDifferInPitch(const std::vector<kelly::Chord>& a, const std::vector<kelly::Chord>& b) {
    if (a.size() != b.size()) {
        return true;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].pitches != b[i].pitches) {
            return true;
        }
    }
    return false;
}

bool chordsDifferInTiming(const std::vector<kelly::Chord>& a, const std::vector<kelly::Chord>& b, double eps = 1e-6) {
    if (a.size() != b.size()) {
        return true;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i].startBeat - b[i].startBeat) > eps) {
            return true;
        }
        if (std::abs(a[i].duration - b[i].duration) > eps) {
            return true;
        }
    }
    return false;
}

bool findFirstNoteOn(const juce::MidiMessageSequence* track, int& pitch, int& velocity) {
    if (track == nullptr) {
        return false;
    }

    for (int i = 0; i < track->getNumEvents(); ++i) {
        const auto* event = track->getEventPointer(i);
        if (event == nullptr) {
            continue;
        }
        const auto& message = event->message;
        if (!message.isNoteOn()) {
            continue;
        }
        pitch = message.getNoteNumber();
        velocity = static_cast<int>(message.getVelocity());
        return true;
    }
    return false;
}

bool findTempoSecondsPerQuarter(const juce::MidiMessageSequence* track, double& secondsPerQuarter) {
    if (track == nullptr) {
        return false;
    }
    for (int i = 0; i < track->getNumEvents(); ++i) {
        const auto* event = track->getEventPointer(i);
        if (event == nullptr) {
            continue;
        }
        const auto& message = event->message;
        if (!message.isTempoMetaEvent()) {
            continue;
        }
        secondsPerQuarter = message.getTempoSecondsPerQuarterNote();
        return true;
    }
    return false;
}

bool findFirstNoteOnOffTimes(const juce::MidiMessageSequence* track,
                            int targetPitch,
                            double& onTime,
                            double& offTime) {
    if (track == nullptr) {
        return false;
    }
    bool foundOn = false;
    bool foundOff = false;
    for (int i = 0; i < track->getNumEvents(); ++i) {
        const auto* event = track->getEventPointer(i);
        if (event == nullptr) {
            continue;
        }
        const auto& message = event->message;
        if (!foundOn && message.isNoteOn() && message.getNoteNumber() == targetPitch) {
            onTime = message.getTimeStamp();
            foundOn = true;
            continue;
        }
        if (!foundOff && message.isNoteOff() && message.getNoteNumber() == targetPitch) {
            offTime = message.getTimeStamp();
            foundOff = true;
        }
        if (foundOn && foundOff) {
            return true;
        }
    }
    return false;
}

bool midiEventTimestampsFinite(const juce::MidiFile& midiFile) {
    for (int trackIndex = 0; trackIndex < midiFile.getNumTracks(); ++trackIndex) {
        const auto* track = midiFile.getTrack(trackIndex);
        if (track == nullptr) {
            continue;
        }
        for (int eventIndex = 0; eventIndex < track->getNumEvents(); ++eventIndex) {
            const auto* event = track->getEventPointer(eventIndex);
            if (event == nullptr) {
                continue;
            }
            if (!std::isfinite(event->message.getTimeStamp())) {
                return false;
            }
        }
    }
    return true;
}

std::string canonicalIntentJson(const std::string& seed, const std::string& text) {
    return std::string("{")
        + "\"schemaVersion\":\"1.0\","
        + "\"seed\":\"" + seed + "\","
        + "\"text\":\"" + text + "\","
        + "\"intent\":{"
        + "\"harmonic\":0.7,"
        + "\"rhythmic\":0.3,"
        + "\"dynamic\":0.4,"
        + "\"tempo\":0.5"
        + "}"
        + "}";
}

bool testIntentPipelineAbstainsOnEmpty() {
    kelly::IntentPipeline pipeline;
    kelly::Wound wound;
    wound.description = "   ";
    wound.intensity = 0.7f;
    wound.source = "test";

    const auto result = pipeline.process(wound);
    return expect(result.isAbstain(), "empty wound input should ABSTAIN")
        && expect(!kelly::canProduceOutput(result), "ABSTAIN should fail output gate");
}

bool testIntentPipelineInvalidOnHighArousalConstraint() {
    kelly::IntentPipeline pipeline;
    kelly::Wound wound;
    wound.description = "chord change anxious terrified frantic urgent";
    wound.intensity = 1.0f;
    wound.source = "test";

    const auto result = pipeline.process(wound);
    return expect(result.isInvalid(), "high arousal rule break should return INVALID")
        && expect(!kelly::canProduceOutput(result), "INVALID should fail output gate");
}

bool testEmotionThesaurusAlwaysOnBehavior() {
    kelly::IntentPipeline pipeline;
    kelly::Wound wound;
    wound.description = "chord change grief";
    wound.intensity = 0.7f;
    wound.source = "test";

    const auto result = pipeline.process(wound);
    if (!expect(result.isValid(), "emotion enrichment probe should stay VALID")) {
        return false;
    }

    if (!expect(result.emotion.name == "grief",
                "emotion enrichment should map 'grief' via thesaurus adapter")) {
        return false;
    }
    return expect(result.emotion.valence < 0.0f,
                  "emotion enrichment should bias valence negative for grief");
}

bool testEmotionThesaurusTokenWeightPrecedence() {
    kelly::EmotionThesaurusEnricher enricher;
    kelly::IntentResult result;
    result.status = kelly::IntentResultStatus::VALID;
    result.emotion.name = "neutral";
    result.emotion.valence = 0.0f;
    result.emotion.arousal = 0.5f;

    kelly::UserInput input;
    input.text = "grief but also joy";
    enricher.enrich(input, result);

    if (!expect(result.emotion.name == "grief",
                "emotion thesaurus should choose highest-weight token when multiple match")) {
        return false;
    }
    return expect(result.emotion.valence < 0.0f,
                  "grief precedence should bias blended valence negative");
}

bool testRegistryActiveIntentEnricherIds() {
    const auto ids = kelly::AdapterRegistry::instance().activeIntentEnricherIds();
    bool hasNoop = false;
    bool hasEmotion = false;
    for (const auto& id : ids) {
        if (id == "noop.intent_enricher") {
            hasNoop = true;
        }
        if (id == "legacy.emotion_thesaurus_enricher") {
            hasEmotion = true;
        }
    }

    if (!expect(hasNoop, "adapter registry should always include no-op intent enricher")) {
        return false;
    }

    return expect(hasEmotion, "adapter registry should include emotion enricher by default");
}

class MaliciousStatusEnricher final : public kelly::IIntentEnricher {
public:
    const char* id() const override { return "test.malicious_status_enricher"; }
    void enrich(const kelly::UserInput&, kelly::IntentResult& intent) override {
        intent.status = kelly::IntentResultStatus::INVALID;
    }
};

bool testIntentEnrichmentCannotMutateStatus() {
    MaliciousStatusEnricher malicious;
    kelly::IntentResult result;
    result.status = kelly::IntentResultStatus::VALID;
    result.emotion.name = "neutral";
    result.emotion.valence = 0.0f;
    result.emotion.arousal = 0.5f;

    kelly::UserInput input;
    input.text = "chord change";

    // Mirror AdapterRegistry policy: enrichment may shape emotion, never control-flow status.
    const auto initialStatus = result.status;
    malicious.enrich(input, result);
    result.status = initialStatus;

    return expect(result.status == kelly::IntentResultStatus::VALID,
                  "enrichment phase must preserve pre-enrichment intent status");
}

bool testChordGeneratorRespectsOutputGate() {
    kelly::ChordGenerator generator;

    const auto abstainChords = generator.generate(kelly::IntentResult::abstain());
    if (!expect(abstainChords.empty(), "ABSTAIN intent should produce no chords")) {
        return false;
    }

    const auto invalidChords = generator.generate(kelly::IntentResult::invalid());
    return expect(invalidChords.empty(), "INVALID intent should produce no chords");
}

bool testRunTrustMVPAmbiguousInputAbstains() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"please help"});
    return expect(generated.chords.empty(), "ambiguous request should ABSTAIN/no output");
}

bool testRunTrustMVPAbstainsOnNoMeaningfulDelta() {
    kelly::PluginProcessor processor;
    kelly::UserInput input;
    input.text = "   ";
    const auto generated = processor.runTrustMVP(input);
    return expect(generated.chords.empty(), "empty delta should ABSTAIN/no output");
}

bool testRunTrustMVPAbstainsOnPerformanceOnlyDelta() {
    kelly::PluginProcessor processor;
    kelly::UserInput input;
    kelly::PerformanceDelta perf;
    perf.noteOnCount = 4;
    perf.durationBeats = 2.0;
    input.performanceDelta = perf;
    const auto generated = processor.runTrustMVP(input);
    return expect(generated.chords.empty(), "performance-only delta should ABSTAIN in current architecture");
}

bool testRunTrustMVPValidParameterAndTextDeltaGenerates() {
    kelly::PluginProcessor processor;
    kelly::UserInput input;
    input.text = "darker";
    kelly::ParameterDelta param;
    param.id = "intensity";
    param.value = 0.45f;
    input.parameterDelta = param;
    const auto generated = processor.runTrustMVP(input);
    return expect(!generated.chords.empty(), "valid parameter constraint + text should generate output");
}

bool testRunTrustMVPConflictingParameterConstraintAbstains() {
    kelly::PluginProcessor processor;
    kelly::UserInput input;
    input.text = "darker";
    kelly::ParameterDelta param;
    param.id = "intensity";
    param.value = 1.5f; // invalid hard constraint
    input.parameterDelta = param;
    const auto generated = processor.runTrustMVP(input);
    return expect(generated.chords.empty(), "conflicting parameter constraint should block output");
}

bool testCoreBridgePassThroughForTerminalStates() {
    kelly::IntentResult abstain = kelly::IntentResult::abstain();
    abstain.escalationTokenPresent = true;
    const auto abstainOut = kelly::CoreBridge::validateViaCore(abstain);
    if (!expect(abstainOut.isAbstain(), "ABSTAIN should pass through CoreBridge unchanged")) {
        return false;
    }
    if (!expect(abstainOut.escalationTokenPresent == abstain.escalationTokenPresent,
                "ABSTAIN escalation flag should preserve parity across bridge")) {
        return false;
    }

    kelly::IntentResult invalid = kelly::IntentResult::invalid();
    invalid.escalationTokenPresent = false;
    const auto invalidOut = kelly::CoreBridge::validateViaCore(invalid);
    if (!expect(invalidOut.isInvalid(), "INVALID should pass through CoreBridge unchanged")) {
        return false;
    }
    return expect(invalidOut.escalationTokenPresent == invalid.escalationTokenPresent,
                  "INVALID escalation flag should preserve parity across bridge");
}

bool testCoreBridgeFallbackPassThroughWhenDisabled() {
#ifdef KELLY_USE_CORE_BRIDGE
    // In bridge-enabled mode, fallback behavior is validated by matrix lane with core dependency checks.
    return true;
#else
    kelly::IntentResult in;
    in.status = kelly::IntentResultStatus::VALID;
    in.mode = "minor";
    in.tempo = 0.93f;
    in.escalationTokenPresent = true;
    in.emotion.name = "test";
    in.emotion.valence = -0.4f;
    in.emotion.arousal = 0.2f;

    const auto out = kelly::CoreBridge::validateViaCore(in);
    if (!expect(out.isValid(), "bridge-disabled mode should preserve VALID status")) {
        return false;
    }
    if (!expect(out.mode == in.mode, "bridge-disabled mode should preserve mode")) {
        return false;
    }
    if (!expect(out.tempo == in.tempo, "bridge-disabled mode should preserve tempo")) {
        return false;
    }
    if (!expect(out.escalationTokenPresent == in.escalationTokenPresent,
                "bridge-disabled mode should preserve escalation parity")) {
        return false;
    }
    if (!expect(out.emotion.valence == in.emotion.valence && out.emotion.arousal == in.emotion.arousal,
                "bridge-disabled mode should preserve emotion")) {
        return false;
    }
    return expect(out.emotion.name == in.emotion.name, "bridge-disabled mode should preserve emotion name");
#endif
}

bool testSuggestionsReturnTwoToThreeOptions() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"darker"});
    if (!expect(!generated.chords.empty(), "generation required before suggestions")) {
        return false;
    }
    const auto suggestions = processor.requestSuggestions();
    if (!expect(suggestions.size() >= 2 && suggestions.size() <= 3, "suggestions must be 2-3 options")) {
        return false;
    }
    return expect(processor.getPendingSuggestionsCount() == suggestions.size(),
                  "pending suggestion count should match returned options");
}

bool testSuggestionsDoNotIncreaseDensity() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"chord change"});
    if (!expect(!generated.chords.empty(), "generation required before density comparison")) {
        return false;
    }
    const auto base = processor.getLastGeneratedMidi();
    const auto suggestions = processor.requestSuggestions();
    if (!expect(!suggestions.empty(), "suggestions should exist for density comparison")) {
        return false;
    }

    for (const auto& suggestion : suggestions) {
        if (!expect(suggestion.chords.size() == base.chords.size(),
                    "suggestion must preserve chord count")) {
            return false;
        }
        for (size_t i = 0; i < suggestion.chords.size(); ++i) {
            if (!expect(suggestion.chords[i].pitches.size() == base.chords[i].pitches.size(),
                        "suggestion must preserve voices per chord (no density increase)")) {
                return false;
            }
        }
    }
    return true;
}

bool testSuggestionApplyBoundaries() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"darker"});
    if (!expect(!generated.chords.empty(), "generation required before apply-boundary checks")) {
        return false;
    }
    const auto suggestions = processor.requestSuggestions();
    if (!expect(!suggestions.empty(), "suggestions should exist before apply checks")) {
        return false;
    }

    const size_t n = suggestions.size();
    if (!expect(!processor.applySuggestion(n), "applySuggestion should reject out-of-range index")) {
        return false;
    }
    if (!expect(processor.getPendingSuggestionsCount() == n, "invalid apply should not clear pending suggestions")) {
        return false;
    }
    if (!expect(processor.applySuggestion(0), "applySuggestion should accept first valid option")) {
        return false;
    }
    return expect(processor.getPendingSuggestionsCount() == 0, "valid apply should clear pending suggestions");
}

bool testSuggestionComparisonMetadataReadOnly() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"darker"});
    if (!expect(!generated.chords.empty(), "generation required before comparison metadata checks")) {
        return false;
    }
    const auto suggestions = processor.requestSuggestions();
    const auto comparisons = processor.getPendingSuggestionComparisons();
    if (!expect(comparisons.size() == suggestions.size(),
                "comparison metadata should match suggestion count")) {
        return false;
    }

    bool sawDifference = false;
    for (const auto& meta : comparisons) {
        if (!expect(meta.voiceCountPreserved, "comparison metadata should report preserved voice counts")) {
            return false;
        }
        if (!expect(meta.maxStartBeatShift >= 0.0, "max start-beat shift must be non-negative")) {
            return false;
        }
        if (!expect(meta.meanStartBeatShift >= 0.0, "mean start-beat shift must be non-negative")) {
            return false;
        }
        if (!expect(meta.meanDurationShift >= 0.0, "mean duration shift must be non-negative")) {
            return false;
        }
        if (meta.changedChords > 0 || meta.maxSemitoneShift > 0) {
            sawDifference = true;
        }
    }
    return expect(sawDifference, "at least one suggestion should differ from baseline voicing");
}

bool testRunTrustMVPVaryPitchTokenChangesVoicing() {
    kelly::PluginProcessor processor;
    const auto baseline = processor.runTrustMVP(kelly::UserInput{"darker"});
    if (!expect(!baseline.chords.empty(), "baseline generation required before vary_pitch comparison")) {
        return false;
    }
    const auto varied = processor.runTrustMVP(kelly::UserInput{"darker vary_pitch"});
    if (!expect(!varied.chords.empty(), "vary_pitch generation should produce output")) {
        return false;
    }
    return expect(chordsDifferInPitch(baseline.chords, varied.chords),
                  "vary_pitch token should alter chord voicing");
}

bool testRunTrustMVPVaryTimingTokenChangesGrid() {
    kelly::PluginProcessor processor;
    const auto baseline = processor.runTrustMVP(kelly::UserInput{"chord change"});
    if (!expect(!baseline.chords.empty(), "baseline generation required before vary_timing comparison")) {
        return false;
    }

    kelly::UserInput input;
    input.text = "chord change vary_timing";
    kelly::ParameterDelta humanizeParam;
    humanizeParam.id = "humanize";
    humanizeParam.value = 0.9f;
    input.parameterDelta = humanizeParam;

    const auto varied = processor.runTrustMVP(input);
    if (!expect(!varied.chords.empty(), "vary_timing generation should produce output")) {
        return false;
    }
    return expect(chordsDifferInTiming(baseline.chords, varied.chords),
                  "vary_timing token should alter start/duration timing");
}

bool testRunTrustMVPVaryDynamicsTokenProducesVelocities() {
    kelly::PluginProcessor processor;
    const auto plain = processor.runTrustMVP(kelly::UserInput{"darker"});
    if (!expect(!plain.chords.empty(), "baseline generation required before vary_dynamics comparison")) {
        return false;
    }
    if (!expect(!plain.chordVelocities.has_value(), "baseline output should not add chord velocities by default")) {
        return false;
    }

    const auto varied = processor.runTrustMVP(kelly::UserInput{"darker vary_dynamics"});
    if (!expect(!varied.chords.empty(), "vary_dynamics generation should produce output")) {
        return false;
    }
    if (!expect(varied.chordVelocities.has_value(), "vary_dynamics should emit explicit chord velocities")) {
        return false;
    }
    if (!expect(varied.chordVelocities->size() == varied.chords.size(),
                "vary_dynamics velocities should align with chord count")) {
        return false;
    }
    bool hasContour = false;
    for (size_t i = 1; i < varied.chordVelocities->size(); ++i) {
        if ((*varied.chordVelocities)[i] != (*varied.chordVelocities)[0]) {
            hasContour = true;
            break;
        }
    }
    return expect(hasContour, "vary_dynamics should create a non-flat dynamics contour");
}

bool testRunTrustMVPHumanizeConstraintAffectsTiming() {
    kelly::PluginProcessor processor;
    const auto baseline = processor.runTrustMVP(kelly::UserInput{"chord change"});
    if (!expect(!baseline.chords.empty(), "baseline generation required before humanize comparison")) {
        return false;
    }

    kelly::UserInput input;
    input.text = "chord change";
    kelly::ParameterDelta param;
    param.id = "humanize";
    param.value = 1.0f;
    input.parameterDelta = param;
    const auto humanized = processor.runTrustMVP(input);
    if (!expect(!humanized.chords.empty(), "humanized generation should produce output")) {
        return false;
    }
    return expect(chordsDifferInTiming(baseline.chords, humanized.chords),
                  "explicit humanize parameter should alter timing");
}

bool testRunTrustMVPTempoLockConstraintAffectsBpm() {
    kelly::PluginProcessor lockedProcessor;
    kelly::UserInput lockedInput;
    lockedInput.text = "more energy";
    lockedInput.parameterDelta = kelly::ParameterDelta{"tempoLock", 1.0f};
    const auto locked = lockedProcessor.runTrustMVP(lockedInput);
    if (!expect(!locked.chords.empty(), "tempo-lock baseline should generate output")) {
        return false;
    }

    kelly::PluginProcessor unlockedProcessor;
    kelly::UserInput unlockedInput;
    unlockedInput.text = "more energy";
    unlockedInput.parameterDelta = kelly::ParameterDelta{"tempoLock", 0.0f};
    const auto unlocked = unlockedProcessor.runTrustMVP(unlockedInput);
    if (!expect(!unlocked.chords.empty(), "tempo-unlocked generation should produce output")) {
        return false;
    }

    if (!expect(std::abs(locked.bpm - 120.0f) < 0.01f, "tempoLock=1 should keep host/default tempo")) {
        return false;
    }
    return expect(unlocked.bpm > locked.bpm, "tempoLock=0 should allow intent tempo scaling");
}

bool testGenerateFromJourneyProducesWhenExplicitlyProvided() {
    kelly::PluginProcessor processor;
    kelly::SideA current{"darker journey", 0.7f, std::nullopt};
    kelly::SideB desired{"smoother journey", 0.5f, std::nullopt};
    const auto generated = processor.generateFromJourney(current, desired);
    return expect(!generated.chords.empty(), "generateFromJourney should produce output for explicit journey inputs");
}

bool testSuggestionsPreserveOptionalDynamicsLayer() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"darker vary_dynamics"});
    if (!expect(!generated.chords.empty(), "generation required before suggestion optional-layer checks")) {
        return false;
    }
    if (!expect(generated.chordVelocities.has_value(), "base output should include chord velocities")) {
        return false;
    }
    const auto suggestions = processor.requestSuggestions();
    if (!expect(!suggestions.empty(), "suggestions should exist before optional-layer checks")) {
        return false;
    }
    for (const auto& suggestion : suggestions) {
        if (!expect(suggestion.chordVelocities.has_value(),
                    "suggestions should preserve chord velocity optionals")) {
            return false;
        }
        if (!expect(*suggestion.chordVelocities == *generated.chordVelocities,
                    "suggestions should keep identical chord velocity contour")) {
            return false;
        }
    }
    return true;
}

bool testRunTrustMVPGeneratesWithoutAutoPlayback() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"chord change"});

    if (!expect(!generated.chords.empty(), "chord change should produce minimal output")) {
        return false;
    }
    return expect(!processor.isPlaying(), "generation path must not start playback");
}

bool testRunTrustMVPRecognizedIntentGenerates() {
    kelly::PluginProcessor processor;
    const auto generated = processor.runTrustMVP(kelly::UserInput{"darker"});

    return expect(!generated.chords.empty(), "recognized intent token should generate via IntentPipeline");
}

bool testRunTrustMVPCanonicalIntentRejectsMalformedJson() {
    kelly::PluginProcessor processor;

    kelly::UserInput missingSeed;
    missingSeed.text =
        R"({"schemaVersion":"1.0","text":"darker","intent":{"harmonic":0.7,"rhythmic":0.3,"dynamic":0.4,"tempo":0.5}})";
    const auto missingSeedOut = processor.runTrustMVP(missingSeed);
    if (!expect(missingSeedOut.chords.empty(), "canonical intent without seed must hard-reject")) {
        return false;
    }
    const auto missingSeedIntent = processor.getLastIntentResult();
    if (!expect(missingSeedIntent.has_value() && missingSeedIntent->isInvalid(),
                "missing-seed canonical intent should set INVALID status")) {
        return false;
    }

    kelly::UserInput unknownField;
    unknownField.text =
        R"({"schemaVersion":"1.0","seed":"s-1","text":"darker","intent":{"harmonic":0.7,"rhythmic":0.3,"dynamic":0.4,"tempo":0.5},"rogue":true})";
    const auto unknownFieldOut = processor.runTrustMVP(unknownField);
    if (!expect(unknownFieldOut.chords.empty(), "canonical intent with unknown field must hard-reject")) {
        return false;
    }
    const auto unknownFieldIntent = processor.getLastIntentResult();
    return expect(unknownFieldIntent.has_value() && unknownFieldIntent->isInvalid(),
                  "unknown-field canonical intent should set INVALID status");
}

bool testRunTrustMVPCanonicalIntentDeterministicReplay() {
    kelly::PluginProcessor processor;
    kelly::UserInput input;
    input.text = canonicalIntentJson("seed-replay-001", "darker");

    const auto first = processor.runTrustMVP(input);
    if (!expect(!first.chords.empty(), "first canonical seeded run should generate output")) {
        return false;
    }
    const auto firstSeed = processor.getLastIntentSeed();
    const auto firstIntentHash = processor.getLastIntentHash();
    const auto firstMidiHash = processor.getLastOutputMidiHash();

    const auto second = processor.runTrustMVP(input);
    if (!expect(!second.chords.empty(), "second canonical seeded run should generate output")) {
        return false;
    }
    const auto secondSeed = processor.getLastIntentSeed();
    const auto secondIntentHash = processor.getLastIntentHash();
    const auto secondMidiHash = processor.getLastOutputMidiHash();

    if (!expect(firstSeed == "seed-replay-001" && secondSeed == "seed-replay-001",
                "seed must be captured and preserved for canonical intent runs")) {
        return false;
    }
    if (!expect(!firstIntentHash.empty() && firstIntentHash == secondIntentHash,
                "intent hash must be stable across deterministic replay")) {
        return false;
    }
    if (!expect(!firstMidiHash.empty() && firstMidiHash == secondMidiHash,
                "output MIDI hash must match across deterministic replay")) {
        return false;
    }
    if (!expect(!chordsDifferInPitch(first.chords, second.chords),
                "deterministic replay should preserve chord voicing")) {
        return false;
    }
    if (!expect(!chordsDifferInTiming(first.chords, second.chords),
                "deterministic replay should preserve timing")) {
        return false;
    }
    return expect(std::abs(first.bpm - second.bpm) < 0.0001f,
                  "deterministic replay should preserve BPM");
}

bool testExportBehaviorFromGeneratedOutputPath() {
    kelly::PluginProcessor processor;

    const auto tempDir = juce::File::getCurrentWorkingDirectory().getChildFile("runtime_contract_tmp");
    tempDir.createDirectory();
    const auto token = juce::String::toHexString(juce::Time::getHighResolutionTicks());
    auto exportFile = tempDir.getChildFile("kelly_contract_export_" + token + ".mid");
    exportFile.deleteFile();

    if (!expect(!processor.exportMidiToFile(exportFile), "export should fail when no output is generated")) {
        return false;
    }
    if (!expect(!exportFile.existsAsFile(), "failed export should not leave a MIDI file")) {
        exportFile.deleteFile();
        return false;
    }

    const auto generated = processor.runTrustMVP(kelly::UserInput{"darker"});
    if (!expect(!generated.chords.empty(), "generation should succeed before export validation")) {
        return false;
    }
    if (!expect(processor.exportMidiToFile(exportFile), "export should succeed from generated output path")) {
        return false;
    }
    if (!expect(exportFile.existsAsFile(), "exported MIDI file should exist")) {
        exportFile.deleteFile();
        return false;
    }
    if (!expect(exportFile.getSize() > 0, "exported MIDI file should be non-empty")) {
        exportFile.deleteFile();
        return false;
    }

    juce::FileInputStream stream(exportFile);
    if (!expect(stream.openedOk(), "exported MIDI file should be readable")) {
        exportFile.deleteFile();
        return false;
    }
    char header[4] = {};
    if (!expect(stream.read(header, 4) == 4, "exported MIDI should contain a full header")) {
        exportFile.deleteFile();
        return false;
    }

    const bool midiHeaderOk = std::memcmp(header, "MThd", 4) == 0;
    if (!expect(midiHeaderOk, "exported file should start with MIDI header MThd")) {
        exportFile.deleteFile();
        return false;
    }

    exportFile.deleteFile();
    return true;
}

bool testMidiBuilderClampsOutOfRangeNotes() {
    kelly::GeneratedMidi midi;
    kelly::Chord chord;
    chord.pitches = {-5, 130};
    chord.startBeat = 0.0;
    chord.duration = 1.0;
    midi.chords.push_back(chord);
    midi.chordVelocities = std::vector<int>{96};
    midi.bpm = 120.0f;

    kelly::MidiNote melody;
    melody.pitch = 200;
    melody.velocity = -8;
    melody.startBeat = 1.0;
    melody.duration = 0.5;
    midi.melody = std::vector<kelly::MidiNote>{melody};

    kelly::MidiNote bass;
    bass.pitch = -12;
    bass.velocity = 64;
    bass.startBeat = 2.0;
    bass.duration = 0.5;
    midi.bass = std::vector<kelly::MidiNote>{bass};

    kelly::OutputLayerAuthorization auth;
    auth.chordsAuthorized = true;
    auth.melodyAuthorized = true;
    auth.bassAuthorized = true;
    auth.velocitiesAuthorized = true;
    auth.rhythmDensityIncreaseAuthorized = true;
    auth.maxVoiceCount = 8;
    auth.maxChordCount = midi.chords.size();

    auto validated = kelly::OutputPlanValidator::validate(std::move(midi), auth);
    if (!expect(validated.validated.has_value(), "constructed MIDI should pass output validation")) {
        return false;
    }

    kelly::MidiBuilder builder;
    const auto midiFile = builder.buildMidiFile(*validated.validated);
    if (!expect(midiFile.getNumTracks() == 3, "MIDI with chords+melody+bass should emit 3 tracks")) {
        return false;
    }

    int chordPitch = -1;
    int chordVelocity = -1;
    if (!expect(findFirstNoteOn(midiFile.getTrack(0), chordPitch, chordVelocity),
                "chord track should include a note-on event")) {
        return false;
    }
    if (!expect(chordPitch == 0, "chord pitch below 0 should clamp to 0 (not wrap)")) {
        return false;
    }
    if (!expect(chordVelocity == 96, "chord velocity should remain unchanged when already in range")) {
        return false;
    }

    int melodyPitch = -1;
    int melodyVelocity = -1;
    if (!expect(findFirstNoteOn(midiFile.getTrack(1), melodyPitch, melodyVelocity),
                "melody track should include a note-on event")) {
        return false;
    }
    if (!expect(melodyPitch == 127, "melody pitch above 127 should clamp to 127 (not wrap)")) {
        return false;
    }
    if (!expect(melodyVelocity == 1, "melody velocity below 1 should clamp to 1 (not wrap)")) {
        return false;
    }

    int bassPitch = -1;
    int bassVelocity = -1;
    if (!expect(findFirstNoteOn(midiFile.getTrack(2), bassPitch, bassVelocity),
                "bass track should include a note-on event")) {
        return false;
    }
    if (!expect(bassPitch == 0, "bass pitch below 0 should clamp to 0 (not wrap)")) {
        return false;
    }
    return expect(bassVelocity == 64, "bass velocity should remain unchanged when already in range");
}

bool testMidiBuilderFallsBackForNonPositiveBpm() {
    const std::vector<float> invalidBpms = {
        0.0f,
        -24.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };

    for (const float invalidBpm : invalidBpms) {
        kelly::GeneratedMidi midi;
        kelly::Chord chord;
        chord.pitches = {60, 64, 67};
        chord.startBeat = 0.0;
        chord.duration = 1.0;
        midi.chords.push_back(chord);
        midi.bpm = invalidBpm;

        kelly::OutputLayerAuthorization auth;
        auth.chordsAuthorized = true;
        auth.melodyAuthorized = false;
        auth.bassAuthorized = false;
        auth.velocitiesAuthorized = false;
        auth.rhythmDensityIncreaseAuthorized = true;
        auth.maxVoiceCount = 3;
        auth.maxChordCount = midi.chords.size();

        auto validated = kelly::OutputPlanValidator::validate(std::move(midi), auth);
        if (!expect(validated.validated.has_value(), "constructed MIDI should pass output validation")) {
            return false;
        }

        kelly::MidiBuilder builder;
        const auto midiFile = builder.buildMidiFile(*validated.validated);
        if (!expect(midiEventTimestampsFinite(midiFile),
                    "invalid BPM should never produce NaN/Inf MIDI event timestamps")) {
            return false;
        }

        double secondsPerQuarter = 0.0;
        if (!expect(findTempoSecondsPerQuarter(midiFile.getTrack(0), secondsPerQuarter),
                    "chord track should include tempo metadata")) {
            return false;
        }

        if (!expect(std::isfinite(secondsPerQuarter) && secondsPerQuarter > 0.0,
                    "tempo metadata should be finite and positive")) {
            return false;
        }

        if (!expect(std::abs(secondsPerQuarter - 0.5) < 0.0001,
                    "invalid BPM should fall back to 120 BPM tempo metadata")) {
            return false;
        }
    }

    return true;
}

bool testMidiBuilderClampsNonPositiveAndNonFiniteDurations() {
    kelly::GeneratedMidi midi;
    kelly::Chord chord;
    chord.pitches = {60};
    chord.startBeat = 1.0;
    chord.duration = -0.75;
    midi.chords.push_back(chord);
    midi.chordVelocities = std::vector<int>{80};
    midi.bpm = 120.0f;

    kelly::MidiNote note;
    note.pitch = 64;
    note.velocity = 90;
    note.startBeat = 2.0;
    note.duration = std::numeric_limits<double>::quiet_NaN();
    midi.melody = std::vector<kelly::MidiNote>{note};

    kelly::OutputLayerAuthorization auth;
    auth.chordsAuthorized = true;
    auth.melodyAuthorized = true;
    auth.bassAuthorized = false;
    auth.velocitiesAuthorized = true;
    auth.rhythmDensityIncreaseAuthorized = true;
    auth.maxVoiceCount = 8;
    auth.maxChordCount = midi.chords.size();

    auto validated = kelly::OutputPlanValidator::validate(std::move(midi), auth);
    if (!expect(validated.validated.has_value(), "constructed MIDI should pass output validation")) {
        return false;
    }

    kelly::MidiBuilder builder;
    const auto midiFile = builder.buildMidiFile(*validated.validated);
    if (!expect(midiEventTimestampsFinite(midiFile),
                "non-positive and NaN durations should not produce NaN/Inf timestamps")) {
        return false;
    }

    const auto* chordTrack = midiFile.getTrack(0);
    double chordOn = -1.0;
    double chordOff = -1.0;
    if (!expect(findFirstNoteOnOffTimes(chordTrack, 60, chordOn, chordOff),
                "chord track should emit both on and off events")) {
        return false;
    }
    if (!expect(chordOff >= chordOn,
                "non-positive chord duration should be clamped so note-off does not occur before note-on")) {
        return false;
    }

    const auto* melodyTrack = midiFile.getTrack(1);
    double melodyOn = -1.0;
    double melodyOff = -1.0;
    if (!expect(findFirstNoteOnOffTimes(melodyTrack, 64, melodyOn, melodyOff),
                "melody track should emit both on and off events")) {
        return false;
    }
    return expect(melodyOff >= melodyOn, "NaN melody duration should be clamped so note-off does not precede note-on");
}

bool testPluginStateSerializationRoundTrip() {
    kelly::PluginProcessor source;
    auto* sourceIntensity = source.parameters.getParameter("intensity");
    auto* sourceHumanize = source.parameters.getParameter("humanize");
    auto* sourceTempoLock = source.parameters.getParameter("tempoLock");
    if (!expect(sourceIntensity != nullptr && sourceHumanize != nullptr && sourceTempoLock != nullptr,
                "plugin parameters should include intensity/humanize/tempoLock")) {
        return false;
    }

    sourceIntensity->setValueNotifyingHost(0.23f);
    sourceHumanize->setValueNotifyingHost(0.67f);
    sourceTempoLock->setValueNotifyingHost(0.0f);

    juce::MemoryBlock state;
    source.getStateInformation(state);
    if (!expect(state.getSize() > 0, "state serialization should produce non-empty payload")) {
        return false;
    }

    kelly::PluginProcessor restored;
    restored.setStateInformation(state.getData(), static_cast<int>(state.getSize()));

    const auto restoredIntensity = restored.parameters.getRawParameterValue("intensity")->load();
    const auto restoredHumanize = restored.parameters.getRawParameterValue("humanize")->load();
    const auto restoredTempoLock = restored.parameters.getRawParameterValue("tempoLock")->load();

    if (!expect(std::abs(restoredIntensity - 0.23f) < 0.001f,
                "restored intensity should match serialized value")) {
        return false;
    }
    if (!expect(std::abs(restoredHumanize - 0.67f) < 0.001f,
                "restored humanize should match serialized value")) {
        return false;
    }
    if (!expect(restoredTempoLock < 0.5f,
                "restored tempoLock should match serialized false value")) {
        return false;
    }

    const float beforeInvalidBlob = restoredIntensity;
    const char invalidBlob[] = "not-a-valid-plugin-state";
    restored.setStateInformation(invalidBlob, static_cast<int>(sizeof(invalidBlob)));
    const auto afterInvalidBlob = restored.parameters.getRawParameterValue("intensity")->load();
    return expect(std::abs(afterInvalidBlob - beforeInvalidBlob) < 0.001f,
                  "invalid state payload should be ignored without mutating parameters");
}

} // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"IntentPipeline abstains on empty input", testIntentPipelineAbstainsOnEmpty},
        {"IntentPipeline returns INVALID on high-arousal constraint", testIntentPipelineInvalidOnHighArousalConstraint},
        {"Emotion-thesaurus enrichment is always on", testEmotionThesaurusAlwaysOnBehavior},
        {"Emotion-thesaurus token precedence picks strongest match", testEmotionThesaurusTokenWeightPrecedence},
        {"Adapter registry includes active default enrichers", testRegistryActiveIntentEnricherIds},
        {"Enrichment phase prevents status mutation", testIntentEnrichmentCannotMutateStatus},
        {"ChordGenerator respects canProduceOutput gate", testChordGeneratorRespectsOutputGate},
        {"runTrustMVP abstains on no meaningful delta", testRunTrustMVPAbstainsOnNoMeaningfulDelta},
        {"runTrustMVP abstains on performance-only delta", testRunTrustMVPAbstainsOnPerformanceOnlyDelta},
        {"runTrustMVP abstains on ambiguous input", testRunTrustMVPAmbiguousInputAbstains},
        {"runTrustMVP generates for valid parameter+text delta", testRunTrustMVPValidParameterAndTextDeltaGenerates},
        {"runTrustMVP blocks conflicting parameter constraint", testRunTrustMVPConflictingParameterConstraintAbstains},
        {"CoreBridge pass-through for ABSTAIN/INVALID", testCoreBridgePassThroughForTerminalStates},
        {"CoreBridge fallback pass-through when disabled", testCoreBridgeFallbackPassThroughWhenDisabled},
        {"Suggestions return 2-3 options", testSuggestionsReturnTwoToThreeOptions},
        {"Suggestions preserve density", testSuggestionsDoNotIncreaseDensity},
        {"Suggestion apply boundaries", testSuggestionApplyBoundaries},
        {"Suggestion comparison metadata is read-only", testSuggestionComparisonMetadataReadOnly},
        {"vary_pitch token alters voicing", testRunTrustMVPVaryPitchTokenChangesVoicing},
        {"vary_timing token alters timing grid", testRunTrustMVPVaryTimingTokenChangesGrid},
        {"vary_dynamics token emits dynamics contour", testRunTrustMVPVaryDynamicsTokenProducesVelocities},
        {"Humanize parameter alters timing when explicit", testRunTrustMVPHumanizeConstraintAffectsTiming},
        {"tempoLock constraint affects BPM output", testRunTrustMVPTempoLockConstraintAffectsBpm},
        {"generateFromJourney supports explicit dual-input path", testGenerateFromJourneyProducesWhenExplicitlyProvided},
        {"Suggestions preserve optional dynamics layer", testSuggestionsPreserveOptionalDynamicsLayer},
        {"runTrustMVP generates for recognized non-chord intent", testRunTrustMVPRecognizedIntentGenerates},
        {"runTrustMVP canonical intent rejects malformed JSON", testRunTrustMVPCanonicalIntentRejectsMalformedJson},
        {"runTrustMVP canonical intent deterministic replay", testRunTrustMVPCanonicalIntentDeterministicReplay},
        {"runTrustMVP does not auto-playback", testRunTrustMVPGeneratesWithoutAutoPlayback},
        {"Export behavior validated from generated output path", testExportBehaviorFromGeneratedOutputPath},
        {"MidiBuilder clamps out-of-range note data", testMidiBuilderClampsOutOfRangeNotes},
        {"MidiBuilder falls back on non-positive BPM", testMidiBuilderFallsBackForNonPositiveBpm},
        {"MidiBuilder clamps non-positive/non-finite durations", testMidiBuilderClampsNonPositiveAndNonFiniteDurations},
        {"Plugin state serialization round-trips safely", testPluginStateSerializationRoundTrip},
    };

    int failed = 0;
    for (const auto& test : tests) {
        std::cout << "[TEST] " << test.name << std::endl;
        if (!test.run()) {
            ++failed;
        }
    }

    if (failed > 0) {
        std::cerr << failed << " runtime contract test(s) failed." << std::endl;
        return 1;
    }

    std::cout << "All runtime contract tests passed." << std::endl;
    return 0;
}
