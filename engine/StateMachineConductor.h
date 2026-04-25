#pragma once

#include "common/DecisionTrace.h"
#include "common/GuardrailValidator.h"
#include "common/Types.h"
#include "common/KellyTypes.h"  // canonical kelly::Wound (PR #150 consolidation)
#include "engine/IntentPipeline.h"
#include "midi/ChordGenerator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#ifndef KELLY_ENABLE_FAULT_INJECTION
#define KELLY_ENABLE_FAULT_INJECTION 0
#endif

namespace kelly {

class StateMachineConductor {
public:
    struct SubmitResult {
        ConductorDecision decision = ConductorDecision::ABSTAIN;
        uint64_t sessionId = 0;
        uint64_t sequenceId = 0;
        std::optional<GeneratedMidi> midi = std::nullopt;
        SuggestedVariants alternatives;
        GuardrailValidationReport validation;
    };

    struct AlternativeResult {
        ConductorDecision decision = ConductorDecision::ABSTAIN;
        uint64_t sessionId = 0;
        uint64_t sequenceId = 0;
        size_t selectedIndex = 0;
    };

    struct PlaybackResult {
        ConductorDecision decision = ConductorDecision::ABSTAIN;
        uint64_t sessionId = 0;
        uint64_t sequenceId = 0;
        bool playbackAuthorized = false;
    };

    SubmitResult submitInput(const UserInput& input);
    AlternativeResult applyAlternative(uint64_t sessionId, size_t index);
    PlaybackResult requestPlaybackStart(uint64_t sessionId, bool userInitiatedFlag);
    PlaybackResult requestPlaybackStop(uint64_t sessionId);

    uint64_t currentSessionId() const;

private:
    struct SessionContext {
        ExecutionState state = ExecutionState::Idle;
        uint64_t nextSequenceId = 1;
        std::optional<ValidatedOutputPlan> validatedPlan;
        SuggestedVariants alternatives;
        std::string lastInputText;
    };

    SubmitResult submitInputForSession(uint64_t sessionId, const UserInput& input);
    SessionContext& ensureSession(uint64_t sessionId);
    static std::string toLower(std::string text);
    static bool containsAny(const std::string& lower, std::initializer_list<const char*> tokens);
    static uint32_t inferRequestedGroupsMask(const std::string& lowerText);
    static SuggestedVariants buildBoundedAlternatives(const std::string& lowerText);
    static void recomputeLengthInBeats(GeneratedMidi& midi);
    static void applyPitchVariation(GeneratedMidi& midi, float intensity);
    static void applyTimingVariation(GeneratedMidi& midi, float humanizeAmount);
    static std::vector<int> buildDynamicsContour(const GeneratedMidi& midi, float intensity, float humanizeAmount);
    static double computeRhythmicDensity(const GeneratedMidi& midi);

    static TraceRecord makeTrace(uint64_t sessionId,
                                 uint64_t sequenceId,
                                 ExecutionState fromState,
                                 ExecutionState toState,
                                 const char* transitionId,
                                 ConductorDecision finalStatus);
    static void markHarnessViolation(TraceRecord& trace, const char* code);
    static ConductorDecision fromStatus(IntentResultStatus status);
    static ConductorDecision fromStatus(ConductorDecision decision);

    mutable std::mutex sessionMutex_;
    std::unordered_map<uint64_t, SessionContext> sessions_;
    std::atomic<uint64_t> nextSessionId_{1};
    uint64_t currentSessionId_ = 0;
    IntentPipeline intentPipeline_;
    ChordGenerator chordGenerator_;
};

} // namespace kelly

namespace kelly {

inline std::string StateMachineConductor::toLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

inline bool StateMachineConductor::containsAny(const std::string& lower,
                                               std::initializer_list<const char*> tokens) {
    for (const auto* token : tokens) {
        if (lower.find(token) != std::string::npos) {
            return true;
        }
    }
    return false;
}

inline uint32_t StateMachineConductor::inferRequestedGroupsMask(const std::string& lowerText) {
    uint32_t maskValue = 0;
    if (containsAny(lowerText, {"chord", "harmony", "darker", "smoother"})) {
        maskValue |= mask(DomainGroup::HARMONY);
    }
    if (containsAny(lowerText, {"rhythm", "timing", "energy", "tempo"})) {
        maskValue |= mask(DomainGroup::RHYTHM);
    }
    if (containsAny(lowerText, {"texture", "layer", "instrument"})) {
        maskValue |= mask(DomainGroup::TEXTURE);
    }
    if (containsAny(lowerText, {"melody", "lead"})) {
        maskValue |= mask(DomainGroup::MELODY);
    }
    if (containsAny(lowerText, {"play", "playback"})) {
        maskValue |= mask(DomainGroup::PLAYBACK);
    }
    return maskValue;
}

inline SuggestedVariants StateMachineConductor::buildBoundedAlternatives(const std::string& lowerText) {
    (void)lowerText;
    SuggestedVariants variants;
    variants.items.push_back({"harmony: keep triads, inversion A", mask(DomainGroup::HARMONY)});
    variants.items.push_back({"harmony: keep triads, inversion B", mask(DomainGroup::HARMONY)});
    variants.items.push_back({"harmony: keep triads, inversion C", mask(DomainGroup::HARMONY)});
    return variants;
}

inline void StateMachineConductor::recomputeLengthInBeats(GeneratedMidi& midi) {
    midi.lengthInBeats = 0.0;
    for (const auto& chord : midi.chords) {
        midi.lengthInBeats = std::max(midi.lengthInBeats, chord.startBeat + chord.duration);
    }
    if (midi.melody.has_value()) {
        for (const auto& note : *midi.melody) {
            midi.lengthInBeats = std::max(midi.lengthInBeats, note.startBeat + note.duration);
        }
    }
    if (midi.bass.has_value()) {
        for (const auto& note : *midi.bass) {
            midi.lengthInBeats = std::max(midi.lengthInBeats, note.startBeat + note.duration);
        }
    }
    if (midi.lengthInBeats <= 0.0) {
        midi.lengthInBeats = 16.0;
    }
}

inline void StateMachineConductor::applyPitchVariation(GeneratedMidi& midi, float intensity) {
    if (midi.chords.empty()) {
        return;
    }
    const int semitoneShift = std::clamp(static_cast<int>(std::round(1.0f + intensity * 3.0f)), 1, 4);
    for (size_t i = 0; i < midi.chords.size(); ++i) {
        auto& chord = midi.chords[i];
        if (chord.pitches.empty()) {
            continue;
        }
        const int dir = (i % 2 == 0) ? 1 : -1;
        const size_t voice = chord.pitches.size() - 1;
        chord.pitches[voice] = std::clamp(chord.pitches[voice] + (dir * semitoneShift), 0, 127);
        std::sort(chord.pitches.begin(), chord.pitches.end());
    }
}

inline void StateMachineConductor::applyTimingVariation(GeneratedMidi& midi, float humanizeAmount) {
    if (midi.chords.empty() || humanizeAmount <= 0.0f) {
        return;
    }
    const double maxShift = static_cast<double>(humanizeAmount) * 0.2;
    for (size_t i = 0; i < midi.chords.size(); ++i) {
        auto& chord = midi.chords[i];
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

inline std::vector<int> StateMachineConductor::buildDynamicsContour(const GeneratedMidi& midi,
                                                                    float intensity,
                                                                    float humanizeAmount) {
    std::vector<int> velocities;
    velocities.reserve(midi.chords.size());
    const int base = std::clamp(static_cast<int>(std::round(56.0f + intensity * 44.0f)), 1, 127);
    const int shape = std::clamp(static_cast<int>(std::round(6.0f + humanizeAmount * 8.0f)), 4, 14);
    for (size_t i = 0; i < midi.chords.size(); ++i) {
        const int offset = (i % 2 == 0) ? shape : -shape / 2;
        velocities.push_back(std::clamp(base + offset, 1, 127));
    }
    return velocities;
}

inline double StateMachineConductor::computeRhythmicDensity(const GeneratedMidi& midi) {
    if (midi.lengthInBeats <= 0.0) {
        return 0.0;
    }
    size_t events = 0;
    for (const auto& chord : midi.chords) {
        events += chord.pitches.size();
    }
    if (midi.melody.has_value()) {
        events += midi.melody->size();
    }
    if (midi.bass.has_value()) {
        events += midi.bass->size();
    }
    return static_cast<double>(events) / midi.lengthInBeats;
}

inline TraceRecord StateMachineConductor::makeTrace(uint64_t sessionId,
                                                    uint64_t sequenceId,
                                                    ExecutionState fromState,
                                                    ExecutionState toState,
                                                    const char* transitionId,
                                                    ConductorDecision finalStatus) {
    TraceRecord trace;
    trace.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    trace.monotonicMicros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    trace.sessionId = sessionId;
    trace.sequenceId = sequenceId;
    trace.fromState = fromState;
    trace.toState = toState;
    trace.transitionId = transitionId != nullptr ? transitionId : "";
    trace.finalStatus = finalStatus;
    return trace;
}

inline void StateMachineConductor::markHarnessViolation(TraceRecord& trace, const char* code) {
    trace.harnessViolation = 1;
    trace.harnessViolationCode = code != nullptr ? code : "harness_violation";
}

inline ConductorDecision StateMachineConductor::fromStatus(IntentResultStatus status) {
    switch (status) {
        case IntentResultStatus::VALID: return ConductorDecision::VALID;
        case IntentResultStatus::ABSTAIN: return ConductorDecision::ABSTAIN;
        case IntentResultStatus::INVALID: return ConductorDecision::INVALID;
    }
    return ConductorDecision::ABSTAIN;
}

inline ConductorDecision StateMachineConductor::fromStatus(ConductorDecision decision) {
    return decision;
}

inline StateMachineConductor::SessionContext& StateMachineConductor::ensureSession(uint64_t sessionId) {
    auto it = sessions_.find(sessionId);
    if (it == sessions_.end()) {
        it = sessions_.emplace(sessionId, SessionContext{}).first;
    }
    return it->second;
}

inline uint64_t StateMachineConductor::currentSessionId() const {
    std::lock_guard<std::mutex> lock(sessionMutex_);
    return currentSessionId_;
}

inline StateMachineConductor::SubmitResult StateMachineConductor::submitInput(const UserInput& input) {
    uint64_t sessionId = 0;
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        if (currentSessionId_ == 0) {
            currentSessionId_ = nextSessionId_.fetch_add(1, std::memory_order_relaxed);
        }
        sessionId = currentSessionId_;
    }
    return submitInputForSession(sessionId, input);
}

inline StateMachineConductor::SubmitResult StateMachineConductor::submitInputForSession(uint64_t sessionId,
                                                                                        const UserInput& input) {
    SubmitResult out;
    out.sessionId = sessionId;

    ExecutionState currentState = ExecutionState::Idle;
    uint64_t sequence = 1;
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        currentState = ctx.state;
        sequence = ctx.nextSequenceId;
    }

    if (currentState == ExecutionState::PlaybackRunning) {
        TraceRecord illegal = makeTrace(sessionId,
                                        sequence++,
                                        currentState,
                                        ExecutionState::InvalidRejected,
                                        "ILLEGAL_SUBMIT",
                                        ConductorDecision::INVALID);
        markHarnessViolation(illegal, "illegal_transition_submit_while_playback_running");
        illegal.invalidViolations.push_back("illegal_transition");
        globalDecisionTrace().append(illegal);
        out.sequenceId = illegal.sequenceId;
        out.decision = ConductorDecision::INVALID;
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::InvalidRejected;
        ctx.nextSequenceId = sequence;
        ctx.validatedPlan.reset();
        return out;
    }

    const auto start = std::chrono::steady_clock::now();

    TraceRecord t1 = makeTrace(sessionId,
                               sequence++,
                               currentState,
                               ExecutionState::InputReceived,
                               "T1",
                               ConductorDecision::ABSTAIN);
    globalDecisionTrace().append(t1);

    const std::string normalizedLower = toLower(input.text);
    IntentResult intent;
    intent.requestedGroupsMask = inferRequestedGroupsMask(normalizedLower);
    if (!input.hasAnyDelta()) {
        intent = IntentResult::abstain(AbstainReason::EMPTY_INPUT);
        intent.requestedGroupsMask = 0;
    } else if (!input.hasTextDelta()) {
        intent = IntentResult::abstain(AbstainReason::UNDERSPECIFIED);
        intent.requestedGroupsMask = 0;
    } else {
        // Use designated initializers — the canonical kelly::Wound (KellyTypes.h)
        // has different field order than the original 3-field common/Types.h
        // layout, so positional aggregate-init would silently mis-map fields.
        Wound wound{};
        wound.description = input.text;
        wound.intensity = 0.7f;
        wound.urgency = 0.7f;  // canonical-name alias for intensity
        wound.source = "conductor";
        if (input.parameterDelta.has_value() && input.parameterDelta->id == "intensity") {
            wound.intensity = std::clamp(input.parameterDelta->value, 0.0f, 1.0f);
            wound.urgency = wound.intensity;
        }
        intent = intentPipeline_.process(wound);
        intent.requestedGroupsMask = inferRequestedGroupsMask(normalizedLower);
    }

#if KELLY_ENABLE_FAULT_INJECTION
    if (containsAny(normalizedLower, {"scope_leak", "leak_scope"})) {
        intent.modifiedGroupsMask = mask(DomainGroup::RHYTHM);
        if (intent.requestedGroupsMask == 0) {
            intent.requestedGroupsMask = mask(DomainGroup::HARMONY);
        }
    } else {
        intent.modifiedGroupsMask = intent.requestedGroupsMask;
    }
#else
    intent.modifiedGroupsMask = intent.requestedGroupsMask;
#endif
    intent.leakedGroupsMask = intent.modifiedGroupsMask & (~intent.requestedGroupsMask);

    GuardrailInputContext guardrailContext(
        input.hasAnyDelta(),
        input.hasTextDelta(),
        input.hasPerformanceDelta(),
        input.hasParameterDelta(),
        intent.requestedGroupsMask,
        normalizedLower);
    auto validation = GuardrailValidator::validate(std::move(intent), guardrailContext);
    out.validation = validation.report;

    TraceRecord t2 = makeTrace(sessionId,
                               sequence++,
                               ExecutionState::InputReceived,
                               ExecutionState::GuardrailsChecked,
                               "T2",
                               fromStatus(validation.report.finalStatus));
    t2.rulesEvaluated = validation.report.rulesEvaluated;
    t2.ruleResults = validation.report.ruleResults;
    t2.abstainReason = validation.report.abstainReason;
    t2.invalidViolations = validation.report.violations;
    t2.requestedGroupsMask = validation.report.requestedGroupsMask;
    t2.modifiedGroupsMask = validation.report.modifiedGroupsMask;
    t2.leakedGroupsMask = validation.report.leakedGroupsMask;
    t2.intentMicros = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count());
    globalDecisionTrace().append(t2);

    if (!validation.validated.has_value()) {
        if (validation.report.finalStatus == IntentResultStatus::INVALID) {
            TraceRecord t6 = makeTrace(sessionId,
                                       sequence++,
                                       ExecutionState::GuardrailsChecked,
                                       ExecutionState::InvalidRejected,
                                       "T6-reject",
                                       ConductorDecision::INVALID);
            t6.rulesEvaluated = validation.report.rulesEvaluated;
            t6.ruleResults = validation.report.ruleResults;
            t6.invalidViolations = validation.report.violations;
            t6.requestedGroupsMask = validation.report.requestedGroupsMask;
            t6.modifiedGroupsMask = validation.report.modifiedGroupsMask;
            t6.leakedGroupsMask = validation.report.leakedGroupsMask;
            globalDecisionTrace().append(t6);
            out.sequenceId = t6.sequenceId;
            out.decision = ConductorDecision::INVALID;

            std::lock_guard<std::mutex> lock(sessionMutex_);
                auto& ctx = ensureSession(sessionId);
                ctx.state = ExecutionState::InvalidRejected;
                ctx.nextSequenceId = sequence;
                ctx.validatedPlan.reset();
                ctx.alternatives.items.clear();
                return out;
        }

        if (validation.report.abstainReason == AbstainReason::UNDERSPECIFIED) {
            SuggestedVariants alternatives = buildBoundedAlternatives(toLower(input.text));
            if (!alternatives.items.empty()) {
                alternatives.items.resize(std::min<size_t>(3, alternatives.items.size()));
                out.alternatives = alternatives;
                TraceRecord t7 = makeTrace(sessionId,
                                           sequence++,
                                           ExecutionState::GuardrailsChecked,
                                           ExecutionState::AlternativesProposed,
                                           "T7",
                                           ConductorDecision::ABSTAIN);
                t7.abstainReason = AbstainReason::UNDERSPECIFIED;
                t7.rulesEvaluated = validation.report.rulesEvaluated;
                t7.ruleResults = validation.report.ruleResults;
                globalDecisionTrace().append(t7);
                out.sequenceId = t7.sequenceId;
                out.decision = ConductorDecision::ABSTAIN;

                std::lock_guard<std::mutex> lock(sessionMutex_);
                    auto& ctx = ensureSession(sessionId);
                    ctx.state = ExecutionState::AlternativesProposed;
                    ctx.nextSequenceId = sequence;
                    ctx.validatedPlan.reset();
                    ctx.alternatives = alternatives;
                    ctx.lastInputText = input.text;
                return out;
            }
        }

        TraceRecord tAbstain = makeTrace(sessionId,
                                         sequence++,
                                         ExecutionState::GuardrailsChecked,
                                         ExecutionState::Abstained,
                                         "T8",
                                         ConductorDecision::ABSTAIN);
        tAbstain.abstainReason = validation.report.abstainReason;
        tAbstain.rulesEvaluated = validation.report.rulesEvaluated;
        tAbstain.ruleResults = validation.report.ruleResults;
        globalDecisionTrace().append(tAbstain);
        out.sequenceId = tAbstain.sequenceId;
        out.decision = ConductorDecision::ABSTAIN;

        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::Abstained;
        ctx.nextSequenceId = sequence;
        ctx.validatedPlan.reset();
        ctx.alternatives.items.clear();
        return out;
    }

    float constrainedIntensity = 0.7f;
    float constrainedHumanize = 0.0f;
    bool tempoLockEnabled = true;
    const std::string lower = normalizedLower;
    const bool varyPitchToken = containsAny(lower, {"vary_pitch", "vary pitch"});
    const bool varyTimingToken = containsAny(lower, {"vary_timing", "vary timing"});
    const bool varyDynamicsToken = containsAny(lower, {"vary_dynamics", "vary dynamics"});

    if (input.parameterDelta.has_value()) {
        const auto& p = *input.parameterDelta;
        if (p.id == "intensity") {
            constrainedIntensity = std::clamp(p.value, 0.0f, 1.0f);
        } else if (p.id == "humanize") {
            constrainedHumanize = std::clamp(p.value, 0.0f, 1.0f);
        } else if (p.id == "tempoLock") {
            tempoLockEnabled = p.value >= 0.5f;
        }
    }

    const auto genStart = std::chrono::steady_clock::now();
    GeneratedMidi generated;
    generated.chords = chordGenerator_.generate(*validation.validated);
    generated.bpm = tempoLockEnabled ? 120.0f : std::clamp(120.0f * validation.validated->tempo(), 40.0f, 220.0f);
#if KELLY_ENABLE_FAULT_INJECTION
    if (containsAny(lower, {"force_generator_empty"})) {
        generated.chords.clear();
    }
#endif

    if (generated.chords.empty()) {
        TraceRecord t8 = makeTrace(sessionId,
                                   sequence++,
                                   ExecutionState::GuardrailsChecked,
                                   ExecutionState::Abstained,
                                   "T8-empty",
                                   ConductorDecision::ABSTAIN);
        t8.abstainReason = AbstainReason::GENERATOR_EMPTY;
        t8.rulesEvaluated = validation.report.rulesEvaluated;
        t8.ruleResults = validation.report.ruleResults;
        globalDecisionTrace().append(t8);
        out.sequenceId = t8.sequenceId;
        out.decision = ConductorDecision::ABSTAIN;

        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::Abstained;
        ctx.nextSequenceId = sequence;
        ctx.validatedPlan.reset();
        return out;
    }

    if (varyPitchToken) {
        applyPitchVariation(generated, constrainedIntensity);
    }
    if (varyTimingToken || constrainedHumanize > 0.0f) {
        applyTimingVariation(generated, constrainedHumanize);
    }
    if (varyDynamicsToken) {
        generated.chordVelocities = buildDynamicsContour(generated, constrainedIntensity, constrainedHumanize);
    }
    recomputeLengthInBeats(generated);

    OutputLayerAuthorization auth;
    auth.chordsAuthorized = true;
    auth.melodyAuthorized = validation.validated->escalationTokenPresent() || containsAny(lower, {"melody"});
    auth.bassAuthorized = validation.validated->escalationTokenPresent() || containsAny(lower, {"bass"});
    auth.velocitiesAuthorized = varyDynamicsToken;
    auth.rhythmDensityIncreaseAuthorized = validation.validated->escalationTokenPresent()
        || containsAny(lower, {"rhythm", "timing", "energy"});
    auth.maxVoiceCount = validation.validated->escalationTokenPresent() ? 8 : 3;
    auth.maxChordCount = generated.chords.size();
    auth.maxRhythmicDensity = computeRhythmicDensity(generated);

    auto outputValidation = OutputPlanValidator::validate(std::move(generated), auth);
    if (!outputValidation.validated.has_value()) {
        TraceRecord reject = makeTrace(sessionId,
                                       sequence++,
                                       ExecutionState::GuardrailsChecked,
                                       ExecutionState::InvalidRejected,
                                       "T6-reject",
                                       ConductorDecision::INVALID);
        reject.rulesEvaluated = validation.report.rulesEvaluated;
        reject.ruleResults = validation.report.ruleResults;
        reject.invalidViolations.push_back("output_plan_rejected");
        globalDecisionTrace().append(reject);
        out.sequenceId = reject.sequenceId;
        out.decision = ConductorDecision::INVALID;

        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::InvalidRejected;
        ctx.nextSequenceId = sequence;
        ctx.validatedPlan.reset();
        return out;
    }

    const auto generationMicros = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - genStart).count());
    const auto totalMicros = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count());

    TraceRecord t5 = makeTrace(sessionId,
                               sequence++,
                               ExecutionState::GuardrailsChecked,
                               ExecutionState::OutputReady,
                               "T5",
                               ConductorDecision::VALID);
    t5.rulesEvaluated = validation.report.rulesEvaluated;
    t5.ruleResults = validation.report.ruleResults;
    t5.requestedGroupsMask = validation.report.requestedGroupsMask;
    t5.modifiedGroupsMask = validation.report.modifiedGroupsMask;
    t5.leakedGroupsMask = validation.report.leakedGroupsMask;
    t5.outputChordCount = static_cast<uint16_t>(outputValidation.validated->midi().chords.size());
    t5.outputMaxVoices = auth.maxVoiceCount;
    t5.outputHasMelody = outputValidation.validated->midi().melody.has_value() ? 1 : 0;
    t5.outputHasBass = outputValidation.validated->midi().bass.has_value() ? 1 : 0;
    t5.intentMicros = t2.intentMicros;
    t5.generationMicros = generationMicros;
    t5.totalMicros = totalMicros;
    globalDecisionTrace().append(t5);

    out.sequenceId = t5.sequenceId;
    out.decision = ConductorDecision::VALID;
    out.midi = outputValidation.validated->midi();

    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::OutputReady;
        ctx.nextSequenceId = sequence;
        ctx.validatedPlan = std::move(*outputValidation.validated);
        ctx.alternatives.items.clear();
        ctx.lastInputText = input.text;
    }
    return out;
}

inline StateMachineConductor::AlternativeResult StateMachineConductor::applyAlternative(uint64_t sessionId,
                                                                                         size_t index) {
    AlternativeResult out;
    out.sessionId = sessionId;
    out.selectedIndex = index;

    ExecutionState currentState = ExecutionState::Idle;
    uint64_t sequence = 1;
    SuggestedVariants alternatives;
    std::string lastInputText;
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        currentState = ctx.state;
        sequence = ctx.nextSequenceId;
        alternatives = ctx.alternatives;
        lastInputText = ctx.lastInputText;
    }

    if (currentState != ExecutionState::AlternativesProposed || index >= alternatives.items.size()) {
        TraceRecord illegal = makeTrace(sessionId,
                                        sequence++,
                                        currentState,
                                        ExecutionState::InvalidRejected,
                                        "ILLEGAL_ALT",
                                        ConductorDecision::INVALID);
        markHarnessViolation(illegal, "illegal_transition_apply_alternative");
        illegal.invalidViolations.push_back("illegal_transition");
        globalDecisionTrace().append(illegal);

        out.sequenceId = illegal.sequenceId;
        out.decision = ConductorDecision::INVALID;
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = currentState;
        ctx.nextSequenceId = sequence;
        return out;
    }

    TraceRecord t10 = makeTrace(sessionId,
                                sequence++,
                                ExecutionState::AlternativesProposed,
                                ExecutionState::InputReceived,
                                "T10",
                                ConductorDecision::VALID);
    globalDecisionTrace().append(t10);

    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.nextSequenceId = sequence;
    }

    UserInput refined;
    refined.text = lastInputText + " " + alternatives.items[index].description;
    auto submit = submitInputForSession(sessionId, refined);
    out.sequenceId = submit.sequenceId;
    out.decision = submit.decision;
    return out;
}

inline StateMachineConductor::PlaybackResult StateMachineConductor::requestPlaybackStart(uint64_t sessionId,
                                                                                          bool userInitiatedFlag) {
    PlaybackResult out;
    out.sessionId = sessionId;

    ExecutionState currentState = ExecutionState::Idle;
    uint64_t sequence = 1;
    bool hasValidatedPlan = false;
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        currentState = ctx.state;
        sequence = ctx.nextSequenceId;
        hasValidatedPlan = ctx.validatedPlan.has_value();
    }

    if (!userInitiatedFlag) {
        TraceRecord blocked = makeTrace(sessionId,
                                        sequence++,
                                        currentState,
                                        ExecutionState::InvalidRejected,
                                        "T11-blocked",
                                        ConductorDecision::INVALID);
        markHarnessViolation(blocked, "playback_without_user_initiation");
        blocked.playbackUserInitiated = 0;
        blocked.playbackAuthorized = 0;
        globalDecisionTrace().append(blocked);

        out.sequenceId = blocked.sequenceId;
        out.decision = ConductorDecision::INVALID;
        out.playbackAuthorized = false;
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = currentState;
        ctx.nextSequenceId = sequence;
        return out;
    }

    if (!hasValidatedPlan) {
        TraceRecord blocked = makeTrace(sessionId,
                                        sequence++,
                                        currentState,
                                        ExecutionState::InvalidRejected,
                                        "T11-blocked",
                                        ConductorDecision::INVALID);
        markHarnessViolation(blocked, "playback_without_validated_output_plan");
        blocked.playbackUserInitiated = 1;
        blocked.playbackAuthorized = 0;
        globalDecisionTrace().append(blocked);

        out.sequenceId = blocked.sequenceId;
        out.decision = ConductorDecision::INVALID;
        out.playbackAuthorized = false;
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::InvalidRejected;
        ctx.nextSequenceId = sequence;
        return out;
    }

    if (currentState != ExecutionState::OutputReady && currentState != ExecutionState::PlaybackReady) {
        TraceRecord illegal = makeTrace(sessionId,
                                        sequence++,
                                        currentState,
                                        ExecutionState::InvalidRejected,
                                        "ILLEGAL_PLAYBACK_START",
                                        ConductorDecision::INVALID);
        markHarnessViolation(illegal, "illegal_transition_playback_start");
        illegal.invalidViolations.push_back("illegal_transition");
        illegal.playbackUserInitiated = 1;
        globalDecisionTrace().append(illegal);

        out.sequenceId = illegal.sequenceId;
        out.decision = ConductorDecision::INVALID;
        out.playbackAuthorized = false;
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::InvalidRejected;
        ctx.nextSequenceId = sequence;
        return out;
    }

    auto token = UserActionAuthority::authorizePlayback(true);
    if (!token.isAuthorized()) {
        TraceRecord abstain = makeTrace(sessionId,
                                        sequence++,
                                        currentState,
                                        ExecutionState::Abstained,
                                        "T11",
                                        ConductorDecision::ABSTAIN);
        abstain.abstainReason = AbstainReason::NON_AUDITABLE_PATH;
        globalDecisionTrace().append(abstain);
        out.sequenceId = abstain.sequenceId;
        out.decision = ConductorDecision::ABSTAIN;
        out.playbackAuthorized = false;
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::Abstained;
        ctx.nextSequenceId = sequence;
        return out;
    }

    TraceRecord t11 = makeTrace(sessionId,
                                sequence++,
                                currentState,
                                ExecutionState::PlaybackReady,
                                "T11",
                                ConductorDecision::VALID);
    t11.playbackUserInitiated = 1;
    t11.playbackAuthorized = 1;
    globalDecisionTrace().append(t11);

    TraceRecord t12 = makeTrace(sessionId,
                                sequence++,
                                ExecutionState::PlaybackReady,
                                ExecutionState::PlaybackRunning,
                                "T12",
                                ConductorDecision::VALID);
    t12.playbackUserInitiated = 1;
    t12.playbackAuthorized = 1;
    t12.playbackStarted = 1;
    globalDecisionTrace().append(t12);

    out.sequenceId = t12.sequenceId;
    out.decision = ConductorDecision::VALID;
    out.playbackAuthorized = true;
    std::lock_guard<std::mutex> lock(sessionMutex_);
    auto& ctx = ensureSession(sessionId);
    ctx.state = ExecutionState::PlaybackRunning;
    ctx.nextSequenceId = sequence;
    return out;
}

inline StateMachineConductor::PlaybackResult StateMachineConductor::requestPlaybackStop(uint64_t sessionId) {
    PlaybackResult out;
    out.sessionId = sessionId;

    ExecutionState currentState = ExecutionState::Idle;
    uint64_t sequence = 1;
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        currentState = ctx.state;
        sequence = ctx.nextSequenceId;
    }

    if (currentState != ExecutionState::PlaybackRunning) {
        TraceRecord illegal = makeTrace(sessionId,
                                        sequence++,
                                        currentState,
                                        ExecutionState::InvalidRejected,
                                        "ILLEGAL_PLAYBACK_STOP",
                                        ConductorDecision::INVALID);
        markHarnessViolation(illegal, "illegal_transition_playback_stop");
        illegal.invalidViolations.push_back("illegal_transition");
        globalDecisionTrace().append(illegal);

        out.sequenceId = illegal.sequenceId;
        out.decision = ConductorDecision::INVALID;
        out.playbackAuthorized = false;
        std::lock_guard<std::mutex> lock(sessionMutex_);
        auto& ctx = ensureSession(sessionId);
        ctx.state = ExecutionState::InvalidRejected;
        ctx.nextSequenceId = sequence;
        return out;
    }

    TraceRecord t13 = makeTrace(sessionId,
                                sequence++,
                                ExecutionState::PlaybackRunning,
                                ExecutionState::Idle,
                                "T13",
                                ConductorDecision::VALID);
    t13.playbackStopped = 1;
    globalDecisionTrace().append(t13);

    out.sequenceId = t13.sequenceId;
    out.decision = ConductorDecision::VALID;
    out.playbackAuthorized = false;
    std::lock_guard<std::mutex> lock(sessionMutex_);
    auto& ctx = ensureSession(sessionId);
    ctx.state = ExecutionState::Idle;
    ctx.nextSequenceId = sequence;
    return out;
}

} // namespace kelly
