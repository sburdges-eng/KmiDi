#pragma once

#include "common/ExecutionState.h"
#include "common/Types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kelly {

enum class ConductorDecision : uint8_t {
    VALID = 0,
    ABSTAIN = 1,
    INVALID = 2
};

struct TraceRecord {
    // Display timestamp only (Unix epoch millis). Not part of integrity hash.
    uint64_t timestamp = 0;
    // Monotonic ordering clock used for integrity hash + ordering.
    uint64_t monotonicMicros = 0;
    uint64_t sessionId = 0;
    uint64_t sequenceId = 0;
    ExecutionState fromState = ExecutionState::UNKNOWN;
    ExecutionState toState = ExecutionState::UNKNOWN;
    std::string transitionId;

    uint8_t rulesEvaluated = 5;
    uint8_t ruleResults = 0;
    ConductorDecision finalStatus = ConductorDecision::ABSTAIN;
    AbstainReason abstainReason = AbstainReason::NONE;
    std::vector<std::string> invalidViolations;

    uint32_t requestedGroupsMask = 0;
    uint32_t modifiedGroupsMask = 0;
    uint32_t leakedGroupsMask = 0;

    uint16_t outputChordCount = 0;
    uint8_t outputMaxVoices = 0;
    uint8_t outputHasMelody = 0;
    uint8_t outputHasBass = 0;

    uint8_t playbackUserInitiated = 0;
    uint8_t playbackAuthorized = 0;
    uint8_t playbackStarted = 0;
    uint8_t playbackStopped = 0;

    uint32_t intentMicros = 0;
    uint32_t generationMicros = 0;
    uint32_t totalMicros = 0;

    uint8_t harnessViolation = 0;
    std::string harnessViolationCode;
    uint8_t auditCommitted = 1;
    uint8_t tokenChainLinked = 1;
    uint64_t prevHash = 0;
    uint64_t recordHash = 0;
};

struct SessionTranscript {
    uint64_t sessionId = 0;
    std::vector<ExecutionState> stateSequence;
    ExecutionState terminalState = ExecutionState::UNKNOWN;
    std::vector<TraceRecord> records;
};

inline const char* toString(ConductorDecision decision) noexcept {
    switch (decision) {
        case ConductorDecision::VALID: return "VALID";
        case ConductorDecision::ABSTAIN: return "ABSTAIN";
        case ConductorDecision::INVALID: return "INVALID";
    }
    return "ABSTAIN";
}

inline const char* toString(AbstainReason reason) noexcept {
    switch (reason) {
        case AbstainReason::NONE: return "NONE";
        case AbstainReason::EMPTY_INPUT: return "EMPTY_INPUT";
        case AbstainReason::UNDERSPECIFIED: return "UNDERSPECIFIED";
        case AbstainReason::GENERATOR_EMPTY: return "GENERATOR_EMPTY";
        case AbstainReason::NON_AUDITABLE_PATH: return "NON_AUDITABLE_PATH";
    }
    return "NONE";
}

} // namespace kelly
