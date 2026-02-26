#pragma once

#include <cstdint>

namespace kelly {

enum class ExecutionState : uint8_t {
    UNKNOWN = 0,
    Idle,
    InputReceived,
    GuardrailsChecked,
    InvalidRejected,
    Abstained,
    AlternativesProposed,
    OutputReady,
    PlaybackReady,
    PlaybackRunning
};

inline const char* toString(ExecutionState state) noexcept {
    switch (state) {
        case ExecutionState::UNKNOWN: return "UNKNOWN";
        case ExecutionState::Idle: return "Idle";
        case ExecutionState::InputReceived: return "InputReceived";
        case ExecutionState::GuardrailsChecked: return "GuardrailsChecked";
        case ExecutionState::InvalidRejected: return "InvalidRejected";
        case ExecutionState::Abstained: return "Abstained";
        case ExecutionState::AlternativesProposed: return "AlternativesProposed";
        case ExecutionState::OutputReady: return "OutputReady";
        case ExecutionState::PlaybackReady: return "PlaybackReady";
        case ExecutionState::PlaybackRunning: return "PlaybackRunning";
    }
    return "UNKNOWN";
}

} // namespace kelly
