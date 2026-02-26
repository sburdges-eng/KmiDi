#include "common/DecisionTrace.h"
#include "engine/StateMachineConductor.h"

#include <iostream>
#include <string>
#include <vector>

#ifndef KELLY_ENABLE_FAULT_INJECTION
#define KELLY_ENABLE_FAULT_INJECTION 0
#endif

namespace {

struct TestCase {
    const char* name;
    bool (*run)();
};

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "  FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

bool testValidPathSequenceAndRuleBits() {
    kelly::globalDecisionTrace().clear();
    kelly::StateMachineConductor conductor;

    kelly::UserInput input;
    input.text = "darker";

    const auto result = conductor.submitInput(input);
    if (!expect(result.decision == kelly::ConductorDecision::VALID, "valid request should succeed")) return false;
    if (!expect(result.midi.has_value(), "valid request should return MIDI")) return false;

    const auto transcript = kelly::globalDecisionTrace().getSessionTranscript(result.sessionId);
    if (!expect(transcript.stateSequence.size() == 4, "valid path should have 4 states")) return false;
    if (!expect(transcript.stateSequence[0] == kelly::ExecutionState::Idle, "state[0] should be Idle")) return false;
    if (!expect(transcript.stateSequence[1] == kelly::ExecutionState::InputReceived, "state[1] should be InputReceived")) return false;
    if (!expect(transcript.stateSequence[2] == kelly::ExecutionState::GuardrailsChecked, "state[2] should be GuardrailsChecked")) return false;
    if (!expect(transcript.stateSequence[3] == kelly::ExecutionState::OutputReady, "state[3] should be OutputReady")) return false;
    if (!expect(transcript.terminalState == kelly::ExecutionState::OutputReady, "terminal state should be OutputReady")) return false;

    bool sawT2 = false;
    bool sawT5 = false;
    for (const auto& record : transcript.records) {
        if (record.transitionId == "T2") {
            sawT2 = true;
            if (!expect(record.ruleResults == 0x1F, "T2 should report full rule bitfield 0x1F")) return false;
        }
        if (record.transitionId == "T5") {
            sawT5 = true;
            if (!expect(record.ruleResults == 0x1F, "T5 should carry full rule bitfield 0x1F")) return false;
        }
    }
    if (!expect(sawT2, "valid path should include T2")) return false;
    if (!expect(sawT5, "valid path should include T5")) return false;
    return true;
}

bool testEmptyInputAbstainsNoOutput() {
    kelly::globalDecisionTrace().clear();
    kelly::StateMachineConductor conductor;

    kelly::UserInput input;
    input.text = "   ";
    const auto result = conductor.submitInput(input);

    if (!expect(result.decision == kelly::ConductorDecision::ABSTAIN, "empty input should ABSTAIN")) return false;
    if (!expect(!result.midi.has_value(), "empty input should not produce MIDI")) return false;

    const auto transcript = kelly::globalDecisionTrace().getSessionTranscript(result.sessionId);
    if (!expect(transcript.terminalState == kelly::ExecutionState::Abstained, "terminal state should be Abstained")) return false;
    if (!expect(!transcript.records.empty(), "trace should exist")) return false;
    const auto& last = transcript.records.back();
    if (!expect(last.abstainReason == kelly::AbstainReason::EMPTY_INPUT, "abstain reason should be EMPTY_INPUT")) return false;
    return true;
}

bool testScopeLeakTriggersT6Reject() {
    kelly::globalDecisionTrace().clear();
    kelly::StateMachineConductor conductor;

    kelly::UserInput input;
    input.text = "darker scope_leak";
    const auto result = conductor.submitInput(input);

#if KELLY_ENABLE_FAULT_INJECTION
    if (!expect(result.decision == kelly::ConductorDecision::INVALID, "scope leak should be INVALID")) return false;
    if (!expect(!result.midi.has_value(), "invalid scope leak should not produce MIDI")) return false;

    const auto transcript = kelly::globalDecisionTrace().getSessionTranscript(result.sessionId);
    if (!expect(!transcript.records.empty(), "trace should exist")) return false;
    const auto& last = transcript.records.back();
    if (!expect(last.transitionId == "T6-reject", "scope leak should end at T6-reject")) return false;
    if (!expect(last.leakedGroupsMask != 0, "scope leak should set leakedGroupsMask")) return false;
    return true;
#else
    if (!expect(result.decision == kelly::ConductorDecision::VALID,
                "scope leak token should be inert when fault injection is disabled")) return false;
    return expect(result.midi.has_value(), "fault injection disabled path should produce MIDI");
#endif
}

bool testGeneratorEmptyTriggersT8Empty() {
    kelly::globalDecisionTrace().clear();
    kelly::StateMachineConductor conductor;

    kelly::UserInput input;
    input.text = "darker force_generator_empty";
    const auto result = conductor.submitInput(input);

#if KELLY_ENABLE_FAULT_INJECTION
    if (!expect(result.decision == kelly::ConductorDecision::ABSTAIN, "generator empty should ABSTAIN")) return false;
    if (!expect(!result.midi.has_value(), "generator empty should not produce MIDI")) return false;

    const auto transcript = kelly::globalDecisionTrace().getSessionTranscript(result.sessionId);
    if (!expect(!transcript.records.empty(), "trace should exist")) return false;
    const auto& last = transcript.records.back();
    if (!expect(last.transitionId == "T8-empty", "empty generation should emit T8-empty")) return false;
    if (!expect(last.abstainReason == kelly::AbstainReason::GENERATOR_EMPTY, "abstain reason should be GENERATOR_EMPTY")) return false;
    return true;
#else
    if (!expect(result.decision == kelly::ConductorDecision::VALID,
                "force_generator_empty token should be inert when fault injection is disabled")) return false;
    return expect(result.midi.has_value(), "fault injection disabled path should produce MIDI");
#endif
}

bool testPlaybackRequiresExplicitUserInitiation() {
    kelly::globalDecisionTrace().clear();
    kelly::StateMachineConductor conductor;

    kelly::UserInput input;
    input.text = "darker";
    const auto submit = conductor.submitInput(input);
    if (!expect(submit.decision == kelly::ConductorDecision::VALID, "setup submit should be VALID")) return false;

    const auto blocked = conductor.requestPlaybackStart(submit.sessionId, false);
    if (!expect(blocked.decision == kelly::ConductorDecision::INVALID, "non-user playback start should be INVALID")) return false;
    if (!expect(!blocked.playbackAuthorized, "blocked start should not authorize playback")) return false;

    const auto started = conductor.requestPlaybackStart(submit.sessionId, true);
    if (!expect(started.decision == kelly::ConductorDecision::VALID, "user-initiated start should be VALID")) return false;
    if (!expect(started.playbackAuthorized, "user-initiated start should authorize playback")) return false;

    const auto stopped = conductor.requestPlaybackStop(submit.sessionId);
    if (!expect(stopped.decision == kelly::ConductorDecision::VALID, "stop from running should be VALID")) return false;

    const auto transcript = kelly::globalDecisionTrace().getSessionTranscript(submit.sessionId);
    bool sawBlocked = false;
    bool sawT11 = false;
    bool sawT12 = false;
    bool sawT13 = false;
    for (const auto& record : transcript.records) {
        if (record.harnessViolation && record.harnessViolationCode == "playback_without_user_initiation") {
            sawBlocked = true;
        }
        if (record.transitionId == "T11" && record.playbackAuthorized == 1) {
            sawT11 = true;
        }
        if (record.transitionId == "T12" && record.playbackStarted == 1) {
            sawT12 = true;
        }
        if (record.transitionId == "T13" && record.playbackStopped == 1) {
            sawT13 = true;
        }
    }
    if (!expect(sawBlocked, "blocked autoplay should emit harness violation")) return false;
    if (!expect(sawT11, "authorized start should emit T11")) return false;
    if (!expect(sawT12, "authorized start should emit T12")) return false;
    if (!expect(sawT13, "stop should emit T13")) return false;
    return true;
}

bool testIllegalTransitionRecorded() {
    kelly::globalDecisionTrace().clear();
    kelly::StateMachineConductor conductor;

    const auto result = conductor.requestPlaybackStop(999);
    if (!expect(result.decision == kelly::ConductorDecision::INVALID, "illegal stop should be INVALID")) return false;

    const auto records = kelly::globalDecisionTrace().getBySessionId(999);
    if (!expect(!records.empty(), "illegal transition should still emit trace")) return false;
    const auto& last = records.back();
    if (!expect(last.harnessViolation == 1, "illegal transition should set harnessViolation")) return false;
    if (!expect(last.harnessViolationCode == "illegal_transition_playback_stop",
                "illegal transition should set expected violation code")) return false;
    return true;
}

} // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"Valid path state sequence + rule bits", testValidPathSequenceAndRuleBits},
        {"Empty input ABSTAIN no output", testEmptyInputAbstainsNoOutput},
        {"Scope leak -> T6-reject INVALID", testScopeLeakTriggersT6Reject},
        {"Generator empty -> T8-empty ABSTAIN", testGeneratorEmptyTriggersT8Empty},
        {"Playback requires explicit user-init", testPlaybackRequiresExplicitUserInitiation},
        {"Illegal transition emits harness violation", testIllegalTransitionRecorded},
    };

    int failed = 0;
    for (const auto& test : tests) {
        std::cout << "[TEST] " << test.name << std::endl;
        if (!test.run()) {
            ++failed;
        }
    }

    if (failed > 0) {
        std::cerr << failed << " state-machine conformance test(s) failed." << std::endl;
        return 1;
    }

    std::cout << "All state-machine conformance tests passed." << std::endl;
    return 0;
}
