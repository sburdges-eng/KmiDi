#include "plugin/PluginProcessor.h"
#include "engine/StateMachineConductor.h"

namespace kelly::creative_interface_layer {

namespace {

StateMachineConductor& uiConductor() {
    static StateMachineConductor conductor;
    return conductor;
}

} // namespace

GeneratedMidi submitIntent(PluginProcessor& processor, const std::string& intentText) {
    UserInput input;
    input.text = intentText;
    const auto submit = uiConductor().submitInput(input);
    if (submit.decision != ConductorDecision::VALID || !submit.midi.has_value()) {
        return {};
    }
    return processor.runTrustMVP(input);
}

bool requestPlaybackStart(PluginProcessor& processor) {
    const auto sessionId = uiConductor().currentSessionId();
    const auto gate = uiConductor().requestPlaybackStart(sessionId, true);
    if (gate.decision != ConductorDecision::VALID || !gate.playbackAuthorized) {
        return false;
    }
    processor.startPlayback(true);
    return processor.isPlaying();
}

void requestPlaybackStop(PluginProcessor& processor) {
    const auto sessionId = uiConductor().currentSessionId();
    const auto gate = uiConductor().requestPlaybackStop(sessionId);
    if (gate.decision == ConductorDecision::VALID) {
        processor.stopPlayback();
    }
}

} // namespace kelly::creative_interface_layer

