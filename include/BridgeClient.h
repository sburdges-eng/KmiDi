#pragma once

#include <juce_core/juce_core.h>
#include <juce_osc/juce_osc.h>
#include <functional>
#include <memory>
#include <atomic>

class VoiceProcessor;

/**
 * OSC-based bridge client for Python DAiW runtime communication.
 *
 * Handles bidirectional OSC communication:
 * - Receives voice model data and phonemes from Python
 * - Sends status updates and completion notifications back
 */
class BridgeClient : public juce::OSCReceiver::Listener<juce::OSCReceiver::MessageLoopCallback>
{
public:
    BridgeClient();
    ~BridgeClient();

    //==========================================================================
    // Connection management

    /** Connect to Python runtime (starts OSC server) */
    bool connect(int receivePort = 9000, const juce::String& sendHost = "127.0.0.1", int sendPort = 9001);

    /** Disconnect from Python runtime */
    void disconnect();

    /** Check if connected */
    bool isConnected() const { return connected_.load(); }

    /** Attempts to reach the DAiW runtime (ping) */
    bool ping();

    //==========================================================================
    // Voice processor integration

    /** Set the voice processor to control */
    void setVoiceProcessor(VoiceProcessor* processor) { voiceProcessor_ = processor; }

    //==========================================================================
    // Outgoing messages (C++ → Python)

    /** Send status update to Python */
    void sendStatus(const juce::String& status);

    /** Send phoneme completion notification */
    void sendPhonemeComplete(int phonemeIndex);

    /** Send error message */
    void sendError(const juce::String& error);

    //==========================================================================
    // Legacy API (kept for compatibility)

    /** Request auto-tune processing */
    bool requestAutoTune(const juce::File& inputFile, juce::File& outputFile);

    /** Send chat message */
    juce::String sendChatMessage(const juce::String& message);

    //==========================================================================
    // OSC message handling

    void oscMessageReceived(const juce::OSCMessage& message) override;

private:
    // OSC communication
    std::unique_ptr<juce::OSCReceiver> oscReceiver_;
    std::unique_ptr<juce::OSCSender> oscSender_;

    std::atomic<bool> connected_{ false };
    juce::String sendHost_;
    int sendPort_ = 9001;

    // Voice processor reference
    VoiceProcessor* voiceProcessor_ = nullptr;

    // Message handlers
    void handleVoiceModelLoad(const juce::OSCMessage& message);
    void handleSpeakText(const juce::OSCMessage& message);
    void handlePhoneme(const juce::OSCMessage& message);
    void handleVowelSet(const juce::OSCMessage& message);
    void handlePitchSet(const juce::OSCMessage& message);
    void handleNoteOn(const juce::OSCMessage& message);
    void handleNoteOff(const juce::OSCMessage& message);
    void handleFormantShift(const juce::OSCMessage& message);
    void handleBreathiness(const juce::OSCMessage& message);
    void handleVibrato(const juce::OSCMessage& message);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BridgeClient)
};
