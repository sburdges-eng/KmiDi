#pragma once

/**
 * OSCBridge - Unified OSC Client for C++ Plugin ↔ Python Brain Communication
 * ===========================================================================
 *
 * Implements OSC client functionality to communicate with the Python brain server.
 * This enables the hybrid architecture:
 * - C++ Body (plugin): Sends requests via OSC
 * - Python Brain (server): Processes requests and responds
 * - OSC Bridge: Communication layer
 *
 * Unified implementation combining OSCBridge and OSCClient features:
 * - Uses RealtimeCallback for better real-time performance
 * - Message ID-based response matching (robust)
 * - Timeout handling for requests
 * - Supports both juce::var and std::string callbacks
 */

#include "bridge/BridgeBase.h"
#include <juce_core/juce_core.h>
#include <juce_osc/juce_osc.h>
#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <map>

namespace kelly {
namespace bridge {

/**
 * OSC Response handler callback (juce::var version)
 */
using OSCResponseHandler = std::function<void(const juce::var& response)>;

/**
 * OSC Response handler callback (string version - simpler)
 */
using OSCStringResponseHandler = std::function<void(const std::string& response)>;

/**
 * OSCBridge - Unified client for communicating with Python brain server
 *
 * Uses RealtimeCallback for better real-time performance.
 * Supports both initialize()/shutdown() and connect()/disconnect() APIs.
 */
class OSCBridge : public BridgeBase, public juce::OSCReceiver::Listener<juce::OSCReceiver::RealtimeCallback> {
public:
    OSCBridge();
    ~OSCBridge() override;

    // BridgeBase interface
    bool initialize() override;
    bool initialize(
        const std::string& brainHost,
        int brainPort,
        int listenPort
    );
    void shutdown() override;

    // Legacy API (for backward compatibility with OSCClient)
    bool connect(const std::string& host = "127.0.0.1",
                 int serverPort = 5005,
                 int responsePort = 5006);
    void disconnect();

    /**
     * Check if bridge is connected
     */
    bool isConnected() const { return connected_; }

    /**
     * Process incoming OSC messages (call from message thread or timer)
     * Cleans up timed-out requests
     */
    void processMessages();

    /**
     * Send generation request to brain (juce::var callback)
     * @param text User input text
     * @param motivation Motivation description
     * @param chaos Chaos level (0.0-1.0)
     * @param vulnerability Vulnerability level (0.0-1.0)
     * @param callback Response handler
     */
    void requestGenerate(
        const std::string& text,
        const std::string& motivation,
        float chaos,
        float vulnerability,
        OSCResponseHandler callback
    );

    /**
     * Send generation request to brain (string callback - simpler API)
     * @param text User input text
     * @param motivation Motivation parameter (0.0-1.0)
     * @param chaos Chaos parameter (0.0-1.0)
     * @param vulnerability Vulnerability parameter (0.0-1.0)
     * @param callback Response handler with JSON string
     */
    void requestGenerate(
        const std::string& text,
        float motivation = 0.5f,
        float chaos = 0.5f,
        float vulnerability = 0.5f,
        OSCStringResponseHandler callback = nullptr
    );

    /**
     * Request chord analysis (juce::var callback)
     * @param progression Chord progression string
     * @param callback Response handler
     */
    void requestAnalyzeChords(
        const std::string& progression,
        OSCResponseHandler callback
    );

    /**
     * Request chord analysis (string callback)
     * @param progression Chord progression string
     * @param callback Response handler with JSON string
     */
    void requestAnalyzeChords(
        const std::string& progression,
        OSCStringResponseHandler callback = nullptr
    );

    /**
     * Request intent processing (juce::var callback)
     * @param intentFile Path to intent file
     * @param callback Response handler
     */
    void requestIntentProcess(
        const std::string& intentFile,
        OSCResponseHandler callback
    );

    /**
     * Request intent processing (string callback)
     * @param intentFile Path to intent file
     * @param callback Response handler with JSON string
     */
    void requestIntentProcess(
        const std::string& intentFile,
        OSCStringResponseHandler callback = nullptr
    );

    /**
     * Request intent suggestions (juce::var callback)
     * @param emotion Emotion name
     * @param callback Response handler
     */
    void requestIntentSuggest(
        const std::string& emotion,
        OSCResponseHandler callback
    );

    /**
     * Request intent suggestions (string callback)
     * @param emotion Emotion name
     * @param callback Response handler with JSON string
     */
    void requestIntentSuggest(
        const std::string& emotion,
        OSCStringResponseHandler callback = nullptr
    );

    /**
     * Send ping to check brain server availability (juce::var callback)
     * @param callback Response handler
     */
    void ping(OSCResponseHandler callback);

    /**
     * Send ping to check brain server availability (string callback)
     * @param callback Response handler with JSON string
     */
    void ping(OSCStringResponseHandler callback = nullptr);

private:
    bool connected_ = false;
    std::string brainHost_;
    int brainPort_ = 5005;
    int listenPort_ = 5006;

    // Response handlers (message ID -> handler)
    struct PendingRequest {
        OSCResponseHandler varCallback;
        OSCStringResponseHandler stringCallback;
        juce::Time timestamp;
    };
    std::map<int, PendingRequest> pendingRequests_;
    int nextMessageId_ = 1;
    static constexpr int REQUEST_TIMEOUT_MS = 5000;

    // OSC sender and receiver
    std::unique_ptr<juce::OSCSender> sender_;
    std::unique_ptr<juce::OSCReceiver> receiver_;

    /**
     * Generate unique message ID
     */
    int generateMessageId() { return nextMessageId_++; }

    /**
     * OSCReceiver::Listener callbacks
     */
    void oscMessageReceived(const juce::OSCMessage& message) override;
    void oscBundleReceived(const juce::OSCBundle& bundle) override;

    /**
     * Handle incoming OSC message by address pattern
     */
    void handleResponseMessage(const juce::OSCMessage& message, int msgId);

    /**
     * Clean up timed-out requests
     */
    void cleanupTimedOutRequests();
};

} // namespace bridge
} // namespace kelly
