#include "bridge/OSCBridge.h"
#include <juce_osc/juce_osc.h>
#include <juce_core/juce_core.h>

namespace kelly {
namespace bridge {

OSCBridge::OSCBridge()
    : BridgeBase("OSCBridge")
{
    sender_ = std::make_unique<juce::OSCSender>();
    receiver_ = std::make_unique<juce::OSCReceiver>();
}

OSCBridge::~OSCBridge() {
    shutdown();
}

bool OSCBridge::initialize() {
    return initialize("127.0.0.1", 5005, 5006);
}

bool OSCBridge::initialize(
    const std::string& brainHost,
    int brainPort,
    int listenPort)
{
    if (connected_) {
        shutdown();
    }

    brainHost_ = brainHost;
    brainPort_ = brainPort;
    listenPort_ = listenPort;

    try {
        // Create OSC sender
        if (!sender_->connect(brainHost, brainPort)) {
            logError("Failed to connect sender to " + brainHost + ":" + std::to_string(brainPort));
            return false;
        }

        // Create OSC receiver
        if (!receiver_->connect(listenPort)) {
            logError("Failed to bind receiver to port " + std::to_string(listenPort));
            sender_->disconnect();
            return false;
        }

        // Register message handler
        receiver_->addListener(this);

        connected_ = true;
        setAvailable(true);
        logInfo("Connected to brain server at " + brainHost + ":" + std::to_string(brainPort));
        return true;
    } catch (const std::exception& e) {
        logError("Exception during initialization: " + std::string(e.what()));
        return false;
    }
}

bool OSCBridge::connect(const std::string& host, int serverPort, int responsePort) {
    return initialize(host, serverPort, responsePort);
}

void OSCBridge::disconnect() {
    shutdown();
}

void OSCBridge::requestGenerate(
    const std::string& text,
    const std::string& motivation,
    float chaos,
    float vulnerability,
    OSCResponseHandler callback)
{
    if (!connected_) {
        logError("Not connected, cannot send generate request");
        return;
    }

    // Create JSON parameters
    juce::DynamicObject::Ptr params = new juce::DynamicObject();
    params->setProperty("text", juce::String(text));
    params->setProperty("motivation", juce::String(motivation));
    params->setProperty("chaos", chaos);
    params->setProperty("vulnerability", vulnerability);

    juce::var paramsVar(params);
    juce::String jsonParams = juce::JSON::toString(paramsVar);

    // Store callback
    int msgId = generateMessageId();
    PendingRequest request;
    request.varCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    // Send OSC message
    juce::OSCMessage msg("/daiw/generate");
    msg.addString(jsonParams.toStdString());
    msg.addInt32(msgId);  // Include message ID for response matching

    if (!sender_->send(msg)) {
        logError("Failed to send generate request");
        pendingRequests_.erase(msgId);
    }
}

void OSCBridge::requestGenerate(
    const std::string& text,
    float motivation,
    float chaos,
    float vulnerability,
    OSCStringResponseHandler callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    // Create JSON request
    juce::String jsonRequest = juce::String::formatted(
        R"({"text":"%s","motivation":%.2f,"chaos":%.2f,"vulnerability":%.2f,"response_port":%d})",
        text.c_str(), motivation, chaos, vulnerability, listenPort_
    );

    // Store callback
    int msgId = generateMessageId();
    PendingRequest request;
    request.stringCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    // Send OSC message
    juce::OSCMessage msg("/daiw/generate");
    msg.addString(jsonRequest.toStdString());
    msg.addInt32(msgId);

    if (!sender_->send(msg)) {
        logError("Failed to send generate request");
        if (callback) {
            callback(R"({"status":"error","message":"Send failed"})");
        }
        pendingRequests_.erase(msgId);
    }
}

void OSCBridge::requestAnalyzeChords(
    const std::string& progression,
    OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    PendingRequest request;
    request.varCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/analyze/chords");
    msg.addString(progression);
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::requestAnalyzeChords(
    const std::string& progression,
    OSCStringResponseHandler callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int msgId = generateMessageId();
    PendingRequest request;
    request.stringCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/analyze/chords");
    msg.addString(progression);
    msg.addInt32(msgId);

    if (!sender_->send(msg)) {
        if (callback) {
            callback(R"({"status":"error","message":"Send failed"})");
        }
        pendingRequests_.erase(msgId);
    }
}

void OSCBridge::requestIntentProcess(
    const std::string& intentFile,
    OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    PendingRequest request;
    request.varCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/intent/process");
    msg.addString(intentFile);
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::requestIntentProcess(
    const std::string& intentFile,
    OSCStringResponseHandler callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int msgId = generateMessageId();
    PendingRequest request;
    request.stringCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/intent/process");
    msg.addString(intentFile);
    msg.addInt32(msgId);

    if (!sender_->send(msg)) {
        if (callback) {
            callback(R"({"status":"error","message":"Send failed"})");
        }
        pendingRequests_.erase(msgId);
    }
}

void OSCBridge::requestIntentSuggest(
    const std::string& emotion,
    OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    PendingRequest request;
    request.varCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/intent/suggest");
    msg.addString(emotion);
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::requestIntentSuggest(
    const std::string& emotion,
    OSCStringResponseHandler callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int msgId = generateMessageId();
    PendingRequest request;
    request.stringCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/intent/suggest");
    msg.addString(emotion);
    msg.addInt32(msgId);

    if (!sender_->send(msg)) {
        if (callback) {
            callback(R"({"status":"error","message":"Send failed"})");
        }
        pendingRequests_.erase(msgId);
    }
}

void OSCBridge::ping(OSCResponseHandler callback)
{
    if (!connected_) return;

    int msgId = generateMessageId();
    PendingRequest request;
    request.varCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/ping");
    msg.addInt32(msgId);

    sender_->send(msg);
}

void OSCBridge::ping(OSCStringResponseHandler callback)
{
    if (!connected_) {
        if (callback) {
            callback(R"({"status":"error","message":"Not connected"})");
        }
        return;
    }

    int msgId = generateMessageId();
    PendingRequest request;
    request.stringCallback = callback;
    request.timestamp = juce::Time::getCurrentTime();
    pendingRequests_[msgId] = request;

    juce::OSCMessage msg("/daiw/ping");
    msg.addInt32(msgId);

    if (!sender_->send(msg)) {
        if (callback) {
            callback(R"({"status":"error","message":"Send failed"})");
        }
        pendingRequests_.erase(msgId);
    }
}

void OSCBridge::shutdown()
{
    if (receiver_) {
        receiver_->removeListener(this);
        receiver_->disconnect();
    }

    if (sender_) {
        sender_->disconnect();
    }

    connected_ = false;
    setAvailable(false);
    pendingRequests_.clear();
}

void OSCBridge::processMessages()
{
    cleanupTimedOutRequests();
}

void OSCBridge::oscMessageReceived(const juce::OSCMessage& message)
{
    juce::String address = message.getAddressPattern().toString();

    // Extract message ID if present (usually first argument)
    int msgId = -1;
    if (message.size() > 0 && message[0].isInt32()) {
        msgId = message[0].getInt32();
    }

    // Handle by address pattern or message ID
    if (msgId > 0 && pendingRequests_.count(msgId)) {
        handleResponseMessage(message, msgId);
    } else if (address == "/daiw/generate/response" ||
               address == "/daiw/analyze/chords/response" ||
               address == "/daiw/intent/process/response" ||
               address == "/daiw/intent/suggest/response" ||
               address == "/daiw/ping/response") {
        // Try to find matching request by address (fallback)
        // This handles cases where message ID matching fails
        for (auto& [id, request] : pendingRequests_) {
            if (request.stringCallback || request.varCallback) {
                handleResponseMessage(message, id);
                break;
            }
        }
    } else {
        logError("Received unknown message: " + address.toStdString());
    }
}

void OSCBridge::oscBundleReceived(const juce::OSCBundle& bundle)
{
    for (const auto& element : bundle) {
        if (element.isMessage()) {
            oscMessageReceived(element.getMessage());
        }
    }
}

void OSCBridge::handleResponseMessage(const juce::OSCMessage& message, int msgId)
{
    auto it = pendingRequests_.find(msgId);
    if (it == pendingRequests_.end()) {
        return;
    }

    PendingRequest request = it->second;
    pendingRequests_.erase(it);

    // Extract response JSON string
    std::string responseJson;
    if (message.size() >= 1 && message[0].isString()) {
        responseJson = message[0].getString().toStdString();
    } else if (message.size() >= 2 && message[1].isString()) {
        responseJson = message[1].getString().toStdString();
    }

    // Call appropriate callback
    if (request.stringCallback) {
        if (!responseJson.empty()) {
            request.stringCallback(responseJson);
        } else {
            request.stringCallback(R"({"status":"error","message":"Empty response"})");
        }
    } else if (request.varCallback) {
        juce::var response;
        if (!responseJson.empty()) {
            response = juce::JSON::parse(juce::String(responseJson));
        }
        request.varCallback(response);
    }
}

void OSCBridge::cleanupTimedOutRequests()
{
    auto now = juce::Time::getCurrentTime();
    auto it = pendingRequests_.begin();
    while (it != pendingRequests_.end()) {
        auto elapsed = now - it->second.timestamp;
        if (elapsed.inMilliseconds() > REQUEST_TIMEOUT_MS) {
            // Call callback with timeout error
            if (it->second.stringCallback) {
                it->second.stringCallback(R"({"status":"error","message":"Request timeout"})");
            } else if (it->second.varCallback) {
                juce::DynamicObject::Ptr errorObj = new juce::DynamicObject();
                errorObj->setProperty("status", juce::String("error"));
                errorObj->setProperty("message", juce::String("Request timeout"));
                it->second.varCallback(juce::var(errorObj));
            }
            it = pendingRequests_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace bridge
} // namespace kelly
