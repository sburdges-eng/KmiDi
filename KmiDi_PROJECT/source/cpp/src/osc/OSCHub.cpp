#include "penta/osc/OSCHub.h"
#include <juce_core/juce_core.h> // For juce::Logger

namespace penta {
namespace osc {

// OSCHub Constructor
OSCHub::OSCHub()
    : config_(),
      server_(std::make_unique<OSCServer>(config_.serverAddress, config_.serverPort)),
      client_(std::make_unique<OSCClient>(config_.clientAddress, config_.clientPort)),
      messageQueue_(std::make_unique<RTMessageQueue>(config_.queueSize))
{
    server_->getMessageQueue().clear(); // Ensure queue is empty on startup
}

// OSCHub Constructor with config
OSCHub::OSCHub(const Config &config)
    : config_(config),
      server_(std::make_unique<OSCServer>(config_.serverAddress, config_.serverPort)),
      client_(std::make_unique<OSCClient>(config_.clientAddress, config_.clientPort)),
      messageQueue_(std::make_unique<RTMessageQueue>(config_.queueSize))
{
    server_->getMessageQueue().clear();
}

// OSCHub Destructor
OSCHub::~OSCHub() {
    stop();
}

bool OSCHub::start() {
    if (server_->start()) {
        juce::Logger::writeToLog("OSCHub: Server started on port " + juce::String(config_.serverPort));
        // Client does not need to explicitly 'start', it connects on first send or setDestination
        return true;
    }
    juce::Logger::writeToLog("OSCHub: Failed to start server");
    return false;
}

void OSCHub::stop() {
    server_->stop();
    juce::Logger::writeToLog("OSCHub: Stopped");
}

bool OSCHub::sendMessage(const OSCMessage &message) noexcept {
    return client_->send(message);
}

bool OSCHub::receiveMessage(OSCMessage &outMessage) noexcept {
    return server_->getMessageQueue().pop(outMessage);
}

void OSCHub::registerCallback(const std::string &pattern, MessageCallback callback) {
    callbacks_[pattern] = std::move(callback);
    juce::Logger::writeToLog("OSCHub: Registered callback for pattern: " + juce::String(pattern));
}

void OSCHub::processCallbacks() {
    OSCMessage message;
    while (messageQueue_->pop(message)) {
        for (auto const& [pattern, callback] : callbacks_) {
            if (matchPattern(message.getAddress(), pattern)) {
                callback(message);
            }
        }
    }
}

void OSCHub::updateConfig(const Config &config) {
    config_ = config;
    server_->stop();
    server_ = std::make_unique<OSCServer>(config_.serverAddress, config_.serverPort);
    client_->setDestination(config_.clientAddress, config_.clientPort);
    server_->start();
}

bool OSCHub::matchPattern(const std::string &address, const std::string &pattern) const {
    if (pattern.empty()) {
        return address.empty(); // Empty pattern only matches empty address
    }
    // Simple wildcard matching for now: only '*' at the end
    if (pattern.back() == '*') {
        std::string prefix = pattern.substr(0, pattern.length() - 1);
        return address.rfind(prefix, 0) == 0; // Check if address starts with prefix
    }
    return address == pattern;
}

} // namespace osc
} // namespace penta
