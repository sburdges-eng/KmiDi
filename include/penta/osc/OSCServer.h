#pragma once

#include "penta/osc/RTMessageQueue.h"
#include <cstdint>
#include <string>
#include <memory>
#include <atomic>

// Forward declarations for JUCE classes
namespace juce {
    class OSCReceiver;
    class OSCMessage;
} // namespace juce

namespace penta {
namespace osc {

struct OSCServerSettings {
    std::string address = "127.0.0.1"; // Not strictly used for binding, but for config
    uint16_t port = 9000;
};

/**
 * OSC server using lock-free message queue
 * Receives OSC messages without blocking RT threads
 */
class OSCServer {
public:
    explicit OSCServer(const OSCServerSettings& settings);
    OSCServer(const std::string& address, uint16_t port);
    ~OSCServer();

    // Non-copyable, non-movable
    OSCServer(const OSCServer&) = delete;
    OSCServer& operator=(const OSCServer&) = delete;

    // Start/stop server
    bool start();
    void stop();

    bool isRunning() const { return running_.load(); }

    // RT-safe: Get message queue for polling
    RTMessageQueue& getMessageQueue();

private:
    std::string address_;
    uint16_t port_;
    std::atomic<bool> running_;
    std::unique_ptr<RTMessageQueue> messageQueue_;

    // Private listener class for JUCE OSCReceiver
    class OSCListener;
    std::unique_ptr<OSCListener> listener_;
    std::unique_ptr<juce::OSCReceiver> receiver_;
};

} // namespace osc
} // namespace penta
