#pragma once

#include "penta/osc/OSCClient.h"
#include "penta/osc/OSCMessage.h"
#include "penta/osc/OSCServer.h"
#include "penta/osc/RTMessageQueue.h"
#include <memory>
#include <string>
#include <functional>
#include <unordered_map>

namespace penta {
namespace osc {

/**
 * OSC communication hub
 * Manages bidirectional OSC communication with DAWs and controllers
 */
class OSCHub {
public:
    struct Config {
        std::string serverAddress = "0.0.0.0";
        uint16_t serverPort = 8000;
        std::string clientAddress = "127.0.0.1";
        uint16_t clientPort = 9000;
        size_t queueSize = 4096;
    };

    using MessageCallback = std::function<void(const OSCMessage&)>;

    OSCHub();
    explicit OSCHub(const Config& config);
    ~OSCHub();

    // Non-copyable
    OSCHub(const OSCHub&) = delete;
    OSCHub& operator=(const OSCHub&) = delete;

    // Non-RT: Start server and client
    bool start();

    // Non-RT: Stop server and client
    void stop();

    // RT-safe: Send OSC message
    bool sendMessage(const OSCMessage& message) noexcept;

    // RT-safe: Poll for received messages
    bool receiveMessage(OSCMessage& outMessage) noexcept;

    // Non-RT: Register callback for specific OSC address pattern
    void registerCallback(const std::string& pattern, MessageCallback callback);

    // Non-RT: Process queued messages and dispatch to callbacks
    void processCallbacks();

    // Non-RT: Configuration
    void updateConfig(const Config& config);

    bool matchPattern(const std::string& address, const std::string& pattern) const;

private:
    Config config_;

    std::unique_ptr<OSCServer> server_;
    std::unique_ptr<OSCClient> client_;
    std::unique_ptr<RTMessageQueue> messageQueue_;
    std::unordered_map<std::string, MessageCallback> callbacks_;
};

} // namespace osc
} // namespace penta
