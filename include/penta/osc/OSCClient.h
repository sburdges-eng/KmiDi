#pragma once
/*
 * OSCClient.h - Lightweight OSC UDP Sender
 * ==========================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - OSC: OSCMessage (penta/osc/OSCMessage.h)
 * - Bridge Layer: OSCBridge.h (higher-level Kelly plugin bridge)
 * - Optional: JUCE OSCSender (private implementation on supported builds)
 *
 * Purpose: Minimal dependency OSC client for fire-and-forget or simple
 *          parameter messages to a remote host.
 *
 * Features:
 * - Configurable host/port
 * - sendFloat, sendInt, sendString helpers
 */

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>

#include "penta/osc/OSCMessage.h"

// Forward declarations for JUCE classes
namespace juce {
    class OSCSender;
} // namespace juce

namespace penta {
namespace osc {

struct OSCClientSettings {
    std::string address = "127.0.0.1";
    uint16_t port = 9000;
};

class OSCClient {
public:
    explicit OSCClient(const OSCClientSettings& settings);
    OSCClient(const std::string& address, uint16_t port);
    ~OSCClient();

    bool send(const OSCMessage& message) noexcept;
    bool sendFloat(const char* address, float value) noexcept;
    bool sendInt(const char* address, int32_t value) noexcept;
    bool sendString(const char* address, const char* value) noexcept;

    void setDestination(const std::string& address, uint16_t port);

private:
    struct SocketImpl; // Private implementation detail
    std::unique_ptr<SocketImpl> socket_;
    std::string address_;
    uint16_t port_;

    // Private connect/disconnect methods
    bool connectInternal();
    void disconnectInternal();
}; // class OSCClient

} // namespace osc
} // namespace penta
