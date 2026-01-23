#include "penta/osc/OSCClient.h"
#include <juce_osc/juce_osc.h>
#include <juce_core/juce_core.h>

namespace penta {
namespace osc {

// Private implementation struct
struct OSCClient::SocketImpl {
    std::unique_ptr<juce::OSCSender> sender_;
    juce::String address_;
    int port_;
};

OSCClient::OSCClient(const OSCClientSettings& settings)
    : socket_(std::make_unique<SocketImpl>())
    , address_(settings.address)
    , port_(settings.port)
{
    socket_->address_ = juce::String(address_);
    socket_->port_ = static_cast<int>(port_);
    connectInternal();
}

OSCClient::OSCClient(const std::string& address, uint16_t port)
    : socket_(std::make_unique<SocketImpl>())
    , address_(address)
    , port_(port)
{
    socket_->address_ = juce::String(address_);
    socket_->port_ = static_cast<int>(port_);
    connectInternal();
}

OSCClient::~OSCClient() {
    disconnectInternal();
}

bool OSCClient::connectInternal() {
    if (!socket_->sender_) {
        socket_->sender_ = std::make_unique<juce::OSCSender>();
    }
    return socket_->sender_->connect(socket_->address_, socket_->port_);
}

void OSCClient::disconnectInternal() {
    if (socket_ && socket_->sender_) {
        socket_->sender_->disconnect();
    }
}

bool OSCClient::send(const OSCMessage& message) noexcept {
    if (!connectInternal()) {
        return false;
    }

    try {
        juce::OSCMessage juceMsg{juce::OSCAddressPattern{message.getAddress()}};

        for (size_t i = 0; i < message.getArgumentCount(); ++i) {
            const auto& arg = message.getArgument(i);

            if (std::holds_alternative<int32_t>(arg)) {
                juceMsg.addInt32(std::get<int32_t>(arg));
            } else if (std::holds_alternative<float>(arg)) {
                juceMsg.addFloat32(std::get<float>(arg));
            } else if (std::holds_alternative<std::string>(arg)) {
                juceMsg.addString(juce::String(std::get<std::string>(arg)));
            } else if (std::holds_alternative<std::vector<uint8_t>>(arg)) {
                const auto& blob = std::get<std::vector<uint8_t>>(arg);
                juceMsg.addBlob(juce::MemoryBlock(blob.data(), blob.size()));
            }
        }
        return socket_->sender_->send(juceMsg);
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("OSCClient::send error: " + juce::String(e.what()));
        return false;
    } catch (...) {
        juce::Logger::writeToLog("OSCClient::send unknown error");
        return false;
    }
}

bool OSCClient::sendFloat(const char* address, float value) noexcept {
    if (!connectInternal() || !socket_->sender_) return false;
    juce::OSCMessage msg{juce::OSCAddressPattern{address}};
    msg.addFloat32(value);
    return socket_->sender_->send(msg);
}

bool OSCClient::sendInt(const char* address, int32_t value) noexcept {
    if (!connectInternal() || !socket_->sender_) return false;
    juce::OSCMessage msg{juce::OSCAddressPattern{address}};
    msg.addInt32(value);
    return socket_->sender_->send(msg);
}

bool OSCClient::sendString(const char* address, const char* value) noexcept {
    if (!connectInternal() || !socket_->sender_) return false;
    juce::OSCMessage msg{juce::OSCAddressPattern{address}};
    msg.addString(juce::String(value));
    return socket_->sender_->send(msg);
}

void OSCClient::setDestination(const std::string& address, uint16_t port) {
    address_ = address;
    port_ = port;
    socket_->address_ = juce::String(address_);
    socket_->port_ = static_cast<int>(port_);
    connectInternal();
}

} // namespace osc
} // namespace penta
