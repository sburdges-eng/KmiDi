#include "penta/osc/OSCServer.h"
#include <juce_core/juce_core.h>
#include <juce_osc/juce_osc.h>

namespace penta {
namespace osc {

// Full definition of OSCListener class (must be defined before use)
class OSCServer::OSCListener
    : public juce::OSCReceiver::Listener<juce::OSCReceiver::RealtimeCallback> {
public:
    explicit OSCListener(OSCServer* owner) : owner_(owner) {}

    void oscMessageReceived(const juce::OSCMessage& message) override {
        if (!owner_ || !owner_->messageQueue_) {
            return;
        }

        OSCMessage pentaMsg;
        pentaMsg.setAddress(message.getAddressPattern().toString().toStdString());
        pentaMsg.setTimestamp(static_cast<uint64_t>(juce::Time::getMillisecondCounterHiRes()));

        for (const auto& arg : message) {
            if (arg.isFloat32()) {
                pentaMsg.addFloat(arg.getFloat32());
            } else if (arg.isInt32()) {
                pentaMsg.addInt(arg.getInt32());
            } else if (arg.isString()) {
                pentaMsg.addString(arg.getString().toStdString());
            } else if (arg.isBlob()) {
                const auto& blob = arg.getBlob();
                pentaMsg.addBlob(std::vector<uint8_t>(
                    static_cast<const uint8_t*>(blob.getData()),
                    static_cast<const uint8_t*>(blob.getData()) + blob.getSize()));
            }
        }
        owner_->messageQueue_->push(pentaMsg);
    }

private:
    OSCServer* owner_;
};

OSCServer::OSCServer(const OSCServerSettings& settings)
    : address_(settings.address)
    , port_(settings.port)
    , running_(false)
    , messageQueue_(std::make_unique<RTMessageQueue>(4096))
    , listener_(std::make_unique<OSCListener>(this))
    , receiver_(std::make_unique<juce::OSCReceiver>())
{
    receiver_->addListener(listener_.get());
}

OSCServer::OSCServer(const std::string& address, uint16_t port)
    : address_(address)
    , port_(port)
    , running_(false)
    , messageQueue_(std::make_unique<RTMessageQueue>(4096))
    , listener_(std::make_unique<OSCListener>(this))
    , receiver_(std::make_unique<juce::OSCReceiver>())
{
    receiver_->addListener(listener_.get());
}

OSCServer::~OSCServer() {
    stop();
    if (receiver_ && listener_) {
        receiver_->removeListener(listener_.get());
    }
}

bool OSCServer::start() {
    if (running_.load()) {
        return true;
    }

    if (!receiver_->connect(static_cast<int>(port_))) {
        juce::Logger::writeToLog("OSCServer: Failed to bind to port " +
                                 juce::String(port_));
        return false;
    }

    running_.store(true);
    juce::Logger::writeToLog("OSCServer: Started on port " + juce::String(port_));
    return true;
}

void OSCServer::stop() {
    if (!running_.load()) {
        return;
    }

    receiver_->disconnect();
    running_.store(false);
    juce::Logger::writeToLog("OSCServer: Stopped");
}

RTMessageQueue& OSCServer::getMessageQueue() {
    return *messageQueue_;
}

} // namespace osc
} // namespace penta
