#pragma once

#include "penta/common/Platform.h"

#include "penta/osc/RTMessageQueue.h"
#include <cstdint>
#include <string>
#include <memory>
#include <atomic>
#include <juce_osc/juce_osc.h>

namespace penta::osc
{

    /**
     * OSC server using lock-free message queue
     * Receives OSC messages without blocking RT threads
     */
    class OSCServer
    {
    public:
        explicit OSCServer(const std::string &address, uint16_t port);
        ~OSCServer();

        // Non-copyable, non-movable
        OSCServer(const OSCServer &) = delete;
        OSCServer &operator=(const OSCServer &) = delete;

        // Start/stop server
        bool start();
        void stop();

        bool isRunning() const { return running_.load(); }

        // RT-safe: Get message queue for polling
        RTMessageQueue &getMessageQueue() { return *messageQueue_; }

    private:
        friend class OSCListener;

        std::string address_;
        uint16_t port_;
        std::atomic<bool> running_;
        std::unique_ptr<RTMessageQueue> messageQueue_;
        std::unique_ptr<class OSCListener> listener_;
        juce::OSCReceiver receiver_;
    };

} // namespace penta::osc
