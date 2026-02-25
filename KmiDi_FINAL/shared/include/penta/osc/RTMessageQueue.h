#pragma once

#include "readerwriterqueue.h"  // Real library
#include "penta/osc/OSCMessage.h" // Include our custom OSCMessage
#include <atomic>
#include <memory>
#include <vector>

namespace penta {
namespace osc {

/**
 * @brief A lock-free, single-producer, single-consumer queue for OSCMessage.
 * Suitable for real-time audio threads.
 */
class RTMessageQueue {
public:
    explicit RTMessageQueue(size_t capacity);
    ~RTMessageQueue();

    /**
     * @brief Pushes a message to the queue (non-blocking, RT-safe).
     * @param message The OSCMessage to push.
     * @return True if the message was pushed, false if the queue was full.
     */
    bool push(const OSCMessage& message) noexcept;

    /**
     * @brief Pops a message from the queue (non-blocking, RT-safe).
     * @param outMessage Reference to store the popped message.
     * @return True if a message was popped, false if the queue was empty.
     */
    bool pop(OSCMessage& outMessage) noexcept;

    /**
     * @brief Checks if the queue is empty (RT-safe).
     */
    bool isEmpty() const noexcept;

    /**
     * @brief Returns the approximate number of messages in the queue (RT-safe).
     */
    size_t size() const noexcept;

    /**
     * @brief Clears all messages from the queue (not RT-safe).
     * Should only be called when audio thread is stopped.
     */
    void clear();

private:
    std::unique_ptr<moodycamel::ReaderWriterQueue<OSCMessage>> queue_;
    size_t capacity_;
    // These are for tracking if push/pop actually happened, not queue state directly
    // moodycamel::ReaderWriterQueue is lock-free by itself.
    std::atomic<size_t> writeIndex_;
    std::atomic<size_t> readIndex_;
};

} // namespace osc
} // namespace penta
