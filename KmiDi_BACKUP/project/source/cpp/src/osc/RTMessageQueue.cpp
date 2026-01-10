#include "penta/osc/RTMessageQueue.h"

namespace penta::osc {

RTMessageQueue::RTMessageQueue(size_t capacity)
    : queue_(std::make_unique<moodycamel::ReaderWriterQueue<OSCMessage>>(capacity))
    , capacity_(capacity)
    , writeIndex_(0)
    , readIndex_(0)
    , buffer_()
{
    // Reserve storage up front to avoid allocations on RT threads
    buffer_.reserve(capacity_);
}

RTMessageQueue::~RTMessageQueue() = default;

bool RTMessageQueue::push(const OSCMessage& message) noexcept {
    if (!queue_) {
        return false;
    }

    const bool success = queue_->try_enqueue(message);
    if (success) {
        writeIndex_.fetch_add(1, std::memory_order_relaxed);
    }

    return success;
}

bool RTMessageQueue::pop(OSCMessage& message) noexcept {
    if (!queue_) {
        return false;
    }

    const bool success = queue_->try_dequeue(message);
    if (success) {
        readIndex_.fetch_add(1, std::memory_order_relaxed);
    }

    return success;
}

bool RTMessageQueue::isEmpty() const noexcept {
    return !queue_ || queue_->size_approx() == 0;
}

size_t RTMessageQueue::size() const noexcept {
    return queue_ ? queue_->size_approx() : 0;
}

} // namespace penta::osc
