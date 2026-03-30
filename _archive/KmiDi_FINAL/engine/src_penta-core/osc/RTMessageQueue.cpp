#include "penta/osc/RTMessageQueue.h"
#include "readerwriterqueue.h"

namespace penta {
namespace osc {

RTMessageQueue::RTMessageQueue(size_t capacity)
    : queue_(std::make_unique<moodycamel::ReaderWriterQueue<OSCMessage>>(capacity))
    , capacity_(capacity)
    , writeIndex_(0)
    , readIndex_(0)
{
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

bool RTMessageQueue::pop(OSCMessage& outMessage) noexcept {
    if (!queue_) {
        return false;
    }

    const bool success = queue_->try_dequeue(outMessage);
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

void RTMessageQueue::clear() {
    if (!queue_) {
        return;
    }
    OSCMessage dummy;
    while (queue_->try_dequeue(dummy)) {
        // Discard messages
    }
    writeIndex_.store(0, std::memory_order_relaxed);
    readIndex_.store(0, std::memory_order_relaxed);
}

} // namespace osc
} // namespace penta
