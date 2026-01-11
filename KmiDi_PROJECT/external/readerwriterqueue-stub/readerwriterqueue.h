#pragma once
// Stub header for readerwriterqueue - actual implementation in Week 6
// This allows the build to proceed without the external dependency

#include <queue>
#include <mutex>

namespace moodycamel {

// Simple mutex-based queue as placeholder
template<typename T>
class ReaderWriterQueue {
public:
    explicit ReaderWriterQueue(size_t capacity = 15) {}

    bool try_enqueue(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        return true;
    }

    bool try_dequeue(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = queue_.front();
        queue_.pop();
        return true;
    }

    size_t size_approx() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
};

} // namespace moodycamel
