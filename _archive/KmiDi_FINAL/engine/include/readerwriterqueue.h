#pragma once

// Offline-friendly fallback for moodycamel::ReaderWriterQueue.
// This keeps API compatibility for current call sites without network fetches.

#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace moodycamel {

template <typename T>
class ReaderWriterQueue {
public:
    explicit ReaderWriterQueue(std::size_t capacity = 1024)
        : capacity_(capacity > 0 ? capacity : 1) {}

    bool try_enqueue(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= capacity_) {
            return false;
        }
        queue_.push_back(item);
        return true;
    }

    bool try_dequeue(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    std::size_t size_approx() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<T> queue_;
    std::size_t capacity_;
};

}  // namespace moodycamel
