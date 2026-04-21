#pragma once

#include <atomic>
#include <array>
#include <cstring>
#include <cstddef>
#include <type_traits>

namespace kelly {

/**
 * LockFreeRingBuffer - Lock-free circular buffer for audio/ML thread communication.
 *
 * Thread-safe ring buffer using atomic operations for producer-consumer pattern.
 * Designed for real-time audio processing where blocking is not acceptable.
 *
 * @tparam T Element type
 * @tparam Capacity Buffer capacity (must be power of 2 for optimal performance)
 */
template<typename T, size_t Capacity>
class LockFreeRingBuffer {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
        "LockFreeRingBuffer requires trivially copyable types");

    LockFreeRingBuffer() : writePos_(0), readPos_(0) {
        buffer_.fill(T{});
    }

    /**
     * Push data into buffer (producer - typically audio thread).
     * @param data Pointer to data to push
     * @param count Number of elements to push
     * @return true if successful, false if buffer full
     */
    bool push(const T* data, size_t count) {
        const size_t currentWrite = writePos_.load(std::memory_order_relaxed);
        const size_t currentRead = readPos_.load(std::memory_order_acquire);

        const size_t available = Capacity - (currentWrite - currentRead);
        if (count > available) {
            return false;  // Buffer full
        }

        const size_t writeIndex = currentWrite & (Capacity - 1);  // Modulo (power of 2)
        const size_t firstPart = std::min(count, Capacity - writeIndex);

        // Copy first part
        std::memcpy(&buffer_[writeIndex], data, firstPart * sizeof(T));

        // Copy wrap-around part if needed
        if (count > firstPart) {
            std::memcpy(&buffer_[0], data + firstPart, (count - firstPart) * sizeof(T));
        }

        writePos_.store(currentWrite + count, std::memory_order_release);
        return true;
    }

    /**
     * Pop data from buffer (consumer - typically ML thread).
     * @param data Pointer to destination buffer
     * @param count Number of elements to pop
     * @return true if successful, false if buffer empty
     */
    bool pop(T* data, size_t count) {
        const size_t currentRead = readPos_.load(std::memory_order_relaxed);
        const size_t currentWrite = writePos_.load(std::memory_order_acquire);

        const size_t available = currentWrite - currentRead;
        if (count > available) {
            return false;  // Not enough data
        }

        const size_t readIndex = currentRead & (Capacity - 1);  // Modulo (power of 2)
        const size_t firstPart = std::min(count, Capacity - readIndex);

        // Copy first part
        std::memcpy(data, &buffer_[readIndex], firstPart * sizeof(T));

        // Copy wrap-around part if needed
        if (count > firstPart) {
            std::memcpy(data + firstPart, &buffer_[0], (count - firstPart) * sizeof(T));
        }

        readPos_.store(currentRead + count, std::memory_order_release);
        return true;
    }

    /**
     * Get number of elements available to read.
     */
    size_t availableToRead() const {
        return writePos_.load(std::memory_order_acquire) -
               readPos_.load(std::memory_order_relaxed);
    }

    /**
     * Get number of elements available to write.
     */
    size_t availableToWrite() const {
        return Capacity - availableToRead();
    }

    /**
     * Check if buffer is empty.
     */
    bool isEmpty() const {
        return availableToRead() == 0;
    }

    /**
     * Check if buffer is full.
     */
    bool isFull() const {
        return availableToWrite() == 0;
    }

    /**
     * Consumer-only: drain all pending elements and return the latest one.
     * Call from the CONSUMER thread only; mutates readPos_ like pop() does.
     *
     * Provides drop-oldest semantics without requiring the producer to touch
     * readPos_.  The audio (consumer) thread calls this instead of pop() when
     * it only needs the freshest available inference result.
     *
     * @param out  Destination for the most recent element.
     * @return true if at least one element was available (out is valid);
     *         false if the buffer was empty (out is unchanged).
     */
    bool popLatest(T* out) noexcept {
        T tmp;
        bool any = false;
        while (pop(&tmp, 1)) {
            *out = tmp;
            any = true;
        }
        return any;
    }

    /**
     * Clear buffer (reset positions).
     */
    void clear() {
        writePos_.store(0, std::memory_order_release);
        readPos_.store(0, std::memory_order_release);
    }

private:
    std::array<T, Capacity> buffer_;
    // Producer and consumer cursors live on separate cache lines so the
    // audio (producer) and ML (consumer) threads don't ping-pong a shared
    // line on every push/pop. 64 bytes is the minimum line size; Apple
    // Silicon treats pairs as 128 (see penta/common/RTState.h::kCacheLine)
    // but double-padding here would bloat the struct without measurable
    // gain — the two atomics are never in the same 64-byte chunk after this.
    alignas(64) std::atomic<size_t> writePos_{0};
    alignas(64) std::atomic<size_t> readPos_{0};
};

} // namespace kelly
