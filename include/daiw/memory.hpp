/**
 * @file memory.hpp
 * @brief Real-time safe memory management for DAiW
 *
 * Provides memory pools, ring buffers, and lock-free structures
 * suitable for audio thread usage.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>
#include <mutex>

namespace daiw {

// =============================================================================
// Memory Pool (Real-Time Safe Allocator)
// =============================================================================

/**
 * @brief Fixed-size memory pool for real-time allocation (mutex-based)
 *
 * Pre-allocates blocks of memory for O(1) allocation and deallocation.
 * NOTE: Uses mutex — NOT suitable for RT audio threads.
 * For RT-safe allocation, use the template MemoryPool in memory_pool.hpp.
 * Renamed from MemoryPool to avoid ODR collision with memory_pool.hpp.
 */
class MutexMemoryPool {
public:
    explicit MutexMemoryPool(size_t blockSize, size_t numBlocks);
    ~MutexMemoryPool();

    // Non-copyable, non-movable
    MutexMemoryPool(const MutexMemoryPool&) = delete;
    MutexMemoryPool& operator=(const MutexMemoryPool&) = delete;
    MutexMemoryPool(MutexMemoryPool&&) = delete;
    MutexMemoryPool& operator=(MutexMemoryPool&&) = delete;

    /**
     * @brief Allocate a block from the pool
     * @return Pointer to allocated block, or nullptr if pool exhausted
     *
     * @note NOT noexcept: takes std::mutex::lock() which can throw
     *       std::system_error. NOT RT-safe. For RT-safe allocation,
     *       use the template MemoryPool in memory_pool.hpp.
     */
    [[nodiscard]] void* allocate();

    /**
     * @brief Return a block to the pool
     * @param ptr Pointer previously returned by allocate()
     *
     * @note NOT noexcept: takes std::mutex::lock() which can throw.
     */
    void deallocate(void* ptr);

    /**
     * @brief Check if a pointer belongs to this pool
     */
    [[nodiscard]] bool contains(void* ptr) const noexcept;

    /**
     * @brief Get number of available blocks
     */
    [[nodiscard]] size_t availableBlocks() const noexcept;

    /**
     * @brief Get total number of blocks
     */
    [[nodiscard]] size_t totalBlocks() const noexcept { return numBlocks_; }

    /**
     * @brief Get block size
     */
    [[nodiscard]] size_t blockSize() const noexcept { return blockSize_; }

private:
    std::unique_ptr<std::byte[]> memory_;
    std::vector<void*> freeList_;
    size_t blockSize_;
    size_t numBlocks_;
    std::atomic<size_t> freeCount_;
    mutable std::mutex mutex_;
};

// =============================================================================
// Lock-Free Ring Buffer
// =============================================================================

/**
 * @brief SPSC (Single-Producer, Single-Consumer) lock-free event ring buffer
 *
 * Single-element push/pop for event passing between threads.
 * For bulk audio streaming, use RingBuffer in ring_buffer.hpp.
 * Renamed from RingBuffer to avoid ODR collision with ring_buffer.hpp.
 */
template<typename T, size_t Capacity>
class EventRingBuffer {
public:
    EventRingBuffer() : head_(0), tail_(0) {
        static_assert((Capacity & (Capacity - 1)) == 0,
                      "Capacity must be power of 2");
    }

    /**
     * @brief Push an element to the buffer
     * @return true if successful, false if buffer full
     */
    bool push(const T& item) noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t nextHead = (head + 1) & (Capacity - 1);

        if (nextHead == tail_.load(std::memory_order_acquire)) {
            return false;  // Buffer full
        }

        buffer_[head] = item;
        head_.store(nextHead, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pop an element from the buffer
     * @param[out] item Destination for the popped element
     * @return true if successful, false if buffer empty
     */
    bool pop(T& item) noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // Buffer empty
        }

        item = buffer_[tail];
        tail_.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }

    /**
     * @brief Check if buffer is empty
     */
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get number of elements in buffer
     */
    [[nodiscard]] size_t size() const noexcept {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & (Capacity - 1);
    }

    /**
     * @brief Clear the buffer
     */
    void clear() noexcept {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
    }

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    T buffer_[Capacity];
};

// =============================================================================
// Lock-Free Queue (MPSC - Multiple Producer, Single Consumer)
// =============================================================================

/**
 * @brief Node for lock-free queue
 */
template<typename T>
struct QueueNode {
    T data;
    std::atomic<QueueNode*> next{nullptr};

    explicit QueueNode(const T& d) : data(d) {}
};

/**
 * @brief MPSC lock-free queue for event passing
 *
 * Useful for sending events from multiple GUI threads to the audio thread.
 */
template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue();
    ~LockFreeQueue();

    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    /**
     * @brief Push an item (multiple producers OK)
     */
    void push(const T& item);

    /**
     * @brief Pop an item (single consumer only)
     * @return true if item was popped
     */
    bool pop(T& item);

    /**
     * @brief Check if queue is empty
     */
    [[nodiscard]] bool empty() const noexcept;

private:
    std::atomic<QueueNode<T>*> head_;
    QueueNode<T>* tail_;
    mutable std::mutex mutex_;
};

}  // namespace daiw
