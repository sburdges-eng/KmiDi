#pragma once

#include "penta/common/Platform.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace penta {

/**
 * Real-time safe memory pool allocator.
 * Pre-allocates memory blocks to avoid malloc/free in audio thread.
 *
 * NOTE: The lock-free free-list is safe for SPSC use (single allocator +
 * single deallocator). For MPMC use, the CAS loop is susceptible to ABA.
 * In KmiDi, the audio thread is the sole user of this pool.
 */
class RTMemoryPool {
public:
    explicit RTMemoryPool(size_t blockSize, size_t numBlocks);
    ~RTMemoryPool();
    
    // Non-copyable, non-movable
    RTMemoryPool(const RTMemoryPool&) = delete;
    RTMemoryPool& operator=(const RTMemoryPool&) = delete;
    
    // RT-safe allocation (lock-free)
    void* allocate() noexcept;
    
    // RT-safe deallocation (lock-free)
    void deallocate(void* ptr) noexcept;
    
    // Statistics (for diagnostics)
    size_t getBlockSize() const noexcept { return blockSize_; }
    size_t getTotalBlocks() const noexcept { return numBlocks_; }
    size_t getAvailableBlocks() const noexcept;
    
    // Debug-only SPSC enforcement: asserts that allocate() is always called from
    // the same thread. The CAS loop in allocate() is ABA-susceptible under MPMC
    // use; this check makes violations visible early in debug builds.
#ifndef NDEBUG
    void assertSingleProducer() {
        static std::atomic<std::thread::id> owner{};
        auto expected = std::thread::id{};
        auto current = std::this_thread::get_id();
        if (!owner.compare_exchange_strong(expected, current))
            assert(expected == current && "RTMemoryPool: SPSC violation — allocate() called from multiple threads");
    }
#else
    void assertSingleProducer() {}
#endif

private:
    struct Block {
        Block* next;
    };

    std::vector<uint8_t> memory_;
    std::atomic<Block*> freeList_;
    size_t blockSize_;
    size_t numBlocks_;
};

/**
 * RAII wrapper for RT memory pool allocation
 */
template<typename T>
class RTPoolPtr {
public:
    RTPoolPtr(RTMemoryPool& pool) 
        : pool_(pool)
        , ptr_(static_cast<T*>(pool.allocate())) {
        if (ptr_) {
            new (ptr_) T(); // Placement new
        }
    }
    
    ~RTPoolPtr() {
        if (ptr_) {
            ptr_->~T(); // Explicit destructor call
            pool_.deallocate(ptr_);
        }
    }
    
    // Non-copyable, movable
    RTPoolPtr(const RTPoolPtr&) = delete;
    RTPoolPtr& operator=(const RTPoolPtr&) = delete;
    
    RTPoolPtr(RTPoolPtr&& other) noexcept
        : pool_(other.pool_)
        , ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    
    T& operator*() noexcept { return *ptr_; }
    const T& operator*() const noexcept { return *ptr_; }
    
    T* operator->() noexcept { return ptr_; }
    const T* operator->() const noexcept { return ptr_; }
    
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    
private:
    RTMemoryPool& pool_;
    T* ptr_;
};

} // namespace penta
