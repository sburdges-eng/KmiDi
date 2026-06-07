/**
 * @file memory.cpp
 * @brief Implementation of memory management utilities
 */

#include "daiw/memory.hpp"
#include "daiw/types.hpp"
#include <algorithm>
#include <cassert>

namespace daiw {

// =============================================================================
// MutexMemoryPool Implementation
// =============================================================================

MutexMemoryPool::MutexMemoryPool(size_t blockSize, size_t numBlocks)
    : blockSize_(blockSize), numBlocks_(numBlocks), freeCount_(numBlocks) {
  // Allocate contiguous memory for all blocks
  memory_ = std::make_unique<std::byte[]>(blockSize_ * numBlocks_);

  // Build free list
  freeList_.reserve(numBlocks_);
  for (size_t i = 0; i < numBlocks_; ++i) {
    freeList_.push_back(memory_.get() + (i * blockSize_));
  }
}

MutexMemoryPool::~MutexMemoryPool() = default;

void *MutexMemoryPool::allocate() {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t available = freeCount_.load(std::memory_order_acquire);
  if (available == 0) {
    return nullptr;
  }
  void *ptr = freeList_[available - 1];
  freeCount_.store(available - 1, std::memory_order_release);
  return ptr;
}

void MutexMemoryPool::deallocate(void *ptr) {
  if (!ptr || !contains(ptr)) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = freeCount_.load(std::memory_order_acquire);
  freeList_[count] = ptr;
  freeCount_.store(count + 1, std::memory_order_release);
}

bool MutexMemoryPool::contains(void *ptr) const noexcept {
  auto *bytePtr = static_cast<std::byte *>(ptr);
  auto *start = memory_.get();
  auto *end = start + (blockSize_ * numBlocks_);
  return bytePtr >= start && bytePtr < end;
}

size_t MutexMemoryPool::availableBlocks() const noexcept {
  return freeCount_.load(std::memory_order_acquire);
}

// =============================================================================
// LockFreeQueue Implementation
// =============================================================================

template <typename T> LockFreeQueue<T>::LockFreeQueue() {
  // Create sentinel node with local RAII until ownership transfers into the
  // queue.
  auto sentinel = std::make_unique<QueueNode<T>>(T{});
  head_.store(sentinel.get(), std::memory_order_relaxed);
  tail_ = sentinel.release();
}

template <typename T> LockFreeQueue<T>::~LockFreeQueue() {
  // Drain the queue
  T item;
  while (pop(item)) {
  }

  // Delete the final sentinel via RAII in case future teardown grows.
  std::unique_ptr<QueueNode<T>> sentinel(tail_);
  tail_ = nullptr;
}

// WARNING: Despite the name, this queue heap-allocates nodes in push().
// NOT safe for real-time audio threads. For RT paths, use
// moodycamel::ReaderWriterQueue or a pre-allocated ring buffer.
template <typename T> void LockFreeQueue<T>::push(const T &item) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto node = std::make_unique<QueueNode<T>>(item);
  auto *rawNode = node.get();

  // Atomically swap the head
  QueueNode<T> *prevHead = head_.exchange(rawNode, std::memory_order_acq_rel);
  prevHead->next.store(rawNode, std::memory_order_release);
  node.release();
}

template <typename T> bool LockFreeQueue<T>::pop(T &item) {
  std::lock_guard<std::mutex> lock(mutex_);
  QueueNode<T> *next = tail_->next.load(std::memory_order_acquire);

  if (next == nullptr) {
    return false; // Empty
  }

  item = next->data;

  std::unique_ptr<QueueNode<T>> oldTail(tail_);
  tail_ = next;
  return true;
}

template <typename T> bool LockFreeQueue<T>::empty() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return tail_->next.load(std::memory_order_acquire) == nullptr;
}

// Explicit instantiations for common types
template class LockFreeQueue<int>;
template class LockFreeQueue<float>;
template class LockFreeQueue<NoteEvent>;

} // namespace daiw
