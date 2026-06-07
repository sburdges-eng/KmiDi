#include "penta/common/RTMemoryPool.h"
#include <cstring>

namespace penta {

RTMemoryPool::RTMemoryPool(size_t blockSize, size_t numBlocks)
    : blockSize_(blockSize), numBlocks_(numBlocks) {
  // Ensure minimum alignment
  constexpr size_t kAlignment = alignof(std::max_align_t);
  blockSize_ = (blockSize + kAlignment - 1) & ~(kAlignment - 1);

  // Allocate memory for all blocks
  memory_.resize(blockSize_ * numBlocks_);

  // Build free list
  Block *head = nullptr;
  for (size_t i = 0; i < numBlocks_; ++i) {
    auto *block = reinterpret_cast<Block *>(memory_.data() + i * blockSize_);
    block->next = head;
    head = block;
  }

  freeList_.store(head, std::memory_order_release);
}

RTMemoryPool::~RTMemoryPool() = default;

bool RTMemoryPool::contains(const void *ptr) const noexcept {
  if (ptr == nullptr || memory_.empty()) {
    return false;
  }

  const auto *bytePtr = static_cast<const uint8_t *>(ptr);
  const auto *start = memory_.data();
  const auto *end = start + memory_.size();
  return bytePtr >= start && bytePtr < end && isBlockAligned(ptr);
}

bool RTMemoryPool::isBlockAligned(const void *ptr) const noexcept {
  if (ptr == nullptr || blockSize_ == 0 || memory_.empty()) {
    return false;
  }

  const auto offset =
      static_cast<size_t>(static_cast<const uint8_t *>(ptr) - memory_.data());
  return offset < memory_.size() && (offset % blockSize_) == 0;
}

void *RTMemoryPool::allocate() noexcept {
  assertSingleProducer();
  Block *head = freeList_.load(std::memory_order_acquire);

  while (head != nullptr) {
    Block *next = head->next;
    if (freeList_.compare_exchange_weak(head, next, std::memory_order_release,
                                        std::memory_order_acquire)) {
      return head;
    }
  }

  return nullptr; // Pool exhausted
}

void RTMemoryPool::deallocate(void *ptr) noexcept {
  if (!contains(ptr))
    return;

  auto *block = static_cast<Block *>(ptr);
  Block *head = freeList_.load(std::memory_order_acquire);

  do {
    block->next = head;
  } while (!freeList_.compare_exchange_weak(
      head, block, std::memory_order_release, std::memory_order_acquire));
}

size_t RTMemoryPool::getAvailableBlocks() const noexcept {
  size_t count = 0;
  Block *current = freeList_.load(std::memory_order_acquire);

  while (current != nullptr) {
    ++count;
    current = current->next;
  }

  return count;
}

} // namespace penta
