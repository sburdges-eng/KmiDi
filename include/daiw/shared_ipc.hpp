#pragma once

// daiw/shared_ipc.hpp — POSIX shared-memory IPC bridge between the audio
// engine and an external observer / controller process (Python brain, MCP
// daemon, telemetry collector, etc.).
//
// Layout of the shared region:
//   struct SharedLayout {
//     SharedHeader   header;            // magic, version, sizes
//     SharedRTState  state;             // engine → world (atomic fields)
//     CommandRing    cmds_in;           // world → engine (MPSC of Command)
//     EventRing      events_out;        // engine → world (MPSC of EventMsg)
//   };
//
// Both processes mmap the same `/kmidi_<id>` named region. The engine side
// owns lifecycle (create/unlink); attachers use `attach_only()`.
//
// RT contract
//   - All cross-process atomics use lock-free types only. A static_assert at
//     namespace scope refuses to compile on platforms where the chosen
//     primitives aren't lock-free, since a syscall in a cross-process lock
//     would be catastrophic on the audio thread.
//   - SharedRTState mirrors the in-process penta::RTState publish protocol
//     (seqlock). Readers use the same retry-snapshot pattern.

#include "daiw/lock_free_queue.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace daiw::ipc {

// Magic + version so attachers can validate the region. Bump KMIDI_IPC_ABI
// whenever the SharedLayout changes shape.
constexpr uint32_t kSharedMagic = 0x4B4D4944u; // 'KMID'
constexpr uint32_t kSharedAbiVersion = 1u;

// One command in (external → engine). Mirrors daiw::rt::Event's intent of
// being a 16-byte POD so it survives the cross-process MPSCQueue without
// any pointer baggage.
struct alignas(16) Command {
  uint64_t timestamp = 0;
  uint8_t opcode = 0;
  uint8_t channel = 0;
  uint8_t data1 = 0;
  uint8_t data2 = 0;
  uint16_t data16 = 0;
  uint16_t _pad = 0;
};
static_assert(sizeof(Command) == 16, "Command must remain 16 bytes");

// One event out (engine → external). Same shape, semantically different
// (status updates, generation completions, error codes).
struct alignas(16) EventMsg {
  uint64_t timestamp = 0;
  uint8_t code = 0;
  uint8_t channel = 0;
  uint8_t data1 = 0;
  uint8_t data2 = 0;
  uint16_t data16 = 0;
  uint16_t _pad = 0;
};
static_assert(sizeof(EventMsg) == 16, "EventMsg must remain 16 bytes");

// Subset of penta::RTState that's worth mirroring across the bridge. Kept
// intentionally small — observers that need richer state should subscribe
// to a more specialised stream over the event ring.
struct SharedRTState {
  alignas(64) std::atomic<uint64_t> sequence{0};
  alignas(64) std::atomic<double> bpm{120.0};
  std::atomic<uint64_t> sample_position{0};
  std::atomic<uint32_t> bar{0};
  std::atomic<uint32_t> beat{0};
  std::atomic<float> valence{0.0f};
  std::atomic<float> arousal{0.5f};
  std::atomic<float> emotion_intensity{0.0f};
  std::atomic<bool> playing{false};
};

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "uint64 atomics must be lock-free for cross-process IPC");
static_assert(std::atomic<double>::is_always_lock_free,
              "double atomics must be lock-free for cross-process IPC");
static_assert(std::atomic<float>::is_always_lock_free,
              "float atomics must be lock-free for cross-process IPC");
static_assert(std::atomic<bool>::is_always_lock_free,
              "bool atomics must be lock-free for cross-process IPC");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "uint32 atomics must be lock-free for cross-process IPC");

constexpr std::size_t kCommandRingCapacity = 256;
constexpr std::size_t kEventRingCapacity = 256;

using CommandRing = daiw::MPSCQueue<Command, kCommandRingCapacity>;
using EventRing = daiw::MPSCQueue<EventMsg, kEventRingCapacity>;

struct SharedHeader {
  std::atomic<uint32_t> magic;
  std::atomic<uint32_t> abi_version;
  std::atomic<uint32_t> ready; // owner writes 1 once layout is initialised
  std::atomic<uint32_t> _pad;
};

struct SharedLayout {
  SharedHeader header;
  SharedRTState state;
  CommandRing cmds_in;
  EventRing events_out;
};

// ----------------------------------------------------------------------
// SharedRegion — POSIX shm_open + mmap wrapper.
//
// Lifecycle:
//   - `create(name, sz)`: shm_open(O_CREAT|O_EXCL), ftruncate, mmap, in-place
//     construct the layout, mark ready. Owner side.
//   - `attach(name, sz)`: shm_open(O_RDWR), mmap, spin until ready=1. Reader/
//     writer side.
//   - `detach()`: munmap. Both sides.
//   - `unlink_name(name)`: shm_unlink. Owner side once everyone has detached.
// ----------------------------------------------------------------------

class SharedRegion {
public:
  SharedRegion() noexcept = default;

  SharedRegion(const SharedRegion &) = delete;
  SharedRegion &operator=(const SharedRegion &) = delete;

  SharedRegion(SharedRegion &&o) noexcept
      : ptr_(o.ptr_), size_(o.size_), owner_(o.owner_), fd_(o.fd_),
        name_(std::move(o.name_)) {
    o.ptr_ = nullptr;
    o.size_ = 0;
    o.owner_ = false;
    o.fd_ = -1;
  }

  ~SharedRegion() { detach(); }

  // Owner side: create the region and initialise the layout.
  bool create(const std::string &name,
              std::size_t size = sizeof(SharedLayout)) {
#if defined(__unix__) || defined(__APPLE__)
    if (ptr_ != nullptr)
      return false;
    name_ = name;
    fd_ = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd_ < 0)
      return false;
    if (::ftruncate(fd_, static_cast<off_t>(size)) != 0) {
      ::close(fd_);
      ::shm_unlink(name.c_str());
      fd_ = -1;
      return false;
    }
    void *p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
      ::close(fd_);
      ::shm_unlink(name.c_str());
      fd_ = -1;
      return false;
    }
    ptr_ = p;
    size_ = size;
    owner_ = true;
    // Zero the region then placement-new the layout so atomics get their
    // proper default-init.
    std::memset(ptr_, 0, size_);
    new (ptr_) SharedLayout{};
    auto *layout = static_cast<SharedLayout *>(ptr_);
    layout->header.magic.store(kSharedMagic, std::memory_order_relaxed);
    layout->header.abi_version.store(kSharedAbiVersion,
                                     std::memory_order_relaxed);
    layout->header.ready.store(1u, std::memory_order_release);
    return true;
#else
    (void)name;
    (void)size;
    return false;
#endif
  }

  // Attacher side: map an existing region. spin_until_ready limits the
  // total wait in milliseconds; 0 means "do not spin".
  bool attach(const std::string &name, std::size_t size = sizeof(SharedLayout),
              int spin_until_ready_ms = 1000) {
#if defined(__unix__) || defined(__APPLE__)
    if (ptr_ != nullptr)
      return false;
    name_ = name;
    fd_ = ::shm_open(name.c_str(), O_RDWR, 0600);
    if (fd_ < 0)
      return false;
    void *p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    ptr_ = p;
    size_ = size;
    owner_ = false;
    auto *layout = static_cast<SharedLayout *>(ptr_);
    // Validate magic + ABI version.
    if (layout->header.magic.load(std::memory_order_acquire) != kSharedMagic) {
      detach();
      return false;
    }
    if (layout->header.abi_version.load(std::memory_order_acquire) !=
        kSharedAbiVersion) {
      detach();
      return false;
    }
    // Wait for owner to finish initialisation.
    for (int waited = 0; waited <= spin_until_ready_ms; ++waited) {
      if (layout->header.ready.load(std::memory_order_acquire) == 1u) {
        return true;
      }
      if (spin_until_ready_ms == 0)
        break;
      ::usleep(1000);
    }
    return layout->header.ready.load(std::memory_order_acquire) == 1u;
#else
    (void)name;
    (void)size;
    (void)spin_until_ready_ms;
    return false;
#endif
  }

  void detach() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    if (ptr_ != nullptr) {
      ::munmap(ptr_, size_);
      ptr_ = nullptr;
      size_ = 0;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
#endif
  }

  static bool unlink_name(const std::string &name) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    return ::shm_unlink(name.c_str()) == 0;
#else
    (void)name;
    return false;
#endif
  }

  bool is_attached() const noexcept { return ptr_ != nullptr; }
  bool is_owner() const noexcept { return owner_; }

  SharedLayout *layout() noexcept { return static_cast<SharedLayout *>(ptr_); }
  const SharedLayout *layout() const noexcept {
    return static_cast<const SharedLayout *>(ptr_);
  }

private:
  void *ptr_ = nullptr;
  std::size_t size_ = 0;
  bool owner_ = false;
  int fd_ = -1;
  std::string name_;
};

} // namespace daiw::ipc
