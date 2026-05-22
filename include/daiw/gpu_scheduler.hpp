#pragma once

// daiw/gpu_scheduler.hpp — abstraction for asynchronous GPU / accelerator
// inference dispatched from the audio plugin, with a working CPU fallback.
//
// Design
//   Inference jobs run off the audio thread. A caller pre-allocates a
//   `GpuJob` (or a subclass) and submits it; the scheduler runs the job's
//   `execute()` on a worker thread and signals completion via an atomic
//   flag inside the job. The audio thread polls `is_done()` and consumes
//   the result through a caller-defined channel (typically the existing
//   RTState seqlock or an MPSC queue carrying a small result struct).
//
// Backends
//   - `CpuGpuScheduler` — always available. Runs jobs on a single worker
//     thread; suitable for ONNX/CoreML CPU inference and for unit tests.
//   - `MetalGpuScheduler` — stub. On macOS the goal is to lazily create
//     an `MTLCommandQueue` per scheduler instance and submit each job's
//     command-buffer on that queue. The header keeps the interface intact
//     so callers compile today; the .mm implementation can be filled in
//     without breaking the API surface.
//
// RT contract
//   - `submit()` is called from a non-RT thread (UI / worker). It enqueues
//     on a lock-free MPSC and returns immediately; no GPU dispatch happens
//     inline.
//   - `GpuJob::is_done()` is a single atomic acquire-load. Safe from the
//     audio thread. Polling is bounded: at most one load per check.
//   - `GpuJob::execute()` runs on the worker thread. It may allocate, lock,
//     and block — it is NOT real-time. The job is expected to write its
//     output into a buffer the audio thread reads after `is_done()` flips.

#include "daiw/lock_free_queue.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace daiw::rt {

class GpuJob {
public:
  GpuJob() noexcept = default;
  virtual ~GpuJob() = default;

  GpuJob(const GpuJob &) = delete;
  GpuJob &operator=(const GpuJob &) = delete;

  // Worker-thread entry point. Implementations write outputs into their
  // own buffers and call mark_done() when complete.
  virtual void execute() = 0;

  // RT-safe polling. Returns true when execute() has finished and the
  // caller may read the job's output.
  bool is_done() const noexcept {
    return done_.load(std::memory_order_acquire);
  }

  // RT-safe reset — call before re-submitting a job. Producer side only.
  void reset() noexcept { done_.store(false, std::memory_order_release); }

protected:
  // Subclasses call this from inside execute() once their output is fully
  // written. The release-store pairs with is_done()'s acquire-load.
  void mark_done() noexcept { done_.store(true, std::memory_order_release); }

private:
  std::atomic<bool> done_{false};
};

class GpuScheduler {
public:
  virtual ~GpuScheduler() = default;
  virtual bool submit(GpuJob *job) noexcept = 0;
  virtual const char *backend_name() const noexcept = 0;
  virtual bool is_available() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// CpuGpuScheduler — single-worker fallback. Always available.
// ---------------------------------------------------------------------------

template <std::size_t QueueCapacity = 256>
class CpuGpuScheduler final : public GpuScheduler {
public:
  CpuGpuScheduler() noexcept : running_(true) {
    worker_ = std::thread([this]() { worker_loop_(); });
  }

  ~CpuGpuScheduler() override {
    running_.store(false, std::memory_order_release);
    // Submit a sentinel so the worker wakes up.
    inbox_.push(nullptr);
    if (worker_.joinable())
      worker_.join();
  }

  CpuGpuScheduler(const CpuGpuScheduler &) = delete;
  CpuGpuScheduler &operator=(const CpuGpuScheduler &) = delete;

  bool submit(GpuJob *job) noexcept override {
    if (!job)
      return false;
    return inbox_.push(job);
  }

  const char *backend_name() const noexcept override { return "cpu"; }
  bool is_available() const noexcept override { return true; }

private:
  void worker_loop_() {
    while (running_.load(std::memory_order_acquire)) {
      auto next = inbox_.pop();
      if (!next) {
        // Cheap back-off — yield rather than spin on the queue.
        std::this_thread::yield();
        continue;
      }
      GpuJob *job = *next;
      if (!job)
        continue; // sentinel
      job->execute();
    }
    // Drain remaining jobs before exit so they don't leak in-flight.
    while (auto last = inbox_.pop()) {
      GpuJob *job = *last;
      if (job)
        job->execute();
    }
  }

  daiw::MPSCQueue<GpuJob *, QueueCapacity> inbox_;
  std::atomic<bool> running_;
  std::thread worker_;
};

// ---------------------------------------------------------------------------
// MetalGpuScheduler — Apple-Silicon stub.
//
// The header keeps the type visible everywhere. The implementation lives in
// a .mm file that's compiled only when KMIDI_ENABLE_METAL is on. Until then
// is_available() returns false and submit() refuses work, so callers can
// safely instantiate it and fall back to CpuGpuScheduler when needed.
// ---------------------------------------------------------------------------

class MetalGpuScheduler final : public GpuScheduler {
public:
  MetalGpuScheduler() noexcept = default;
  ~MetalGpuScheduler() override = default;

  bool submit(GpuJob *) noexcept override { return false; }
  const char *backend_name() const noexcept override { return "metal"; }
  bool is_available() const noexcept override {
#ifdef KMIDI_ENABLE_METAL
    return true;
#else
    return false;
#endif
  }
};

// ---------------------------------------------------------------------------
// Convenience: pick the best available scheduler.
// ---------------------------------------------------------------------------

inline GpuScheduler *select_default_scheduler(MetalGpuScheduler &metal,
                                              CpuGpuScheduler<> &cpu) noexcept {
  return metal.is_available() ? static_cast<GpuScheduler *>(&metal)
                              : static_cast<GpuScheduler *>(&cpu);
}

} // namespace daiw::rt
