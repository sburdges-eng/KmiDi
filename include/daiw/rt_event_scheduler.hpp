#pragma once

// daiw/rt_event_scheduler.hpp — sample-accurate event scheduler for the RT
// audio thread.
//
// Architecture
//   Producers (UI / message thread / inference worker) call submit() to push
//   timestamped Events into a lock-free MPSC inbox. The audio thread calls
//   drain_block() once per processBlock, which:
//     1. drains the inbox into a bounded min-heap (sorted by sample_time),
//     2. pops every heap entry whose sample_time falls inside the current
//        block window and dispatches it via a caller-supplied sink.
//
// Transport-jump handling
//   reconcile_clock() compares the host's reported sample position against
//   the scheduler's expected position. When the delta exceeds a half-block
//   tolerance (loop / seek / rewind), the scheduler emits AllNotesOff to
//   the sink for every channel it has heard from this session, clears the
//   pending heap, and drains+discards the entire inbox so stale events
//   from the pre-jump timeline cannot fire.
//
// RT contract
//   - submit(): wait-free MPSC enqueue (returns false on overflow).
//   - drain_block(): allocation-free, lock-free, bounded by MaxPending.
//   - reconcile_clock(): allocation-free; bounded by 16 (one AllNotesOff
//     per MIDI channel) plus inbox capacity for the discard sweep.

#include "daiw/lock_free_queue.hpp"
#include "daiw/rt_event.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace daiw::rt {

template <std::size_t InboxCapacity = 2048, std::size_t MaxPending = 1024>
class EventScheduler {
  static_assert((InboxCapacity & (InboxCapacity - 1)) == 0,
                "InboxCapacity must be power of 2 (MPSCQueue constraint)");
  static_assert(MaxPending >= 16, "MaxPending too small to be useful");

public:
  EventScheduler() noexcept : pending_count_(0), expected_sample_pos_(0) {
    for (auto &touched : channel_touched_) {
      touched.store(false, std::memory_order_relaxed);
    }
  }

  EventScheduler(const EventScheduler &) = delete;
  EventScheduler &operator=(const EventScheduler &) = delete;

  // ---------------------------------------------------------------------
  // Producer side (any thread)
  // ---------------------------------------------------------------------

  // Wait-free push. Returns false if the inbox is full (caller's
  // responsibility to back-pressure or drop).
  bool submit(const Event &e) noexcept {
    const bool ok = inbox_.push(e);
    if (ok && e.kind != EventKind::TransportPlay &&
        e.kind != EventKind::TransportStop &&
        e.kind != EventKind::TransportSeek &&
        e.kind != EventKind::TempoChange && e.channel < 16) {
      channel_touched_[e.channel].store(true, std::memory_order_release);
    }
    return ok;
  }

  // ---------------------------------------------------------------------
  // Audio thread — drain a single processBlock window
  // ---------------------------------------------------------------------
  //
  // Sink signature: void(uint32_t sample_offset, const Event&)
  // sample_offset is the relative offset inside the current block.

  template <typename Sink>
  void drain_block(uint64_t block_start_samples, uint32_t num_samples,
                   Sink &&sink) noexcept {
    // 1. Move any newly-submitted events into the pending heap. Bounded
    //    by InboxCapacity; if the heap fills first, leave the rest for
    //    next block.
    //
    //    MPSCQueue::pop() returns nullopt both for an empty queue AND for
    //    a slot that's been reserved but not yet written by a producer
    //    (publish race). Retrying a small fixed number of times absorbs
    //    that race without breaking the RT bound — kIngestRetries adds at
    //    most a handful of nanoseconds per drain.
    constexpr int kIngestRetries = 4;
    int empty_streak = 0;
    while (pending_count_ < MaxPending && empty_streak < kIngestRetries) {
      auto next = inbox_.pop();
      if (!next) {
        ++empty_streak;
        continue;
      }
      empty_streak = 0;
      heap_push_(*next);
    }

    const uint64_t block_end = block_start_samples + num_samples;

    // 2. Pop every event whose sample_time falls inside this window.
    //    Events scheduled for the past (sample_time < block_start) fire
    //    at offset 0 — this matches DAW behaviour for events that arrive
    //    late and is safer than silently dropping mutations.
    while (pending_count_ > 0 && pending_[0].sample_time < block_end) {
      Event e = heap_pop_();
      uint32_t offset =
          (e.sample_time <= block_start_samples)
              ? 0u
              : static_cast<uint32_t>(e.sample_time - block_start_samples);
      sink(offset, e);
    }

    expected_sample_pos_ = block_end;
  }

  // Returns true if a jump was detected; in that case the sink received
  // AllNotesOff events at offset 0 and all pending state was cleared.
  // tolerance_samples controls how strict the detector is — a typical
  // setting is the current block size (drift of one block is tolerated).
  template <typename Sink>
  bool reconcile_clock(uint64_t host_sample_pos, uint32_t tolerance_samples,
                       Sink &&sink) noexcept {
    const uint64_t expected = expected_sample_pos_;
    const uint64_t delta = host_sample_pos > expected
                               ? host_sample_pos - expected
                               : expected - host_sample_pos;

    if (delta <= tolerance_samples) {
      // No jump; just keep the expected pos honest if the host is
      // running slightly behind.
      expected_sample_pos_ = host_sample_pos;
      return false;
    }

    // Emit AllNotesOff for every channel we've ever seen activity on.
    for (uint8_t ch = 0; ch < 16; ++ch) {
      if (channel_touched_[ch].load(std::memory_order_acquire)) {
        sink(0u, make_all_notes_off(host_sample_pos, ch));
      }
    }

    // Discard pending heap.
    pending_count_ = 0;

    // Drain inbox of stale entries. Bounded by InboxCapacity so this
    // remains real-time-safe.
    for (std::size_t i = 0; i < InboxCapacity; ++i) {
      if (!inbox_.pop())
        break;
    }

    expected_sample_pos_ = host_sample_pos;
    return true;
  }

  // Diagnostics (non-RT; pure reads).
  std::size_t pending_size() const noexcept { return pending_count_; }
  uint64_t expected_sample_pos() const noexcept { return expected_sample_pos_; }

private:
  // -------- min-heap over pending_, keyed by sample_time --------

  void heap_push_(const Event &e) noexcept {
    std::size_t i = pending_count_++;
    pending_[i] = e;
    while (i > 0) {
      std::size_t parent = (i - 1) / 2;
      if (pending_[parent].sample_time <= pending_[i].sample_time)
        break;
      Event tmp = pending_[parent];
      pending_[parent] = pending_[i];
      pending_[i] = tmp;
      i = parent;
    }
  }

  Event heap_pop_() noexcept {
    Event top = pending_[0];
    --pending_count_;
    if (pending_count_ > 0) {
      pending_[0] = pending_[pending_count_];
      std::size_t i = 0;
      while (true) {
        const std::size_t l = 2 * i + 1;
        const std::size_t r = 2 * i + 2;
        std::size_t smallest = i;
        if (l < pending_count_ &&
            pending_[l].sample_time < pending_[smallest].sample_time) {
          smallest = l;
        }
        if (r < pending_count_ &&
            pending_[r].sample_time < pending_[smallest].sample_time) {
          smallest = r;
        }
        if (smallest == i)
          break;
        Event tmp = pending_[i];
        pending_[i] = pending_[smallest];
        pending_[smallest] = tmp;
        i = smallest;
      }
    }
    return top;
  }

  daiw::MPSCQueue<Event, InboxCapacity> inbox_;

  // Pending heap (audio-thread only — no synchronisation needed).
  Event pending_[MaxPending];
  std::size_t pending_count_;
  uint64_t expected_sample_pos_;

  // Per-channel "have we ever seen MIDI on this channel" flag. Set by
  // producers via submit(), read by the audio thread on reconcile_clock()
  // to know which channels need AllNotesOff. Release/acquire pair is
  // sufficient — the flag is monotonic (false → true), false positives
  // (extra AllNotesOff) are harmless, no false negatives possible.
  std::atomic<bool> channel_touched_[16];
};

} // namespace daiw::rt
