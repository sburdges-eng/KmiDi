#pragma once

// daiw/rt_transaction.hpp — producer-side transactional staging for the RT
// event scheduler.
//
// Use case: the UI/message thread groups a logical edit (e.g. "drag a clip
// 100ms earlier" = N NoteOff cancellations + N NoteOn re-schedules) and
// either commits it as a whole or aborts before the RT thread has seen
// anything.
//
// Semantics
//   - Transactions are NOT RT-safe — they live entirely on the producer
//     side and may allocate. The RT thread is unaware of them.
//   - commit() pushes staged events into the scheduler's lock-free inbox in
//     submission order. If the inbox runs out of room mid-commit, the
//     remaining events are kept in the staging buffer and commit() returns
//     the count actually submitted; the caller can retry later. This is
//     **not** atomic-or-none — a partial commit is observable by the RT
//     thread. For true atomicity, ensure the inbox has capacity before
//     committing.
//   - abort() drops the staged buffer; the RT thread never observes it.
//
// Rollback-safe scheduling
//   Once events are committed (in the inbox), they cannot be retracted
//   from the producer side. To "rollback" already-committed future events,
//   submit a matching cancel event (NoteOff for an outstanding NoteOn,
//   AllNotesOff for a panic) so the RT thread sees the corrective action.
//   The scheduler dispatches in sample_time order, so a cancel with an
//   earlier or equal timestamp will fire first.

#include "daiw/rt_event.hpp"
#include "daiw/rt_event_scheduler.hpp"

#include <cstddef>
#include <vector>

namespace daiw::rt {

template <typename Scheduler> class Transaction {
public:
  explicit Transaction(Scheduler &sched) noexcept
      : sched_(&sched), committed_(false) {}

  Transaction(const Transaction &) = delete;
  Transaction &operator=(const Transaction &) = delete;

  Transaction(Transaction &&other) noexcept
      : sched_(other.sched_), staged_(std::move(other.staged_)),
        committed_(other.committed_) {
    other.sched_ = nullptr;
    other.committed_ = true; // moved-from is inert
  }

  ~Transaction() {
    // Abort on scope exit if neither commit nor abort was called.
    if (!committed_ && sched_) {
      staged_.clear();
    }
  }

  // Stage one event. No interaction with the scheduler until commit().
  void add(const Event &e) { staged_.push_back(e); }

  // Reserve staging capacity in advance (useful when batching a known
  // number of edits).
  void reserve(std::size_t n) { staged_.reserve(n); }

  // Push staged events into the scheduler's inbox in order. Returns the
  // number of events successfully submitted. If less than size(), the
  // remaining events are retained — call commit() again later, or
  // abort() to drop them.
  std::size_t commit() {
    if (!sched_)
      return 0;
    std::size_t submitted = 0;
    while (submitted < staged_.size()) {
      if (!sched_->submit(staged_[submitted]))
        break;
      ++submitted;
    }
    if (submitted == staged_.size()) {
      staged_.clear();
      committed_ = true;
    } else if (submitted > 0) {
      staged_.erase(staged_.begin(),
                    staged_.begin() + static_cast<std::ptrdiff_t>(submitted));
    }
    return submitted;
  }

  // Drop staged events without touching the scheduler.
  void abort() noexcept {
    staged_.clear();
    committed_ = true; // mark inert so destructor does nothing
  }

  std::size_t size() const noexcept { return staged_.size(); }
  bool empty() const noexcept { return staged_.empty(); }

private:
  Scheduler *sched_;
  std::vector<Event> staged_;
  bool committed_;
};

} // namespace daiw::rt
