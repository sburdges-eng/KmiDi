#pragma once
/**
 * BlockLatencyInstrument.h — Lock-free per-block latency capture.
 *
 * Drop into processBlock() to measure P50/P90/P99 processing times
 * with zero allocations on the audio thread.
 *
 * Usage:
 *   BlockLatencyInstrument inst;
 *
 *   void processBlock(...) {
 *       auto scope = inst.measure();  // RAII — captures start/end
 *       // ... DSP work ...
 *   }
 *
 *   auto stats = inst.getStats();  // call from UI/timer thread
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>

#ifdef __APPLE__
#include <mach/mach_time.h>
#else
#include <chrono>
#include <cstddef>
#endif

namespace penta {
namespace diagnostics {

class BlockLatencyInstrument {
public:
    static constexpr size_t kHistorySize = 1024; // power of 2

    struct Stats {
        double p50Us  = 0.0;
        double p90Us  = 0.0;
        double p99Us  = 0.0;
        double maxUs  = 0.0;
        double meanUs = 0.0;
        uint64_t sampleCount = 0;
        uint32_t overruns = 0; // exceeded deadline
    };

    // RAII scope for measuring a single block
    class Scope {
    public:
        explicit Scope(BlockLatencyInstrument& inst) : inst_(inst) {
            start_ = nowTicks();
        }
        ~Scope() {
            uint64_t end = nowTicks();
            inst_.record(end - start_);
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    private:
        BlockLatencyInstrument& inst_;
        uint64_t start_;
    };

    BlockLatencyInstrument() {
        history_.fill(0);
#ifdef __APPLE__
        mach_timebase_info(&timebase_);
#endif
    }

    /** Create an RAII measurement scope. Call at top of processBlock(). */
    Scope measure() { return Scope(*this); }

    /** Set the deadline in microseconds (for overrun detection). */
    void setDeadlineUs(double us) { deadlineUs_ = us; }

    /**
     * Get stats snapshot. Call from UI/timer thread, not audio thread.
     * Copies the circular buffer and sorts — O(N log N) but off-RT.
     */
    Stats getStats() const {
        Stats s{};
        uint64_t count = count_.load(std::memory_order_acquire);
        if (count == 0) return s;

        size_t n = std::min(count, static_cast<uint64_t>(kHistorySize));
        std::array<double, kHistorySize> sorted{};

        for (size_t i = 0; i < n; ++i) {
            sorted[i] = ticksToUs(history_[i]);
        }
        std::sort(sorted.begin(), sorted.begin() + n);

        s.p50Us = sorted[n * 50 / 100];
        s.p90Us = sorted[n * 90 / 100];
        s.p99Us = sorted[std::min(n - 1, n * 99 / 100)];
        s.maxUs = sorted[n - 1];

        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) sum += sorted[i];
        s.meanUs = sum / static_cast<double>(n);

        s.sampleCount = count;
        s.overruns = overruns_.load(std::memory_order_relaxed);
        return s;
    }

    /** Reset all counters. Call from non-RT thread. */
    void reset() {
        count_.store(0, std::memory_order_release);
        overruns_.store(0, std::memory_order_release);
        history_.fill(0);
    }

private:
    void record(uint64_t durationTicks) {
        uint64_t idx = count_.fetch_add(1, std::memory_order_relaxed);
        history_[idx & (kHistorySize - 1)] = durationTicks;

        if (deadlineUs_ > 0.0 && ticksToUs(durationTicks) > deadlineUs_) {
            overruns_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static uint64_t nowTicks() {
#ifdef __APPLE__
        return mach_absolute_time();
#else
        return static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
#endif
    }

    double ticksToUs(uint64_t ticks) const {
#ifdef __APPLE__
        return static_cast<double>(ticks) * timebase_.numer / timebase_.denom / 1000.0;
#else
        // chrono nanoseconds → microseconds
        return static_cast<double>(ticks) / 1000.0;
#endif
    }

    std::array<uint64_t, kHistorySize> history_{};
    std::atomic<uint64_t> count_{0};
    std::atomic<uint32_t> overruns_{0};
    double deadlineUs_ = 0.0;

#ifdef __APPLE__
    mach_timebase_info_data_t timebase_{};
#endif
};

} // namespace diagnostics
} // namespace penta
