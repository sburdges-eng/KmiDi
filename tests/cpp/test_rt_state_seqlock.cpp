// Seqlock torture test for penta::RTState.
//
// One writer thread bumps begin_publish / end_publish around a batch of
// paired stores (valence == arousal == dominance). A reader thread snapshots
// 1,000,000 times. Every observed snapshot must satisfy v == a == d — a
// mismatch means the seqlock leaked a torn window.
//
// This test deliberately does NOT link Catch2 or KellyCore. It is a tiny
// standalone executable that depends only on the penta::RTState header
// (which is header-only). This keeps it runnable even when JUCE is absent
// from the worktree.

#include "penta/common/RTState.h"

#include <atomic>
#include <cstdio>
#include <thread>

int main() {
    penta::RTState s;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> tearings{0};
    std::atomic<uint64_t> snapshot_failures{0};
    std::atomic<uint64_t> pre_publish_skipped{0};

    std::thread writer([&] {
        uint64_t n = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            // v is chosen so v == a == d always, and nonzero (to distinguish
            // from the RTState default-initialised pre-publish state where
            // valence=0.0, arousal=0.5, dominance=0.5 — differ by design).
            const float v = 1.0f + static_cast<float>(n & 0xFFFF);
            s.begin_publish();
            s.valence.store(v, std::memory_order_relaxed);
            s.arousal.store(v, std::memory_order_relaxed);
            s.dominance.store(v, std::memory_order_relaxed);
            s.end_publish();
            ++n;
        }
    });

    // Wait for at least one complete publish before the reader starts, so
    // we don't count pre-publish reads of the (intentionally different)
    // default field values as tearings.
    while (s.sequence.load(std::memory_order_acquire) < 2) {
        std::this_thread::yield();
    }

    constexpr int kIterations = 1'000'000;
    for (int i = 0; i < kIterations; ++i) {
        float v = 0.0f, a = 0.0f, d = 0.0f;
        const bool ok = penta::rt_state_snapshot(s, [&]() noexcept {
            v = s.valence.load(std::memory_order_relaxed);
            a = s.arousal.load(std::memory_order_relaxed);
            d = s.dominance.load(std::memory_order_relaxed);
        });
        if (!ok) {
            snapshot_failures.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        // Guard against the writer hitting its very first publish cycle
        // between our `sequence >= 2` wait and our first snapshot — if all
        // three are still defaults (v=0, a=0.5, d=0.5), skip rather than
        // count as a tear.
        if (v == 0.0f && a == 0.5f && d == 0.5f) {
            pre_publish_skipped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (v != a || a != d) {
            tearings.fetch_add(1, std::memory_order_relaxed);
        }
    }

    stop.store(true, std::memory_order_relaxed);
    writer.join();

    const auto t = tearings.load();
    const auto f = snapshot_failures.load();
    const auto p = pre_publish_skipped.load();
    std::printf("iterations=%d tearings=%llu snapshot_failures=%llu pre_publish=%llu\n",
                kIterations,
                static_cast<unsigned long long>(t),
                static_cast<unsigned long long>(f),
                static_cast<unsigned long long>(p));

    // Tearings must be zero. Snapshot failures (8-retry exhaustion) are
    // tolerable under extreme thrashing but should be very rare; we surface
    // them for visibility but do not fail the test on them.
    return t == 0 ? 0 : 1;
}
