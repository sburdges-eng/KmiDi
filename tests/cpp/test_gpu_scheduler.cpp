// Unit test for daiw::rt::GpuScheduler / CpuGpuScheduler.
//
// Coverage:
//   1. CpuGpuScheduler runs submitted jobs and marks them done
//   2. RT-style polling pattern: producer submits, audio-thread proxy
//      polls is_done() and consumes the result
//   3. MetalGpuScheduler stub reports the right backend name and is not
//      available unless KMIDI_ENABLE_METAL is set
//   4. select_default_scheduler picks Metal when available, CPU otherwise
//   5. Submitting null or to a destroyed scheduler is safe

#include "daiw/gpu_scheduler.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;
void check(bool condition, const char *label) {
  if (!condition) {
    std::printf("FAIL: %s\n", label);
    ++g_failures;
  }
}

class CountingJob : public daiw::rt::GpuJob {
public:
  explicit CountingJob(std::atomic<int> *counter) : counter_(counter) {}
  void execute() override {
    // Simulate a tiny inference workload.
    for (int i = 0; i < 1024; ++i) {
      volatile float x = std::sqrt(static_cast<float>(i + 1));
      (void)x;
    }
    counter_->fetch_add(1, std::memory_order_relaxed);
    mark_done();
  }

private:
  std::atomic<int> *counter_;
};

// Helper: wait up to `timeout_ms` for predicate to become true, polling at
// 100us intervals. Returns whether it eventually held.
template <typename Pred> bool wait_for(Pred &&pred, int timeout_ms = 1000) {
  using namespace std::chrono_literals;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred())
      return true;
    std::this_thread::sleep_for(100us);
  }
  return pred();
}

void test_cpu_scheduler_runs_jobs() {
  daiw::rt::CpuGpuScheduler<128> sched;
  check(sched.is_available(), "cpu: scheduler available");

  std::atomic<int> counter{0};
  std::vector<std::unique_ptr<CountingJob>> jobs;
  jobs.reserve(64);
  for (int i = 0; i < 64; ++i) {
    jobs.push_back(std::make_unique<CountingJob>(&counter));
  }

  for (auto &j : jobs) {
    check(sched.submit(j.get()), "cpu: submit succeeds");
  }

  const bool all_done = wait_for([&]() {
    for (const auto &j : jobs)
      if (!j->is_done())
        return false;
    return true;
  });
  check(all_done, "cpu: all 64 jobs completed within timeout");
  check(counter == 64, "cpu: counter incremented for every job");
}

void test_rt_polling_pattern() {
  daiw::rt::CpuGpuScheduler<32> sched;
  std::atomic<int> counter{0};
  CountingJob job(&counter);

  // Producer (UI / inference orchestrator) submits.
  check(sched.submit(&job), "rt-poll: submit");

  // Audio-thread proxy: polls is_done() until set, then reads result.
  int polls = 0;
  while (!job.is_done()) {
    ++polls;
    if (polls > 100000)
      break; // safety
    std::this_thread::yield();
  }
  check(job.is_done(), "rt-poll: is_done() flips after execute()");
  check(counter.load() == 1, "rt-poll: counter incremented exactly once");

  // Reset + re-submit semantics.
  job.reset();
  check(!job.is_done(), "rt-poll: is_done() false after reset()");
  sched.submit(&job);
  wait_for([&]() { return job.is_done(); });
  check(counter.load() == 2, "rt-poll: re-submit re-runs the job");
}

void test_metal_stub_reports_correctly() {
  daiw::rt::MetalGpuScheduler m;
  const char *name = m.backend_name();
  check(name != nullptr, "metal: backend_name not null");
  check(name[0] == 'm' && name[1] == 'e' && name[2] == 't',
        "metal: backend_name starts with 'met'");

#ifdef KMIDI_ENABLE_METAL
  check(m.is_available(), "metal: available when flag enabled");
#else
  check(!m.is_available(), "metal: not available without KMIDI_ENABLE_METAL");
  std::atomic<int> dummy{0};
  CountingJob job(&dummy);
  check(!m.submit(&job), "metal: submit refuses work in stub mode");
#endif
}

void test_select_default() {
  daiw::rt::MetalGpuScheduler m;
  daiw::rt::CpuGpuScheduler<> c;
  auto *chosen = daiw::rt::select_default_scheduler(m, c);
  check(chosen != nullptr, "select: returns a scheduler");
#ifdef KMIDI_ENABLE_METAL
  check(chosen == &m, "select: prefers metal when available");
#else
  check(chosen == &c, "select: falls back to cpu when metal unavailable");
#endif
}

void test_null_submit_safe() {
  daiw::rt::CpuGpuScheduler<32> sched;
  check(!sched.submit(nullptr),
        "null: submit(nullptr) rejected without crashing");
}

} // namespace

int main() {
  test_cpu_scheduler_runs_jobs();
  test_rt_polling_pattern();
  test_metal_stub_reports_correctly();
  test_select_default();
  test_null_submit_safe();

  if (g_failures == 0) {
    std::printf("OK: all gpu-scheduler checks passed\n");
    return 0;
  }
  std::printf("FAIL: %d check(s) failed\n", g_failures);
  return 1;
}
