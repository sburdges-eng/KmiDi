#pragma once
/**
 * AudioWorkerThread.h — RT-safe worker that joins the host's audio workgroup.
 *
 * Usage:
 *   auto worker = AudioWorkerThread([](){ processMLInference(); });
 *   worker.start(pluginProcessor.getAudioWorkgroup());
 *   // ... later
 *   worker.stop();
 *
 * macOS-specific: uses os_workgroup + mach thread_policy for RT scheduling.
 * Falls back to high-priority thread on other platforms.
 */

#include <atomic>
#include <functional>
#include <thread>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#if __has_include(<os/workgroup.h>)
#include <os/workgroup.h>
#define PENTA_HAS_WORKGROUP 1
#endif
#endif

namespace penta {
namespace rt {

class AudioWorkerThread {
public:
    using WorkFn = std::function<void()>;

    explicit AudioWorkerThread(WorkFn work, uint32_t computationUs = 2000,
                               uint32_t constraintUs = 5000)
        : work_(std::move(work))
        , computationUs_(computationUs)
        , constraintUs_(constraintUs) {}

    ~AudioWorkerThread() { stop(); }

    /**
     * Start the worker thread. On macOS, joins the audio workgroup for
     * cooperative scheduling with the audio render thread.
     *
     * @param workgroup  Opaque workgroup handle from the host (e.g. JUCE AudioWorkgroup).
     *                   Pass nullptr for standalone/non-AU contexts.
     */
    void start(void* workgroup = nullptr) {
        if (running_.exchange(true)) return;
        workgroup_ = workgroup;
        thread_ = std::thread([this]() { run(); });
    }

    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
    }

    bool isRunning() const { return running_.load(); }

private:
    void run() {
        setRealtimePolicy();
        joinWorkgroup();

        while (running_.load(std::memory_order_relaxed)) {
            work_();
        }

        leaveWorkgroup();
    }

    void setRealtimePolicy() {
#ifdef __APPLE__
        thread_time_constraint_policy_data_t policy{};
        policy.period      = 0;
        policy.computation = computationUs_ * 1000; // ns
        policy.constraint  = constraintUs_ * 1000;
        policy.preemptible = true;

        thread_policy_set(mach_thread_self(),
                          THREAD_TIME_CONSTRAINT_POLICY,
                          reinterpret_cast<thread_policy_t>(&policy),
                          THREAD_TIME_CONSTRAINT_POLICY_COUNT);
#endif
        // Linux/Windows: rely on thread priority set by JUCE or caller
    }

    void joinWorkgroup() {
#if defined(PENTA_HAS_WORKGROUP)
        if (!workgroup_) return;
        auto wg = static_cast<os_workgroup_t>(workgroup_);
        joinedWorkgroup_ = (os_workgroup_join(wg, &wgToken_) == 0);
#endif
    }

    void leaveWorkgroup() {
#if defined(PENTA_HAS_WORKGROUP)
        if (joinedWorkgroup_ && workgroup_) {
            auto wg = static_cast<os_workgroup_t>(workgroup_);
            os_workgroup_leave(wg, &wgToken_);
            joinedWorkgroup_ = false;
        }
#endif
    }

    WorkFn work_;
    uint32_t computationUs_;
    uint32_t constraintUs_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    void* workgroup_ = nullptr;

#if defined(PENTA_HAS_WORKGROUP)
    os_workgroup_join_token_s wgToken_{};
    bool joinedWorkgroup_ = false;
#endif
};

} // namespace rt
} // namespace penta
