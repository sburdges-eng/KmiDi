#include "penta/diagnostics/PerformanceMonitor.h"
#include <algorithm>
#include <numeric>

namespace penta::diagnostics
{

    // Latency measurement uses std::chrono (Clock::now()) in beginMeasurement/endMeasurement.
    // Platform-specific getHighResTimestamp() was removed as dead code.

    PerformanceMonitor::PerformanceMonitor()
        : measurementStart_(), latencyHistory_(kHistorySize, 0), historyIndex_(0), peakLatencyUs_(0), xrunCount_(0)
    {
    }

    void PerformanceMonitor::beginMeasurement() noexcept
    {
        // Store start time using high-resolution timer
        // This is RT-safe as it only reads a timestamp
        measurementStart_ = Clock::now();
    }

    void PerformanceMonitor::endMeasurement() noexcept
    {
        // Calculate elapsed time
        auto end = Clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - measurementStart_);
        uint64_t latencyUs = elapsed.count() / 1000;
        // Preserve a tiny-but-nonzero measurement for very fast code paths.
        // Without this, sub-microsecond measurements will quantize to 0us and make
        // averages/peaks appear as 0 on fast machines.
        if (elapsed.count() > 0 && latencyUs == 0)
        {
            latencyUs = 1;
        }

        // Update circular buffer (RT-safe)
        size_t idx = historyIndex_.fetch_add(1, std::memory_order_relaxed) % kHistorySize;
        latencyHistory_[idx] = latencyUs;

        // Update peak (RT-safe atomic)
        uint64_t currentPeak = peakLatencyUs_.load(std::memory_order_relaxed);
        while (latencyUs > currentPeak &&
               !peakLatencyUs_.compare_exchange_weak(currentPeak, latencyUs,
                                                     std::memory_order_relaxed))
        {
            // Spin until we successfully update or another thread sets a higher peak
        }
    }

    void PerformanceMonitor::recordXrun() noexcept
    {
        xrunCount_.fetch_add(1, std::memory_order_relaxed);
    }

    float PerformanceMonitor::getAverageLatencyUs() const
    {
        // Non-RT: Calculate average from history
        size_t count = std::min(historyIndex_.load(std::memory_order_relaxed), kHistorySize);
        if (count == 0)
            return 0.0f;

        uint64_t sum = 0;
        for (size_t i = 0; i < count; ++i)
        {
            sum += latencyHistory_[i];
        }

        return static_cast<float>(sum) / static_cast<float>(count);
    }

    float PerformanceMonitor::getPeakLatencyUs() const
    {
        return static_cast<float>(peakLatencyUs_.load(std::memory_order_relaxed));
    }

    float PerformanceMonitor::getCpuUsagePercent(size_t bufferSize, double sampleRate) const
    {
        // Calculate CPU usage as percentage of buffer duration
        // Formula: (average_latency_us / buffer_duration_us) * 100

        if (sampleRate <= 0.0 || bufferSize == 0)
        {
            return 0.0f;
        }

        float avgLatencyUs = getAverageLatencyUs();
        float bufferDurationUs = (static_cast<float>(bufferSize) / static_cast<float>(sampleRate)) * 1000000.0f;

        if (bufferDurationUs <= 0.0f)
        {
            return 0.0f;
        }

        return (avgLatencyUs / bufferDurationUs) * 100.0f;
    }

    void PerformanceMonitor::reset()
    {
        // Non-RT: Reset all statistics
        historyIndex_.store(0, std::memory_order_relaxed);
        peakLatencyUs_.store(0, std::memory_order_relaxed);
        xrunCount_.store(0, std::memory_order_relaxed);
        std::fill(latencyHistory_.begin(), latencyHistory_.end(), 0);
    }

} // namespace penta::diagnostics
