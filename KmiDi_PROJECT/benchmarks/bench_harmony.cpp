/**
 * @file bench_harmony.cpp
 * @brief Harmony Engine Performance Benchmarks
 *
 * Target: <100μs latency @ 48kHz/512 samples
 *
 * Build:
 *   cmake -DDAIW_BUILD_BENCHMARKS=ON ..
 *   cmake --build . --target bench_harmony
 *
 * Run:
 *   ./benchmarks/bench_harmony
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

// Target latency in microseconds
constexpr double TARGET_LATENCY_US = 100.0;

// Test parameters
constexpr size_t NUM_ITERATIONS = 10000;
constexpr size_t WARMUP_ITERATIONS = 1000;
constexpr double SAMPLE_RATE = 48000.0;
constexpr size_t BUFFER_SIZE = 512;

// High-resolution timer
class HighResTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedMicroseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Simulated chord analysis (placeholder for actual implementation)
 *
 * In production, this would call the actual HarmonyEngine::analyzeChord()
 * from penta-core.
 */
struct ChordAnalysisResult {
    int rootNote;       // MIDI note number
    int quality;        // 0=major, 1=minor, etc.
    float confidence;   // 0.0 to 1.0
};

ChordAnalysisResult analyzeChord(const std::vector<int>& midiNotes) {
    ChordAnalysisResult result;

    if (midiNotes.empty()) {
        result.rootNote = 60;  // C4
        result.quality = 0;
        result.confidence = 0.0f;
        return result;
    }

    // Simplified chord detection
    // Find the lowest note as potential root
    result.rootNote = *std::min_element(midiNotes.begin(), midiNotes.end());

    // Check intervals to determine quality
    std::vector<int> intervals;
    for (int note : midiNotes) {
        intervals.push_back((note - result.rootNote) % 12);
    }

    // Sort and remove duplicates
    std::sort(intervals.begin(), intervals.end());
    intervals.erase(std::unique(intervals.begin(), intervals.end()), intervals.end());

    // Check for major/minor third
    bool hasMinorThird = std::find(intervals.begin(), intervals.end(), 3) != intervals.end();
    bool hasMajorThird = std::find(intervals.begin(), intervals.end(), 4) != intervals.end();

    result.quality = hasMinorThird ? 1 : (hasMajorThird ? 0 : 2);
    result.confidence = static_cast<float>(intervals.size()) / 4.0f;

    return result;
}

/**
 * @brief Simulated scale detection
 */
struct ScaleResult {
    int root;
    int mode;  // 0=ionian, 1=dorian, etc.
    float score;
};

ScaleResult detectScale(const std::vector<int>& midiNotes) {
    ScaleResult result;
    result.root = 0;
    result.mode = 0;
    result.score = 0.0f;

    if (midiNotes.empty()) return result;

    // Build pitch class histogram
    std::array<int, 12> histogram = {0};
    for (int note : midiNotes) {
        histogram[note % 12]++;
    }

    // Find most common pitch class
    auto maxIt = std::max_element(histogram.begin(), histogram.end());
    result.root = static_cast<int>(std::distance(histogram.begin(), maxIt));

    // Score based on histogram correlation with major scale template
    const std::array<int, 7> majorScale = {0, 2, 4, 5, 7, 9, 11};
    for (int degree : majorScale) {
        int pc = (result.root + degree) % 12;
        result.score += histogram[pc];
    }
    result.score /= static_cast<float>(midiNotes.size());

    return result;
}

/**
 * @brief Statistics calculation
 */
struct BenchmarkStats {
    double mean;
    double median;
    double p95;
    double p99;
    double min;
    double max;
    double stddev;
};

BenchmarkStats computeStats(std::vector<double>& latencies) {
    BenchmarkStats stats;

    std::sort(latencies.begin(), latencies.end());

    stats.min = latencies.front();
    stats.max = latencies.back();
    stats.mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    size_t mid = latencies.size() / 2;
    stats.median = (latencies.size() % 2 == 0)
        ? (latencies[mid - 1] + latencies[mid]) / 2.0
        : latencies[mid];

    stats.p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    stats.p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    // Standard deviation
    double sq_sum = std::inner_product(latencies.begin(), latencies.end(),
                                       latencies.begin(), 0.0,
                                       std::plus<>(),
                                       [mean=stats.mean](double a, double b) {
                                           return (a - mean) * (b - mean);
                                       });
    stats.stddev = std::sqrt(sq_sum / latencies.size());

    return stats;
}

void printStats(const BenchmarkStats& stats, double target) {
    bool passed = stats.mean < target;

    std::cout << "\n  Results (" << NUM_ITERATIONS << " iterations):\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    Mean:     " << stats.mean << " μs\n";
    std::cout << "    Median:   " << stats.median << " μs\n";
    std::cout << "    Stddev:   " << stats.stddev << " μs\n";
    std::cout << "    P95:      " << stats.p95 << " μs\n";
    std::cout << "    P99:      " << stats.p99 << " μs\n";
    std::cout << "    Min:      " << stats.min << " μs\n";
    std::cout << "    Max:      " << stats.max << " μs\n";

    std::cout << "\n  " << (passed ? "✅ PASS" : "❌ FAIL")
              << ": Mean " << stats.mean << "μs "
              << (passed ? "<" : ">=") << " target " << target << "μs\n";
}

/**
 * @brief Benchmark chord analysis
 */
void benchmarkChordAnalysis() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Chord Analysis Performance Benchmark\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\n  Target: <" << TARGET_LATENCY_US << "μs latency\n";
    std::cout << "  Sample rate: " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer size: " << BUFFER_SIZE << " samples\n";

    // Generate test data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> noteDist(48, 84);  // C3 to C6
    std::uniform_int_distribution<int> chordSizeDist(3, 6);

    std::vector<std::vector<int>> testChords;
    for (size_t i = 0; i < 100; ++i) {
        std::vector<int> chord;
        int size = chordSizeDist(rng);
        for (int j = 0; j < size; ++j) {
            chord.push_back(noteDist(rng));
        }
        testChords.push_back(chord);
    }

    // Warmup
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        analyzeChord(testChords[i % testChords.size()]);
    }

    // Benchmark
    std::vector<double> latencies;
    latencies.reserve(NUM_ITERATIONS);

    HighResTimer timer;
    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        timer.start();
        analyzeChord(testChords[i % testChords.size()]);
        latencies.push_back(timer.elapsedMicroseconds());
    }

    auto stats = computeStats(latencies);
    printStats(stats, TARGET_LATENCY_US);
}

/**
 * @brief Benchmark scale detection
 */
void benchmarkScaleDetection() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Scale Detection Performance Benchmark\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\n  Target: <" << TARGET_LATENCY_US << "μs latency\n";

    // Generate test data (melodies)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> noteDist(48, 84);

    std::vector<std::vector<int>> testMelodies;
    for (size_t i = 0; i < 100; ++i) {
        std::vector<int> melody;
        for (int j = 0; j < 16; ++j) {  // 16-note phrases
            melody.push_back(noteDist(rng));
        }
        testMelodies.push_back(melody);
    }

    // Warmup
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        detectScale(testMelodies[i % testMelodies.size()]);
    }

    // Benchmark
    std::vector<double> latencies;
    latencies.reserve(NUM_ITERATIONS);

    HighResTimer timer;
    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        timer.start();
        detectScale(testMelodies[i % testMelodies.size()]);
        latencies.push_back(timer.elapsedMicroseconds());
    }

    auto stats = computeStats(latencies);
    printStats(stats, TARGET_LATENCY_US);
}

/**
 * @brief Combined harmony processing benchmark
 */
void benchmarkCombinedHarmony() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Combined Harmony Processing Benchmark\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\n  Target: <" << TARGET_LATENCY_US << "μs latency\n";
    std::cout << "  Simulating real-time audio processing\n";

    // Generate test data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> noteDist(48, 84);

    std::vector<int> testBuffer;
    for (size_t i = 0; i < BUFFER_SIZE / 16; ++i) {  // ~32 notes per buffer
        testBuffer.push_back(noteDist(rng));
    }

    // Warmup
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        analyzeChord(testBuffer);
        detectScale(testBuffer);
    }

    // Benchmark combined processing
    std::vector<double> latencies;
    latencies.reserve(NUM_ITERATIONS);

    HighResTimer timer;
    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        timer.start();
        auto chord = analyzeChord(testBuffer);
        auto scale = detectScale(testBuffer);
        (void)chord;  // Prevent optimization
        (void)scale;
        latencies.push_back(timer.elapsedMicroseconds());
    }

    auto stats = computeStats(latencies);
    printStats(stats, TARGET_LATENCY_US);
}

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "KmiDi Harmony Engine Benchmark Suite\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\nConfiguration:\n";
    std::cout << "  Target latency: " << TARGET_LATENCY_US << " μs\n";
    std::cout << "  Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "  Warmup: " << WARMUP_ITERATIONS << "\n";
    std::cout << "  Sample rate: " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer size: " << BUFFER_SIZE << " samples\n";

    // Run benchmarks
    benchmarkChordAnalysis();
    benchmarkScaleDetection();
    benchmarkCombinedHarmony();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Benchmark complete.\n";
    std::cout << std::string(70, '=') << "\n\n";

    return 0;
}
