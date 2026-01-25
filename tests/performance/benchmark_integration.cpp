#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>

#include "bridge/kelly_ffi.h"

// =============================================================================
// Performance Benchmark Suite for KmiDi Integration
// =============================================================================

class PerformanceBenchmark {
public:
    static void run_all_benchmarks() {
        std::cout << "KmiDi Integration Performance Benchmarks\n";
        std::cout << "========================================\n\n";
        
        benchmark_ffi_call_overhead();
        benchmark_memory_management();
        benchmark_text_processing();
        benchmark_emotion_parameter_updates();
        benchmark_concurrent_access();
        benchmark_state_queries();
        
        std::cout << "\nBenchmark Summary:\n";
        std::cout << "FFI calls should be < 10µs\n";
        std::cout << "Memory operations should be < 100µs\n";
        std::cout << "Text processing should be < 10ms\n";
        std::cout << "Parameter updates should be < 1ms\n";
    }
    
private:
    static void benchmark_ffi_call_overhead() {
        std::cout << "1. FFI Call Overhead Benchmark\n";
        std::cout << "--------------------------------\n";
        
        const int iterations = 10000;
        
        // Measure version call (simple FFI)
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            const char* version = kelly_get_version();
            (void)version;  // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double avg_ns = double(duration.count()) / iterations;
        
        std::cout << "Version call: " << avg_ns << " ns per call\n";
        std::cout << "             " << (avg_ns / 1000.0) << " µs per call\n";
        
        if (avg_ns < 10000) {  // < 10µs
            std::cout << "✅ FFI overhead: EXCELLENT\n";
        } else if (avg_ns < 50000) {  // < 50µs
            std::cout << "⚠️  FFI overhead: ACCEPTABLE\n";
        } else {
            std::cout << "❌ FFI overhead: TOO HIGH\n";
        }
        
        std::cout << "\n";
    }
    
    static void benchmark_memory_management() {
        std::cout << "2. Memory Management Benchmark\n";
        std::cout << "-------------------------------\n";
        
        const int iterations = 1000;
        
        // Measure KellyBrain creation/destruction
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            KellyBrain* brain = kelly_brain_create();
            if (brain) {
                kelly_brain_destroy(brain);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_us = double(duration.count()) / iterations;
        
        std::cout << "Create/Destroy: " << avg_us << " µs per cycle\n";
        
        if (avg_us < 100) {
            std::cout << "✅ Memory management: EXCELLENT\n";
        } else if (avg_us < 500) {
            std::cout << "⚠️  Memory management: ACCEPTABLE\n";
        } else {
            std::cout << "❌ Memory management: TOO SLOW\n";
        }
        
        std::cout << "\n";
    }
    
    static void benchmark_text_processing() {
        std::cout << "3. Text Processing Benchmark\n";
        std::cout << "-----------------------------\n";
        
        KellyBrain* brain = kelly_brain_create();
        if (!brain) {
            std::cout << "❌ Failed to create KellyBrain for benchmark\n\n";
            return;
        }
        
        // Note: This will likely fail without initialization, but we can measure the call time
        const char* test_texts[] = {
            "I feel happy and energetic",
            "Feeling sad and lonely today",
            "Excited about new possibilities",
            "Overwhelmed by recent changes",
            "Peaceful and content"
        };
        
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            const char* text = test_texts[i % 5];
            char* result = kelly_brain_from_text(brain, text);
            if (result) {
                kelly_free_string(result);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double avg_ms = double(duration.count()) / iterations;
        
        std::cout << "Text processing: " << avg_ms << " ms per call\n";
        
        if (avg_ms < 10) {
            std::cout << "✅ Text processing: EXCELLENT\n";
        } else if (avg_ms < 50) {
            std::cout << "⚠️  Text processing: ACCEPTABLE\n";
        } else {
            std::cout << "❌ Text processing: TOO SLOW\n";
        }
        
        kelly_brain_destroy(brain);
        std::cout << "\n";
    }
    
    static void benchmark_emotion_parameter_updates() {
        std::cout << "4. Emotion Parameter Update Benchmark\n";
        std::cout << "--------------------------------------\n";
        
        KellyBrain* brain = kelly_brain_create();
        if (!brain) {
            std::cout << "❌ Failed to create KellyBrain for benchmark\n\n";
            return;
        }
        
        const int iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            float valence = (float(i) / iterations) * 2.0f - 1.0f;  // -1 to 1
            float arousal = float(i % 100) / 100.0f;                // 0 to 1
            float dominance = 0.5f;
            
            kelly_brain_set_emotion_parameters(brain, valence, arousal, dominance);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_us = double(duration.count()) / iterations;
        
        std::cout << "Parameter updates: " << avg_us << " µs per update\n";
        
        if (avg_us < 100) {
            std::cout << "✅ Parameter updates: EXCELLENT\n";
        } else if (avg_us < 1000) {
            std::cout << "⚠️  Parameter updates: ACCEPTABLE\n";
        } else {
            std::cout << "❌ Parameter updates: TOO SLOW\n";
        }
        
        kelly_brain_destroy(brain);
        std::cout << "\n";
    }
    
    static void benchmark_concurrent_access() {
        std::cout << "5. Concurrent Access Benchmark\n";
        std::cout << "-------------------------------\n";
        
        KellyBrain* brain = kelly_brain_create();
        if (!brain) {
            std::cout << "❌ Failed to create KellyBrain for benchmark\n\n";
            return;
        }
        
        const int num_threads = 4;
        const int iterations_per_thread = 250;
        std::atomic<int> completed_operations{0};
        std::vector<std::thread> threads;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Spawn threads that perform read operations
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([brain, iterations_per_thread, &completed_operations]() {
                for (int i = 0; i < iterations_per_thread; ++i) {
                    // Perform read-only operations
                    bool initialized = kelly_brain_is_initialized(brain);
                    const char* version = kelly_get_version();
                    (void)initialized;
                    (void)version;
                    
                    completed_operations++;
                }
            });
        }
        
        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        int total_operations = completed_operations.load();
        double ops_per_second = double(total_operations) / (duration.count() / 1000.0);
        
        std::cout << "Concurrent operations: " << total_operations << " total\n";
        std::cout << "Throughput: " << ops_per_second << " ops/second\n";
        std::cout << "Duration: " << duration.count() << " ms\n";
        
        if (ops_per_second > 100000) {
            std::cout << "✅ Concurrent access: EXCELLENT\n";
        } else if (ops_per_second > 10000) {
            std::cout << "⚠️  Concurrent access: ACCEPTABLE\n";
        } else {
            std::cout << "❌ Concurrent access: TOO SLOW\n";
        }
        
        kelly_brain_destroy(brain);
        std::cout << "\n";
    }
    
    static void benchmark_state_queries() {
        std::cout << "6. State Query Benchmark\n";
        std::cout << "-------------------------\n";
        
        KellyBrain* brain = kelly_brain_create();
        if (!brain) {
            std::cout << "❌ Failed to create KellyBrain for benchmark\n\n";
            return;
        }
        
        const int iterations = 500;
        
        // Benchmark emotion state queries
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            char* state = kelly_brain_get_emotion_state(brain);
            if (state) {
                kelly_free_string(state);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_us = double(duration.count()) / iterations;
        
        std::cout << "Emotion state queries: " << avg_us << " µs per query\n";
        
        // Benchmark available emotions query
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            char* emotions = kelly_brain_get_available_emotions(brain);
            if (emotions) {
                kelly_free_string(emotions);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        avg_us = double(duration.count()) / iterations;
        
        std::cout << "Available emotions queries: " << avg_us << " µs per query\n";
        
        if (avg_us < 500) {
            std::cout << "✅ State queries: EXCELLENT\n";
        } else if (avg_us < 2000) {
            std::cout << "⚠️  State queries: ACCEPTABLE\n";
        } else {
            std::cout << "❌ State queries: TOO SLOW\n";
        }
        
        kelly_brain_destroy(brain);
        std::cout << "\n";
    }
};

int main() {
    try {
        PerformanceBenchmark::run_all_benchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }
}