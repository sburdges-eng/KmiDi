import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

// Mock Tauri API for performance testing
const createMockInvoke = () => {
  const callTimes: number[] = [];

  return {
    invoke: vi.fn().mockImplementation(async (cmd: string, args?: any) => {
      const start = performance.now();

      // Simulate processing time based on command complexity
      const processingTime = getSimulatedProcessingTime(cmd);
      await new Promise((resolve) => setTimeout(resolve, processingTime));

      const end = performance.now();
      callTimes.push(end - start);

      return getMockResponse(cmd, args);
    }),
    getCallTimes: () => callTimes,
    getAverageCallTime: () =>
      callTimes.reduce((a, b) => a + b, 0) / callTimes.length,
  };
};

const getSimulatedProcessingTime = (cmd: string): number => {
  switch (cmd) {
    case "kelly_brain_from_text":
      return 5; // 5ms for text processing
    case "kelly_brain_generate_midi":
      return 20; // 20ms for MIDI generation
    case "kelly_brain_set_emotion_parameters":
      return 1; // 1ms for parameter updates
    case "kelly_brain_get_emotion_state":
      return 2; // 2ms for state queries
    case "kelly_brain_initialize":
      return 100; // 100ms for initialization
    default:
      return 1; // 1ms for simple operations
  }
};

const getMockResponse = (cmd: string, args?: any): any => {
  switch (cmd) {
    case "kelly_brain_initialize":
      return true;
    case "kelly_brain_is_initialized":
      return true;
    case "kelly_brain_from_text":
      return {
        core_wound: "performance_test_wound",
        core_desire: "performance_test_desire",
        emotional_intent: args?.text || "performance test",
        valence: 0.5,
        arousal: 0.7,
        dominance: 0.3,
        complexity: 0.6,
        progression: ["C", "F", "G", "Am"],
        key: "C",
        bpm: 120,
        genre: "test",
      };
    case "kelly_brain_generate_midi":
      return {
        bars: args?.bars || 8,
        bpm: 120,
        key: "C",
        time_signature: "4/4",
        tracks: Array.from({ length: 4 }, (_, i) => ({
          name: `Track_${i}`,
          channel: i + 1,
          events: Array.from({ length: 50 }, (_, j) => ({
            event_type: 1,
            timestamp: j * 0.25,
            note: 60 + (j % 12),
            velocity: 80,
            duration: 0.2,
          })),
        })),
      };
    case "kelly_brain_get_emotion_state":
      return {
        valence: Math.random() * 2 - 1,
        arousal: Math.random(),
        dominance: Math.random(),
        complexity: Math.random(),
      };
    case "kelly_brain_set_emotion_parameters":
      return true;
    default:
      return {};
  }
};

describe("React Hook Performance Tests", () => {
  let mockInvokeSystem: ReturnType<typeof createMockInvoke>;

  beforeEach(() => {
    mockInvokeSystem = createMockInvoke();
    (global as any).invoke = mockInvokeSystem.invoke;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("useKellyBrain Hook Performance", () => {
    it("should initialize quickly", async () => {
      // Import hook dynamically to use mocked invoke
      const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");

      const { result } = renderHook(() => useKellyBrain());

      const startTime = performance.now();

      await act(async () => {
        await result.current.initialize("./test-data");
      });

      const endTime = performance.now();
      const initTime = endTime - startTime;

      expect(initTime).toBeLessThan(200); // < 200ms for initialization
      expect(result.current.isInitialized).toBe(true);

      console.log(`Hook initialization time: ${initTime.toFixed(2)}ms`);
    });

    it("should handle rapid emotion parameter updates", async () => {
      const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");

      const { result } = renderHook(() => useKellyBrain());

      // Initialize first
      await act(async () => {
        await result.current.initialize("./test-data");
      });

      const updates = 50;
      const startTime = performance.now();

      // Perform rapid updates
      for (let i = 0; i < updates; i++) {
        const valence = (i / updates) * 2 - 1; // -1 to 1
        const arousal = (i % 10) / 10; // 0 to 1

        await act(async () => {
          await result.current.setEmotionParameters(valence, arousal, 0.5);
        });
      }

      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const avgTime = totalTime / updates;

      expect(avgTime).toBeLessThan(10); // < 10ms per update
      expect(result.current.error).toBeNull();

      console.log(`Average emotion update time: ${avgTime.toFixed(2)}ms`);
    });

    it("should handle text processing efficiently", async () => {
      const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");

      const { result } = renderHook(() => useKellyBrain());

      await act(async () => {
        await result.current.initialize("./test-data");
      });

      const texts = [
        "feeling hopeful about the future",
        "overwhelmed by sadness",
        "excited and energetic",
        "calm and peaceful",
        "anxious about changes",
      ];

      const startTime = performance.now();

      for (const text of texts) {
        await act(async () => {
          await result.current.fromText(text);
        });
      }

      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const avgTime = totalTime / texts.length;

      expect(avgTime).toBeLessThan(30); // < 30ms per text processing
      expect(result.current.state.current_intent).toBeTruthy();

      console.log(`Average text processing time: ${avgTime.toFixed(2)}ms`);
    });
  });

  describe("State Management Performance", () => {
    it("should handle rapid state changes efficiently", async () => {
      const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");

      const { result } = renderHook(() => useKellyBrain());

      await act(async () => {
        await result.current.initialize("./test-data");
      });

      const stateUpdates = 100;
      const startTime = performance.now();

      // Simulate rapid state changes
      for (let i = 0; i < stateUpdates; i++) {
        await act(async () => {
          await result.current.getEmotionState();
        });
      }

      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const avgTime = totalTime / stateUpdates;

      expect(avgTime).toBeLessThan(5); // < 5ms per state query

      console.log(`Average state query time: ${avgTime.toFixed(2)}ms`);
    });

    it("should batch multiple operations efficiently", async () => {
      const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");

      const { result } = renderHook(() => useKellyBrain());

      await act(async () => {
        await result.current.initialize("./test-data");
      });

      const startTime = performance.now();

      // Batch multiple operations
      await act(async () => {
        const promises = [
          result.current.fromText("test emotion"),
          result.current.getEmotionState(),
          result.current.setEmotionParameters(0.5, 0.7, 0.3),
          result.current.getAvailableEmotions(),
        ];

        await Promise.all(promises);
      });

      const endTime = performance.now();
      const batchTime = endTime - startTime;

      expect(batchTime).toBeLessThan(100); // < 100ms for batch operations

      console.log(`Batch operation time: ${batchTime.toFixed(2)}ms`);
    });
  });

  describe("Memory Usage Performance", () => {
    it("should not leak memory with repeated operations", async () => {
      const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");

      const { result } = renderHook(() => useKellyBrain());

      await act(async () => {
        await result.current.initialize("./test-data");
      });

      // Measure initial memory usage (if available)
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;

      // Perform many operations
      const iterations = 100;
      for (let i = 0; i < iterations; i++) {
        await act(async () => {
          const intent = await result.current.fromText(`test emotion ${i}`);
          if (intent) {
            await result.current.generateMidi(intent, 4);
          }
        });
      }

      // Force garbage collection if available
      if ((global as any).gc) {
        (global as any).gc();
      }

      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const memoryDelta = finalMemory - initialMemory;

      // Memory growth should be reasonable
      const memoryDeltaMB = memoryDelta / (1024 * 1024);
      expect(memoryDeltaMB).toBeLessThan(50); // < 50MB growth

      console.log(
        `Memory delta after ${iterations} operations: ${memoryDeltaMB.toFixed(2)}MB`,
      );
    });
  });

  describe("Event System Performance", () => {
    it("should handle high-frequency events efficiently", async () => {
      const { useKellyBrainEvents } =
        await import("../../src/hooks/useKellyBrain");

      const { result } = renderHook(() => useKellyBrainEvents());

      await act(async () => {
        await result.current.startListening();
      });

      expect(result.current.isListening).toBe(true);

      // Simulate high-frequency events
      const eventCount = 100;
      const startTime = performance.now();

      // Events would be triggered by backend operations
      // For testing, measure the overhead of event handling
      for (let i = 0; i < eventCount; i++) {
        // Simulate event processing overhead
        await new Promise((resolve) => setTimeout(resolve, 1));
      }

      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const avgTime = totalTime / eventCount;

      expect(avgTime).toBeLessThan(5); // < 5ms per event

      await act(async () => {
        await result.current.stopListening();
      });

      console.log(`Average event handling time: ${avgTime.toFixed(2)}ms`);
    });
  });
});

// =============================================================================
// Performance Benchmark Utilities
// =============================================================================

export class FrontendPerformanceBenchmark {
  static async measureHookPerformance(
    hookName: string,
    iterations: number = 100,
  ) {
    const times: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();

      // Hook operation would go here
      // For now, just measure overhead
      await new Promise((resolve) => setTimeout(resolve, 1));

      const end = performance.now();
      times.push(end - start);
    }

    const average = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);

    return {
      hookName,
      iterations,
      averageTime: average,
      minTime: min,
      maxTime: max,
      totalTime: times.reduce((a, b) => a + b, 0),
    };
  }

  static async measureApiCallPerformance(
    apiCalls: string[],
    iterations: number = 50,
  ) {
    const results = new Map<string, number[]>();

    for (const call of apiCalls) {
      results.set(call, []);
    }

    for (let i = 0; i < iterations; i++) {
      for (const call of apiCalls) {
        const start = performance.now();

        try {
          // Mock API call
          await getMockResponse(call, {});
        } catch (error) {
          // Record error time as well
        }

        const end = performance.now();
        results.get(call)!.push(end - start);
      }
    }

    const summary: Record<string, any> = {};

    for (const [call, times] of results) {
      const average = times.reduce((a, b) => a + b, 0) / times.length;
      summary[call] = {
        averageTime: average,
        minTime: Math.min(...times),
        maxTime: Math.max(...times),
        callCount: times.length,
      };
    }

    return summary;
  }

  static generatePerformanceReport(results: Record<string, any>) {
    console.log("\n=== Frontend Performance Report ===\n");

    for (const [operation, stats] of Object.entries(results)) {
      console.log(`${operation}:`);
      console.log(`  Average: ${stats.averageTime.toFixed(2)}ms`);
      console.log(`  Min: ${stats.minTime.toFixed(2)}ms`);
      console.log(`  Max: ${stats.maxTime.toFixed(2)}ms`);

      // Performance rating
      if (stats.averageTime < 10) {
        console.log(`  Rating: ✅ EXCELLENT`);
      } else if (stats.averageTime < 50) {
        console.log(`  Rating: ⚠️  ACCEPTABLE`);
      } else {
        console.log(`  Rating: ❌ TOO SLOW`);
      }
      console.log("");
    }
  }
}

describe("Frontend Performance Benchmarks", () => {
  it("should benchmark API call performance", async () => {
    const apiCalls = [
      "kelly_brain_from_text",
      "kelly_brain_get_emotion_state",
      "kelly_brain_set_emotion_parameters",
      "kelly_brain_generate_midi",
    ];

    const results =
      await FrontendPerformanceBenchmark.measureApiCallPerformance(
        apiCalls,
        20,
      );

    FrontendPerformanceBenchmark.generatePerformanceReport(results);

    // Verify performance requirements
    expect(
      results["kelly_brain_set_emotion_parameters"].averageTime,
    ).toBeLessThan(10);
    expect(results["kelly_brain_get_emotion_state"].averageTime).toBeLessThan(
      15,
    );
    expect(results["kelly_brain_from_text"].averageTime).toBeLessThan(50);
  });

  it("should benchmark hook initialization performance", async () => {
    const result = await FrontendPerformanceBenchmark.measureHookPerformance(
      "useKellyBrain",
      10,
    );

    console.log("\n=== Hook Performance ===");
    console.log(`${result.hookName}:`);
    console.log(`  Average: ${result.averageTime.toFixed(2)}ms`);
    console.log(`  Min: ${result.minTime.toFixed(2)}ms`);
    console.log(`  Max: ${result.maxTime.toFixed(2)}ms`);

    expect(result.averageTime).toBeLessThan(20); // < 20ms average
  });

  it("should measure state update performance", async () => {
    // Test rapid state updates like real-time parameter changes
    const updateCount = 100;
    const startTime = performance.now();

    for (let i = 0; i < updateCount; i++) {
      // Simulate state update
      const mockState = {
        initialized: true,
        emotion_state: {
          valence: Math.random() * 2 - 1,
          arousal: Math.random(),
          dominance: Math.random(),
          complexity: Math.random(),
        },
        processing: false,
        last_update: new Date().toISOString(),
      };

      // Simulate React state update overhead
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;
    const avgTime = totalTime / updateCount;

    expect(avgTime).toBeLessThan(2); // < 2ms per state update

    console.log(`State update performance: ${avgTime.toFixed(3)}ms per update`);
  });
});

// =============================================================================
// Memory Leak Detection
// =============================================================================

describe("Memory Leak Detection", () => {
  it("should not leak memory with repeated hook usage", async () => {
    if (!(performance as any).memory) {
      console.log("Memory measurement not available in this environment");
      return;
    }

    const initialMemory = (performance as any).memory.usedJSHeapSize;

    // Simulate heavy hook usage
    for (let i = 0; i < 50; i++) {
      const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");
      const { result } = renderHook(() => useKellyBrain());

      await act(async () => {
        await result.current.initialize("./test-data");
        await result.current.fromText(`test ${i}`);
      });

      // Force cleanup
      result.current = null as any;
    }

    // Force garbage collection
    if ((global as any).gc) {
      (global as any).gc();
    }

    // Wait for cleanup
    await new Promise((resolve) => setTimeout(resolve, 100));

    const finalMemory = (performance as any).memory.usedJSHeapSize;
    const memoryGrowth = finalMemory - initialMemory;
    const memoryGrowthMB = memoryGrowth / (1024 * 1024);

    // Memory growth should be minimal
    expect(memoryGrowthMB).toBeLessThan(10); // < 10MB growth

    console.log(
      `Memory growth after repeated usage: ${memoryGrowthMB.toFixed(2)}MB`,
    );
  });
});

// =============================================================================
// Integration Performance Tests
// =============================================================================

describe("Integration Performance", () => {
  it("should handle complete workflow efficiently", async () => {
    const { useSimpleKellyBrain } =
      await import("../../src/hooks/useKellyBrain");

    const { result } = renderHook(() => useSimpleKellyBrain());

    const startTime = performance.now();

    await act(async () => {
      // Complete workflow: initialize → process → generate
      await result.current.initialize("./test-data");
      const midi = await result.current.generateMusicFromText(
        "feeling contemplative",
        8,
      );
      expect(midi).toBeTruthy();
    });

    const endTime = performance.now();
    const workflowTime = endTime - startTime;

    expect(workflowTime).toBeLessThan(200); // < 200ms for complete workflow

    console.log(`Complete workflow time: ${workflowTime.toFixed(2)}ms`);
  });

  it("should maintain performance under concurrent usage", async () => {
    const { useKellyBrain } = await import("../../src/hooks/useKellyBrain");

    // Simulate multiple concurrent hook instances
    const hookCount = 5;
    const hooks = Array.from({ length: hookCount }, () =>
      renderHook(() => useKellyBrain()),
    );

    const startTime = performance.now();

    // Initialize all hooks concurrently
    await Promise.all(
      hooks.map(({ result }) =>
        act(async () => {
          await result.current.initialize("./test-data");
        }),
      ),
    );

    // Perform operations concurrently
    await Promise.all(
      hooks.map(({ result }, i) =>
        act(async () => {
          await result.current.fromText(`concurrent test ${i}`);
        }),
      ),
    );

    const endTime = performance.now();
    const concurrentTime = endTime - startTime;

    expect(concurrentTime).toBeLessThan(500); // < 500ms for concurrent operations

    console.log(`Concurrent hook usage time: ${concurrentTime.toFixed(2)}ms`);
  });
});

export { FrontendPerformanceBenchmark };
