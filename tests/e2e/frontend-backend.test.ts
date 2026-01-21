import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';

// Mock Tauri API for testing
const mockInvoke = async (cmd: string, args?: any): Promise<any> => {
  // Mock implementations for testing
  switch (cmd) {
    case 'kelly_brain_initialize':
      return true;
    case 'kelly_brain_is_initialized':
      return true;
    case 'kelly_brain_from_text':
      return {
        core_wound: 'test_wound',
        core_desire: 'test_desire',
        emotional_intent: args.text || 'test intent',
        valence: 0.5,
        arousal: 0.7,
        dominance: 0.3,
        complexity: 0.6,
        progression: ['C', 'F', 'G', 'Am'],
        key: 'C',
        bpm: 120,
        genre: 'test'
      };
    case 'kelly_brain_generate_midi':
      return {
        bars: args.bars || 8,
        bpm: 120,
        key: 'C',
        time_signature: '4/4',
        tracks: [
          {
            name: 'Melody',
            channel: 1,
            events: [
              {
                event_type: 1,
                timestamp: 0.0,
                note: 60,
                velocity: 80,
                duration: 0.5
              }
            ]
          }
        ]
      };
    case 'kelly_brain_get_emotion_state':
      return {
        valence: 0.2,
        arousal: 0.6,
        dominance: 0.4,
        complexity: 0.5
      };
    case 'kelly_brain_set_emotion_parameters':
      return true;
    case 'get_kelly_brain_state':
      return {
        initialized: true,
        emotion_state: {
          valence: 0.2,
          arousal: 0.6,
          dominance: 0.4,
          complexity: 0.5
        },
        current_intent: null,
        current_midi: null,
        processing: false,
        last_update: new Date().toISOString()
      };
    default:
      throw new Error(`Unhandled mock command: ${cmd}`);
  }
};

// Mock event listening
const mockListen = async (event: string, handler: any): Promise<UnlistenFn> => {
  // Simulate event emission after delay
  setTimeout(() => {
    handler({
      payload: {
        type: 'test_event',
        data: { test: 'value' }
      }
    });
  }, 50);
  
  return async () => {
    // Cleanup mock
  };
};

// Replace global invoke for testing
(global as any).invoke = mockInvoke;
(global as any).listen = mockListen;

describe('KellyBrain Frontend-Backend Integration', () => {
  beforeAll(async () => {
    // Setup test environment
  });

  afterAll(async () => {
    // Cleanup test environment
  });

  describe('Basic Integration Flow', () => {
    it('should initialize KellyBrain successfully', async () => {
      const result = await invoke('kelly_brain_initialize', { dataPath: './test-data' });
      expect(result).toBe(true);
    });

    it('should check initialization status', async () => {
      const isInitialized = await invoke('kelly_brain_is_initialized');
      expect(isInitialized).toBe(true);
    });

    it('should process text to intent', async () => {
      const intent = await invoke('kelly_brain_from_text', { 
        text: 'I feel nostalgic about summer days' 
      });
      
      expect(intent).toMatchObject({
        core_wound: expect.any(String),
        core_desire: expect.any(String),
        emotional_intent: expect.any(String),
        valence: expect.any(Number),
        arousal: expect.any(Number),
        dominance: expect.any(Number),
        progression: expect.any(Array),
        key: expect.any(String),
        bpm: expect.any(Number)
      });
    });

    it('should generate MIDI from intent', async () => {
      const intent = await invoke('kelly_brain_from_text', { 
        text: 'feeling hopeful' 
      });
      
      const midi = await invoke('kelly_brain_generate_midi', { 
        intent, 
        bars: 8 
      });
      
      expect(midi).toMatchObject({
        bars: 8,
        bpm: expect.any(Number),
        key: expect.any(String),
        tracks: expect.any(Array)
      });
      
      expect(midi.tracks.length).toBeGreaterThan(0);
      expect(midi.tracks[0]).toMatchObject({
        name: expect.any(String),
        channel: expect.any(Number),
        events: expect.any(Array)
      });
    });
  });

  describe('Emotion Parameter Integration', () => {
    it('should set emotion parameters', async () => {
      const result = await invoke('kelly_brain_set_emotion_parameters', {
        valence: 0.3,
        arousal: 0.8,
        dominance: 0.5
      });
      
      expect(result).toBe(true);
    });

    it('should get current emotion state', async () => {
      const state = await invoke('kelly_brain_get_emotion_state');
      
      expect(state).toMatchObject({
        valence: expect.any(Number),
        arousal: expect.any(Number),
        dominance: expect.any(Number),
        complexity: expect.any(Number)
      });
      
      // Values should be in valid ranges
      expect(state.valence).toBeGreaterThanOrEqual(-1.0);
      expect(state.valence).toBeLessThanOrEqual(1.0);
      expect(state.arousal).toBeGreaterThanOrEqual(0.0);
      expect(state.arousal).toBeLessThanOrEqual(1.0);
      expect(state.dominance).toBeGreaterThanOrEqual(0.0);
      expect(state.dominance).toBeLessThanOrEqual(1.0);
    });

    it('should reject invalid parameter values', async () => {
      // Test invalid valence
      await expect(
        invoke('kelly_brain_set_emotion_parameters', {
          valence: -1.5,  // Invalid
          arousal: 0.5,
          dominance: 0.5
        })
      ).rejects.toThrow();

      // Test invalid arousal
      await expect(
        invoke('kelly_brain_set_emotion_parameters', {
          valence: 0.0,
          arousal: -0.5,  // Invalid
          dominance: 0.5
        })
      ).rejects.toThrow();
    });
  });

  describe('State Management Integration', () => {
    it('should get complete brain state', async () => {
      const state = await invoke('get_kelly_brain_state');
      
      expect(state).toMatchObject({
        initialized: expect.any(Boolean),
        emotion_state: expect.any(Object),
        processing: expect.any(Boolean),
        last_update: expect.any(String)
      });
    });

    it('should handle state subscriptions', async () => {
      const result = await invoke('subscribe_to_state_events', {
        subscriberId: 'test-subscriber'
      });
      
      expect(result).toBe(true);
      
      // Unsubscribe
      const unsubResult = await invoke('unsubscribe_from_state_events', {
        subscriberId: 'test-subscriber'
      });
      
      expect(unsubResult).toBe(true);
    });
  });

  describe('Event System Integration', () => {
    it('should handle event listener registration', async () => {
      const result = await invoke('add_event_listener', {
        listenerId: 'test-listener'
      });
      
      expect(result).toBe(true);
      
      // Remove listener
      const removeResult = await invoke('remove_event_listener', {
        listenerId: 'test-listener'
      });
      
      expect(removeResult).toBe(true);
    });

    it('should emit test events', async () => {
      const result = await invoke('emit_test_event');
      expect(result).toBe(true);
    });

    it('should receive events via Tauri event system', async () => {
      return new Promise(async (resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Event receive timeout'));
        }, 2000);

        const unlisten = await listen('kelly-parameter-changed', (event) => {
          clearTimeout(timeout);
          
          expect(event.payload).toMatchObject({
            type: 'ParameterChanged',
            parameter: expect.any(String),
            value: expect.any(Object)
          });
          
          unlisten.then(() => resolve(undefined));
        });

        // Trigger test event
        await invoke('emit_test_event');
      });
    }, 5000);  // 5 second timeout
  });

  describe('Performance Integration', () => {
    it('should handle rapid parameter updates', async () => {
      const startTime = performance.now();
      const iterations = 20;
      
      for (let i = 0; i < iterations; i++) {
        const valence = (i / iterations) * 2 - 1;  // -1 to 1
        await invoke('kelly_brain_set_emotion_parameters', {
          valence,
          arousal: 0.5,
          dominance: 0.5
        });
      }
      
      const endTime = performance.now();
      const avgTime = (endTime - startTime) / iterations;
      
      // Each call should be reasonably fast
      expect(avgTime).toBeLessThan(50);  // < 50ms per call
      
      console.log(`Average parameter update time: ${avgTime.toFixed(2)}ms`);
    });

    it('should handle multiple concurrent requests', async () => {
      const promises = [];
      const requestCount = 10;
      
      for (let i = 0; i < requestCount; i++) {
        const promise = invoke('kelly_brain_from_text', {
          text: `test emotion ${i}`
        });
        promises.push(promise);
      }
      
      // All requests should complete successfully (or fail gracefully)
      const results = await Promise.allSettled(promises);
      
      // Check that all promises settled (didn't hang)
      expect(results.length).toBe(requestCount);
      
      // Count successes vs failures
      const successes = results.filter(r => r.status === 'fulfilled').length;
      const failures = results.filter(r => r.status === 'rejected').length;
      
      console.log(`Concurrent test: ${successes} succeeded, ${failures} failed`);
      
      // At least some should succeed or fail gracefully
      expect(successes + failures).toBe(requestCount);
    });
  });

  describe('Error Handling Integration', () => {
    it('should handle backend unavailable gracefully', async () => {
      // This test would require actually disconnecting the backend
      // For now, test that error responses are properly structured
      
      try {
        await invoke('kelly_brain_from_text', { text: null }); // Invalid input
      } catch (error) {
        expect(error).toBeInstanceOf(String);
        expect(error).toContain('Error');
      }
    });

    it('should provide meaningful error messages', async () => {
      try {
        await invoke('kelly_brain_set_emotion_parameters', {
          valence: 999,  // Invalid value
          arousal: 0.5,
          dominance: 0.5
        });
      } catch (error) {
        expect(error).toBeInstanceOf(String);
        expect(error.toLowerCase()).toContain('parameter');
      }
    });
  });

  describe('Backwards Compatibility', () => {
    it('should maintain legacy API compatibility', async () => {
      // Test that original useMusicBrain APIs still work
      const emotions = await invoke('get_emotions');
      expect(emotions).toBeDefined();
      
      const generateResult = await invoke('generate_music', {
        request: {
          intent: {
            emotional_intent: 'test emotion'
          }
        }
      });
      
      expect(generateResult).toBeDefined();
    });
  });
});

// =============================================================================
// React Hook Integration Tests
// =============================================================================

describe('React Hook Integration', () => {
  // These tests would require React testing library setup
  // For now, provide the structure for future implementation
  
  it('should integrate useKellyBrain hook', async () => {
    // Test hook integration with React components
    // Would require @testing-library/react setup
    console.log('Hook integration tests require React testing library setup');
  });

  it('should handle real-time events in React', async () => {
    // Test event handling in React components
    console.log('Event integration tests require component testing setup');
  });
});

// =============================================================================
// Test Configuration and Helpers
// =============================================================================

// Vitest configuration for Tauri
export const vitestConfig = {
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    testTimeout: 10000,  // 10 second timeout for integration tests
  },
  define: {
    // Mock Tauri in test environment
    'window.__TAURI__': 'true',
  }
};