import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

// =============================================================================
// Types (matching Rust types)
// =============================================================================

export interface IntentResult {
  core_wound: string;
  core_desire: string;
  emotional_intent: string;
  valence: number;
  arousal: number;
  dominance: number;
  complexity: number;
  progression: string[];
  key: string;
  bpm: number;
  genre: string;
}

export interface GeneratedMidi {
  bars: number;
  bpm: number;
  key: string;
  time_signature: string;
  tracks: MidiTrack[];
}

export interface MidiTrack {
  name: string;
  channel: number;
  events: MidiEvent[];
}

export interface MidiEvent {
  event_type: number;
  timestamp: number;
  note: number;
  velocity: number;
  duration: number;
}

export interface EmotionState {
  valence: number;
  arousal: number;
  dominance: number;
  complexity: number;
}

export interface KellyBrainState {
  initialized: boolean;
  emotion_state: EmotionState | null;
  current_intent: IntentResult | null;
  current_midi: GeneratedMidi | null;
  processing: boolean;
  last_update: string | null;
}

// =============================================================================
// Main KellyBrain Hook
// =============================================================================

export const useKellyBrain = () => {
  const [state, setState] = useState<KellyBrainState>({
    initialized: false,
    emotion_state: null,
    current_intent: null,
    current_midi: null,
    processing: false,
    last_update: null,
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Initialize KellyBrain
  const initialize = useCallback(async (dataPath: string) => {
    try {
      setLoading(true);
      setError(null);
      const success = await invoke<boolean>("kelly_brain_initialize", {
        dataPath,
      });

      if (success) {
        // Get initial state
        const currentState = await invoke<KellyBrainState>(
          "get_kelly_brain_state",
        );
        setState(currentState);
      }

      return success;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  // Check if initialized
  const checkInitialized = useCallback(async () => {
    try {
      const initialized = await invoke<boolean>("kelly_brain_is_initialized");
      setState((prev) => ({ ...prev, initialized }));
      return initialized;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
      return false;
    }
  }, []);

  // Generate intent from text
  const fromText = useCallback(
    async (text: string): Promise<IntentResult | null> => {
      try {
        setLoading(true);
        setError(null);
        setState((prev) => ({ ...prev, processing: true }));

        const intent = await invoke<IntentResult>("kelly_brain_from_text", {
          text,
        });
        setState((prev) => ({
          ...prev,
          current_intent: intent,
          processing: false,
        }));

        return intent;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setError(errorMsg);
        setState((prev) => ({ ...prev, processing: false }));
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // Generate intent from emotion
  const fromEmotion = useCallback(
    async (
      emotionName: string,
      intensity: number,
    ): Promise<IntentResult | null> => {
      try {
        setLoading(true);
        setError(null);
        setState((prev) => ({ ...prev, processing: true }));

        const intent = await invoke<IntentResult>("kelly_brain_from_emotion", {
          emotionName,
          intensity,
        });
        setState((prev) => ({
          ...prev,
          current_intent: intent,
          processing: false,
        }));

        return intent;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setError(errorMsg);
        setState((prev) => ({ ...prev, processing: false }));
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // Generate MIDI from intent
  const generateMidi = useCallback(
    async (
      intent: IntentResult,
      bars: number = 8,
    ): Promise<GeneratedMidi | null> => {
      try {
        setLoading(true);
        setError(null);
        setState((prev) => ({ ...prev, processing: true }));

        const midi = await invoke<GeneratedMidi>("kelly_brain_generate_midi", {
          intent,
          bars,
        });
        setState((prev) => ({
          ...prev,
          current_midi: midi,
          processing: false,
        }));

        return midi;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setError(errorMsg);
        setState((prev) => ({ ...prev, processing: false }));
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // Generate MIDI with parameters
  const generateMidiWithParams = useCallback(
    async (
      intent: IntentResult,
      bars: number,
      bpm: number,
      keySignature: string,
    ): Promise<GeneratedMidi | null> => {
      try {
        setLoading(true);
        setError(null);
        setState((prev) => ({ ...prev, processing: true }));

        const midi = await invoke<GeneratedMidi>(
          "kelly_brain_generate_midi_with_params",
          {
            intent,
            bars,
            bpm,
            keySignature,
          },
        );
        setState((prev) => ({
          ...prev,
          current_midi: midi,
          processing: false,
        }));

        return midi;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setError(errorMsg);
        setState((prev) => ({ ...prev, processing: false }));
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // Set emotion parameters
  const setEmotionParameters = useCallback(
    async (valence: number, arousal: number, dominance: number) => {
      try {
        setError(null);
        const success = await invoke<boolean>(
          "kelly_brain_set_emotion_parameters",
          {
            valence,
            arousal,
            dominance,
          },
        );

        if (success) {
          // Update local state immediately for responsiveness
          const newEmotionState: EmotionState = {
            valence,
            arousal,
            dominance,
            complexity: state.emotion_state?.complexity || 0.5,
          };
          setState((prev) => ({ ...prev, emotion_state: newEmotionState }));
        }

        return success;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setError(errorMsg);
        return false;
      }
    },
    [state.emotion_state],
  );

  // Get current emotion state
  const getEmotionState =
    useCallback(async (): Promise<EmotionState | null> => {
      try {
        const emotionState = await invoke<EmotionState>(
          "kelly_brain_get_emotion_state",
        );
        setState((prev) => ({ ...prev, emotion_state: emotionState }));
        return emotionState;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setError(errorMsg);
        return null;
      }
    }, []);

  // Get available emotions
  const getAvailableEmotions = useCallback(async () => {
    try {
      const emotions = await invoke<any>("kelly_brain_get_available_emotions");
      return emotions;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
      return null;
    }
  }, []);

  // Get version
  const getVersion = useCallback(async (): Promise<string | null> => {
    try {
      const version = await invoke<string>("kelly_brain_get_version");
      return version;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
      return null;
    }
  }, []);

  // Refresh state from backend
  const refreshState = useCallback(async () => {
    try {
      const currentState = await invoke<KellyBrainState>(
        "get_kelly_brain_state",
      );
      setState(currentState);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
    }
  }, []);

  return {
    // State
    state,
    error,
    loading,

    // Actions
    initialize,
    checkInitialized,
    fromText,
    fromEmotion,
    generateMidi,
    generateMidiWithParams,
    setEmotionParameters,
    getEmotionState,
    getAvailableEmotions,
    getVersion,
    refreshState,

    // Computed values
    isInitialized: state.initialized,
    isProcessing: state.processing || loading,
    currentIntent: state.current_intent,
    currentMidi: state.current_midi,
    emotionState: state.emotion_state,
  };
};

// =============================================================================
// Real-time Event Hook
// =============================================================================

export const useKellyBrainEvents = () => {
  const [events, setEvents] = useState<any[]>([]);
  const [isListening, setIsListening] = useState(false);
  const unlistenersRef = useRef<UnlistenFn[]>([]);

  const startListening = useCallback(async () => {
    if (isListening) return;

    try {
      // Add event listener on backend
      await invoke("add_event_listener", {
        listenerId: "react-hook-" + Date.now(),
      });

      // Listen to various event types
      const eventTypes = [
        "kelly-brain-initialized",
        "kelly-emotion-update",
        "kelly-intent-generated",
        "kelly-midi-generated",
        "kelly-processing-started",
        "kelly-processing-completed",
        "kelly-error",
        "kelly-connection-status",
        "kelly-parameter-changed",
      ];

      const unlisteners: UnlistenFn[] = [];

      for (const eventType of eventTypes) {
        const unlisten = await listen(eventType, (event) => {
          setEvents((prev) => [
            ...prev.slice(-99),
            {
              // Keep last 100 events
              type: eventType,
              data: event.payload,
              timestamp: new Date().toISOString(),
            },
          ]);
        });
        unlisteners.push(unlisten);
      }

      unlistenersRef.current = unlisteners;
      setIsListening(true);
    } catch (err) {
      console.error("Failed to start listening to KellyBrain events:", err);
    }
  }, [isListening]);

  const stopListening = useCallback(async () => {
    if (!isListening) return;

    try {
      // Remove event listeners
      for (const unlisten of unlistenersRef.current) {
        await unlisten();
      }
      unlistenersRef.current = [];

      // Remove event listener on backend
      await invoke("remove_event_listener", { listenerId: "react-hook" });

      setIsListening(false);
    } catch (err) {
      console.error("Failed to stop listening to KellyBrain events:", err);
    }
  }, [isListening]);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  // Auto-cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, [stopListening]);

  return {
    events,
    isListening,
    startListening,
    stopListening,
    clearEvents,
  };
};

// =============================================================================
// Simplified hook for basic usage
// =============================================================================

export const useSimpleKellyBrain = () => {
  const {
    state,
    error,
    loading,
    initialize,
    fromText,
    generateMidi,
    setEmotionParameters,
    isInitialized,
    isProcessing,
  } = useKellyBrain();

  const generateMusicFromText = useCallback(
    async (text: string, bars: number = 8) => {
      const intent = await fromText(text);
      if (intent) {
        return await generateMidi(intent, bars);
      }
      return null;
    },
    [fromText, generateMidi],
  );

  return {
    // Simplified state
    initialized: isInitialized,
    processing: isProcessing,
    error,
    loading,
    emotionState: state.emotion_state,
    currentMidi: state.current_midi,

    // Simplified actions
    initialize,
    generateMusicFromText,
    setEmotion: setEmotionParameters,
  };
};

// =============================================================================
// Auto-initialization hook
// =============================================================================

export const useKellyBrainAutoInit = (dataPath: string = "./data") => {
  const { initialize, isInitialized, error } = useKellyBrain();
  const [initAttempted, setInitAttempted] = useState(false);

  useEffect(() => {
    if (!initAttempted && !isInitialized) {
      setInitAttempted(true);
      initialize(dataPath).catch(console.error);
    }
  }, [initialize, isInitialized, initAttempted, dataPath]);

  return {
    isInitialized,
    initError: error,
    initAttempted,
  };
};
