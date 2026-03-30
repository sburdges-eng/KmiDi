import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getLogger } from '../diagnostics/IntentLogger';

/**
 * useIntentIR - React hook for subscribing to IntentFrame snapshots
 * 
 * Rules:
 * - Never derive emotion in hooks
 * - No memoization of IR data
 * - Freeze UI state on audio ticks
 */
export interface IntentFrame {
  meta: {
    ir_version: number;
    intent_id: number;
    session_id: number;
  };
  emotion: {
    valence: number;
    arousal: number;
    dominance: number;
    discrete_id: number;
    intensity: number;
    confidence: number;
  };
  music: {
    tempo_bias: number;
    rhythmic_density: number;
    groove_strength: number;
    harmonic_tension: number;
    harmonic_motion: number;
    mode_preference: number;
    melodic_activity: number;
    contour_variance: number;
    dynamic_range: number;
    texture_density: number;
  };
  time: {
    start_bar: number;
    end_bar: number;
    fade_in_beats: number;
    fade_out_beats: number;
  };
  constraints: {
    allowed_engines_mask: number;
    forbidden_engines_mask: number;
    max_cpu_cost: number;
    max_event_rate: number;
  };
  provenance: {
    source: number;
    user_override_weight: number;
  };
}

export function useIntentIR() {
  const [frame, setFrame] = useState<IntentFrame | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const frameRef = useRef<IntentFrame | null>(null);
  const logger = getLogger();

  // Freeze UI state on audio ticks (snapshot-based updates)
  const updateFrame = useCallback((newFrame: IntentFrame) => {
    // Log at UI boundary
    logger.logUIBoundary(newFrame);
    
    // Update ref immediately (for synchronous access)
    frameRef.current = newFrame;
    
    // Update state (triggers re-render)
    setFrame(newFrame);
  }, [logger]);

  // Subscribe to IntentFrame updates
  useEffect(() => {
    let isMounted = true;
    let intervalId: number | null = null;

    const fetchFrame = async () => {
      try {
        const snapshot = await invoke<IntentFrame>('get_intent_ir_snapshot');
        if (isMounted) {
          updateFrame(snapshot);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Unknown error');
          logger.logError('ui_boundary', null, String(err));
        }
      }
    };

    // Initial fetch
    fetchFrame();

    // Poll for updates (could be replaced with event subscription)
    intervalId = window.setInterval(fetchFrame, 100); // 10 Hz update rate

    return () => {
      isMounted = false;
      if (intervalId !== null) {
        clearInterval(intervalId);
      }
    };
  }, [updateFrame, logger]);

  // Get current frame synchronously (from ref)
  const getCurrentFrame = useCallback((): IntentFrame | null => {
    return frameRef.current;
  }, []);

  // Update emotion (emits IR update)
  const updateEmotion = useCallback(async (
    valence: number,
    arousal: number,
    dominance: number,
    discreteId?: number,
    intensity: number = 1.0,
    confidence: number = 1.0,
  ) => {
    setIsLoading(true);
    try {
      await invoke('update_intent_ir_emotion', {
        valence,
        arousal,
        dominance,
        discreteId: discreteId ?? -1,
        intensity,
        confidence,
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      logger.logError('ui_update_emotion', frameRef.current, String(err));
    } finally {
      setIsLoading(false);
    }
  }, [logger]);

  // Update music (emits IR update)
  const updateMusic = useCallback(async (
    tempoBias: number,
    rhythmicDensity: number,
    grooveStrength: number,
    harmonicTension: number,
    harmonicMotion: number,
    modePreference: number,
    melodicActivity: number,
    contourVariance: number,
    dynamicRange: number,
    textureDensity: number,
  ) => {
    setIsLoading(true);
    try {
      await invoke('update_intent_ir_music', {
        tempoBias,
        rhythmicDensity,
        grooveStrength,
        harmonicTension,
        harmonicMotion,
        modePreference,
        melodicActivity,
        contourVariance,
        dynamicRange,
        textureDensity,
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      logger.logError('ui_update_music', frameRef.current, String(err));
    } finally {
      setIsLoading(false);
    }
  }, [logger]);

  return {
    frame,
    isLoading,
    error,
    getCurrentFrame,
    updateEmotion,
    updateMusic,
  };
}
