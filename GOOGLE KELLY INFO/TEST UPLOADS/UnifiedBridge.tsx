/**
 * UnifiedBridge - Connects emotion state to DAW parameters
 *
 * This is the key integration point that makes Side A and Side B
 * work as a single unified entity. Emotion drives technical choices.
 */

import React, { useEffect, useRef } from 'react';
import { useUnifiedStore } from '../../store/useUnifiedStore';
import { useMusicBrain } from '../../hooks/useMusicBrain';

interface UnifiedBridgeProps {
  emotion: string | null;
  intent: Record<string, unknown> | null;
}

export const UnifiedBridge: React.FC<UnifiedBridgeProps> = ({ emotion, intent }) => {
  const { processIntent } = useMusicBrain();
  const {
    setGeneratedMusic,
    setTempo,
    applyEmotionToMixer,
    tracks,
    updateTrack,
  } = useUnifiedStore();

  const lastProcessedIntentRef = useRef<string | null>(null);
  const isProcessingRef = useRef(false);

  // Process intent when it changes
  useEffect(() => {
    const intentKey = intent ? JSON.stringify(intent) : null;

    if (
      intent &&
      intentKey !== lastProcessedIntentRef.current &&
      !isProcessingRef.current
    ) {
      isProcessingRef.current = true;

      processIntent(intent)
        .then((result) => {
          setGeneratedMusic(result);
          lastProcessedIntentRef.current = intentKey;

          // Log the bridge action
          console.log('[UnifiedBridge] Generated music from intent:', {
            emotion,
            harmony: result.harmony,
            tempo: result.tempo,
            key: result.key,
          });
        })
        .catch((error) => {
          console.error('[UnifiedBridge] Failed to process intent:', error);
        })
        .finally(() => {
          isProcessingRef.current = false;
        });
    }
  }, [intent, emotion, processIntent, setGeneratedMusic]);

  // Apply emotion to mixer when emotion changes
  useEffect(() => {
    if (emotion) {
      applyEmotionToMixer();

      console.log('[UnifiedBridge] Applied emotion to mixer:', emotion);
    }
  }, [emotion, applyEmotionToMixer]);

  // This component is invisible - it's purely for state bridging
  return null;
};

/**
 * Utility functions for bridging emotion to DAW parameters
 */

export const CHORD_TO_MIDI: Record<string, number[]> = {
  // Major chords
  'C': [60, 64, 67],
  'D': [62, 66, 69],
  'E': [64, 68, 71],
  'F': [65, 69, 72],
  'G': [67, 71, 74],
  'A': [69, 73, 76],
  'B': [71, 75, 78],

  // Minor chords
  'Am': [57, 60, 64],
  'Bm': [59, 62, 66],
  'Cm': [60, 63, 67],
  'Dm': [62, 65, 69],
  'Em': [64, 67, 71],
  'Fm': [65, 68, 72],
  'Gm': [67, 70, 74],

  // Extended chords
  'Cmaj7': [60, 64, 67, 71],
  'Am7': [57, 60, 64, 67],
  'Dm7': [62, 65, 69, 72],
  'G7': [67, 71, 74, 77],
  'Fmaj7': [65, 69, 72, 76],

  // Slash chords (common in emotional music)
  'Am/E': [52, 57, 60, 64],
  'F/A': [57, 65, 69, 72],
  'C/G': [55, 60, 64, 67],
  'G/B': [59, 67, 71, 74],
};

export function chordToMidi(chordName: string): number[] {
  return CHORD_TO_MIDI[chordName] || [60, 64, 67]; // Default to C major
}

export function harmonyToMidiNotes(
  harmony: string[],
  ppq: number = 480,
  beatsPerChord: number = 4
): Array<{ pitch: number; velocity: number; start_tick: number; duration_ticks: number }> {
  const notes: Array<{ pitch: number; velocity: number; start_tick: number; duration_ticks: number }> = [];

  harmony.forEach((chord, index) => {
    const midiNotes = chordToMidi(chord);
    const startTick = index * beatsPerChord * ppq;
    const durationTicks = beatsPerChord * ppq;

    midiNotes.forEach((pitch) => {
      notes.push({
        pitch,
        velocity: 80,
        start_tick: startTick,
        duration_ticks: durationTicks,
      });
    });
  });

  return notes;
}
