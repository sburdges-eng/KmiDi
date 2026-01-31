/**
 * useAudioEngineBridge - Connects UI state to audio engine
 *
 * This hook bridges the gap between:
 * - Side A (DAW) components
 * - Side B (Emotion) state
 * - The underlying audio engine (Tauri/Rust/C++)
 *
 * In production, this would use Tauri's invoke() to communicate
 * with the Rust audio backend, which interfaces with the C++ penta-core.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useUnifiedStore, Track, GeneratedMusic } from '../store/useUnifiedStore';

// ============================================================================
// TYPES
// ============================================================================

interface AudioEngineState {
  isInitialized: boolean;
  isProcessing: boolean;
  sampleRate: number;
  bufferSize: number;
  latency: number;
  cpuUsage: number;
}

interface MIDINote {
  pitch: number;
  velocity: number;
  channel: number;
  startTime: number; // In seconds
  duration: number; // In seconds
}

interface AudioEngineMessage {
  type: 'play' | 'stop' | 'pause' | 'seek' | 'set_tempo' | 'load_midi' | 'apply_emotion';
  payload?: unknown;
}

// ============================================================================
// EMOTION TO AUDIO MAPPING
// ============================================================================

const EMOTION_AUDIO_PARAMS: Record<string, {
  reverbMix: number;
  reverbDecay: number;
  delayMix: number;
  delayTime: number;
  compressorThreshold: number;
  compressorRatio: number;
  filterCutoff: number;
  saturation: number;
}> = {
  Grief: {
    reverbMix: 0.7,
    reverbDecay: 3.5,
    delayMix: 0.4,
    delayTime: 0.5,
    compressorThreshold: -18,
    compressorRatio: 2,
    filterCutoff: 2000,
    saturation: 0.1,
  },
  Joy: {
    reverbMix: 0.3,
    reverbDecay: 1.5,
    delayMix: 0.2,
    delayTime: 0.25,
    compressorThreshold: -12,
    compressorRatio: 4,
    filterCutoff: 8000,
    saturation: 0.3,
  },
  Rage: {
    reverbMix: 0.2,
    reverbDecay: 0.8,
    delayMix: 0.1,
    delayTime: 0.1,
    compressorThreshold: -6,
    compressorRatio: 8,
    filterCutoff: 12000,
    saturation: 0.8,
  },
  Anxiety: {
    reverbMix: 0.45,
    reverbDecay: 2.0,
    delayMix: 0.7,
    delayTime: 0.375, // Dotted eighth
    compressorThreshold: -15,
    compressorRatio: 3,
    filterCutoff: 4000,
    saturation: 0.2,
  },
  Longing: {
    reverbMix: 0.65,
    reverbDecay: 4.0,
    delayMix: 0.5,
    delayTime: 0.666, // Triplet feel
    compressorThreshold: -20,
    compressorRatio: 2,
    filterCutoff: 3000,
    saturation: 0.15,
  },
  Passion: {
    reverbMix: 0.4,
    reverbDecay: 2.5,
    delayMix: 0.3,
    delayTime: 0.333,
    compressorThreshold: -12,
    compressorRatio: 3,
    filterCutoff: 6000,
    saturation: 0.4,
  },
  Defiance: {
    reverbMix: 0.3,
    reverbDecay: 1.0,
    delayMix: 0.2,
    delayTime: 0.125,
    compressorThreshold: -8,
    compressorRatio: 6,
    filterCutoff: 10000,
    saturation: 0.6,
  },
  Hope: {
    reverbMix: 0.35,
    reverbDecay: 2.0,
    delayMix: 0.25,
    delayTime: 0.5,
    compressorThreshold: -14,
    compressorRatio: 3,
    filterCutoff: 7000,
    saturation: 0.25,
  },
};

// ============================================================================
// HOOK IMPLEMENTATION
// ============================================================================

export function useAudioEngineBridge() {
  const [engineState, setEngineState] = useState<AudioEngineState>({
    isInitialized: false,
    isProcessing: false,
    sampleRate: 44100,
    bufferSize: 512,
    latency: 11.6, // ms
    cpuUsage: 0,
  });

  const {
    isPlaying,
    tempo,
    currentTime,
    tracks,
    selectedEmotion,
    generatedMusic,
    setCurrentTime,
  } = useUnifiedStore();

  const audioContextRef = useRef<AudioContext | null>(null);
  const schedulerRef = useRef<number | null>(null);

  // Initialize audio engine
  const initializeEngine = useCallback(async () => {
    try {
      // In production: await invoke('audio_engine_init', { sampleRate: 44100, bufferSize: 512 });
      audioContextRef.current = new AudioContext({ sampleRate: 44100 });

      setEngineState(prev => ({
        ...prev,
        isInitialized: true,
        sampleRate: audioContextRef.current?.sampleRate || 44100,
      }));

      console.log('[AudioBridge] Engine initialized');
    } catch (error) {
      console.error('[AudioBridge] Failed to initialize:', error);
    }
  }, []);

  // Send message to audio engine
  const sendMessage = useCallback(async (message: AudioEngineMessage) => {
    // In production: await invoke('audio_engine_message', { message });
    console.log('[AudioBridge] Message:', message);

    switch (message.type) {
      case 'play':
        setEngineState(prev => ({ ...prev, isProcessing: true }));
        break;
      case 'stop':
      case 'pause':
        setEngineState(prev => ({ ...prev, isProcessing: false }));
        break;
      case 'apply_emotion':
        // Apply emotion-based audio parameters
        const emotion = message.payload as string;
        const params = EMOTION_AUDIO_PARAMS[emotion];
        if (params) {
          console.log('[AudioBridge] Applying emotion params:', emotion, params);
          // In production: await invoke('audio_engine_apply_emotion', { params });
        }
        break;
    }
  }, []);

  // Apply emotion to audio engine when it changes
  useEffect(() => {
    if (selectedEmotion && engineState.isInitialized) {
      sendMessage({ type: 'apply_emotion', payload: selectedEmotion });
    }
  }, [selectedEmotion, engineState.isInitialized, sendMessage]);

  // Sync playback state
  useEffect(() => {
    if (engineState.isInitialized) {
      if (isPlaying) {
        sendMessage({ type: 'play' });
      } else {
        sendMessage({ type: 'pause' });
      }
    }
  }, [isPlaying, engineState.isInitialized, sendMessage]);

  // Sync tempo
  useEffect(() => {
    if (engineState.isInitialized) {
      sendMessage({ type: 'set_tempo', payload: tempo });
    }
  }, [tempo, engineState.isInitialized, sendMessage]);

  // Load generated MIDI when available
  useEffect(() => {
    if (generatedMusic && engineState.isInitialized) {
      sendMessage({ type: 'load_midi', payload: generatedMusic });
    }
  }, [generatedMusic, engineState.isInitialized, sendMessage]);

  // Initialize on mount
  useEffect(() => {
    initializeEngine();

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (schedulerRef.current) {
        cancelAnimationFrame(schedulerRef.current);
      }
    };
  }, [initializeEngine]);

  // Convert tracks to MIDI notes for the engine
  const tracksToMIDI = useCallback((): MIDINote[] => {
    const notes: MIDINote[] = [];
    const PPQ = 480;
    const secondsPerBeat = 60 / tempo;

    tracks.forEach((track, trackIndex) => {
      const channel = track.name.toLowerCase().includes('drum') ? 9 : trackIndex;

      track.clips.forEach(clip => {
        // Simple conversion - in production would use full MIDI data
        notes.push({
          pitch: 60 + (trackIndex * 5), // Placeholder pitch
          velocity: Math.round(track.volume * 127),
          channel,
          startTime: clip.startTime * secondsPerBeat,
          duration: clip.duration * secondsPerBeat,
        });
      });
    });

    return notes;
  }, [tracks, tempo]);

  // Get audio params for current emotion
  const getEmotionAudioParams = useCallback(() => {
    if (!selectedEmotion) return null;
    return EMOTION_AUDIO_PARAMS[selectedEmotion] || null;
  }, [selectedEmotion]);

  return {
    engineState,
    initializeEngine,
    sendMessage,
    tracksToMIDI,
    getEmotionAudioParams,
    EMOTION_AUDIO_PARAMS,
  };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Convert beats to samples
 */
export function beatsToSamples(beats: number, tempo: number, sampleRate: number): number {
  const secondsPerBeat = 60 / tempo;
  return Math.round(beats * secondsPerBeat * sampleRate);
}

/**
 * Convert samples to beats
 */
export function samplesToBeats(samples: number, tempo: number, sampleRate: number): number {
  const secondsPerBeat = 60 / tempo;
  return samples / (secondsPerBeat * sampleRate);
}

/**
 * Convert MIDI note to frequency (Hz)
 */
export function midiToFrequency(midiNote: number): number {
  return 440 * Math.pow(2, (midiNote - 69) / 12);
}

/**
 * Convert frequency to MIDI note
 */
export function frequencyToMidi(frequency: number): number {
  return Math.round(69 + 12 * Math.log2(frequency / 440));
}
