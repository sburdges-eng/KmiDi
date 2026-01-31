/**
 * Unified Store - Combines Side A (DAW) and Side B (Emotion) state
 *
 * No longer separate entities - emotion state directly influences DAW state
 * through the bridge functions.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// ============================================================================
// TYPES
// ============================================================================

export interface Clip {
  id: string;
  name: string;
  startTime: number;
  duration: number;
  color?: string;
  // Emotion metadata
  emotionTag?: string;
  generatedFrom?: 'manual' | 'ghost_writer' | 'rule_break';
}

export interface Track {
  id: string;
  name: string;
  type: 'audio' | 'midi' | 'aux';
  color: string;
  volume: number;
  pan: number;
  muted: boolean;
  solo: boolean;
  armed: boolean;
  clips: Clip[];
  // Emotion-driven parameters
  emotionOverrides?: {
    reverb?: number;
    delay?: number;
    compression?: number;
    warmth?: number;
  };
}

export interface SongIntent {
  coreEmotion: string | null;
  subEmotion: string | null;
  ruleToBreak: string | null;
  intent: Record<string, unknown> | null;
  // Phase 0: Core Wound
  coreEvent?: string;
  coreResistance?: string;
  coreLonging?: string;
  // Phase 1: Emotional Intent
  vulnerabilityScale?: 'Low' | 'Medium' | 'High' | 'Devastating';
  // Phase 2: Technical
  technicalKey?: string;
  ruleBreakJustification?: string;
}

export interface GeneratedMusic {
  harmony: string[];
  tempo: number;
  key: string;
  mixer_params: Record<string, number | string>;
  midiData?: unknown; // For Logic Pro export
}

type PanelState = 'collapsed' | 'minimal' | 'expanded';

// ============================================================================
// STORE STATE
// ============================================================================

interface UnifiedStoreState {
  // === PLAYBACK (Side A) ===
  isPlaying: boolean;
  isRecording: boolean;
  currentTime: number;
  tempo: number;
  timeSignature: [number, number];

  // === PROJECT (Side A) ===
  projectName: string;
  tracks: Track[];
  masterVolume: number;
  selectedTrackId: string | null;

  // === EMOTION (Side B) ===
  selectedEmotion: string | null;
  completedIntent: Record<string, unknown> | null;
  songIntent: SongIntent;
  generatedMusic: GeneratedMusic | null;

  // === UI LAYOUT ===
  emotionPanelState: PanelState;
  mixerPanelState: PanelState;

  // === ACTIONS: Playback ===
  setPlaying: (playing: boolean) => void;
  setRecording: (recording: boolean) => void;
  setCurrentTime: (time: number | ((prev: number) => number)) => void;
  setTempo: (tempo: number) => void;
  setMasterVolume: (volume: number) => void;
  play: () => void;
  stop: () => void;
  pause: () => void;

  // === ACTIONS: Tracks ===
  addTrack: (track: Omit<Track, 'id'>) => void;
  updateTrack: (id: string, updates: Partial<Track>) => void;
  removeTrack: (id: string) => void;
  selectTrack: (id: string | null) => void;
  addClipToTrack: (trackId: string, clip: Omit<Clip, 'id'>) => void;

  // === ACTIONS: Emotion (Side B) ===
  setSelectedEmotion: (emotion: string | null) => void;
  setCompletedIntent: (intent: Record<string, unknown> | null) => void;
  updateSongIntent: (updates: Partial<SongIntent>) => void;
  clearSongIntent: () => void;
  setGeneratedMusic: (music: GeneratedMusic | null) => void;

  // === ACTIONS: UI ===
  setEmotionPanelState: (state: PanelState) => void;
  setMixerPanelState: (state: PanelState) => void;

  // === ACTIONS: Bridge (Unified) ===
  applyEmotionToMixer: () => void;
  applyGeneratedHarmony: () => void;
  exportToLogicPro: () => LogicProProject;
}

// ============================================================================
// LOGIC PRO EXPORT TYPES
// ============================================================================

export interface LogicProProject {
  name: string;
  tempo_bpm: number;
  time_signature: [number, number];
  ppq: number;
  key: string;
  mode: string;
  tracks: LogicProTrack[];
  metadata: {
    emotion: string | null;
    intent: Record<string, unknown> | null;
    generated_at: string;
    idawi_version: string;
  };
}

export interface LogicProTrack {
  name: string;
  channel: number;
  instrument: number | null;
  notes: LogicProNote[];
  emotion_params?: {
    reverb: number;
    delay: number;
    compression: number;
  };
}

export interface LogicProNote {
  pitch: number;
  velocity: number;
  start_tick: number;
  duration_ticks: number;
}

// ============================================================================
// DEFAULT DATA
// ============================================================================

const defaultTracks: Track[] = [
  {
    id: '1',
    name: 'Drums',
    type: 'audio',
    color: '#ff5500',
    volume: 0.8,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [
      { id: 'c1', name: 'Beat 1', startTime: 0, duration: 4 },
      { id: 'c2', name: 'Beat 2', startTime: 4, duration: 4 },
    ],
  },
  {
    id: '2',
    name: 'Bass',
    type: 'midi',
    color: '#00aaff',
    volume: 0.7,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [
      { id: 'c3', name: 'Bass Line', startTime: 0, duration: 8 },
    ],
  },
  {
    id: '3',
    name: 'Synth Lead',
    type: 'midi',
    color: '#aa00ff',
    volume: 0.6,
    pan: 0.2,
    muted: false,
    solo: false,
    armed: false,
    clips: [
      { id: 'c4', name: 'Melody', startTime: 4, duration: 4 },
    ],
  },
  {
    id: '4',
    name: 'Pad',
    type: 'midi',
    color: '#00ff88',
    volume: 0.5,
    pan: -0.2,
    muted: false,
    solo: false,
    armed: false,
    clips: [
      { id: 'c5', name: 'Atmosphere', startTime: 0, duration: 8 },
    ],
  },
  {
    id: '5',
    name: 'Vocals',
    type: 'audio',
    color: '#ffaa00',
    volume: 0.9,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [],
  },
  {
    id: '6',
    name: 'FX',
    type: 'aux',
    color: '#ff00aa',
    volume: 0.4,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [],
  },
];

const defaultSongIntent: SongIntent = {
  coreEmotion: null,
  subEmotion: null,
  ruleToBreak: null,
  intent: null,
  coreEvent: '',
  coreResistance: '',
  coreLonging: '',
  vulnerabilityScale: 'Medium',
  technicalKey: 'C',
  ruleBreakJustification: '',
};

// ============================================================================
// EMOTION-TO-MIXER MAPPING
// ============================================================================

const EMOTION_MIXER_MAP: Record<string, Partial<Track['emotionOverrides']>> = {
  Grief: { reverb: 0.7, delay: 0.4, compression: 0.3, warmth: 0.2 },
  Melancholy: { reverb: 0.6, delay: 0.3, compression: 0.3, warmth: 0.3 },
  Longing: { reverb: 0.65, delay: 0.5, compression: 0.25, warmth: 0.35 },
  Joy: { reverb: 0.3, delay: 0.2, compression: 0.5, warmth: 0.7 },
  Euphoria: { reverb: 0.4, delay: 0.3, compression: 0.6, warmth: 0.8 },
  Hope: { reverb: 0.35, delay: 0.25, compression: 0.4, warmth: 0.6 },
  Rage: { reverb: 0.2, delay: 0.1, compression: 0.9, warmth: 0.1 },
  Frustration: { reverb: 0.25, delay: 0.15, compression: 0.7, warmth: 0.2 },
  Defiance: { reverb: 0.3, delay: 0.2, compression: 0.8, warmth: 0.15 },
  Terror: { reverb: 0.5, delay: 0.6, compression: 0.4, warmth: 0.1 },
  Anxiety: { reverb: 0.45, delay: 0.7, compression: 0.5, warmth: 0.15 },
  Dread: { reverb: 0.55, delay: 0.5, compression: 0.35, warmth: 0.1 },
  Passion: { reverb: 0.4, delay: 0.3, compression: 0.5, warmth: 0.7 },
  Tenderness: { reverb: 0.5, delay: 0.35, compression: 0.25, warmth: 0.8 },
  Devotion: { reverb: 0.45, delay: 0.3, compression: 0.35, warmth: 0.75 },
};

const EMOTION_TEMPO_MAP: Record<string, number> = {
  Grief: 72,
  Melancholy: 80,
  Longing: 75,
  Joy: 120,
  Euphoria: 128,
  Hope: 110,
  Rage: 160,
  Frustration: 140,
  Defiance: 130,
  Terror: 90,
  Anxiety: 140,
  Dread: 70,
  Passion: 90,
  Tenderness: 65,
  Devotion: 85,
};

// ============================================================================
// STORE IMPLEMENTATION
// ============================================================================

export const useUnifiedStore = create<UnifiedStoreState>()(
  subscribeWithSelector((set, get) => ({
    // === INITIAL STATE ===

    // Playback
    isPlaying: false,
    isRecording: false,
    currentTime: 0,
    tempo: 120,
    timeSignature: [4, 4],

    // Project
    projectName: 'Untitled Project',
    tracks: defaultTracks,
    masterVolume: 0.8,
    selectedTrackId: null,

    // Emotion
    selectedEmotion: null,
    completedIntent: null,
    songIntent: defaultSongIntent,
    generatedMusic: null,

    // UI
    emotionPanelState: 'expanded',
    mixerPanelState: 'expanded',

    // === ACTIONS: Playback ===

    setPlaying: (playing) => set({ isPlaying: playing }),
    setRecording: (recording) => set({ isRecording: recording }),
    setCurrentTime: (time) => set((state) => ({
      currentTime: typeof time === 'function' ? time(state.currentTime) : time
    })),
    setTempo: (tempo) => set({ tempo }),
    setMasterVolume: (volume) => set({ masterVolume: volume }),
    play: () => set({ isPlaying: true }),
    stop: () => set({ isPlaying: false, currentTime: 0 }),
    pause: () => set({ isPlaying: false }),

    // === ACTIONS: Tracks ===

    addTrack: (track) => set((state) => ({
      tracks: [...state.tracks, { ...track, id: `track-${Date.now()}` }]
    })),

    updateTrack: (id, updates) => set((state) => ({
      tracks: state.tracks.map((t) =>
        t.id === id ? { ...t, ...updates } : t
      )
    })),

    removeTrack: (id) => set((state) => ({
      tracks: state.tracks.filter((t) => t.id !== id)
    })),

    selectTrack: (id) => set({ selectedTrackId: id }),

    addClipToTrack: (trackId, clip) => set((state) => ({
      tracks: state.tracks.map((t) =>
        t.id === trackId
          ? { ...t, clips: [...t.clips, { ...clip, id: `clip-${Date.now()}` }] }
          : t
      )
    })),

    // === ACTIONS: Emotion ===

    setSelectedEmotion: (emotion) => {
      set({ selectedEmotion: emotion });
      // Auto-apply emotion effects when emotion changes
      if (emotion) {
        get().applyEmotionToMixer();
      }
    },

    setCompletedIntent: (intent) => set({ completedIntent: intent }),

    updateSongIntent: (updates) => set((state) => ({
      songIntent: { ...state.songIntent, ...updates }
    })),

    clearSongIntent: () => set({
      songIntent: defaultSongIntent,
      selectedEmotion: null,
      completedIntent: null,
      generatedMusic: null,
    }),

    setGeneratedMusic: (music) => {
      set({ generatedMusic: music });
      if (music) {
        // Auto-update tempo from generated music
        set({ tempo: music.tempo });
        // Apply harmony to tracks
        get().applyGeneratedHarmony();
      }
    },

    // === ACTIONS: UI ===

    setEmotionPanelState: (state) => set({ emotionPanelState: state }),
    setMixerPanelState: (state) => set({ mixerPanelState: state }),

    // === ACTIONS: Bridge ===

    applyEmotionToMixer: () => {
      const { selectedEmotion, tracks } = get();
      if (!selectedEmotion) return;

      const emotionParams = EMOTION_MIXER_MAP[selectedEmotion];
      const emotionTempo = EMOTION_TEMPO_MAP[selectedEmotion];

      if (emotionParams) {
        // Apply emotion overrides to all tracks
        const updatedTracks = tracks.map((track) => ({
          ...track,
          emotionOverrides: emotionParams,
        }));
        set({ tracks: updatedTracks });
      }

      if (emotionTempo) {
        set({ tempo: emotionTempo });
      }
    },

    applyGeneratedHarmony: () => {
      const { generatedMusic, tracks } = get();
      if (!generatedMusic?.harmony) return;

      // Find the first MIDI track (likely chords/pad)
      const harmonyTrack = tracks.find(t => t.type === 'midi' && t.name.toLowerCase().includes('pad'));
      if (!harmonyTrack) return;

      // Create clips for the harmony progression
      const ppq = 480; // Logic Pro standard
      const beatsPerChord = 4;

      const newClips: Clip[] = generatedMusic.harmony.map((chord, index) => ({
        id: `harmony-${Date.now()}-${index}`,
        name: chord,
        startTime: index * beatsPerChord,
        duration: beatsPerChord,
        emotionTag: get().selectedEmotion || undefined,
        generatedFrom: 'ghost_writer',
      }));

      set((state) => ({
        tracks: state.tracks.map((t) =>
          t.id === harmonyTrack.id
            ? { ...t, clips: [...t.clips, ...newClips] }
            : t
        )
      }));
    },

    // === LOGIC PRO EXPORT ===

    exportToLogicPro: (): LogicProProject => {
      const state = get();
      const PPQ = 480;

      // Convert tracks to Logic Pro format
      const logicTracks: LogicProTrack[] = state.tracks.map((track, index) => {
        // Determine MIDI channel (drums on 10)
        const channel = track.name.toLowerCase().includes('drum') ? 10 : index + 1;

        // Convert clips to notes (simplified - would need full MIDI data in production)
        const notes: LogicProNote[] = [];
        track.clips.forEach((clip) => {
          // Generate placeholder notes for each clip
          const startTick = Math.round(clip.startTime * PPQ);
          const durationTicks = Math.round(clip.duration * PPQ);

          // Add a root note for now (would be replaced with actual MIDI data)
          notes.push({
            pitch: 60, // Middle C
            velocity: 100,
            start_tick: startTick,
            duration_ticks: durationTicks,
          });
        });

        return {
          name: track.name,
          channel,
          instrument: track.type === 'midi' ? 0 : null,
          notes,
          emotion_params: track.emotionOverrides ? {
            reverb: track.emotionOverrides.reverb || 0.3,
            delay: track.emotionOverrides.delay || 0.2,
            compression: track.emotionOverrides.compression || 0.4,
          } : undefined,
        };
      });

      return {
        name: state.projectName,
        tempo_bpm: state.tempo,
        time_signature: state.timeSignature,
        ppq: PPQ,
        key: state.songIntent.technicalKey || 'C',
        mode: state.selectedEmotion?.toLowerCase() || 'neutral',
        tracks: logicTracks,
        metadata: {
          emotion: state.selectedEmotion,
          intent: state.completedIntent,
          generated_at: new Date().toISOString(),
          idawi_version: '1.0.0',
        },
      };
    },
  }))
);

// ============================================================================
// SUBSCRIPTIONS - Auto-sync emotion to DAW
// ============================================================================

// When emotion changes, auto-apply mixer settings
useUnifiedStore.subscribe(
  (state) => state.selectedEmotion,
  (emotion) => {
    if (emotion) {
      console.log(`[Bridge] Emotion changed to: ${emotion}`);
      // The applyEmotionToMixer is called in setSelectedEmotion
    }
  }
);

// When generated music is set, auto-apply to timeline
useUnifiedStore.subscribe(
  (state) => state.generatedMusic,
  (music) => {
    if (music) {
      console.log(`[Bridge] Generated music applied:`, music);
    }
  }
);
