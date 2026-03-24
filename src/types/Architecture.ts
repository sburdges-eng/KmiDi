/**
 * KMiDi Architecture Types
 *
 * Defines the complete type system for the Mind/Body/Cascade architecture.
 * These types are the contract between frontend, gateway API, plugins, and LLMs.
 *
 * Latency note: All types are pure data — no computation, no side effects.
 * Serialization is JSON over WebSocket. No impact on audio thread.
 */

// ─── Cascade ────────────────────────────────────────────────────────────────

/** Cascade tier — where execution stops based on ML Intensity */
export type CascadeTier =
  | 'raw_dsp'        // 0%   — pure DSP, no intelligence, <0.1ms
  | 'embedded_ml'    // 10%  — ONNX/CoreML/RTNeural, <1ms
  | 'penta_core'     // 25%  — C++ SIMD engines, 1-2ms
  | 'local_engine'   // 50%  — Python companion engines + thesaurus, 5-10ms
  | 'cloud_t1'       // 75%  — Symbolic LLM (3B-7B), 50-200ms
  | 'cloud_t2'       // 85%  — Text+Symbolic LLM (3B-13B), 200-500ms
  | 'cloud_t3'       // 95%  — Audio+Symbolic LLM (7B-30B), 200-500ms
  | 'cloud_t4';      // 100% — Audio Generation (30B+), 500ms-2s (Parrot only)

/** Maps ML Intensity (0-100) to the highest cascade tier to attempt */
export function intensityToTier(intensity: number): CascadeTier {
  if (intensity <= 0) return 'raw_dsp';
  if (intensity <= 10) return 'embedded_ml';
  if (intensity <= 25) return 'penta_core';
  if (intensity <= 50) return 'local_engine';
  if (intensity <= 75) return 'cloud_t1';
  if (intensity <= 85) return 'cloud_t2';
  if (intensity <= 99) return 'cloud_t3';
  return 'cloud_t4';
}

/** Domain influence sliders — what aspects get AI power */
export interface DomainInfluences {
  melody: number;   // 0-100
  harmony: number;  // 0-100
  groove: number;   // 0-100
  dynamics: number; // 0-100
}

// ─── The Thought ────────────────────────────────────────────────────────────

/** Structured performance direction — output of the Mind, input to the Body */
export interface Thought {
  chord_sequence: string[];
  melody_contour: number[];
  rhythm_pattern: string;
  dynamics: {
    start: string;
    peak: string;
    end: string;
  };
  vocal_params: VocalParams | null;
  arrangement: string;
  instrument_assignments: Record<string, string>;
  /** Which cascade tier actually produced this thought */
  source_tier: CascadeTier;
  /** Generation latency in ms */
  latency_ms: number;
}

export interface VocalParams {
  breathiness: number;      // 0-1
  tremor: number;           // 0-1
  vibrato: number;          // 0-1
  emotion_vector: [number, number, number]; // [valence, arousal, dominance]
  lyrics_phrasing: LyricPhrase[];
}

export interface LyricPhrase {
  text: string;
  emphasis: number;  // 0-1
}

// ─── Emotion ────────────────────────────────────────────────────────────────

/** A node in the 216-node emotion thesaurus */
export interface EmotionNode {
  id: number;
  label: string;
  valence: number;      // -1 to +1
  arousal: number;      // 0 to 1
  dominance: number;    // -1 to +1
  vocal_params: Partial<VocalParams>;
  harmony_params: {
    mode: 'major' | 'minor' | 'diminished' | 'augmented' | 'suspended';
    extensions: string[];
  };
  rhythm_params: {
    density: 'sparse' | 'moderate' | 'dense';
    feel: 'strict' | 'rubato' | 'swing' | 'shuffle';
  };
}

// ─── Plugin ─────────────────────────────────────────────────────────────────

export type PluginId =
  | 'Kl01' | 'Kl02' | 'Kl03' | 'Kl04' | 'Kl05'
  | 'Kl06' | 'Kl07' | 'Kl08' | 'Kl09' | 'Kl10';

export interface PluginInfo {
  id: PluginId;
  name: string;
  dsp_effect: string;
  engines: string[];
  ml_model: string;
}

export const PLUGIN_CATALOG: Record<PluginId, PluginInfo> = {
  Kl01: { id: 'Kl01', name: 'SaturationDynamics', dsp_effect: 'Tube Saturation', engines: ['DynamicsEngine'], ml_model: 'dynamics_influence' },
  Kl02: { id: 'Kl02', name: 'CrushFill', dsp_effect: 'Lo-fi Bitcrusher', engines: ['FillEngine'], ml_model: 'groove_influence' },
  Kl03: { id: 'Kl03', name: 'GrooveGate', dsp_effect: 'Spectral Gating', engines: ['GrooveEngine'], ml_model: 'groove_influence' },
  Kl04: { id: 'Kl04', name: 'SynthMelodyPad', dsp_effect: 'Wavetable Synth', engines: ['MelodyEngine', 'PadEngine'], ml_model: 'melody_influence' },
  Kl05: { id: 'Kl05', name: 'Parrot', dsp_effect: 'Voice Companion', engines: ['CounterMelodyEngine', 'StringEngine'], ml_model: 'melody_influence' },
  Kl06: { id: 'Kl06', name: 'CompressorDynamics', dsp_effect: 'VCA Compressor', engines: ['DynamicsEngine'], ml_model: 'dynamics_influence' },
  Kl07: { id: 'Kl07', name: 'ReverbTension', dsp_effect: 'Convolution Reverb', engines: ['TensionEngine'], ml_model: 'harmony_influence' },
  Kl08: { id: 'Kl08', name: 'StutterRhythm', dsp_effect: 'Stutter Repeater', engines: ['RhythmEngine'], ml_model: 'groove_influence' },
  Kl09: { id: 'Kl09', name: 'DuckTransition', dsp_effect: 'Sidechain Ducker', engines: ['TransitionEngine'], ml_model: 'ml_intensity' },
  Kl10: { id: 'Kl10', name: 'DelayVariation', dsp_effect: 'Delay', engines: ['VariationEngine'], ml_model: 'melody_influence' },
};

// ─── Gateway API ────────────────────────────────────────────────────────────

/** Plugin → Gateway request (Tier 2: Pro) */
export interface PluginGatewayRequest {
  plugin_id: PluginId;
  ml_intensity: number;         // 0-100
  domain_influences: DomainInfluences;
  context: MusicalContext;
  timeout_ms: number;           // max wait before auto-fallback
}

/** Gateway → Plugin response */
export interface PluginGatewayResponse {
  tier_used: CascadeTier;
  latency_ms: number;
  thought: Thought;
}

/** Consumer → Gateway request (Tier 1: Consumer) */
export interface ConsumerGenerateRequest {
  prompt: string;
  mood?: string;
  genre?: string;
  tempo?: number;
}

/** Gateway → Consumer response */
export interface ConsumerGenerateResponse {
  audio_url: string;
  format: 'wav' | 'mp3';
  duration_seconds: number;
  thought_summary: string;
}

/** DAiW → Gateway request (Tier 3: Premium) */
export interface DAiWRequest {
  intent: EmotionalIntentFull;
  audio_context?: Float32Array;     // JEPA embedding of current mix
  lyrics?: string;
  arrangement?: ArrangementState;
  session_id: string;
  render_formats: RenderFormat[];
}

/** Gateway → DAiW response */
export interface DAiWResponse {
  thought: Thought;
  rendered_files: Record<RenderFormat, string>;  // format → file URL
  session_id: string;
}

export type RenderFormat = 'midi' | 'wav' | 'mp3' | 'aif' | 'stems' | 'spectocloud';

// ─── Musical Context ────────────────────────────────────────────────────────

export interface MusicalContext {
  key: string;
  tempo: number;
  time_signature: string;
  current_chord: string;
  emotion_node: number;           // 0-215, index into thesaurus
  audio_embedding?: number[];     // JEPA vector (if T3 requested)
  lyrics?: string;
  intent?: string;                // natural language instruction
}

export interface EmotionalIntentFull {
  core_wound?: string;
  core_desire?: string;
  emotional_intent: string;
  technical?: {
    key?: string;
    bpm?: number;
    progression?: string[];
    genre?: string;
    duration?: number;
    structure?: Array<{ name: string; bars: number; repetitions?: number }>;
    instruments?: Array<{ instrument: string; techniques?: string[] }>;
    groove_feel?: string;
    rule_to_break?: string;
    rule_justification?: string;
  };
  vulnerability_scale?: number;
  narrative_arc?: string;
}

export interface ArrangementState {
  sections: Array<{
    name: string;
    bars: number;
    active_plugins: PluginId[];
  }>;
  active_engines: string[];
}

// ─── Gateway Connection ─────────────────────────────────────────────────────

export type GatewayConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface GatewayHealth {
  status: 'healthy' | 'degraded' | 'offline';
  available_tiers: CascadeTier[];
  latency_estimates: Record<CascadeTier, number>;
  model_status: {
    t1_symbolic: boolean;
    t2_text_symbolic: boolean;
    t3_audio_symbolic: boolean;
    t4_audio_gen: boolean;
  };
}

// ─── Product Tiers ──────────────────────────────────────────────────────────

export type ProductTier = 'consumer' | 'pro' | 'premium';

export const PRODUCT_TIER_INFO: Record<ProductTier, {
  name: string;
  description: string;
  outputs: string[];
  max_cascade_tier: CascadeTier;
}> = {
  consumer: {
    name: 'AI Music Generator',
    description: 'Vibe-based prompt → audio output. No DAW needed.',
    outputs: ['wav', 'mp3'],
    max_cascade_tier: 'cloud_t2',
  },
  pro: {
    name: 'Plugin Suite',
    description: 'Fully customizable MIDI + audio in your DAW.',
    outputs: ['midi', 'audio_passthrough'],
    max_cascade_tier: 'cloud_t3',
  },
  premium: {
    name: 'DAiW',
    description: 'Everything. Full workstation with all outputs and services.',
    outputs: ['midi', 'wav', 'mp3', 'aif', 'stems', 'spectocloud'],
    max_cascade_tier: 'cloud_t4',
  },
};

// ─── Training / Model Metadata ──────────────────────────────────────────────

export type ModelTask =
  | 'emotion_classifier' | 'emotion_node_216' | 'chord_predictor'
  | 'melody_generator' | 'groove_predictor' | 'dynamics_engine'
  | 'audio_jepa' | 'chord_jepa' | 'texture_density' | 'instrument_classifier'
  | 'voice_parrot';

export type ModelBackend = 'onnx' | 'coreml' | 'rtneural' | 'torchscript' | 'pytorch';

export type TrainingTier = 0 | 1 | 2 | 3 | 4;

export interface ModelInfo {
  task: ModelTask;
  backend: ModelBackend;
  training_tier: TrainingTier;
  input_dim: number;
  output_dim: number;
  size_mb: number;
  latency_ms: number;
  datasets: string[];
}

// ─── LLM Tiers ──────────────────────────────────────────────────────────────

export type LLMTier = 'T1' | 'T2' | 'T3' | 'T4';

export const LLM_TIER_INFO: Record<LLMTier, {
  name: string;
  params: string;
  modalities_in: string[];
  modalities_out: string[];
  purpose: string;
  feeds: string;
  est_cost: string;
}> = {
  T1: {
    name: 'Symbolic',
    params: '3B-7B',
    modalities_in: ['MIDI tokens', 'chord symbols', 'text descriptions'],
    modalities_out: ['chords', 'melody', 'arrangement'],
    purpose: 'Core music generation',
    feeds: 'All plugins + standalone + consumer',
    est_cost: '~$2K',
  },
  T2: {
    name: 'Text + Symbolic',
    params: '3B-13B',
    modalities_in: ['natural language', 'MIDI tokens'],
    modalities_out: ['symbolic + instruction response'],
    purpose: 'Instruction following ("make this darker")',
    feeds: 'All plugins + standalone + consumer',
    est_cost: '~$8K',
  },
  T3: {
    name: 'Audio + Symbolic',
    params: '7B-30B',
    modalities_in: ['audio embeddings (JEPA)', 'MIDI', 'text'],
    modalities_out: ['symbolic + context-aware adjustments'],
    purpose: 'Listens to mix, context-aware edits',
    feeds: 'Plugins (high intensity) + standalone',
    est_cost: '~$30K',
  },
  T4: {
    name: 'Audio Generation',
    params: '30B+',
    modalities_in: ['all above', 'voice parameters'],
    modalities_out: ['audio waveforms'],
    purpose: 'Neural voice synthesis, stem generation',
    feeds: 'Parrot (Kl05) + standalone + consumer',
    est_cost: '~$150K',
  },
};
