import type { CompleteSongIntentRequest, StructureSection, TrackIntent } from '../types/Intent';

/** External API is deny-by-default; set VITE_KMIDI_USE_API=true to allow (e.g. dev). Unset in freeze/CI. */
export const USE_EXTERNAL_API = import.meta.env.VITE_KMIDI_USE_API === 'true';
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000';

/** PRD TTG v1 — matches music_brain.api_schemas.ttg_v1 (docs/prd/KMIDI_PRD.md §8). */
export type TtgBoundaryEventV1 = 'drum_fill_1bar';
export type TtgHarmonicRhythmV1 = 'slow' | 'fast' | 'medium';

export interface TTGPhraseV1 {
  id?: string;
  bars: number;
  harmonic_rhythm?: TtgHarmonicRhythmV1;
  motifs?: string[];
  boundary_event?: TtgBoundaryEventV1;
  /** Engine section tag (intro | verse | chorus | …). */
  section_role?: string;
}

export interface TTGMovementV1 {
  type: 'movement';
  id: string;
  /** Must equal sum(phrase bars) + count(drum_fill_1bar) when provided. */
  bars?: number;
  children: TTGPhraseV1[];
}

export interface OrchestrationRoleSpecV1 {
  patch: string;
  active_threshold: number;
}

export interface TTGOrchestrationV1 {
  roles: Record<string, OrchestrationRoleSpecV1>;
}

export interface EnergyCurvePointV1 {
  bar: number;
  value: number;
}

export interface EnergyCurveV1 {
  points: EnergyCurvePointV1[];
}

export interface TechnicalIntent {
  key?: string;
  bpm?: number;
  progression?: string[];
  genre?: string;
  duration?: number;
  structure?: Array<{ name: string; bars: number; repetitions?: number }>;
  instruments?: Array<{ instrument: string; techniques?: string[] }>;
  techniques?: string[];
  groove_feel?: string;
  narrative_arc?: string;
  rule_to_break?: string;
  rule_justification?: string;
  /** TTG root movement; when set, overrides flat `structure` on the server. */
  timeline?: TTGMovementV1;
  /** Role → patch + energy threshold; when set, overrides flat `instruments`. */
  orchestration?: TTGOrchestrationV1;
  energy_curve?: EnergyCurveV1;
}

export interface EmotionalIntent {
  core_wound?: string;
  core_desire?: string;
  emotional_intent: string;
  technical?: TechnicalIntent;
  vulnerability_scale?: number;
  narrative_arc?: string;
  imagery_texture?: string;
}

export interface GenerateRequest {
  /** PRD §7 — only 1 is supported; omit defaults to 1 on the server. */
  intent_schema_version?: number;
  intent: EmotionalIntent;
  output_format?: string;
}

export interface InterrogateRequest {
  message: string;
  session_id?: string;
  context?: Record<string, unknown>;
}

export interface HumanizerConfig {
  default_style: string;
  ppq: number;
  bpm: number;
  analysis: {
    flam_threshold_ms: number;
    buzz_threshold_ms: number;
    drag_threshold_ms: number;
    alternation_window_ms: number;
  };
}

export interface UpdateHumanizerConfigInput extends Omit<Partial<HumanizerConfig>, 'analysis'> {
  analysis?: Partial<HumanizerConfig["analysis"]>;
}

export type SpectocloudMode = "static" | "animation";

export interface SpectocloudRenderRequest {
  midi_events?: Array<Record<string, unknown>>;
  midi_file_path?: string;
  audio_file_path?: string;
  duration?: number;
  emotion_trajectory?: Array<Record<string, unknown>>;
  mode?: SpectocloudMode;
  frame_idx?: number;
  output_path?: string;
  fps?: number;
  rotate?: boolean;
  anchor_density?: string;
  n_particles?: number;
}

export interface SpectocloudRenderResponse {
  status: string;
  mode: SpectocloudMode;
  output_path: string;
  frames: number;
}

export interface LyricsState {
  lyrics?: string;
  source?: string;
  generated?: string;
}

export interface LyricsUpdateResponse {
  status: string;
  source: string;
  lines: number;
  word_count: number;
  preview?: string;
}

export interface AudioClassifyResult {
  emotion: string;
  confidence: number;
  valence: number;
  arousal: number;
  top_predictions: Array<{ emotion: string; confidence: number }>;
  model_type: string;
}

async function apiCall<T>(endpoint: string, options?: RequestInit): Promise<T> {
  if (!USE_EXTERNAL_API) {
    throw new Error(
      'External Music Brain API is disabled. Set VITE_KMIDI_USE_API=true to enable (dev only; leave unset in freeze/CI).'
    );
  }
  const resp = await fetch(`${API_BASE}${endpoint}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`API error (${resp.status}): ${errorText}`);
  }
  return resp.json();
}

export function buildGeneratePayload(intent: CompleteSongIntentRequest): GenerateRequest {
  return {
    intent_schema_version: 1,
    intent: {
      core_desire: intent.core_desire,
      emotional_intent: intent.mood_primary,
      narrative_arc: intent.narrative_arc,
      technical: {
        key: intent.key_mode,
        bpm: intent.tempo ?? 120,
        genre: intent.genre,
        structure: intent.structure.map(s => ({
          name: s.name,
          bars: s.bars,
          repetitions: s.repetitions ?? 1,
        })),
        instruments: intent.instruments.map(i => ({
          instrument: i.instrument,
          techniques: i.techniques ?? [],
        })),
        groove_feel: intent.groove_feel,
        rule_to_break: intent.rule_to_break ?? undefined,
        rule_justification: intent.rule_justification ?? undefined,
      },
    },
  };
}

/** Bars per movement including 1 bar per `drum_fill_1bar` boundary event. */
export function derivedTtgMovementBars(children: TTGPhraseV1[]): number {
  let total = 0;
  for (const p of children) {
    total += p.bars;
    if (p.boundary_event === 'drum_fill_1bar') total += 1;
  }
  return total;
}

/**
 * Flat UI sections → single TTG movement (one phrase per section).
 * Repetitions are folded into phrase bar counts.
 */
export function structureToTtgMovement(
  structure: StructureSection[],
  opts?: { movementId?: string },
): TTGMovementV1 {
  const id = opts?.movementId ?? 'A';
  const children: TTGPhraseV1[] = structure.map(s => ({
    bars: s.bars * (s.repetitions ?? 1),
    section_role: s.name,
  }));
  const bars = derivedTtgMovementBars(children);
  return { type: 'movement', id, bars, children };
}

/**
 * Instrument list → orchestration roles (patch = instrument name).
 * Unique role keys avoid collisions when the same patch appears twice.
 */
export function instrumentsToOrchestration(
  instruments: TrackIntent[],
  defaultThreshold = 0.5,
): TTGOrchestrationV1 {
  const roles: Record<string, OrchestrationRoleSpecV1> = {};
  instruments.forEach((inst, i) => {
    const patch = inst.instrument.trim() || `instrument_${i}`;
    const slug = patch.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_').replace(/[^a-z0-9_]/g, '_');
    const role = `${slug || 'role'}_${i}`;
    roles[role] = { patch, active_threshold: defaultThreshold };
  });
  return { roles };
}

/**
 * PRD v1 payload: timeline + orchestration; POST to `/v1/generate` in the browser.
 * Server adapts TTG → flat engine schema.
 */
export function buildGeneratePayloadAsTtg(intent: CompleteSongIntentRequest): GenerateRequest {
  const timeline = structureToTtgMovement(intent.structure);
  const orchestration = instrumentsToOrchestration(intent.instruments);
  return {
    intent_schema_version: 1,
    intent: {
      core_desire: intent.core_desire,
      emotional_intent: intent.mood_primary,
      narrative_arc: intent.narrative_arc,
      technical: {
        key: intent.key_mode,
        bpm: intent.tempo ?? 120,
        genre: intent.genre,
        timeline,
        orchestration,
        groove_feel: intent.groove_feel,
        rule_to_break: intent.rule_to_break ?? undefined,
        rule_justification: intent.rule_justification ?? undefined,
      },
    },
  };
}

export const useMusicBrain = () => {
  const getEmotions = async (): Promise<string[]> => {
    return apiCall<string[]>('/emotions');
  };

  const generateMusic = async (request: GenerateRequest) => {
    return apiCall('/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  };

  /** Versioned PRD path — same handler as `/generate`; prefer with TTG payloads. */
  const generateMusicV1 = async (request: GenerateRequest) => {
    return apiCall('/v1/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  };

  const generateFromIntent = async (intent: CompleteSongIntentRequest) => {
    const payload = buildGeneratePayload(intent);
    return generateMusic(payload);
  };

  const generateFromIntentAsTtg = async (intent: CompleteSongIntentRequest) => {
    return generateMusicV1(buildGeneratePayloadAsTtg(intent));
  };

  const interrogate = async (request: InterrogateRequest) => {
    return apiCall<{ status: string; reply: string; session_id?: string }>('/interrogate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  };

  const getHumanizerConfig = async (): Promise<HumanizerConfig> => {
    return apiCall<HumanizerConfig>('/config/humanizer');
  };

  const updateHumanizerConfig = async (
    payload: UpdateHumanizerConfigInput,
  ): Promise<HumanizerConfig> => {
    return apiCall<HumanizerConfig>('/config/humanizer', {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
  };

  const renderSpectocloud = async (
    payload: SpectocloudRenderRequest,
  ): Promise<SpectocloudRenderResponse> => {
    return apiCall<SpectocloudRenderResponse>('/spectocloud/render', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  };

  const setUserLyrics = async (lyrics: string): Promise<LyricsUpdateResponse> => {
    return apiCall<LyricsUpdateResponse>('/lyrics', {
      method: 'POST',
      body: JSON.stringify({ lyrics, source: 'user' }),
    });
  };

  const getUserLyrics = async (): Promise<LyricsState> => {
    return apiCall<LyricsState>('/lyrics');
  };

  const classifyAudio = async (
    audioPath: string,
    modelType: string = 'emotion_7',
  ): Promise<AudioClassifyResult> => {
    const result = await apiCall<{ status: string; result: AudioClassifyResult }>('/audio/classify', {
      method: 'POST',
      body: JSON.stringify({ audio_path: audioPath, model_type: modelType }),
    });
    return result.result;
  };

  const getAudioValenceArousal = async (
    audioPath: string,
  ): Promise<{ valence: number; arousal: number; emotion: string; confidence: number }> => {
    return apiCall('/audio/valence-arousal', {
      method: 'POST',
      body: JSON.stringify({ audio_path: audioPath }),
    });
  };

  const getAudioModels = async (): Promise<{
    models: Array<{ name: string; path: string }>;
    supported_types: string[];
  }> => {
    return apiCall('/audio/models');
  };

  const healthCheck = async (): Promise<{ status: string; version: string }> => {
    if (!USE_EXTERNAL_API) {
      return Promise.reject(new Error('API disabled (no cloud required)'));
    }
    return apiCall('/health');
  };

  return {
    getEmotions,
    generateMusic,
    generateMusicV1,
    generateFromIntent,
    generateFromIntentAsTtg,
    interrogate,
    getHumanizerConfig,
    updateHumanizerConfig,
    renderSpectocloud,
    setUserLyrics,
    getUserLyrics,
    classifyAudio,
    getAudioValenceArousal,
    getAudioModels,
    healthCheck,
  };
};
