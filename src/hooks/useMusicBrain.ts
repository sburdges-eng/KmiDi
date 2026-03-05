// Music Brain API client - connects directly to Python backend at http://127.0.0.1:8000

import { StructureSection, TrackIntent } from '../types/Intent';

const API_BASE = 'http://127.0.0.1:8000';

export interface EmotionalIntent {
  core_wound?: string;
  core_desire?: string;
  emotional_intent: string;
  vulnerability_scale?: number;
  secondary_tension?: number;
  imagery_texture?: string;
  technical?: {
    key?: string;
    bpm?: number;
    progression?: string[];
    genre?: string;
    duration?: number;
    structure?: StructureSection[];
    instruments?: TrackIntent[];
    techniques?: string[];
    groove_feel?: string;
    narrative_arc?: string;
    rule_to_break?: string | null;
    rule_justification?: string | null;
  };
}

export interface GenerateRequest {
  intent: EmotionalIntent;
  output_format?: string;
}

export interface InterrogateRequest {
  message: string;
  session_id?: string;
  context?: any;
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
  midi_events?: Array<Record<string, any>>;
  midi_file_path?: string;
  duration?: number;
  emotion_trajectory?: Array<Record<string, any>>;
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

export const useMusicBrain = () => {
  const getEmotions = async (): Promise<string[]> => {
    try {
      const result = await apiCall<string[]>('/emotions');
      return result;
    } catch (error) {
      console.error('Failed to get emotions:', error);
      throw error;
    }
  };

  const generateMusic = async (request: GenerateRequest) => {
    try {
      const result = await apiCall('/generate', {
        method: 'POST',
        body: JSON.stringify(request),
      });
      return result;
    } catch (error) {
      console.error('Failed to generate music:', error);
      throw error;
    }
  };

  const interrogate = async (request: InterrogateRequest) => {
    try {
      const result = await apiCall('/interrogate', {
        method: 'POST',
        body: JSON.stringify(request),
      });
      return result;
    } catch (error) {
      console.error('Failed to interrogate:', error);
      throw error;
    }
  };

  const getHumanizerConfig = async (): Promise<HumanizerConfig> => {
    try {
      const result = await apiCall<HumanizerConfig>('/config/humanizer');
      return result;
    } catch (error) {
      console.error('Failed to load humanizer config:', error);
      throw error;
    }
  };

  const updateHumanizerConfig = async (
    payload: UpdateHumanizerConfigInput,
  ): Promise<HumanizerConfig> => {
    const result = await apiCall<HumanizerConfig>('/config/humanizer', {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
    return result;
  };

  const renderSpectocloud = async (
    payload: SpectocloudRenderRequest,
  ): Promise<SpectocloudRenderResponse> => {
    if ((!payload.midi_events || payload.midi_events.length === 0) && !payload.midi_file_path) {
      throw new Error("provide midi_events or midi_file_path");
    }
    if (payload.midi_events && payload.midi_events.length === 0) {
      throw new Error("midi_events cannot be empty");
    }
    if (payload.duration !== undefined && payload.duration <= 0) {
      throw new Error("duration must be greater than 0 when provided");
    }
    const result = await apiCall<SpectocloudRenderResponse>('/spectocloud/render', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    return result;
  };

  const setUserLyrics = async (lyrics: string): Promise<LyricsUpdateResponse> => {
    try {
      const result = await apiCall<LyricsUpdateResponse>('/lyrics', {
        method: 'POST',
        body: JSON.stringify({ lyrics, source: 'user' }),
      });
      return result;
    } catch (error) {
      console.error('Failed to set user lyrics:', error);
      throw error;
    }
  };

  const getUserLyrics = async (): Promise<LyricsState> => {
    try {
      const result = await apiCall<LyricsState>('/lyrics');
      return result;
    } catch (error) {
      console.error('Failed to get user lyrics:', error);
      throw error;
    }
  };

  // Audio emotion classification
  const classifyAudio = async (audioPath: string, modelType: string = 'emotion_7'): Promise<AudioClassifyResult> => {
    try {
      const result = await apiCall<{ status: string; result: AudioClassifyResult }>('/audio/classify', {
        method: 'POST',
        body: JSON.stringify({ audio_path: audioPath, model_type: modelType }),
      });
      return result.result;
    } catch (error) {
      console.error('Failed to classify audio:', error);
      throw error;
    }
  };

  const getAudioValenceArousal = async (audioPath: string): Promise<{ valence: number; arousal: number; emotion: string; confidence: number }> => {
    try {
      const result = await apiCall<any>('/audio/valence-arousal', {
        method: 'POST',
        body: JSON.stringify({ audio_path: audioPath }),
      });
      return result;
    } catch (error) {
      console.error('Failed to get audio valence-arousal:', error);
      throw error;
    }
  };

  const getAudioModels = async (): Promise<{ models: Array<{ name: string; path: string }>; supported_types: string[] }> => {
    try {
      const result = await apiCall<any>('/audio/models');
      return result;
    } catch (error) {
      console.error('Failed to get audio models:', error);
      throw error;
    }
  };

  return {
    getEmotions,
    generateMusic,
    interrogate,
    getHumanizerConfig,
    updateHumanizerConfig,
    renderSpectocloud,
    setUserLyrics,
    getUserLyrics,
    classifyAudio,
    getAudioValenceArousal,
    getAudioModels,
  };
};
