import { invoke } from '@tauri-apps/api/core';

export interface EmotionalIntent {
  core_wound?: string;
  core_desire?: string;
  emotional_intent: string;
  technical?: {
    key?: string;
    bpm?: number;
    progression?: string[];
    genre?: string;
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

export interface UpdateHumanizerConfigInput extends Partial<HumanizerConfig> {
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

export const useMusicBrain = () => {
  const getEmotions = async () => {
    try {
      const result = await invoke('get_emotions');
      return result;
    } catch (error) {
      console.error('Failed to get emotions:', error);
      throw error;
    }
  };

  const generateMusic = async (request: GenerateRequest) => {
    try {
      const result = await invoke('generate_music', { request });
      return result;
    } catch (error) {
      console.error('Failed to generate music:', error);
      throw error;
    }
  };

  const interrogate = async (request: InterrogateRequest) => {
    try {
      const result = await invoke('interrogate', { request });
      return result;
    } catch (error) {
      console.error('Failed to interrogate:', error);
      throw error;
    }
  };

  const getHumanizerConfig = async (): Promise<HumanizerConfig> => {
    try {
      const result = await invoke('get_humanizer_config');
      return result as HumanizerConfig;
    } catch (error) {
      console.error('Failed to load humanizer config:', error);
      throw error;
    }
  };

  const updateHumanizerConfig = async (
    payload: UpdateHumanizerConfigInput,
  ): Promise<HumanizerConfig> => {
    const resp = await fetch('http://127.0.0.1:8000/config/humanizer', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      throw new Error(`Failed to update config (${resp.status})`);
    }
    return resp.json();
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
    const resp = await fetch('http://127.0.0.1:8000/spectocloud/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      throw new Error(`Spectocloud render failed (${resp.status})`);
    }
    return resp.json();
  };

  const setUserLyrics = async (lyrics: string): Promise<LyricsUpdateResponse> => {
    try {
      const result = await invoke('set_user_lyrics', { lyrics });
      return result as LyricsUpdateResponse;
    } catch (error) {
      console.error('Failed to set user lyrics:', error);
      throw error;
    }
  };

  const getUserLyrics = async (): Promise<LyricsState> => {
    try {
      const result = await invoke('get_user_lyrics');
      return result as LyricsState;
    } catch (error) {
      console.error('Failed to get user lyrics:', error);
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
  };
};
