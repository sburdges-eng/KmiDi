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

  return {
    getEmotions,
    generateMusic,
    interrogate,
    getHumanizerConfig,
    updateHumanizerConfig,
  };
};
