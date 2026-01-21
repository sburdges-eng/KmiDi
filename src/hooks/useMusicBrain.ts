import { invoke } from '@tauri-apps/api/core';
import { useKellyBrain, IntentResult, GeneratedMidi } from './useKellyBrain';

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
  audio_file_path?: string; // MP3 or WAV file path
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
  const kellyBrain = useKellyBrain();

  const getEmotions = async () => {
    try {
      // Try KellyBrain first if initialized
      if (kellyBrain.isInitialized) {
        const emotions = await kellyBrain.getAvailableEmotions();
        if (emotions) return emotions;
      }
      
      // Fallback to original API
      const result = await invoke('get_emotions');
      return result;
    } catch (error) {
      console.error('Failed to get emotions:', error);
      throw error;
    }
  };

  const generateMusic = async (request: GenerateRequest) => {
    try {
      // Try KellyBrain first if initialized
      if (kellyBrain.isInitialized) {
        const intent = await kellyBrain.fromText(request.intent.emotional_intent);
        if (intent) {
          const midi = await kellyBrain.generateMidi(intent, 8);
          if (midi) {
            // Convert to expected format
            return {
              success: true,
              intent,
              midi,
              source: 'kelly_brain_cpp'
            };
          }
        }
      }
      
      // Try REST API first (works in both browser and Tauri mode)
      try {
        console.log('Attempting REST API call to http://127.0.0.1:8000/generate');
        const resp = await fetch('http://127.0.0.1:8000/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        });
        
        if (!resp.ok) {
          const errorText = await resp.text();
          throw new Error(`API returned ${resp.status}: ${resp.statusText}. ${errorText}`);
        }
        
        const result = await resp.json();
        console.log('REST API response:', result);
        
        // The API returns { status, result, lyrics, midi_path?, audio_path?, output_path? }
        // Extract audio file path from result if available
        if (result?.audio_path) {
          return result;
        } else if (result?.output_path) {
          return {
            ...result,
            audio_path: result.output_path,
          };
        } else if (result?.midi_path) {
          // If we only have MIDI, use it as audio path (conversion would happen server-side)
          return {
            ...result,
            audio_path: result.midi_path,
            output_path: result.midi_path,
          };
        } else if (result?.result?.midi_path) {
          // Nested in result object
          return {
            ...result,
            audio_path: result.result.midi_path,
            output_path: result.result.midi_path,
          };
        }
        
        return result;
      } catch (fetchError: any) {
        console.error('REST API call failed:', fetchError);
        
        // If fetch fails, try Tauri invoke (only works in Tauri app)
        if (typeof window !== 'undefined' && (window as any).__TAURI__) {
          console.log('Falling back to Tauri invoke');
          try {
            const result = await invoke('generate_music', { request });
            return result;
          } catch (tauriError: any) {
            // If both fail, throw helpful error
            if (fetchError.message?.includes('fetch') || fetchError.message?.includes('Failed to fetch') || fetchError.message?.includes('ECONNREFUSED')) {
              throw new Error('Music Brain API is not running. Start it with: python -m music_brain.api\n\nMake sure the API is running at http://127.0.0.1:8000');
            }
            throw fetchError;
          }
        } else {
          // In browser mode, throw helpful error
          if (fetchError.message?.includes('fetch') || fetchError.message?.includes('Failed to fetch') || fetchError.message?.includes('ECONNREFUSED')) {
            throw new Error('Music Brain API is not running. Start it with: python -m music_brain.api\n\nMake sure the API is running at http://127.0.0.1:8000');
          }
          throw fetchError;
        }
      }
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
    // Accept audio_file_path (MP3/WAV), midi_file_path, or midi_events
    const hasInput = 
      payload.audio_file_path || 
      payload.midi_file_path || 
      (payload.midi_events && payload.midi_events.length > 0);
    
    if (!hasInput) {
      throw new Error("provide audio_file_path, midi_file_path, or midi_events");
    }
    if (payload.midi_events && payload.midi_events.length === 0) {
      throw new Error("midi_events cannot be empty");
    }
    if (payload.duration !== undefined && payload.duration <= 0) {
      throw new Error("duration must be greater than 0 when provided");
    }
    
    console.log('Calling spectocloud render with:', payload);
    
    const resp = await fetch('http://127.0.0.1:8000/spectocloud/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    
    if (!resp.ok) {
      const errorText = await resp.text();
      throw new Error(`Spectocloud render failed (${resp.status}): ${errorText}`);
    }
    
    const result = await resp.json();
    console.log('Spectocloud render response:', result);
    return result;
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
    
    // KellyBrain integration
    kellyBrain,
    
    // Direct KellyBrain methods for advanced usage
    initializeKellyBrain: kellyBrain.initialize,
    isKellyBrainInitialized: kellyBrain.isInitialized,
    generateFromText: kellyBrain.fromText,
    generateFromEmotion: kellyBrain.fromEmotion,
    generateMidiDirect: kellyBrain.generateMidi,
    setEmotionParameters: kellyBrain.setEmotionParameters,
    getEmotionState: kellyBrain.getEmotionState,
  };
};
