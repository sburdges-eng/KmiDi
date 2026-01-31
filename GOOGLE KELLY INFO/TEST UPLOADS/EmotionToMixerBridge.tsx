/**
 * EmotionToMixerBridge - Visual feedback for emotion-to-mixer mapping
 *
 * Shows the user how their emotional choices affect the mixer parameters.
 * This makes the Side A/B integration visible and understandable.
 */

import React from 'react';
import { useUnifiedStore } from '../../store/useUnifiedStore';
import { Sparkles, Volume2, Clock, Waves } from 'lucide-react';

interface EmotionToMixerBridgeProps {
  emotion: string | null;
  intent: Record<string, unknown> | null;
}

// Emotion to mixer parameter mappings (visual reference)
const EMOTION_EFFECTS: Record<string, {
  description: string;
  reverb: string;
  delay: string;
  compression: string;
  warmth: string;
}> = {
  Grief: {
    description: 'Spacious, distant, unresolved',
    reverb: 'High (70%)',
    delay: 'Medium (40%)',
    compression: 'Low (30%)',
    warmth: 'Low (20%)',
  },
  Joy: {
    description: 'Bright, punchy, present',
    reverb: 'Low (30%)',
    delay: 'Low (20%)',
    compression: 'Medium (50%)',
    warmth: 'High (70%)',
  },
  Rage: {
    description: 'Aggressive, compressed, in your face',
    reverb: 'Low (20%)',
    delay: 'Minimal (10%)',
    compression: 'Very High (90%)',
    warmth: 'Low (10%)',
  },
  Anxiety: {
    description: 'Unsettling, delayed, claustrophobic',
    reverb: 'Medium (45%)',
    delay: 'High (70%)',
    compression: 'Medium (50%)',
    warmth: 'Low (15%)',
  },
  Longing: {
    description: 'Yearning, spacious, warm undertones',
    reverb: 'High (65%)',
    delay: 'Medium (50%)',
    compression: 'Low (25%)',
    warmth: 'Medium (35%)',
  },
  Passion: {
    description: 'Intimate, warm, breathing room',
    reverb: 'Medium (40%)',
    delay: 'Low (30%)',
    compression: 'Medium (50%)',
    warmth: 'High (70%)',
  },
  Defiance: {
    description: 'Bold, present, punchy',
    reverb: 'Low (30%)',
    delay: 'Low (20%)',
    compression: 'High (80%)',
    warmth: 'Low (15%)',
  },
  Hope: {
    description: 'Open, ascending, warm',
    reverb: 'Medium (35%)',
    delay: 'Low (25%)',
    compression: 'Medium (40%)',
    warmth: 'Medium-High (60%)',
  },
};

export const EmotionToMixerBridge: React.FC<EmotionToMixerBridgeProps> = ({
  emotion,
  intent,
}) => {
  const { generatedMusic } = useUnifiedStore();

  if (!emotion) {
    return null;
  }

  const effects = EMOTION_EFFECTS[emotion] || {
    description: 'Custom emotional mapping',
    reverb: 'Auto',
    delay: 'Auto',
    compression: 'Auto',
    warmth: 'Auto',
  };

  return (
    <div className="p-3 bg-ableton-surface-light border-t border-ableton-border">
      <div className="flex items-center gap-2 mb-2">
        <Sparkles size={14} className="text-ableton-accent" />
        <span className="text-xs font-medium">Emotion Bridge Active</span>
      </div>

      <div className="text-xs text-ableton-text-dim mb-2 italic">
        "{effects.description}"
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="flex items-center gap-1">
          <Waves size={12} className="text-blue-400" />
          <span>Reverb: {effects.reverb}</span>
        </div>
        <div className="flex items-center gap-1">
          <Clock size={12} className="text-green-400" />
          <span>Delay: {effects.delay}</span>
        </div>
        <div className="flex items-center gap-1">
          <Volume2 size={12} className="text-yellow-400" />
          <span>Compression: {effects.compression}</span>
        </div>
        <div className="flex items-center gap-1">
          <Sparkles size={12} className="text-orange-400" />
          <span>Warmth: {effects.warmth}</span>
        </div>
      </div>

      {/* Generated Music Info */}
      {generatedMusic && (
        <div className="mt-3 pt-3 border-t border-ableton-border">
          <div className="text-xs text-ableton-text-dim mb-1">Generated Parameters:</div>
          <div className="flex gap-4 text-xs">
            <span>
              Key: <span className="text-ableton-accent">{generatedMusic.key}</span>
            </span>
            <span>
              Tempo: <span className="text-ableton-accent">{generatedMusic.tempo} BPM</span>
            </span>
            <span>
              Chords: <span className="text-ableton-accent">{generatedMusic.harmony.join(' - ')}</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
