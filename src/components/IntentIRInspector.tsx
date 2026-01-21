import React, { useState, useEffect } from 'react';
import './IntentIRInspector.css';

/**
 * IntentIRInspector - Display full IntentFrame (read-only)
 * 
 * This component displays the complete IntentFrame structure with:
 * - Raw values, no formatting
 * - Color-coding by provenance
 * - Real-time updates
 */
interface IntentFrame {
  meta: {
    ir_version: number;
    intent_id: number;
    session_id: number;
  };
  emotion: {
    valence: number;
    arousal: number;
    dominance: number;
    discrete_id: number;
    intensity: number;
    confidence: number;
  };
  music: {
    tempo_bias: number;
    rhythmic_density: number;
    groove_strength: number;
    harmonic_tension: number;
    harmonic_motion: number;
    mode_preference: number;
    melodic_activity: number;
    contour_variance: number;
    dynamic_range: number;
    texture_density: number;
  };
  time: {
    start_bar: number;
    end_bar: number;
    fade_in_beats: number;
    fade_out_beats: number;
  };
  constraints: {
    allowed_engines_mask: number;
    forbidden_engines_mask: number;
    max_cpu_cost: number;
    max_event_rate: number;
  };
  provenance: {
    source: number;
    user_override_weight: number;
  };
}

type IntentSource = 'UI_DIRECT' | 'UI_EDIT' | 'ML_TEXT' | 'ML_AUDIO' | 'PRESET' | 'AUTOMATION';

const SOURCE_NAMES: Record<number, IntentSource> = {
  0: 'UI_DIRECT',
  1: 'UI_EDIT',
  2: 'ML_TEXT',
  3: 'ML_AUDIO',
  4: 'PRESET',
  5: 'AUTOMATION',
};

// Source color mappings using semantic token RGB values
const SOURCE_COLORS: Record<IntentSource, string> = {
  UI_DIRECT: 'rgb(52, 199, 89)',        // accent-success - user direct input
  UI_EDIT: 'rgb(0, 122, 255)',         // accent-primary - user editing
  ML_TEXT: 'rgb(255, 149, 0)',         // accent-warning - ML text analysis
  ML_AUDIO: 'rgb(255, 59, 48)',         // accent-error - ML audio analysis
  PRESET: 'rgb(156, 39, 176)',         // accent-purple - preset
  AUTOMATION: 'rgb(96, 125, 139)',     // accent-bluegrey - automation
};

interface IntentIRInspectorProps {
  frame?: IntentFrame;
  onFrameUpdate?: (frame: IntentFrame) => void;
}

export const IntentIRInspector: React.FC<IntentIRInspectorProps> = ({ frame, onFrameUpdate }) => {
  const [currentFrame, setCurrentFrame] = useState<IntentFrame | null>(frame || null);
  const [isExpanded, setIsExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (frame) {
      setCurrentFrame(frame);
    }
  }, [frame]);

  const toggleSection = (section: string) => {
    setIsExpanded(prev => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  if (!currentFrame) {
    return (
      <div className="intent-ir-inspector">
        <div className="intent-ir-inspector-empty">
          No IntentFrame available
        </div>
      </div>
    );
  }

  const sourceName = SOURCE_NAMES[currentFrame.provenance.source] || 'UNKNOWN';
  const sourceColor = SOURCE_COLORS[sourceName] || 'rgb(52, 52, 56)'; // bg-tertiary fallback

  return (
    <div className="intent-ir-inspector">
      <div className="intent-ir-inspector-header">
        <h3>Intent IR Inspector</h3>
        <div 
          className="intent-ir-provenance-badge"
          style={{ backgroundColor: sourceColor }}
        >
          {sourceName}
        </div>
      </div>

      <div className="intent-ir-inspector-content">
        {/* Meta */}
        <div className="intent-ir-section">
          <div 
            className="intent-ir-section-header"
            onClick={() => toggleSection('meta')}
          >
            <span>Meta</span>
            <span>{isExpanded['meta'] ? '−' : '+'}</span>
          </div>
          {isExpanded['meta'] && (
            <div className="intent-ir-section-content">
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">ir_version:</span>
                <span className="intent-ir-field-value">{currentFrame.meta.ir_version}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">intent_id:</span>
                <span className="intent-ir-field-value">{currentFrame.meta.intent_id.toString(16)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">session_id:</span>
                <span className="intent-ir-field-value">{currentFrame.meta.session_id.toString(16)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Emotion */}
        <div className="intent-ir-section">
          <div 
            className="intent-ir-section-header"
            onClick={() => toggleSection('emotion')}
          >
            <span>Emotion</span>
            <span>{isExpanded['emotion'] ? '−' : '+'}</span>
          </div>
          {isExpanded['emotion'] && (
            <div className="intent-ir-section-content">
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">valence:</span>
                <span className="intent-ir-field-value">{currentFrame.emotion.valence.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">arousal:</span>
                <span className="intent-ir-field-value">{currentFrame.emotion.arousal.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">dominance:</span>
                <span className="intent-ir-field-value">{currentFrame.emotion.dominance.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">discrete_id:</span>
                <span className="intent-ir-field-value">{currentFrame.emotion.discrete_id}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">intensity:</span>
                <span className="intent-ir-field-value">{currentFrame.emotion.intensity.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">confidence:</span>
                <span className="intent-ir-field-value">{currentFrame.emotion.confidence.toFixed(3)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Music */}
        <div className="intent-ir-section">
          <div 
            className="intent-ir-section-header"
            onClick={() => toggleSection('music')}
          >
            <span>Music</span>
            <span>{isExpanded['music'] ? '−' : '+'}</span>
          </div>
          {isExpanded['music'] && (
            <div className="intent-ir-section-content">
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">tempo_bias:</span>
                <span className="intent-ir-field-value">{currentFrame.music.tempo_bias.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">rhythmic_density:</span>
                <span className="intent-ir-field-value">{currentFrame.music.rhythmic_density.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">groove_strength:</span>
                <span className="intent-ir-field-value">{currentFrame.music.groove_strength.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">harmonic_tension:</span>
                <span className="intent-ir-field-value">{currentFrame.music.harmonic_tension.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">harmonic_motion:</span>
                <span className="intent-ir-field-value">{currentFrame.music.harmonic_motion.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">mode_preference:</span>
                <span className="intent-ir-field-value">{currentFrame.music.mode_preference}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">melodic_activity:</span>
                <span className="intent-ir-field-value">{currentFrame.music.melodic_activity.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">contour_variance:</span>
                <span className="intent-ir-field-value">{currentFrame.music.contour_variance.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">dynamic_range:</span>
                <span className="intent-ir-field-value">{currentFrame.music.dynamic_range.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">texture_density:</span>
                <span className="intent-ir-field-value">{currentFrame.music.texture_density.toFixed(3)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Time */}
        <div className="intent-ir-section">
          <div 
            className="intent-ir-section-header"
            onClick={() => toggleSection('time')}
          >
            <span>Time</span>
            <span>{isExpanded['time'] ? '−' : '+'}</span>
          </div>
          {isExpanded['time'] && (
            <div className="intent-ir-section-content">
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">start_bar:</span>
                <span className="intent-ir-field-value">{currentFrame.time.start_bar}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">end_bar:</span>
                <span className="intent-ir-field-value">{currentFrame.time.end_bar}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">fade_in_beats:</span>
                <span className="intent-ir-field-value">{currentFrame.time.fade_in_beats.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">fade_out_beats:</span>
                <span className="intent-ir-field-value">{currentFrame.time.fade_out_beats.toFixed(3)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Constraints */}
        <div className="intent-ir-section">
          <div 
            className="intent-ir-section-header"
            onClick={() => toggleSection('constraints')}
          >
            <span>Constraints</span>
            <span>{isExpanded['constraints'] ? '−' : '+'}</span>
          </div>
          {isExpanded['constraints'] && (
            <div className="intent-ir-section-content">
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">allowed_engines_mask:</span>
                <span className="intent-ir-field-value">0x{currentFrame.constraints.allowed_engines_mask.toString(16)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">forbidden_engines_mask:</span>
                <span className="intent-ir-field-value">0x{currentFrame.constraints.forbidden_engines_mask.toString(16)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">max_cpu_cost:</span>
                <span className="intent-ir-field-value">{currentFrame.constraints.max_cpu_cost.toFixed(3)}</span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">max_event_rate:</span>
                <span className="intent-ir-field-value">{currentFrame.constraints.max_event_rate === Infinity ? '∞' : currentFrame.constraints.max_event_rate.toFixed(3)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Provenance */}
        <div className="intent-ir-section">
          <div 
            className="intent-ir-section-header"
            onClick={() => toggleSection('provenance')}
          >
            <span>Provenance</span>
            <span>{isExpanded['provenance'] ? '−' : '+'}</span>
          </div>
          {isExpanded['provenance'] && (
            <div className="intent-ir-section-content">
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">source:</span>
                <span className="intent-ir-field-value" style={{ color: sourceColor }}>
                  {sourceName}
                </span>
              </div>
              <div className="intent-ir-field">
                <span className="intent-ir-field-name">user_override_weight:</span>
                <span className="intent-ir-field-value">{currentFrame.provenance.user_override_weight.toFixed(3)}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
