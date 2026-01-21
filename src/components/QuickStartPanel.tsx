import { useState } from "react";

export type QuickStartTemplate = {
  id: string;
  name: string;
  description: string;
  genre?: string;
  emotion?: string;
  icon: string;
  config: {
    key?: string;
    bpm?: number;
    progression?: string[];
    style?: string;
  };
};

const quickStartTemplates: QuickStartTemplate[] = [
  {
    id: "lo-fi-chill",
    name: "Lo-Fi Chill",
    description: "Relaxed, nostalgic vibes perfect for studying or background music",
    genre: "lo-fi",
    emotion: "calm",
    icon: "🌙",
    config: {
      key: "C major",
      bpm: 80,
      progression: ["C", "Am", "F", "G"],
      style: "lo-fi bedroom"
    }
  },
  {
    id: "upbeat-pop",
    name: "Upbeat Pop",
    description: "Energetic and catchy, great for commercial projects",
    genre: "pop",
    emotion: "happy",
    icon: "✨",
    config: {
      key: "G major",
      bpm: 120,
      progression: ["G", "D", "Em", "C"],
      style: "commercial pop"
    }
  },
  {
    id: "emotional-ballad",
    name: "Emotional Ballad",
    description: "Slow, heartfelt melodies for emotional expression",
    genre: "ballad",
    emotion: "sad",
    icon: "💔",
    config: {
      key: "D minor",
      bpm: 70,
      progression: ["Dm", "Bb", "F", "C"],
      style: "emotional ballad"
    }
  },
  {
    id: "electronic-dance",
    name: "Electronic Dance",
    description: "High-energy electronic music for clubs and parties",
    genre: "edm",
    emotion: "excited",
    icon: "🎧",
    config: {
      key: "A minor",
      bpm: 128,
      progression: ["Am", "F", "C", "G"],
      style: "electronic dance"
    }
  },
  {
    id: "acoustic-folk",
    name: "Acoustic Folk",
    description: "Warm, organic sounds with acoustic instruments",
    genre: "folk",
    emotion: "peaceful",
    icon: "🎸",
    config: {
      key: "G major",
      bpm: 90,
      progression: ["G", "C", "D", "Em"],
      style: "acoustic folk"
    }
  },
  {
    id: "cinematic-epic",
    name: "Cinematic Epic",
    description: "Dramatic, orchestral arrangements for film and media",
    genre: "cinematic",
    emotion: "awe",
    icon: "🎬",
    config: {
      key: "E minor",
      bpm: 100,
      progression: ["Em", "C", "G", "D"],
      style: "cinematic orchestral"
    }
  }
];

type Props = {
  onTemplateSelect?: (template: QuickStartTemplate) => void;
  onGenerateWithTemplate?: (template: QuickStartTemplate) => void;
};

export function QuickStartPanel({ onTemplateSelect, onGenerateWithTemplate }: Props) {
  const [selectedTemplate, setSelectedTemplate] = useState<QuickStartTemplate | null>(null);

  const handleTemplateClick = (template: QuickStartTemplate) => {
    setSelectedTemplate(template);
    if (onTemplateSelect) {
      onTemplateSelect(template);
    }
  };

  const handleUseTemplate = () => {
    if (selectedTemplate) {
      if (onTemplateSelect) {
        onTemplateSelect(selectedTemplate);
      }
      if (onGenerateWithTemplate) {
        onGenerateWithTemplate(selectedTemplate);
      }
    }
  };

  return (
    <div className="quickstart-panel">
      <div className="section-header">
        <div className="section-header-inline">
          <h2>Quick Start</h2>
          <span className="section-badge">Templates</span>
        </div>
        <p className="section-description">
          Choose a ready-made template to get started quickly. Each template includes 
          pre-configured settings optimized for different genres and moods.
        </p>
      </div>

      <div className="template-grid">
        {quickStartTemplates.map((template) => (
          <div
            key={template.id}
            className={`template-card ${selectedTemplate?.id === template.id ? 'template-card-selected' : ''}`}
            onClick={() => handleTemplateClick(template)}
          >
            <div className="template-icon">{template.icon}</div>
            <div className="template-content">
              <h3 className="template-name">{template.name}</h3>
              <p className="template-description">{template.description}</p>
              <div className="template-tags">
                {template.genre && (
                  <span className="template-tag">{template.genre}</span>
                )}
                {template.emotion && (
                  <span className="template-tag">{template.emotion}</span>
                )}
              </div>
            </div>
            {selectedTemplate?.id === template.id && (
              <div className="template-check">✓</div>
            )}
          </div>
        ))}
      </div>

      {selectedTemplate && (
        <div className="template-preview-card">
          <div className="template-preview-header">
            <div>
              <h3>Selected Template: {selectedTemplate.name}</h3>
              <p className="template-preview-description">{selectedTemplate.description}</p>
            </div>
            <button
              onClick={() => setSelectedTemplate(null)}
              className="template-clear-btn"
            >
              ✕
            </button>
          </div>
          <div className="template-config-display">
            <div className="config-item">
              <span className="config-label">Key:</span>
              <span className="config-value">{selectedTemplate.config.key}</span>
            </div>
            <div className="config-item">
              <span className="config-label">BPM:</span>
              <span className="config-value">{selectedTemplate.config.bpm}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Progression:</span>
              <span className="config-value">{selectedTemplate.config.progression?.join(" - ")}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Style:</span>
              <span className="config-value">{selectedTemplate.config.style}</span>
            </div>
          </div>
          <button
            onClick={handleUseTemplate}
            className="primary-action-btn"
          >
            Create Music with This Template
          </button>
          <p className="action-hint">
            This will automatically generate music using the template settings. 
            You can always adjust the emotion or lyrics to personalize it.
          </p>
        </div>
      )}

      <div className="quickstart-actions">
        <div className="quickstart-action-card">
          <div className="action-icon">🎵</div>
          <div className="action-content">
            <h4>Start from Emotion</h4>
            <p>Use the Emotion Wheel above to create music based on how you feel</p>
          </div>
        </div>
        <div className="quickstart-action-card">
          <div className="action-icon">📝</div>
          <div className="action-content">
            <h4>Start from Lyrics</h4>
            <p>Write or paste your lyrics to guide the musical generation</p>
          </div>
        </div>
        <div className="quickstart-action-card">
          <div className="action-icon">💬</div>
          <div className="action-content">
            <h4>Start from Conversation</h4>
            <p>Describe what you want and let the Interrogator help refine your idea</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default QuickStartPanel;
