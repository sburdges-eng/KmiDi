import { useState } from "react";

export type SongSection = {
  id: string;
  name: string;
  icon: string;
  enabled: boolean;
  repetitions?: number;
  order?: number;
};

export type Instrument = {
  id: string;
  name: string;
  icon: string;
  category: string;
  techniques: string[];
};

const defaultSections: SongSection[] = [
  { id: "intro", name: "Intro", icon: "🎬", enabled: true, repetitions: 1, order: 1 },
  { id: "verse", name: "Verse", icon: "📝", enabled: true, repetitions: 2, order: 2 },
  { id: "chorus", name: "Chorus", icon: "🎵", enabled: true, repetitions: 2, order: 3 },
  { id: "bridge", name: "Bridge", icon: "🌉", enabled: false, repetitions: 1, order: 4 },
  { id: "solo", name: "Solo", icon: "🎸", enabled: false, repetitions: 1, order: 5 },
  { id: "outro", name: "Outro", icon: "🎯", enabled: true, repetitions: 1, order: 6 },
];

const instruments: Instrument[] = [
  {
    id: "drums",
    name: "Drums",
    icon: "🥁",
    category: "Rhythm",
    techniques: [
      "Ghost Notes",
      "Fills",
      "Syncopation",
      "Dynamics",
      "Hi-Hat Patterns",
      "Kick Variations",
      "Crash Patterns",
      "Ride Patterns",
      "Snare Rolls",
      "Double Bass",
      "Toms",
      "Cymbal Chokes"
    ]
  },
  {
    id: "bass",
    name: "Bass",
    icon: "🎸",
    category: "Rhythm",
    techniques: [
      "Slap",
      "Fingerstyle",
      "Muted",
      "Harmonics",
      "Glissando",
      "Palm Mute",
      "Pick",
      "Tapping",
      "Pop",
      "Thumb",
      "Double Stops",
      "Octave Jumps"
    ]
  },
  {
    id: "guitar",
    name: "Guitar",
    icon: "🎸",
    category: "Melody",
    techniques: [
      "Strumming",
      "Picking",
      "Bend",
      "Vibrato",
      "Slide",
      "Palm Mute",
      "Harmonics",
      "Arpeggio",
      "Hammer-On",
      "Pull-Off",
      "Tapping",
      "Fingerpicking",
      "Chords",
      "Power Chords",
      "Barre Chords"
    ]
  },
  {
    id: "piano",
    name: "Piano",
    icon: "🎹",
    category: "Harmony",
    techniques: [
      "Arpeggio",
      "Chords",
      "Staccato",
      "Legato",
      "Glissando",
      "Pedal",
      "Octaves",
      "Trills",
      "Tremolo",
      "Broken Chords",
      "Block Chords",
      "Bass Notes",
      "Melody in Right Hand"
    ]
  },
  {
    id: "strings",
    name: "Strings",
    icon: "🎻",
    category: "Harmony",
    techniques: [
      "Pizzicato",
      "Arco",
      "Vibrato",
      "Tremolo",
      "Harmonics",
      "Sul Ponticello",
      "Sul Tasto",
      "Staccato",
      "Legato",
      "Portamento",
      "Double Stops",
      "Trills",
      "Violin",
      "Viola",
      "Cello",
      "Double Bass"
    ]
  },
  {
    id: "synth",
    name: "Synth",
    icon: "🎹",
    category: "Melody",
    techniques: [
      "Arpeggio",
      "Lead",
      "Pad",
      "Pluck",
      "Bass",
      "Modulation",
      "Filter Sweep",
      "Pulse Width",
      "LFO",
      "Envelope",
      "Sequencer",
      "Chords"
    ]
  },
  {
    id: "vocals",
    name: "Vocals",
    icon: "🎤",
    category: "Melody",
    techniques: [
      "Male Voice",
      "Female Voice",
      "Duet",
      "Choir",
      "Soprano",
      "Alto",
      "Tenor",
      "Bass",
      "Melisma",
      "Harmony",
      "Falsetto",
      "Whisper",
      "Belting",
      "Runs",
      "Vibrato",
      "Growl",
      "Fry"
    ]
  },
];

const MAX_SONG_LENGTH_SECONDS = 600; // 10 minutes - practical maximum for tokenization
const MIN_SONG_LENGTH_SECONDS = 30; // 30 seconds minimum

type Props = {
  songLength: number; // in seconds
  selectedSections: SongSection[];
  selectedInstruments: string[];
  instrumentTechniques: Record<string, string[]>; // instrumentId -> techniqueIds
  onSongLengthChange: (length: number) => void;
  onSectionsChange: (sections: SongSection[]) => void;
  onInstrumentsChange: (instruments: string[]) => void;
  onInstrumentTechniquesChange: (instrumentId: string, techniques: string[]) => void;
};

export function SongStructureEditor({
  songLength,
  selectedSections,
  selectedInstruments,
  instrumentTechniques,
  onSongLengthChange,
  onSectionsChange,
  onInstrumentsChange,
  onInstrumentTechniquesChange,
}: Props) {
  const [activeInstrument, setActiveInstrument] = useState<string | null>(null);

  const handleSectionToggle = (sectionId: string) => {
    const updated = selectedSections.map(section => {
      if (section.id === sectionId) {
        return { ...section, enabled: !section.enabled };
      }
      return section;
    });
    onSectionsChange(updated);
  };

  const handleRepetitionChange = (sectionId: string, delta: number) => {
    const updated = selectedSections.map(section => {
      if (section.id === sectionId) {
        const currentReps = section.repetitions || 1;
        const newReps = Math.max(0, currentReps + delta);
        // If set to 0, also disable the section
        if (newReps === 0) {
          return { ...section, repetitions: 0, enabled: false };
        }
        // If going from 0 to 1, enable it
        if (currentReps === 0 && newReps === 1) {
          return { ...section, repetitions: 1, enabled: true };
        }
        return { ...section, repetitions: newReps };
      }
      return section;
    });
    onSectionsChange(updated);
  };

  const handleInstrumentToggle = (instrumentId: string) => {
    if (selectedInstruments.includes(instrumentId)) {
      onInstrumentsChange(selectedInstruments.filter(id => id !== instrumentId));
      setActiveInstrument(null);
    } else {
      onInstrumentsChange([...selectedInstruments, instrumentId]);
      setActiveInstrument(instrumentId);
    }
  };

  const handleTechniqueToggle = (instrumentId: string, technique: string) => {
    const current = instrumentTechniques[instrumentId] || [];
    if (current.includes(technique)) {
      onInstrumentTechniquesChange(instrumentId, current.filter(t => t !== technique));
    } else {
      onInstrumentTechniquesChange(instrumentId, [...current, technique]);
    }
  };

  const calculateEstimatedLength = () => {
    // Rough estimate: each section type has base length
    const baseLengths: Record<string, number> = {
      intro: 8,
      verse: 16,
      chorus: 8,
      bridge: 16,
      solo: 16,
      outro: 8,
    };

    let total = 0;
    selectedSections.filter(s => s.enabled && (s.repetitions || 0) > 0).forEach(section => {
      const baseLength = baseLengths[section.id] || 8;
      total += baseLength * (section.repetitions || 1);
    });

    return Math.min(total, MAX_SONG_LENGTH_SECONDS);
  };

  return (
    <div className="song-structure-editor">
      <div className="section-header">
        <div className="section-header-inline">
          <h2>Song Structure & Instruments</h2>
          <span className="section-badge">Advanced</span>
        </div>
        <p className="section-description">
          Control the length, sections, and instruments in your song. Select instruments 
          and choose specific techniques for each.
        </p>
      </div>

      <div className="song-length-section">
        <div className="song-length-control">
          <label className="length-label">
            Song Length: <strong>{songLength} seconds</strong> ({Math.round(songLength / 60)}:{String(songLength % 60).padStart(2, '0')})
          </label>
          <div className="length-slider-container">
            <input
              type="range"
              min={MIN_SONG_LENGTH_SECONDS}
              max={MAX_SONG_LENGTH_SECONDS}
              step={5}
              value={songLength}
              onChange={(e) => onSongLengthChange(Number(e.target.value))}
              className="length-slider"
            />
            <div className="length-presets">
              <button
                onClick={() => onSongLengthChange(30)}
                className="length-preset-btn"
              >
                30s
              </button>
              <button
                onClick={() => onSongLengthChange(90)}
                className="length-preset-btn"
              >
                1:30
              </button>
              <button
                onClick={() => onSongLengthChange(180)}
                className="length-preset-btn"
              >
                3:00
              </button>
              <button
                onClick={() => onSongLengthChange(240)}
                className="length-preset-btn"
              >
                4:00
              </button>
            </div>
          </div>
          <p className="length-hint">
            Maximum: {MAX_SONG_LENGTH_SECONDS / 60} minutes (for optimal tokenization)
          </p>
        </div>
      </div>

      <div className="sections-section">
        <div className="customizer-header">
          <h3>Song Sections</h3>
          <p className="customizer-hint">Enable sections and set how many times they repeat</p>
        </div>
        <div className="sections-grid">
          {selectedSections
            .sort((a, b) => (a.order || 0) - (b.order || 0))
            .map((section) => (
              <div
                key={section.id}
                className={`section-card ${section.enabled ? 'section-card-enabled' : ''}`}
              >
                <div className="section-card-header">
                  <button
                    onClick={() => handleSectionToggle(section.id)}
                    className={`section-toggle ${section.enabled ? 'section-toggle-enabled' : ''}`}
                  >
                    <span className="section-icon">{section.icon}</span>
                    <span className="section-name">{section.name}</span>
                  </button>
                </div>
                {section.enabled && (
                  <div className="section-controls">
                    <label className="repetition-label">Repetitions:</label>
                    <div className="repetition-controls">
                      <button
                        onClick={() => handleRepetitionChange(section.id, -1)}
                        className="repetition-btn"
                        disabled={(section.repetitions || 1) <= 0}
                      >
                        −
                      </button>
                      <span className="repetition-count">{section.repetitions || (section.enabled ? 1 : 0)}</span>
                      <button
                        onClick={() => handleRepetitionChange(section.id, 1)}
                        className="repetition-btn"
                      >
                        +
                      </button>
                      {(section.repetitions || 1) === 0 && (
                        <span className="repetition-zero-hint">(Disabled)</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
        </div>
      </div>

      <div className="instruments-section">
        <div className="customizer-header">
          <h3>Instruments</h3>
          <p className="customizer-hint">Select instruments and choose techniques for each</p>
        </div>
        <div className="instruments-grid">
          {instruments.map((instrument) => {
            const isSelected = selectedInstruments.includes(instrument.id);
            const isActive = activeInstrument === instrument.id;
            const techniques = instrumentTechniques[instrument.id] || [];

            return (
              <div key={instrument.id} className="instrument-wrapper">
                <button
                  onClick={() => {
                    handleInstrumentToggle(instrument.id);
                    if (!isSelected) {
                      setActiveInstrument(instrument.id);
                    } else if (isActive) {
                      setActiveInstrument(null);
                    } else {
                      setActiveInstrument(instrument.id);
                    }
                  }}
                  className={`instrument-btn ${isSelected ? 'instrument-btn-selected' : ''} ${isActive ? 'instrument-btn-active' : ''}`}
                >
                  <span className="instrument-icon">{instrument.icon}</span>
                  <span className="instrument-name">{instrument.name}</span>
                  {isSelected && (
                    <span className="instrument-check">✓</span>
                  )}
                </button>
                {isActive && isSelected && (
                  <div className="techniques-panel">
                    <div className="techniques-header">
                      <strong>{instrument.name} Techniques</strong>
                    </div>
                    <div className="techniques-list">
                      {instrument.techniques.map((technique) => (
                        <button
                          key={technique}
                          onClick={() => handleTechniqueToggle(instrument.id, technique)}
                          className={`technique-item-btn ${techniques.includes(technique) ? 'technique-item-btn-selected' : ''}`}
                        >
                          {technique}
                          {techniques.includes(technique) && <span className="technique-check">✓</span>}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {selectedInstruments.length > 0 && (
        <div className="instrument-summary">
          <div className="summary-header">Selected Instruments & Techniques:</div>
          <div className="instrument-summary-list">
            {selectedInstruments.map(instrumentId => {
              const instrument = instruments.find(i => i.id === instrumentId);
              const techniques = instrumentTechniques[instrumentId] || [];
              if (!instrument) return null;

              return (
                <div key={instrumentId} className="instrument-summary-item">
                  <span className="summary-instrument">
                    {instrument.icon} {instrument.name}
                  </span>
                  {techniques.length > 0 && (
                    <div className="summary-techniques">
                      {techniques.map(tech => (
                        <span key={tech} className="summary-technique-tag">{tech}</span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default SongStructureEditor;
