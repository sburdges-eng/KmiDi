import { useState, useEffect } from "react";
import { useMusicBrain } from "./hooks/useMusicBrain";
import { EmotionWheel, SelectedEmotion } from "./components/EmotionWheel";
import QuickStartPanel, { QuickStartTemplate } from "./components/QuickStartPanel";
import MusicCustomizer from "./components/MusicCustomizer";
import SongStructureEditor, { SongSection } from "./components/SongStructureEditor";
import SpectoCloudPanel from "./components/SpectoCloudPanel";
import LyricPanel from "./components/LyricPanel";
import IntentBuilder from "../../../src/components/IntentBuilder";
import "./App.css";

function App() {
  console.log("App: Component rendering");
  
  const [sideA, setSideA] = useState(true);
  const [emotions, setEmotions] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [selectedEmotion, setSelectedEmotion] = useState<SelectedEmotion | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<QuickStartTemplate | null>(null);
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([]);
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);
  const [selectedQuickEmotion, setSelectedQuickEmotion] = useState<string | null>(null);
  const [lastGeneratedAudioPath, setLastGeneratedAudioPath] = useState<string | null>(null);
  const [songLength, setSongLength] = useState<number>(180); // 3 minutes default
  const [selectedSections, setSelectedSections] = useState<SongSection[]>([
    { id: "intro", name: "Intro", icon: "🎬", enabled: true, repetitions: 1, order: 1 },
    { id: "verse", name: "Verse", icon: "📝", enabled: true, repetitions: 2, order: 2 },
    { id: "chorus", name: "Chorus", icon: "🎵", enabled: true, repetitions: 2, order: 3 },
    { id: "bridge", name: "Bridge", icon: "🌉", enabled: false, repetitions: 1, order: 4 },
    { id: "solo", name: "Solo", icon: "🎸", enabled: false, repetitions: 1, order: 5 },
    { id: "outro", name: "Outro", icon: "🎯", enabled: true, repetitions: 1, order: 6 },
  ]);
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>([]);
  const [instrumentTechniques, setInstrumentTechniques] = useState<Record<string, string[]>>({});
  
  // Always call hooks unconditionally
  const musicBrain = useMusicBrain();
  const {
    getEmotions,
    generateMusic,
    interrogate,
    setUserLyrics,
    getUserLyrics,
  } = musicBrain;
  console.log("App: useMusicBrain hook initialized");
  
  // Initialize Tauri API check and Music Brain API status
  useEffect(() => {
    console.log("App: useEffect running, checking Tauri API");
    try {
      // Check if we're in Tauri environment
      if (typeof window !== 'undefined' && (window as any).__TAURI__) {
        console.log("App: Tauri API detected");
      } else {
        console.warn("App: Tauri API not detected - running in browser mode");
      }
    } catch (e) {
      console.error("App: Error checking Tauri API:", e);
    }

    // Check Music Brain API status
    const checkApiStatus = async () => {
      try {
        await getEmotions();
        setApiStatus('online');
        console.log("App: Music Brain API is online");
      } catch (error) {
        setApiStatus('offline');
        console.warn("App: Music Brain API is offline:", error);
      }
    };
    
    checkApiStatus();
    // Check API status every 30 seconds
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, [getEmotions]);

  const toggleSide = () => {
    setSideA(!sideA);
  };

  const handleGetEmotions = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getEmotions();
      setEmotions(result);
      setApiStatus('online');
      console.log('Emotions loaded:', result);
    } catch (error) {
      console.error('Error loading emotions:', error);
      setApiStatus('offline');
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (errorMsg.includes('Connection') || errorMsg.includes('Failed to fetch') || errorMsg.includes('network')) {
        setError('Music Brain API is not running. Start it with: python -m music_brain.api');
      } else {
        setError(`Failed to load emotions: ${errorMsg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateMusic = async () => {
    console.log('Generate Music button clicked');
    setLoading(true);
    setError(null);
    try {
      console.log('Building request with:', {
        selectedEmotion,
        selectedQuickEmotion,
        selectedTemplate,
        selectedGenre,
        selectedTechniques,
        songLength,
        selectedSections: selectedSections.filter(s => s.enabled),
        selectedInstruments
      });
      // Build emotional intent from selected emotion, quick emotion, template, or use default
      let emotionalIntent = "grief hidden as love";
      let technicalConfig = {
        key: "F major",
        bpm: 82,
        progression: ["F", "C", "Am", "Dm"],
        genre: "lo-fi bedroom emo"
      };

      if (selectedEmotion) {
        emotionalIntent = `${selectedEmotion.base} (${selectedEmotion.intensity}: ${selectedEmotion.sub})`;
      } else if (selectedQuickEmotion) {
        emotionalIntent = selectedQuickEmotion;
      } else if (selectedTemplate) {
        emotionalIntent = selectedTemplate.emotion || selectedTemplate.name.toLowerCase();
      }

      // Apply genre selection
      if (selectedGenre) {
        technicalConfig.genre = selectedGenre;
      } else if (selectedTemplate) {
        technicalConfig.genre = selectedTemplate.config.style || technicalConfig.genre;
      }

      // Apply template configuration
      if (selectedTemplate) {
        technicalConfig = {
          key: selectedTemplate.config.key || technicalConfig.key,
          bpm: selectedTemplate.config.bpm || technicalConfig.bpm,
          progression: selectedTemplate.config.progression || technicalConfig.progression,
          genre: technicalConfig.genre
        };
      }

      // Add techniques to the intent
      const techniquesStr = selectedTechniques.length > 0 
        ? ` with ${selectedTechniques.join(", ")}`
        : "";
      
      // Build structure configuration - only include sections that are enabled and have repetitions > 0
      const enabledSections = selectedSections.filter(s => s.enabled && (s.repetitions || 0) > 0);
      const structure = enabledSections.map(s => ({
        type: s.id,
        repetitions: s.repetitions || 1
      }));

      const requestPayload = {
        intent: {
          emotional_intent: emotionalIntent + techniquesStr,
          technical: {
            ...technicalConfig,
            techniques: selectedTechniques,
            duration: songLength,
            structure: structure,
            instruments: selectedInstruments.map(id => ({
              id,
              techniques: instrumentTechniques[id] || []
            }))
          }
        },
        output_format: 'wav' // Request audio output
      };
      
      console.log('Calling generateMusic with payload:', JSON.stringify(requestPayload, null, 2));
      
      const result = await generateMusic(requestPayload);
      setApiStatus('online');
      console.log('Music generated successfully! Result:', result);
      
      // Extract audio file path from result if available
      if (result && typeof result === 'object') {
        const audioPath = (result as any).audio_path || 
                         (result as any).output_path || 
                         (result as any).file_path ||
                         (result as any).midi_path ||
                         ((result as any).result && (result as any).result.midi_path);
        if (audioPath) {
          setLastGeneratedAudioPath(audioPath);
          console.log('Audio file path captured:', audioPath);
        } else {
          console.warn('No audio path found in result:', result);
        }
      }
      
      alert('Music generated! Check the console for details. You can now visualize it below.');
    } catch (error) {
      console.error('Error generating music:', error);
      console.error('Error details:', {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        error
      });
      setApiStatus('offline');
      const errorMsg = error instanceof Error ? error.message : String(error);
      
      // Show detailed error to user
      let userErrorMessage = `Failed to generate music: ${errorMsg}`;
      if (errorMsg.includes('Connection') || errorMsg.includes('Failed to fetch') || errorMsg.includes('network') || errorMsg.includes('ECONNREFUSED')) {
        userErrorMessage = 'Music Brain API is not running. Start it with: python -m music_brain.api';
      } else if (errorMsg.includes('Tauri not available')) {
        userErrorMessage = 'Running in browser mode. Some features require the Tauri app.';
      }
      
      setError(userErrorMessage);
      alert(`Error: ${userErrorMessage}\n\nCheck the browser console (F12) for more details.`);
    } finally {
      setLoading(false);
      console.log('Generate music completed, loading set to false');
    }
  };

  const handleInterrogate = async () => {
    setLoading(true);
    setError(null);
    try {
      // Build message with selected emotion context if available
      let message = "I want to write a song about loss";
      if (selectedEmotion) {
        message = `I want to write a song about ${selectedEmotion.base} (${selectedEmotion.intensity}: ${selectedEmotion.sub})`;
      }
      
      const result = await interrogate({
        message: message
      });
      setApiStatus('online');
      console.log('Interrogation response:', result);
      alert('Interrogation response received! Check console.');
    } catch (error) {
      console.error('Error interrogating:', error);
      setApiStatus('offline');
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (errorMsg.includes('Connection') || errorMsg.includes('Failed to fetch') || errorMsg.includes('network')) {
        setError('Music Brain API is not running. Start it with: python -m music_brain.api');
      } else {
        setError(`Failed to interrogate: ${errorMsg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {error && (
        <div className="p-3 bg-accent-error/10 border border-accent-error rounded m-3">
          <strong>Error:</strong> {error}
          <button onClick={() => setError(null)} className="ml-3 px-2 py-1 bg-bg-secondary text-text-primary rounded min-h-touch min-w-touch">
            Dismiss
          </button>
        </div>
      )}
      <div className="cassette-header">
        <h1>iDAW - Kelly Project</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div className={`flex items-center gap-2 px-3 py-1 rounded border ${
            apiStatus === 'online' ? 'bg-status-online/10 border-status-online' :
            apiStatus === 'offline' ? 'bg-status-offline/10 border-status-offline' :
            'bg-status-checking/10 border-status-checking'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              apiStatus === 'online' ? 'bg-status-online' :
              apiStatus === 'offline' ? 'bg-status-offline' :
              'bg-status-checking animate-pulse'
            }`}></span>
            <span className="text-sm text-text-primary">
              API: {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}
            </span>
          </div>
          <button onClick={toggleSide} className="toggle-btn px-4 py-2 bg-accent-primary text-white rounded min-h-touch min-w-touch hover:bg-accent-secondary transition-colors">
            {sideA ? "⏭ Side B" : "⏮ Side A"}
          </button>
        </div>
      </div>

      {sideA ? (
        <div className="side-a">
          <div className="section-header">
            <h2>Side A: Professional DAW</h2>
            <p className="section-description">
              Full-featured digital audio workstation with mixer, timeline, and transport controls. 
              The professional production environment for mixing, mastering, and arrangement.
            </p>
          </div>
          <div className="coming-soon-card">
            <div className="coming-soon-icon">🎛️</div>
            <h3>Studio Tools</h3>
            <p>Multi-track recording, real-time effects processing, and professional mixing console coming soon.</p>
            <div className="feature-list">
              <div className="feature-item">✓ Multi-track timeline</div>
              <div className="feature-item">✓ Professional mixer</div>
              <div className="feature-item">✓ VST/AU plugin support</div>
              <div className="feature-item">✓ Real-time effects</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="side-b">
          <div className="section-header">
            <h2>Side B: Therapeutic Interface</h2>
            <p className="section-description">
              A creative space for emotional expression through music. Start with your feelings, 
              add your words, and let the system translate your intent into musical form.
            </p>
          </div>

<LyricPanel onSave={setUserLyrics} loadLyrics={getUserLyrics} />

          <div className="intent-builder-section" style={{ marginTop: 'var(--spacing-8)' }}>
            <div className="section-header-inline">
              <h3>Intent Builder</h3>
              <span className="section-badge">Schema</span>
            </div>
            <p className="section-description">
              Design your generation intent with strict API constraints. Core desire, structure, and instruments match the engine schema.
            </p>
            <IntentBuilder />
          </div>

          <div className="emotion-section">
            <div className="section-header-inline">
              <h3>Emotion Wheel</h3>
              <span className="section-badge">6×6×6 Grid</span>
            </div>
            <p className="section-description">
              Navigate through 216 emotional states across 6 base emotions, 6 intensity levels, 
              and 6 specific variations. Your selection guides the musical generation process.
            </p>
            <div className="action-card">
              <button 
                onClick={handleGetEmotions} 
                disabled={loading}
                className="primary-action-btn"
              >
                {loading ? "Loading Emotion Database..." : "Load Emotion Wheel"}
              </button>
              <p className="action-hint">
                Loads the complete emotion taxonomy. Once loaded, select your emotional state in three steps.
              </p>
            </div>
            {emotions && (
              <div style={{ marginTop: '20px' }}>
                <EmotionWheel emotions={emotions} onEmotionSelected={setSelectedEmotion} />
              </div>
            )}
          </div>

          <div className="ghostwriter-section">
            <div className="section-header-inline">
              <h3>GhostWriter</h3>
              <span className="section-badge">AI Composer</span>
            </div>
            <p className="section-description">
              Generates musical arrangements based on your selected emotion. The system creates 
              melodies, harmonies, and rhythms that express your chosen emotional state.
            </p>
            {selectedEmotion && (
              <div className="emotion-display-card">
                <div className="emotion-display-label">Current Selection:</div>
                <div className="emotion-display-value">
                  <span className="emotion-base">{selectedEmotion.base}</span>
                  <span className="emotion-arrow">→</span>
                  <span className="emotion-intensity">{selectedEmotion.intensity}</span>
                  <span className="emotion-arrow">→</span>
                  <span className="emotion-sub">{selectedEmotion.sub}</span>
                </div>
              </div>
            )}
            <div className="action-card">
              <button 
                onClick={handleGenerateMusic} 
                disabled={loading}
                className="primary-action-btn"
              >
                {loading 
                  ? "Composing..." 
                  : "Generate Musical Arrangement"}
              </button>
              <p className="action-hint">
                {selectedEmotion 
                  ? "Creates a complete musical piece expressing your selected emotion."
                  : selectedQuickEmotion
                    ? `Creates music with ${selectedQuickEmotion} mood.`
                    : selectedTemplate
                      ? `Creates music using the ${selectedTemplate.name} template.`
                      : "Uses your customizations above to generate music. Select emotion, template, or customize settings."}
              </p>
            </div>
          </div>

          <div className="interrogator-section">
            <div className="section-header-inline">
              <h3>Interrogator</h3>
              <span className="section-badge">Conversational</span>
            </div>
            <p className="section-description">
              Have a conversation about your musical intent. Describe what you want to express, 
              and the system will help refine your creative direction through dialogue.
            </p>
            <div className="action-card">
              <button 
                onClick={handleInterrogate} 
                disabled={loading}
                className="primary-action-btn"
              >
                {loading ? "Processing..." : "Start Conversation"}
              </button>
              <p className="action-hint">
                Opens a dialogue to explore your musical ideas and refine the creative direction.
              </p>
            </div>
          </div>
        </div>
      )}

      <MusicCustomizer
        selectedTechniques={selectedTechniques}
        selectedGenre={selectedGenre}
        selectedEmotion={selectedQuickEmotion}
        onTechniquesChange={setSelectedTechniques}
        onGenreChange={setSelectedGenre}
        onEmotionChange={setSelectedQuickEmotion}
      />

      <SongStructureEditor
        songLength={songLength}
        selectedSections={selectedSections}
        selectedInstruments={selectedInstruments}
        instrumentTechniques={instrumentTechniques}
        onSongLengthChange={setSongLength}
        onSectionsChange={setSelectedSections}
        onInstrumentsChange={setSelectedInstruments}
        onInstrumentTechniquesChange={(instrumentId, techniques) => {
          setInstrumentTechniques(prev => ({
            ...prev,
            [instrumentId]: techniques
          }));
        }}
      />

      <div className="generate-section">
        <div className="section-header">
          <div className="section-header-inline">
            <h2>Generate Your Music</h2>
            <span className="section-badge">Ready</span>
          </div>
          <p className="section-description">
            Review your selections above, then click the button below to generate your music. 
            All your customizations will be applied automatically.
          </p>
        </div>
        <div className="generate-action-card">
          <button 
            onClick={handleGenerateMusic} 
            disabled={loading}
            className="generate-main-btn"
          >
            {loading ? "Generating Your Music..." : "🎵 Generate Music"}
          </button>
          <div className="generate-summary">
            {selectedGenre && (
              <span className="generate-tag">Genre: {selectedGenre}</span>
            )}
            {(selectedEmotion || selectedQuickEmotion) && (
              <span className="generate-tag">
                Emotion: {selectedEmotion 
                  ? `${selectedEmotion.base} (${selectedEmotion.intensity}: ${selectedEmotion.sub})`
                  : selectedQuickEmotion}
              </span>
            )}
            {selectedTemplate && (
              <span className="generate-tag">Template: {selectedTemplate.name}</span>
            )}
            {songLength && (
              <span className="generate-tag">Length: {Math.round(songLength / 60)}:{String(songLength % 60).padStart(2, '0')}</span>
            )}
            {selectedInstruments.length > 0 && (
              <span className="generate-tag">{selectedInstruments.length} Instrument(s)</span>
            )}
          </div>
        </div>
      </div>

      <QuickStartPanel 
        onTemplateSelect={setSelectedTemplate}
        onGenerateWithTemplate={async (template) => {
          setSelectedTemplate(template);
          setSelectedEmotion(null); // Clear emotion to use template
          // Trigger music generation with template
          await handleGenerateMusic();
        }}
      />

      <SpectoCloudPanel lastGeneratedAudioPath={lastGeneratedAudioPath} />
    </div>
  );
}

export default App;
