import { useState, useEffect } from "react";
import { useMusicBrain } from "./hooks/useMusicBrain";
import { EmotionWheel, SelectedEmotion } from "./components/EmotionWheel";
import GuideNav, { Guide } from "./components/GuideNav";
import GuideViewer from "./components/GuideViewer";
import SpectoCloudPanel from "./components/SpectoCloudPanel";
import LyricPanel from "./components/LyricPanel";
import "./App.css";

function App() {
  console.log("App: Component rendering");
  
  const [sideA, setSideA] = useState(true);
  const [emotions, setEmotions] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [selectedEmotion, setSelectedEmotion] = useState<SelectedEmotion | null>(null);
  const [selectedGuide, setSelectedGuide] = useState<Guide | null>(null);
  
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
    setLoading(true);
    setError(null);
    try {
      // Build emotional intent from selected emotion or use default
      const emotionalIntent = selectedEmotion 
        ? `${selectedEmotion.base} (${selectedEmotion.intensity}: ${selectedEmotion.sub})`
        : "grief hidden as love";
      
      const result = await generateMusic({
        intent: {
          emotional_intent: emotionalIntent,
          technical: {
            key: "F major",
            bpm: 82,
            progression: ["F", "C", "Am", "Dm"],
            genre: "lo-fi bedroom emo"
          }
        }
      });
      setApiStatus('online');
      console.log('Music generated:', result);
      alert('Music generated! Check console for details.');
    } catch (error) {
      console.error('Error generating music:', error);
      setApiStatus('offline');
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (errorMsg.includes('Connection') || errorMsg.includes('Failed to fetch') || errorMsg.includes('network')) {
        setError('Music Brain API is not running. Start it with: python -m music_brain.api');
      } else {
        setError(`Failed to generate music: ${errorMsg}`);
      }
    } finally {
      setLoading(false);
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
          <h2>Side A: Professional DAW</h2>
          <p>Mixer, Timeline, Transport controls coming soon...</p>
          <div className="test-buttons">
            <button onClick={handleGenerateMusic} disabled={loading}>
              {loading ? "Generating..." : "Test Generate Music"}
            </button>
          </div>
        </div>
      ) : (
        <div className="side-b">
          <h2>Side B: Therapeutic Interface</h2>

          <LyricPanel onSave={setUserLyrics} loadLyrics={getUserLyrics} />
          
          <div className="emotion-section">
            <h3>Emotion Wheel (6×6×6)</h3>
            <button onClick={handleGetEmotions} disabled={loading}>
              {loading ? "Loading..." : "Load Emotions"}
            </button>
            {emotions && (
              <div style={{ marginTop: '20px' }}>
                <EmotionWheel emotions={emotions} onEmotionSelected={setSelectedEmotion} />
              </div>
            )}
          </div>

          <div className="ghostwriter-section">
            <h3>GhostWriter</h3>
            {selectedEmotion && (
              <div className="mb-3 p-2 bg-accent-primary/10 rounded text-sm text-text-primary">
                Selected: {selectedEmotion.base} → {selectedEmotion.intensity} → {selectedEmotion.sub}
              </div>
            )}
            <button onClick={handleGenerateMusic} disabled={loading || !selectedEmotion}>
              {loading ? "Generating..." : selectedEmotion ? "Generate Music" : "Select an emotion first"}
            </button>
          </div>

          <div className="interrogator-section">
            <h3>Interrogator</h3>
            <button onClick={handleInterrogate} disabled={loading}>
              {loading ? "Interrogating..." : "Start Interrogation"}
            </button>
          </div>
        </div>
      )}

      <div className="docs-section">
        <div className="docs-header">
          <div>
            <h2>Production Workflow Guides</h2>
            <p className="docs-subtitle">
              Search, filter, preview, and open the curated production workflow docs.
            </p>
          </div>
          <span className="docs-badge">Local Markdown</span>
        </div>
        <div className="docs-layout">
          <GuideNav onSelect={setSelectedGuide} />
          <GuideViewer guide={selectedGuide} />
        </div>
      </div>

      <SpectoCloudPanel />
    </div>
  );
}

export default App;
