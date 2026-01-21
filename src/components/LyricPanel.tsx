import { useEffect, useState } from "react";
import { LyricsState, LyricsUpdateResponse } from "../hooks/useMusicBrain";

type Props = {
  onSave: (lyrics: string) => Promise<LyricsUpdateResponse>;
  loadLyrics: () => Promise<LyricsState>;
};

const isTauri = () => typeof window !== "undefined" && Boolean((window as any).__TAURI__);

// Lazy load Tauri APIs only when needed (browser mode compatible)
const getTauriDialog = async () => {
  if (!isTauri()) return null;
  try {
    // Use dynamic import with error handling for browser mode
    const dialogModule = await import("@tauri-apps/api/dialog" as any).catch(() => null);
    return dialogModule?.open || null;
  } catch {
    return null;
  }
};

const getTauriFs = async () => {
  if (!isTauri()) return null;
  try {
    // Use dynamic import with error handling for browser mode
    const fsModule = await import("@tauri-apps/api/fs" as any).catch(() => null);
    return fsModule?.readTextFile || null;
  } catch {
    return null;
  }
};

const LyricPanel = ({ onSave, loadLyrics }: Props) => {
  const [lyrics, setLyrics] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [source, setSource] = useState<string>("none");
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      setError(null);
      const payload = await loadLyrics();
      if (payload?.lyrics) {
        setLyrics(payload.lyrics);
        setSource(payload.source || "user");
        setStatus("Loaded from backend");
      } else if (payload?.generated) {
        setLyrics(payload.generated);
        setSource("generated");
        setStatus("Loaded generated draft");
      } else {
        setLyrics("");
        setSource("none");
        setStatus("No lyrics stored");
      }
    } catch (err) {
      console.error("Failed to load lyrics", err);
      setError("Music Brain API not reachable for lyrics");
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const persistLyrics = async (text: string) => {
    try {
      const resp = await onSave(text);
      setStatus(`Saved (${resp.lines} lines)`);
      setSource(resp.source);
      setError(null);
    } catch (err) {
      console.error("Failed to persist lyrics", err);
      setError("Could not save lyrics to backend");
    }
  };

  const handleLoadFromFile = async () => {
    if (!isTauri()) {
      setError("File picker requires Tauri; paste lyrics instead.");
      return;
    }
    try {
      const open = await getTauriDialog();
      const readTextFile = await getTauriFs();
      
      if (!open || !readTextFile) {
        setError("Tauri APIs not available");
        return;
      }
      
      const selected = await open({
        title: "Load lyrics",
        multiple: false,
        filters: [{ name: "Lyrics", extensions: ["txt", "lrc"] }],
      });
      if (!selected || Array.isArray(selected)) {
        return;
      }
      const text = await readTextFile(selected);
      setLyrics(text);
      await persistLyrics(text);
    } catch (err) {
      console.error("Failed to load lyrics from file", err);
      setError("Could not read the selected file");
    }
  };

  const handleClear = async () => {
    setLyrics("");
    await persistLyrics("");
  };

  return (
    <div className="lyric-panel">
      <div className="lyric-panel__header">
        <div>
          <div className="section-header-inline">
            <h3>Lyrics</h3>
            <span className="section-badge">User Priority</span>
          </div>
          <p className="section-description">
            Your lyrics guide the musical generation. The system analyzes your words for emotional content, 
            stress patterns, and prosody to create music that matches your text. If no lyrics are provided, 
            the system will generate draft lyrics based on your selected emotion.
          </p>
        </div>
        <span className={`lyric-badge lyric-badge--${source || "none"}`}>
          Source: {source || "none"}
        </span>
      </div>

      {status && <div className="lyric-status">{status}</div>}
      {error && <div className="lyric-error">{error}</div>}

      <div className="lyric-actions">
        <button 
          onClick={handleLoadFromFile}
          className="lyric-action-btn"
          title="Load lyrics from a text or LRC file"
        >
          📄 Load from File
        </button>
        <button 
          onClick={() => persistLyrics(lyrics)}
          className="lyric-action-btn primary"
          title="Save your lyrics to the backend"
        >
          💾 Save Lyrics
        </button>
        <button 
          onClick={refresh}
          className="lyric-action-btn"
          title="Reload lyrics from the backend"
        >
          🔄 Refresh
        </button>
        <button 
          onClick={handleClear}
          className="lyric-action-btn"
          title="Clear all lyrics"
        >
          🗑️ Clear
        </button>
      </div>

      <textarea
        className="lyric-textarea"
        value={lyrics}
        onChange={(e) => setLyrics(e.target.value)}
        placeholder="Paste or type your lyrics here. Downbeats will align to stressed syllables."
      />
    </div>
  );
};

export default LyricPanel;
