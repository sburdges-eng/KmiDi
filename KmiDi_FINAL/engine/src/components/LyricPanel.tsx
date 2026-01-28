import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { LyricsState, LyricsUpdateResponse } from "../hooks/useMusicBrain";

type Props = {
  onSave: (lyrics: string) => Promise<LyricsUpdateResponse>;
  loadLyrics: () => Promise<LyricsState>;
};

const isTauri = () => typeof window !== "undefined" && Boolean((window as any).__TAURI__);

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
          <h3>Lyric Priority (User First)</h3>
          <p className="lyric-panel__subtitle">
            Load or type lyrics to drive intent and prosody. Empty lyrics fall back to generated drafts.
          </p>
        </div>
        <span className={`lyric-badge lyric-badge--${source || "none"}`}>
          Source: {source || "none"}
        </span>
      </div>

      {status && <div className="lyric-status">{status}</div>}
      {error && <div className="lyric-error">{error}</div>}

      <div className="lyric-actions">
        <button onClick={handleLoadFromFile}>Load .txt/.lrc</button>
        <button onClick={() => persistLyrics(lyrics)}>Save Lyrics</button>
        <button onClick={refresh}>Refresh</button>
        <button onClick={handleClear}>Clear</button>
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
