import { useEffect, useState, useRef } from "react";
import { useMusicBrain } from "../hooks/useMusicBrain";

const LyricPanel = () => {
  const { setUserLyrics, getUserLyrics } = useMusicBrain();
  const [lyrics, setLyrics] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [source, setSource] = useState<string>("none");
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    try {
      setError(null);
      const payload = await getUserLyrics();
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
      const resp = await setUserLyrics(text);
      setStatus(`Saved (${resp.lines} lines)`);
      setSource(resp.source);
      setError(null);
    } catch (err) {
      console.error("Failed to persist lyrics", err);
      setError("Could not save lyrics to backend");
    }
  };

  const handleLoadFromFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      setLyrics(text);
      await persistLyrics(text);
    } catch (err) {
      console.error("Failed to read file", err);
      setError("Could not read the selected file");
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
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

      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.lrc"
        style={{ display: "none" }}
        onChange={handleFileSelected}
      />

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
