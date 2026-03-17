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
        setStatus("Loaded.");
      } else if (payload?.generated) {
        setLyrics(payload.generated);
        setSource("generated");
        setStatus("Generated draft loaded.");
      } else {
        setLyrics("");
        setSource("none");
        setStatus("No lyrics yet.");
      }
    } catch (err) {
      console.error("Failed to load lyrics", err);
      setError("Can't reach the studio. Check your connection.");
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const persistLyrics = async (text: string) => {
    try {
      const resp = await setUserLyrics(text);
      setStatus(`Saved. ${resp.lines} lines.`);
      setSource(resp.source);
      setError(null);
    } catch (err) {
      console.error("Failed to persist lyrics", err);
      setError("Couldn't save. Check your connection.");
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
      setError("Couldn't read that file.");
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
          <h3>Lyrics</h3>
          <p className="lyric-panel__subtitle">
            Your words first. Type or load lyrics; leave blank to use a generated draft.
          </p>
        </div>
        <span className={`lyric-badge lyric-badge--${source || "none"}`}>
          {source === 'user' ? 'You' : source === 'generated' ? 'Generated' : '—'}
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
        <button onClick={handleLoadFromFile}>Load file</button>
        <button onClick={() => persistLyrics(lyrics)}>Save</button>
        <button onClick={refresh}>Reload</button>
        <button onClick={handleClear}>Clear</button>
      </div>

      <textarea
        className="lyric-textarea"
        value={lyrics}
        onChange={(e) => setLyrics(e.target.value)}
        placeholder="Paste or type. Stressed syllables can align to downbeats."
      />
    </div>
  );
};

export default LyricPanel;
