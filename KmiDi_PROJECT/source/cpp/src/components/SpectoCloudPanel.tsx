import React, { useState } from "react";
import { useMusicBrain, SpectocloudRenderRequest, SpectocloudRenderResponse } from "../hooks/useMusicBrain";

const presets: Record<string, Partial<SpectocloudRenderRequest>> = {
  preview: { anchor_density: "sparse", n_particles: 600, fps: 8 },
  standard: { anchor_density: "normal", n_particles: 1200, fps: 15 },
  high: { anchor_density: "dense", n_particles: 1800, fps: 24 },
};

export function SpectoCloudPanel() {
  const { renderSpectocloud, getHumanizerConfig } = useMusicBrain();
  const [mode, setMode] = useState<"static" | "animation">("static");
  const [preset, setPreset] = useState<string>("standard");
  const [fps, setFps] = useState<number>(15);
  const [rotate, setRotate] = useState<boolean>(true);
  const [anchorDensity, setAnchorDensity] = useState<string>("normal");
  const [particles, setParticles] = useState<number>(1200);
  const [duration, setDuration] = useState<number>(1.0);
  const [frameIdx, setFrameIdx] = useState<number>(0);
  const [midiEventsJson, setMidiEventsJson] = useState<string>(
    '[{"time":0,"type":"note_on","note":60,"velocity":90}]'
  );
  const [midiFilePath, setMidiFilePath] = useState<string>("");
  const [output, setOutput] = useState<SpectocloudRenderResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [humanizerConfig, setHumanizerConfig] = useState<any>(null);
  const [uploadName, setUploadName] = useState<string>("");

  const applyPreset = (name: string) => {
    setPreset(name);
    const p = presets[name];
    if (p.fps) setFps(p.fps);
    if (p.anchor_density) setAnchorDensity(p.anchor_density);
    if (p.n_particles) setParticles(p.n_particles);
  };

  const loadConfig = async () => {
    try {
      const cfg = await getHumanizerConfig();
      setHumanizerConfig(cfg);
    } catch (e: any) {
      setError(e.message || String(e));
    }
  };

  const render = async () => {
    setError("");
    setOutput(null);
    setLoading(true);
    try {
      const payload: SpectocloudRenderRequest = {
        mode,
        fps,
        rotate,
        anchor_density: anchorDensity,
        n_particles: particles,
        duration,
        frame_idx: frameIdx,
      };
      if (midiFilePath.trim()) {
        payload.midi_file_path = midiFilePath.trim();
      } else {
        const events = JSON.parse(midiEventsJson);
        payload.midi_events = events;
      }
      const resp = await renderSpectocloud(payload);
      setOutput(resp);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadName(file.name);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = reader.result?.toString() || "[]";
        JSON.parse(text); // validate
        setMidiEventsJson(text);
        setMidiFilePath("");
        setError("");
      } catch (err: any) {
        setError(`Invalid JSON in uploaded file: ${err.message || err}`);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 16, marginTop: 16, background: "#fafafa" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h3>SpectoCloud Render</h3>
          <p style={{ margin: 0 }}>Choose preset, set mode, and render static or animation.</p>
        </div>
        <button onClick={loadConfig} style={{ padding: "6px 10px" }}>Load Humanizer Config</button>
      </div>

      {humanizerConfig && (
        <div style={{ background: "#f5f5f5", padding: 8, borderRadius: 6, marginTop: 8, fontSize: "0.9em" }}>
          <strong>Humanizer</strong>: style {humanizerConfig.default_style}, ppq {humanizerConfig.ppq}, bpm {humanizerConfig.bpm}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, marginTop: 12 }}>
        <div>
          <label>Preset</label>
          <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
            {Object.keys(presets).map((p) => (
              <button
                key={p}
                onClick={() => applyPreset(p)}
                style={{ padding: "6px 10px", background: preset === p ? "#444" : "#eee", color: preset === p ? "#fff" : "#000" }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label>Mode</label>
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="static">Static</option>
            <option value="animation">Animation</option>
          </select>
        </div>
        <div>
          <label>FPS</label>
          <input type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} />
        </div>
        <div>
          <label>Rotate</label>
          <input type="checkbox" checked={rotate} onChange={(e) => setRotate(e.target.checked)} />
        </div>
        <div>
          <label>Anchor Density</label>
          <select value={anchorDensity} onChange={(e) => setAnchorDensity(e.target.value)}>
            <option value="sparse">sparse</option>
            <option value="normal">normal</option>
            <option value="dense">dense</option>
          </select>
        </div>
        <div>
          <label>Particles</label>
          <input type="number" value={particles} onChange={(e) => setParticles(Number(e.target.value))} />
        </div>
        <div>
          <label>Duration (s)</label>
          <input type="number" step="0.1" value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
        </div>
        {mode === "static" && (
          <div>
            <label>Frame Index</label>
            <input type="number" value={frameIdx} onChange={(e) => setFrameIdx(Number(e.target.value))} />
          </div>
        )}
      </div>

      <div style={{ marginTop: 12 }}>
        <label>MIDI events (JSON array)</label>
        <textarea
          rows={4}
          style={{ width: "100%" }}
          value={midiEventsJson}
          onChange={(e) => setMidiEventsJson(e.target.value)}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
          <label style={{ margin: 0 }}>Upload JSON events:</label>
          <input type="file" accept=".json" onChange={handleUpload} />
          {uploadName && <span style={{ fontSize: "0.9em", color: "#555" }}>{uploadName}</span>}
        </div>
        <div style={{ marginTop: 6 }}>or provide MIDI file path:</div>
        <input
          type="text"
          placeholder="/abs/path/to/file.mid"
          style={{ width: "100%" }}
          value={midiFilePath}
          onChange={(e) => setMidiFilePath(e.target.value)}
        />
      </div>

      {error && (
        <div style={{ marginTop: 10, padding: 8, background: "#fee", border: "1px solid #f88" }}>
          {error}
        </div>
      )}

      <button onClick={render} disabled={loading} style={{ marginTop: 12, padding: "8px 12px" }}>
        {loading ? "Rendering..." : mode === "static" ? "Render Static" : "Render Animation"}
      </button>

      {output && (
        <div style={{ marginTop: 12, padding: 8, background: "#f0f7ff", border: "1px solid #c3e0ff" }}>
          <div><strong>Output</strong></div>
          <div>Mode: {output.mode}</div>
          <div>Frames: {output.frames}</div>
          <div>Path: <code>{output.output_path}</code></div>
        </div>
      )}
    </div>
  );
}

export default SpectoCloudPanel;
