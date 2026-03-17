import { useState } from "react";
import { useMusicBrain, SpectocloudRenderRequest, SpectocloudRenderResponse } from "../hooks/useMusicBrain";

const presets: Record<string, Partial<SpectocloudRenderRequest>> = {
  preview: { anchor_density: "sparse", n_particles: 600, fps: 8 },
  standard: { anchor_density: "normal", n_particles: 1200, fps: 15 },
  high: { anchor_density: "dense", n_particles: 1800, fps: 24 },
};

type Props = {
  lastGeneratedAudioPath?: string | null;
};

export function SpectoCloudPanel({ lastGeneratedAudioPath }: Props) {
  const { renderSpectocloud } = useMusicBrain();
  const [mode, setMode] = useState<"static" | "animation">("static");
  const [preset, setPreset] = useState<string>("standard");
  const [fps, setFps] = useState<number>(15);
  const [rotate, setRotate] = useState<boolean>(true);
  const [anchorDensity, setAnchorDensity] = useState<string>("normal");
  const [particles, setParticles] = useState<number>(1200);
  const [duration, setDuration] = useState<number>(1.0);
  const [frameIdx, setFrameIdx] = useState<number>(0);
  const [output, setOutput] = useState<SpectocloudRenderResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const applyPreset = (name: string) => {
    setPreset(name);
    const p = presets[name];
    if (p.fps) setFps(p.fps);
    if (p.anchor_density) setAnchorDensity(p.anchor_density);
    if (p.n_particles) setParticles(p.n_particles);
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

      if (lastGeneratedAudioPath) {
        payload.audio_file_path = lastGeneratedAudioPath;
      } else {
        throw new Error("No piece yet. Generate one in Compose, then return here.");
      }

      const resp = await renderSpectocloud(payload);
      setOutput(resp);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="spectocloud-panel">
      <div className="spectocloud-header">
        <div>
          <div className="section-header-inline">
            <h3>Visualize</h3>
            <span className="section-badge">Particle Cloud</span>
          </div>
          <p className="section-description">
            Turn your piece into a visual. Share it or use it as art.
          </p>
        </div>
      </div>

      <div className="spectocloud-simple-controls">
        <div className="simple-control-group">
          <label className="simple-label">Quality:</label>
          <div className="preset-buttons-simple">
            {Object.keys(presets).map((p) => (
              <button
                key={p}
                onClick={() => applyPreset(p)}
                className={`preset-btn-simple ${preset === p ? 'preset-btn-simple-active' : ''}`}
              >
                {p === 'preview' ? 'Fast' : p === 'standard' ? 'Balanced' : 'High Quality'}
              </button>
            ))}
          </div>
        </div>
        <div className="simple-control-group">
          <label className="simple-label">Type:</label>
          <div className="mode-buttons-simple">
            <button
              onClick={() => setMode("static")}
              className={`mode-btn ${mode === "static" ? 'mode-btn-active' : ''}`}
            >
              📷 Image
            </button>
            <button
              onClick={() => setMode("animation")}
              className={`mode-btn ${mode === "animation" ? 'mode-btn-active' : ''}`}
            >
              🎬 Animation
            </button>
          </div>
        </div>
      </div>

      <div className="spectocloud-simple-section">
        <p className="simple-description">
          After you generate a piece, return here to turn it into a visual.
        </p>
        {lastGeneratedAudioPath ? (
          <div className="auto-detection-notice success">
            <span className="notice-icon">✓</span>
            <span>Ready. Build an image or animation below.</span>
          </div>
        ) : (
          <div className="auto-detection-notice warning">
            <span className="notice-icon">ℹ️</span>
            <span>Generate a piece first, then return here.</span>
          </div>
        )}
      </div>

      {error && (
        <div className="error-display-card">
          {error}
        </div>
      )}

      <div className="render-action-section">
        <button 
          onClick={render} 
          disabled={loading} 
          className="primary-action-btn"
        >
          {loading 
            ? "Building…" 
            : mode === "static" 
              ? "Build image" 
              : "Build animation"}
        </button>
        <p className="action-hint">
          {mode === "static" 
            ? "A still image of your piece."
            : "An animated take on your piece."}
        </p>
      </div>

      {output && (
        <div className="output-display-card">
          <div className="output-display-label">Done.</div>
          <div className="output-display-details">
            <div>Saved.</div>
            <div className="output-location">
              <span className="output-path-simple">{output.output_path}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default SpectoCloudPanel;
