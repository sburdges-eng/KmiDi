import './KmidiShell.css';

export default function KmidiShell() {
  return (
    <div className="kmidi-app">
      <header className="top-bar">
        <div className="transport-controls">
          <button>Play</button>
          <button>Stop</button>
          <button>Record</button>
        </div>

        <div className="mixer">
          <div className="mixer-track">
            <span className="track-label">Kick</span>
            <div className="fader-vertical" />
            <span className="track-value">-3.7</span>
          </div>
          <div className="mixer-track">
            <span className="track-label">Snare</span>
            <div className="fader-vertical" />
            <span className="track-value">-5.2</span>
          </div>
        </div>
      </header>

      <main className="workspace">
        <section className="panel mood-panel">
          <h2>Mood</h2>
          <div className="mood-wheel" />
        </section>

        <section className="center-column">
          <div className="panel song-details">
            <h2>Song Details</h2>
            <textarea className="input-field" placeholder="Describe your song..." />

            <div className="parameter-grid">
              <select className="input-field select-with-chevron">
                <option>Genre</option>
              </select>
              <select className="input-field select-with-chevron">
                <option>Key</option>
              </select>
            </div>
          </div>

          <button className="generate-btn">Generate</button>
        </section>

        <section className="panel arrangement-panel">
          <h2>Arrangement</h2>
          <div className="arrangement-blocks" />
          <h2>Instruments</h2>
          <div className="instrument-list" />
        </section>
      </main>
    </div>
  );
}
