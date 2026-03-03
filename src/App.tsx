import { useEffect, useMemo, useRef, useState } from 'react';
import { Transport } from './components/SideA/Transport';
import { Mixer } from './components/SideA/Mixer';
import { Timeline } from './components/SideA/Timeline';
import { VUMeter } from './components/SideA/VUMeter';
import { EmotionWheel } from './components/SideB/EmotionWheel';
import { GhostWriter } from './components/SideB/GhostWriter';
import { Interrogator } from './components/SideB/Interrogator';

type Side = 'side-a' | 'side-b';

type Channel = {
  id: string;
  name: string;
  level: number;
  pan: number;
};

type SelectedEmotion = {
  base: string;
  intensity: string;
  detail: string;
};

export default function App() {
  const [side, setSide] = useState<Side>('side-a');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [tempo, setTempo] = useState(120);
  const [masterVu, setMasterVu] = useState(0);
  const playTickRef = useRef(0);
  const [channels, setChannels] = useState<Channel[]>([
    { id: 'kick', name: 'Kick', level: 0.65, pan: 0.5 },
    { id: 'snare', name: 'Snare', level: 0.55, pan: 0.5 },
    { id: 'bass', name: 'Bass', level: 0.72, pan: 0.5 },
    { id: 'pad', name: 'Pad', level: 0.42, pan: 0.5 },
  ]);
  const [selectedEmotion, setSelectedEmotion] = useState<SelectedEmotion | null>(null);
  const [ghostText, setGhostText] = useState('');
  const [interactions, setInteractions] = useState<string[]>([
    'KmiDi loaded. Ask it for an arrangement brief.',
  ]);

  const timelineBars = useMemo(() => Math.max(8, tempo > 140 ? 24 : tempo > 95 ? 16 : 12), [tempo]);

  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      playTickRef.current += 1;
      setMasterVu((prev) => {
        const wave =
          Math.sin((playTickRef.current) * 0.35) * 0.4 + Math.cos((playTickRef.current) * 0.18) * 0.2;
        const next = 0.35 + wave * 0.4;
        return Number(Math.min(0.94, Math.max(0.12, next)).toFixed(3));
      });

      setChannels((prev) =>
        prev.map((channel, index) => {
          const wave = Math.sin(playTickRef.current * 0.3 + index * 0.65);
          return {
            ...channel,
            level: Number((0.35 + Math.max(-0.18, Math.min(0.58, wave * 0.28))).toFixed(3)),
          };
        }),
      );
    }, 120);

    return () => {
      window.clearInterval(timer);
    };
  }, [isPlaying]);

  const activeTitle = side === 'side-a' ? 'Side A Studio' : 'Side B Studio';

  return (
    <div className="km-frame">
      <header className="km-header">
        <h1 className="app-title">KmiDi UI</h1>
        <div className="km-toggle" role="tablist" aria-label="Studio mode">
          <button
            type="button"
            className={side === 'side-a' ? 'tab active' : 'tab'}
            onClick={() => setSide('side-a')}
          >
            Side A
          </button>
          <button
            type="button"
            className={side === 'side-b' ? 'tab active' : 'tab'}
            onClick={() => setSide('side-b')}
          >
            Side B
          </button>
        </div>
      </header>

      <section className="km-titlebar">
        <p className="km-subtitle">Current mode: {activeTitle}</p>
        <button
          type="button"
          className="mode-reset-btn"
          onClick={() => setIsPlaying(false)}
        >
          Reset playback
        </button>
      </section>

      <main>
        {side === 'side-a' ? (
          <section className="km-side-grid" aria-label="Side A console">
            <article className="panel">
              <h2 className="panel-title">Transport</h2>
              <Transport
                isPlaying={isPlaying}
                isRecording={isRecording}
                tempo={tempo}
                onPlayPause={() => {
                  setIsPlaying((value) => !value);
                  setIsRecording(false);
                }}
                onStop={() => {
                  setIsPlaying(false);
                  playTickRef.current = 0;
                  setMasterVu(0);
                }}
                onRecord={() => setIsRecording((value) => !value)}
                onTempoChange={setTempo}
              />
            </article>
            <article className="panel">
              <h2 className="panel-title">Mixer</h2>
              <Mixer
                channels={channels}
                onChannelChange={(channelId, patch) => {
                  setChannels((prev) =>
                    prev.map((channel) =>
                      channel.id === channelId
                        ? {
                            ...channel,
                            ...patch,
                          }
                        : channel,
                    ),
                  );
                }}
              />
            </article>
            <article className="panel wide">
              <h2 className="panel-title">Timeline</h2>
              <Timeline bars={timelineBars} tempo={tempo} />
            </article>
            <article className="panel">
              <h2 className="panel-title">Master Output</h2>
              <VUMeter value={masterVu} isActive={isPlaying} />
              <p className="hint">This is a deterministic UI mock for immediate visual feedback.</p>
            </article>
          </section>
        ) : (
          <section className="km-side-grid" aria-label="Side B creative station">
            <article className="panel">
              <h2 className="panel-title">Emotion Wheel</h2>
              <EmotionWheel
                onSelect={(emotion) => setSelectedEmotion(emotion)}
                selected={selectedEmotion}
              />
            </article>
            <article className="panel">
              <h2 className="panel-title">Ghost Writer</h2>
              <GhostWriter
                seed={selectedEmotion ? `${selectedEmotion.base} ${selectedEmotion.intensity}` : ''}
                onGenerate={(value) => setGhostText(value)}
                output={ghostText}
              />
            </article>
            <article className="panel wide">
              <h2 className="panel-title">Interrogator</h2>
              <Interrogator
                starter={selectedEmotion ? `How should this feel: ${selectedEmotion.base}` : 'Start a prompt'}
                onAsk={(question) =>
                  setInteractions((prev) => [
                    ...prev,
                    `You: ${question}`,
                    `KmiDi: Drafted approach aligned to ${selectedEmotion?.base ?? 'global intent'} path.`,
                  ])
                }
              />
            </article>
            <article className="panel">
              <h2 className="panel-title">Session Log</h2>
              <ul className="log-list">
                {interactions.map((entry, index) => (
                  <li key={`${entry}-${index}`}>
                    {entry}
                  </li>
                ))}
              </ul>
            </article>
          </section>
        )}
      </main>
    </div>
  );
}
