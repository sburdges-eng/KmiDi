import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Transport } from './components/SideA/Transport';
import { Mixer } from './components/SideA/Mixer';
import { Timeline } from './components/SideA/Timeline';
import { VUMeter } from './components/SideA/VUMeter';
import { EmotionWheel } from './components/SideB/EmotionWheel';
import { GhostWriter } from './components/SideB/GhostWriter';
import { Interrogator } from './components/SideB/Interrogator';
import IntentBuilder from './components/IntentBuilder';
import { QuickStartPanel } from './components/QuickStartPanel';
import LyricPanel from './components/LyricPanel';
import { SpectoCloudPanel } from './components/SpectoCloudPanel';
import { MusicCustomizer } from './components/MusicCustomizer';
import { useMusicBrain } from './hooks/useMusicBrain';

type Side = 'side-a' | 'side-b' | 'create' | 'intent';

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
    'What are you making?',
  ]);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [lastAudioPath, setLastAudioPath] = useState<string | undefined>();
  const logRef = useRef<HTMLUListElement>(null);
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([]);

  const brain = useMusicBrain();

  useEffect(() => {
    brain.healthCheck()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const timelineBars = useMemo(() => Math.max(8, tempo > 140 ? 24 : tempo > 95 ? 16 : 12), [tempo]);

  useEffect(() => {
    if (!isPlaying) return undefined;

    const timer = window.setInterval(() => {
      playTickRef.current += 1;
      setMasterVu(() => {
        const wave = Math.sin(playTickRef.current * 0.35) * 0.4 + Math.cos(playTickRef.current * 0.18) * 0.2;
        const next = 0.35 + wave * 0.4;
        return Number(Math.min(0.94, Math.max(0.12, next)).toFixed(3));
      });
      setChannels((prev) =>
        prev.map((channel, index) => {
          const wave = Math.sin(playTickRef.current * 0.3 + index * 0.65);
          return { ...channel, level: Number((0.35 + Math.max(-0.18, Math.min(0.58, wave * 0.28))).toFixed(3)) };
        }),
      );
    }, 120);

    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const handleGhostGenerate = useCallback(async (localText: string) => {
    if (apiStatus === 'online') {
      try {
        await brain.setUserLyrics(localText);
        const lyrics = await brain.getUserLyrics();
        setGhostText(lyrics.lyrics ?? lyrics.generated ?? localText);
        return;
      } catch { /* fall through */ }
    }
    setGhostText(localText);
  }, [apiStatus, brain]);

  const handleInterrogatorAsk = useCallback(async (question: string) => {
    setInteractions((prev) => [...prev, `You: ${question}`]);
    if (apiStatus === 'online') {
      try {
        const response = await brain.interrogate({ message: question });
        setInteractions((prev) => [...prev, `KmiDi: ${response.reply}`]);
        return;
      } catch { /* fall through */ }
    }
    setInteractions((prev) => [
      ...prev,
      `KmiDi: Shaped around ${selectedEmotion?.base ?? 'your direction'}.`,
    ]);
  }, [apiStatus, brain, selectedEmotion]);

  const handleQuickStart = useCallback(async (template: {
    id: string; name: string; config: { bpm?: number; key?: string };
  }) => {
    setInteractions((prev) => [...prev, `Started: ${template.name} (${template.config.key ?? 'C'}, ${template.config.bpm ?? 120} BPM)`]);
    if (template.config.bpm) setTempo(template.config.bpm);
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [interactions]);

  const activeTitle = side === 'side-a' ? 'Mix'
    : side === 'side-b' ? 'Inspire'
    : side === 'create' ? 'Create'
    : 'Compose';

  return (
    <div className="km-frame">
      <header className="km-header">
        <h1 className="app-title">KmiDi</h1>
        <nav className="km-toggle" role="tablist" aria-label="Studio mode">
          {(['side-a', 'side-b', 'create', 'intent'] as Side[]).map((s, i) => (
            <button
              key={s}
              type="button"
              role="tab"
              aria-selected={side === s}
              aria-controls="main-content"
              id={`tab-${s}`}
              tabIndex={side === s ? 0 : -1}
              className={side === s ? 'tab active' : 'tab'}
              onClick={() => setSide(s)}
            >
              {s === 'side-a' ? 'Mix' : s === 'side-b' ? 'Inspire' : s === 'create' ? 'Create' : 'Compose'}
            </button>
          ))}
        </nav>
      </header>

      <section className="km-titlebar" aria-hidden="true">
        <p className="km-subtitle">{activeTitle}</p>
        <button
          type="button"
          className="mode-reset-btn"
          onClick={() => setIsPlaying(false)}
          aria-label="Reset playback"
        >
          Reset
        </button>
      </section>

      <main id="main-content" role="tabpanel" aria-labelledby={`tab-${side}`} tabIndex={0}>
        {side === 'side-a' && (
          <section className="km-side-grid" aria-label="Mix console">
            <article className="panel">
              <h2 className="panel-title">Transport</h2>
              <Transport
                isPlaying={isPlaying}
                isRecording={isRecording}
                tempo={tempo}
                onPlayPause={() => { setIsPlaying((v) => !v); setIsRecording(false); }}
                onStop={() => { setIsPlaying(false); playTickRef.current = 0; setMasterVu(0); }}
                onRecord={() => setIsRecording((v) => !v)}
                onTempoChange={setTempo}
              />
            </article>
            <article className="panel">
              <h2 className="panel-title">Mixer</h2>
              <Mixer
                channels={channels}
                onChannelChange={(channelId, patch) => {
                  setChannels((prev) => prev.map((ch) => ch.id === channelId ? { ...ch, ...patch } : ch));
                }}
              />
            </article>
            <article className="panel wide">
              <h2 className="panel-title">Timeline</h2>
              <Timeline bars={timelineBars} tempo={tempo} />
            </article>
            <article className="panel">
              <h2 className="panel-title">Master</h2>
              <VUMeter value={masterVu} isActive={isPlaying} />
            </article>
          </section>
        )}

        {side === 'side-b' && (
          <section className="km-side-grid" aria-label="Inspiration board">
            <article className="panel">
              <h2 className="panel-title">Mood</h2>
              <EmotionWheel onSelect={(emotion) => setSelectedEmotion(emotion)} selected={selectedEmotion} />
            </article>
            <article className="panel">
              <h2 className="panel-title">Lyric Spark</h2>
              <GhostWriter
                seed={selectedEmotion ? `${selectedEmotion.base} ${selectedEmotion.intensity}` : ''}
                onGenerate={handleGhostGenerate}
                output={ghostText}
              />
            </article>
            <article className="panel wide">
              <h2 className="panel-title">Ask</h2>
              <Interrogator
                starter={selectedEmotion ? `How should this feel: ${selectedEmotion.base}` : 'Ask something'}
                onAsk={handleInterrogatorAsk}
              />
            </article>
            <article className="panel">
              <h2 className="panel-title">Session</h2>
              <ul className="log-list" ref={logRef}>
                {interactions.map((entry, index) => (
                  <li key={`${entry.slice(0, 20)}-${index}`}>{entry}</li>
                ))}
              </ul>
            </article>
          </section>
        )}

        {side === 'create' && (
          <section className="km-side-grid" aria-label="Quick creation tools">
            <article className="panel wide">
              <h2 className="panel-title">Starters</h2>
              <QuickStartPanel onTemplateSelect={handleQuickStart} />
            </article>
            <article className="panel wide">
              <h2 className="panel-title">Sound Palette</h2>
              <MusicCustomizer
                selectedGenre={selectedGenre}
                selectedEmotion={selectedMood}
                selectedTechniques={selectedTechniques}
                onGenreChange={(genre) => { setSelectedGenre(genre); setInteractions((p) => [...p, `Genre: ${genre}`]); }}
                onEmotionChange={(emotion) => { setSelectedMood(emotion); setInteractions((p) => [...p, `Emotion: ${emotion}`]); }}
                onTechniquesChange={(techs) => { setSelectedTechniques(techs); }}
              />
            </article>
            <article className="panel">
              <h2 className="panel-title">Lyrics</h2>
              <LyricPanel />
            </article>
            <article className="panel">
              <h2 className="panel-title">Spectocloud</h2>
              <SpectoCloudPanel lastGeneratedAudioPath={lastAudioPath} />
            </article>
          </section>
        )}

        {side === 'intent' && (
          <section className="km-intent-section" aria-label="Intent Builder">
            <IntentBuilder />
          </section>
        )}
      </main>
    </div>
  );
}
