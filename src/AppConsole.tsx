import { useEffect, useMemo, useRef, useState, useCallback, type ReactNode } from 'react';
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
import { RotaryModeSelector } from './components/RotaryModeSelector';
import { useMusicBrain } from './hooks/useMusicBrain';
import './console-shell.css';

type Mode = 'mix-detail' | 'inspire' | 'create' | 'compose';

/**
 * Feature flag: swap the vertical NavRail for the tactile rotary mode
 * selector (see design/prototype-tactile.html). Default OFF so this
 * ship is a no-op for existing users. Enable at build time with
 *   VITE_ROTARY_NAV=true npm run dev
 * or at runtime with
 *   localStorage.setItem('kmidi.rotaryNav', 'true'); location.reload();
 */
const USE_ROTARY_NAV: boolean = (() => {
  if (typeof window === 'undefined') return false;
  // Vite exposes env at import.meta.env; guard for environments without it.
  try {
    const viteEnv = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
    if (viteEnv?.VITE_ROTARY_NAV === 'true') return true;
  } catch { /* noop */ }
  try {
    return window.localStorage.getItem('kmidi.rotaryNav') === 'true';
  } catch {
    return false;
  }
})();

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

/* ─── Nav Rail Icons (inline SVG) ─── */

const NAV_ITEMS: { id: Mode; label: string; icon: ReactNode }[] = [
  {
    id: 'mix-detail',
    label: 'Mix Detail',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="10" cy="10" r="7" />
        <circle cx="10" cy="10" r="2" />
      </svg>
    ),
  },
  {
    id: 'inspire',
    label: 'Inspire',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M10 3a7 7 0 0 1 0 14" />
        <circle cx="10" cy="10" r="7" />
      </svg>
    ),
  },
  {
    id: 'create',
    label: 'Create',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M10 3l2 5h5l-4 3.5 1.5 5L10 13l-4.5 3.5L7 11.5 3 8h5z" />
      </svg>
    ),
  },
  {
    id: 'compose',
    label: 'Compose',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="3" y="3" width="6" height="6" rx="1" />
        <rect x="11" y="3" width="6" height="6" rx="1" />
        <rect x="3" y="11" width="6" height="6" rx="1" />
        <rect x="11" y="11" width="6" height="6" rx="1" />
      </svg>
    ),
  },
];

function NavRail({
  activeMode,
  onModeChange,
}: {
  activeMode: Mode;
  onModeChange: (mode: Mode) => void;
}) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    const currentIndex = NAV_ITEMS.findIndex((item) => item.id === activeMode);
    let nextIndex = currentIndex;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      nextIndex = (currentIndex + 1) % NAV_ITEMS.length;
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      nextIndex = (currentIndex - 1 + NAV_ITEMS.length) % NAV_ITEMS.length;
    } else {
      return;
    }

    onModeChange(NAV_ITEMS[nextIndex].id);
  };

  return (
    <nav
      className="nav-rail"
      role="tablist"
      aria-orientation="vertical"
      aria-label="Studio mode"
      onKeyDown={handleKeyDown}
    >
      {NAV_ITEMS.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={activeMode === item.id}
          aria-controls="workspace-panel"
          id={`tab-${item.id}`}
          tabIndex={activeMode === item.id ? 0 : -1}
          className="nav-rail__item"
          onClick={() => onModeChange(item.id)}
        >
          {item.icon}
          <span className="nav-rail__tooltip">{item.label}</span>
        </button>
      ))}
    </nav>
  );
}

function SessionBar({
  lastInteraction,
  apiStatus,
}: {
  lastInteraction: string;
  apiStatus: 'checking' | 'online' | 'offline';
}) {
  const statusLabel = apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking';
  const dotClass = apiStatus === 'online'
    ? 'session-bar__dot session-bar__dot--online'
    : apiStatus === 'offline'
      ? 'session-bar__dot session-bar__dot--offline'
      : 'session-bar__dot';

  return (
    <footer className="session-bar" aria-live="polite">
      <p className="session-bar__text">{lastInteraction}</p>
      <span className="session-bar__status">
        <span className={dotClass} aria-hidden="true" />
        {statusLabel}
      </span>
    </footer>
  );
}

export default function AppConsole() {
  const [mode, setMode] = useState<Mode>('inspire');
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
  const [interactions, setInteractions] = useState<string[]>(['What are you making?']);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [lastAudioPath, setLastAudioPath] = useState<string | undefined>();
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([]);

  const brain = useMusicBrain();

  useEffect(() => {
    brain.healthCheck()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

  // DAW-standard transport shortcuts: Space = play/pause, Shift+Space = record.
  // Skipped while typing or when a button has focus (its native Space click
  // already fires the action — handling it here too would double-toggle).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' ||
          t.tagName === 'BUTTON' || t.isContentEditable)
      ) return;
      e.preventDefault();
      if (e.shiftKey) {
        setIsRecording((v) => !v);
      } else {
        setIsPlaying((v) => !v);
        setIsRecording(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const timelineBars = useMemo(() => Math.max(8, tempo > 140 ? 24 : tempo > 95 ? 16 : 12), [tempo]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    // rAF instead of setInterval so meter updates land on frame boundaries —
    // a 120ms interval beats against the 60Hz refresh and reads as stutter.
    let raf = 0;
    let last = 0;
    const METER_MS = 100;
    const loop = (t: number) => {
      raf = window.requestAnimationFrame(loop);
      if (t - last < METER_MS) return;
      last = t;
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
    };
    raf = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(raf);
  }, [isPlaying]);

  const handleGhostGenerate = useCallback(async (localText: string) => {
    if (apiStatus === 'online') {
      try {
        await brain.setUserLyrics(localText);
        const lyrics = await brain.getUserLyrics();
        setGhostText(lyrics.lyrics ?? lyrics.generated ?? localText);
        return;
      } catch {
        // The session bar is the app's feedback channel — flip the status dot
        // and say what happened instead of silently using the local draft.
        setApiStatus('offline');
        setInteractions((prev) => [...prev, 'KmiDi: Studio offline — kept your draft as-is.']);
      }
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
      } catch {
        setApiStatus('offline');
      }
    }
    setInteractions((prev) => [
      ...prev,
      `KmiDi: Shaped around ${selectedEmotion?.base ?? 'your direction'}.`,
    ]);
  }, [apiStatus, brain, selectedEmotion]);

  const handleQuickStart = useCallback((template: {
    id: string; name: string; config: { bpm?: number; key?: string };
  }) => {
    setInteractions((prev) => [...prev, `Started: ${template.name} (${template.config.key ?? 'C'}, ${template.config.bpm ?? 120} BPM)`]);
    if (template.config.bpm) setTempo(template.config.bpm);
  }, []);

  const lastInteraction = interactions[interactions.length - 1] ?? '';

  const [displayMode, setDisplayMode] = useState<Mode>(mode);
  const [transitioning, setTransitioning] = useState(false);

  useEffect(() => {
    if (mode === displayMode) return;
    setTransitioning(true);
    const timer = setTimeout(() => {
      setDisplayMode(mode);
      setTransitioning(false);
    }, 80);
    return () => clearTimeout(timer);
  }, [mode, displayMode]);

  return (
    <div className="console-shell">
      <header className="upper-deck">
        <div className="upper-deck__zone">
          <h1 className="console-wordmark">KmiDi</h1>
          <Transport
            isPlaying={isPlaying}
            isRecording={isRecording}
            tempo={tempo}
            onPlayPause={() => { setIsPlaying((v) => !v); setIsRecording(false); }}
            onStop={() => { setIsPlaying(false); playTickRef.current = 0; setMasterVu(0); }}
            onRecord={() => setIsRecording((v) => !v)}
            onTempoChange={setTempo}
          />
        </div>
        <div className="upper-deck__zone">
          <Timeline bars={timelineBars} tempo={tempo} />
        </div>
        <div className="upper-deck__zone">
          <VUMeter value={masterVu} isActive={isPlaying} />
          <Mixer
            channels={channels}
            onChannelChange={(channelId, patch) => {
              setChannels((prev) => prev.map((ch) => ch.id === channelId ? { ...ch, ...patch } : ch));
            }}
          />
        </div>
      </header>

      <div className={`lower-deck${USE_ROTARY_NAV ? ' lower-deck--rotary' : ''}`}>
        {USE_ROTARY_NAV ? (
          <RotaryModeSelector<Mode>
            items={NAV_ITEMS}
            activeMode={mode}
            onModeChange={setMode}
            panelId="workspace-panel"
          />
        ) : (
          <NavRail activeMode={mode} onModeChange={setMode} />
        )}

        <div
          id="workspace-panel"
          role="tabpanel"
          aria-labelledby={`tab-${mode}`}
          tabIndex={0}
          className={`workspace workspace--${displayMode} ${transitioning ? 'workspace--exiting' : ''}`}
        >
          {displayMode === 'mix-detail' && (
            <section className="mode-mix-detail mode-enter" aria-label="Mix detail">
              <article className="console-panel">
                <h2 className="console-panel__title">Mixer</h2>
                <Mixer
                  channels={channels}
                  onChannelChange={(channelId, patch) => {
                    setChannels((prev) => prev.map((ch) => ch.id === channelId ? { ...ch, ...patch } : ch));
                  }}
                />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Master</h2>
                <VUMeter value={masterVu} isActive={isPlaying} />
              </article>
            </section>
          )}

          {displayMode === 'inspire' && (
            <section className="mode-inspire mode-enter" aria-label="Inspiration board">
              <article className="console-panel">
                <h2 className="console-panel__title">Mood</h2>
                <EmotionWheel onSelect={(emotion) => setSelectedEmotion(emotion)} selected={selectedEmotion} />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Ask</h2>
                <Interrogator
                  starter={selectedEmotion ? `How should this feel: ${selectedEmotion.base}` : ''}
                  onAsk={handleInterrogatorAsk}
                />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Lyric Spark</h2>
                <GhostWriter
                  seed={selectedEmotion ? `${selectedEmotion.base} ${selectedEmotion.intensity}` : ''}
                  onGenerate={handleGhostGenerate}
                  output={ghostText}
                />
              </article>
            </section>
          )}

          {displayMode === 'create' && (
            <section className="mode-create mode-enter" aria-label="Quick creation tools">
              <article className="console-panel">
                <h2 className="console-panel__title">Starters</h2>
                <QuickStartPanel onTemplateSelect={handleQuickStart} />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Sound Palette</h2>
                <MusicCustomizer
                  selectedGenre={selectedGenre}
                  selectedEmotion={selectedMood}
                  selectedTechniques={selectedTechniques}
                  onGenreChange={(genre) => { setSelectedGenre(genre); setInteractions((p) => [...p, `Genre: ${genre}`]); }}
                  onEmotionChange={(emotion) => { setSelectedMood(emotion); setInteractions((p) => [...p, `Emotion: ${emotion}`]); }}
                  onTechniquesChange={(techs) => { setSelectedTechniques(techs); }}
                />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Lyrics</h2>
                <LyricPanel />
              </article>
              <article className="console-panel">
                <h2 className="console-panel__title">Spectocloud</h2>
                <SpectoCloudPanel lastGeneratedAudioPath={lastAudioPath} />
              </article>
            </section>
          )}

          {displayMode === 'compose' && (
            <section className="mode-compose mode-enter" aria-label="Intent Builder">
              <IntentBuilder />
            </section>
          )}
        </div>

        <SessionBar lastInteraction={lastInteraction} apiStatus={apiStatus} />
      </div>
    </div>
  );
}
