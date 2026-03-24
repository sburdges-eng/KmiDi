import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Transport } from './components/SideA/Transport';
import { Mixer } from './components/SideA/Mixer';
import { Timeline } from './components/SideA/Timeline';
import { VUMeter } from './components/SideA/VUMeter';
import { EmotionWheel } from './components/SideB/EmotionWheel';
import { GhostWriter } from './components/SideB/GhostWriter';
import { Interrogator } from './components/SideB/Interrogator';
import LyricPanel from './components/LyricPanel';
import { SpectoCloudPanel } from './components/SpectoCloudPanel';
import UniversalMusicInput from './components/UniversalMusicInput/UniversalMusicInput';
import { useMusicBrain } from './hooks/useMusicBrain';
import { useGateway } from './hooks/useGateway';

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
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [lastAudioPath, setLastAudioPath] = useState<string | undefined>();

  const brain = useMusicBrain();
  const gateway = useGateway('daiw-main');

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

  // Tool drawer content — existing components re-parented into the new layout
  const toolDrawerContent = (
    <div className="umi-drawer-grid">
      <article className="panel">
        <h3 className="panel-title">Transport</h3>
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
        <h3 className="panel-title">Mixer</h3>
        <Mixer
          channels={channels}
          onChannelChange={(channelId, patch) => {
            setChannels((prev) => prev.map((ch) => ch.id === channelId ? { ...ch, ...patch } : ch));
          }}
        />
      </article>
      <article className="panel">
        <h3 className="panel-title">Master</h3>
        <VUMeter value={masterVu} isActive={isPlaying} />
      </article>
      <article className="panel wide">
        <h3 className="panel-title">Timeline</h3>
        <Timeline bars={timelineBars} tempo={tempo} />
      </article>
      <article className="panel">
        <h3 className="panel-title">Mood</h3>
        <EmotionWheel onSelect={(emotion) => setSelectedEmotion(emotion)} selected={selectedEmotion} />
      </article>
      <article className="panel">
        <h3 className="panel-title">Lyric Spark</h3>
        <GhostWriter
          seed={selectedEmotion ? `${selectedEmotion.base} ${selectedEmotion.intensity}` : ''}
          onGenerate={handleGhostGenerate}
          output={ghostText}
        />
      </article>
      <article className="panel">
        <h3 className="panel-title">Lyrics</h3>
        <LyricPanel />
      </article>
      <article className="panel">
        <h3 className="panel-title">Spectocloud</h3>
        <SpectoCloudPanel lastGeneratedAudioPath={lastAudioPath} />
      </article>
    </div>
  );

  return (
    <div className="km-frame">
      <header className="km-header">
        <h1 className="app-title">KmiDi</h1>
        <span className="km-subtitle">Universal Music Input</span>
        <span className="km-gateway-status" title={`Gateway: ${gateway.connection}${gateway.lastTierUsed ? ` | Last: ${gateway.lastTierUsed} (${gateway.lastLatencyMs}ms)` : ''}`}>
          <span className={`status-dot ${gateway.isConnected ? 'connected' : gateway.connection === 'error' ? 'error' : 'disconnected'}`} />
        </span>
      </header>

      <main id="main-content" tabIndex={0}>
        <UniversalMusicInput
          apiStatus={apiStatus}
          toolDrawerContent={toolDrawerContent}
        />
      </main>
    </div>
  );
}
