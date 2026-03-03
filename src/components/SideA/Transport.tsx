import { ChangeEvent } from 'react';

type Props = {
  isPlaying: boolean;
  isRecording: boolean;
  tempo: number;
  onPlayPause: () => void;
  onStop: () => void;
  onRecord: () => void;
  onTempoChange: (tempo: number) => void;
};

export function Transport({ isPlaying, isRecording, tempo, onPlayPause, onStop, onRecord, onTempoChange }: Props) {
  const updateTempo = (event: ChangeEvent<HTMLInputElement>) => {
    const next = Number(event.target.value);
    onTempoChange(Math.max(40, Math.min(300, next)));
  };

  return (
    <div className="transport-grid">
      <div className="control-row">
        <button type="button" className="cta" onClick={onPlayPause}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button type="button" className="outline" onClick={onStop}>
          Stop
        </button>
        <button
          type="button"
          className={isRecording ? 'recording' : 'outline'}
          onClick={onRecord}
        >
          {isRecording ? 'Stop Record' : 'Record'}
        </button>
      </div>
      <label className="tempo">
        Tempo {tempo} BPM
        <input type="range" min={40} max={300} value={tempo} onChange={updateTempo} />
      </label>
    </div>
  );
}
