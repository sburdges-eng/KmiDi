type Props = {
  bars: number;
  tempo: number;
};

function formatDurationSeconds(totalSeconds: number): string {
  const sec = Math.round(totalSeconds);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function Timeline({ bars, tempo }: Props) {
  const totalSeconds = (bars * 4 * 60) / tempo;
  const durationMmSs = formatDurationSeconds(totalSeconds);

  return (
    <div className="timeline">
      <p className="timeline-meta">
        {bars} bars · {durationMmSs}
      </p>
      <div className="timeline-ruler" aria-label="bars">
        {Array.from({ length: bars }).map((_, index) => (
          <span key={`bar-${index}`} className={index % 4 === 0 ? 'beat major' : 'beat'}>
            {index + 1}
          </span>
        ))}
      </div>
    </div>
  );
}
