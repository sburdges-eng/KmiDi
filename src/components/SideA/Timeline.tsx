type Props = {
  bars: number;
  tempo: number;
};

export function Timeline({ bars, tempo }: Props) {
  return (
    <div className="timeline">
      <p className="timeline-meta">Bars: {bars} | Estimated cycle: {Math.round((bars * 4 * 60) / tempo)} s</p>
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
