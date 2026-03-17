type Props = {
  value: number;
  isActive: boolean;
};

export function VUMeter({ value, isActive }: Props) {
  return (
    <div className="vu-wrap">
      <label className="vu-title" aria-live="polite" aria-label={isActive ? 'Master level, active' : 'Master level, idle'}>Master</label>
      <div className="vu-track" aria-label="vu meter">
        <span className="vu-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <p className="vu-value">{Math.round(value * 100)}%</p>
    </div>
  );
}
