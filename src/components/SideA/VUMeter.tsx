type Props = {
  value: number;
  isActive: boolean;
};

export function VUMeter({ value, isActive }: Props) {
  return (
    <div className="vu-wrap">
      <label className="vu-title">Master {isActive ? 'active' : 'idle'}</label>
      <div className="vu-track" aria-label="vu meter">
        <span className="vu-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <p className="vu-value">{Math.round(value * 100)}%</p>
    </div>
  );
}
