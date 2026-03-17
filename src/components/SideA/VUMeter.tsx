import { CompactMeter } from './CompactMeter';

type Props = {
  value: number;
  isActive: boolean;
};

export function VUMeter({ value, isActive }: Props) {
  return (
    <div className="vu-wrap">
      <CompactMeter
        label="Master"
        value={value}
        valueIsLinear
        disabled={!isActive}
        aria-label={isActive ? 'Master level, active' : 'Master level, idle'}
      />
    </div>
  );
}
