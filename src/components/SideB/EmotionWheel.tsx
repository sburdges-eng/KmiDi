import { ChangeEvent } from 'react';

type EmotionNode = {
  base: string;
  intensity: string;
  detail: string;
};

type Selected = EmotionNode | null;

type Props = {
  selected: Selected;
  onSelect: (emotion: Selected) => void;
};

const EMOTIONS = {
  calm: ['muted', 'hopeful', 'serene'],
  bright: ['bright', 'playful', 'energetic'],
  intense: ['pensive', 'urgent', 'fierce'],
  reflective: ['nostalgic', 'grateful', 'wistful'],
  mystical: ['ethereal', 'dreamy', 'haunting'],
};

const INTENSITIES = ['low', 'medium', 'high'];

export function EmotionWheel({ selected, onSelect }: Props) {
  const baseKeys = Object.keys(EMOTIONS) as Array<keyof typeof EMOTIONS>;

  const handleBase = (event: ChangeEvent<HTMLSelectElement>) => {
    onSelect(null);
    const base = event.target.value;
    if (!base) {
      onSelect(null);
      return;
    }
    const intensity = INTENSITIES[0];
    const detail = EMOTIONS[base as keyof typeof EMOTIONS][0];
    onSelect({ base, intensity, detail });
  };

  const handleIntensity = (event: ChangeEvent<HTMLSelectElement>) => {
    if (!selected) return;
    const intensity = event.target.value;
    onSelect({
      ...selected,
      intensity,
    });
  };

  const handleDetail = (event: ChangeEvent<HTMLSelectElement>) => {
    if (!selected) return;
    onSelect({
      ...selected,
      detail: event.target.value,
    });
  };

  return (
    <div className="emotion-wheel">
      <label>
        Feeling
        <select value={selected?.base ?? ''} onChange={handleBase}>
          <option value="">choose a mood</option>
          {baseKeys.map((base) => (
            <option key={base} value={base}>
              {base}
            </option>
          ))}
        </select>
      </label>

      <label>
        Strength
        <select
          value={selected?.intensity ?? ''}
          onChange={handleIntensity}
          disabled={!selected}
        >
          <option value="">intensity</option>
          {INTENSITIES.map((intensity) => (
            <option key={intensity} value={intensity}>
              {intensity}
            </option>
          ))}
        </select>
      </label>

      <label>
        Color
        <select value={selected?.detail ?? ''} onChange={handleDetail} disabled={!selected}>
          <option value="">shade</option>
          {(selected ? EMOTIONS[selected.base as keyof typeof EMOTIONS] : []).map((detail) => (
            <option key={detail} value={detail}>
              {detail}
            </option>
          ))}
        </select>
      </label>

      <p className="emotion-output">
        {selected
          ? `${selected.base} · ${selected.intensity} · ${selected.detail}`
          : 'Set a feeling to begin'}
      </p>
    </div>
  );
}
