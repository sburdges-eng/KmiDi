interface InterpretiveSliderProps {
  value: number; // 0 = literal, 1 = surprise
  onChange: (value: number) => void;
}

export function InterpretiveSlider({ value, onChange }: InterpretiveSliderProps) {
  return (
    <div className="umi-interpretive-slider">
      <label className="umi-slider-label">
        <span className="umi-slider-end">Literal</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="umi-slider-input"
        />
        <span className="umi-slider-end">Surprise</span>
      </label>
    </div>
  );
}
