import { useCallback, useEffect, useRef, useState } from 'react';

const SIZE = 40;
const DEFAULT_STEP = 1;
const FINE_STEP_COARSE_RATIO = 0.1;

type KnobProps = {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  stepFine?: number;
  onChange: (value: number) => void;
  label?: string;
  'aria-label'?: string;
  disabled?: boolean;
};

export function Knob({
  value,
  min = 0,
  max = 1,
  step = DEFAULT_STEP,
  stepFine,
  onChange,
  label,
  'aria-label': ariaLabel,
  disabled = false,
}: KnobProps) {
  const stepFineResolved = stepFine ?? step * FINE_STEP_COARSE_RATIO;
  const [isDragging, setIsDragging] = useState(false);
  const startYRef = useRef(0);
  const startValueRef = useRef(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const clamp = useCallback(
    (v: number) => Math.max(min, Math.min(max, v)),
    [min, max]
  );

  const roundToStep = useCallback(
    (v: number, useFine: boolean) => {
      const s = useFine ? stepFineResolved : step;
      return Math.round((v - min) / s) * s + min;
    },
    [min, step, stepFineResolved]
  );

  const valueToAngle = useCallback(() => {
    const t = (value - min) / (max - min || 1);
    return -135 + t * 270;
  }, [value, min, max]);

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e: MouseEvent) => {
      const dy = startYRef.current - e.clientY;
      const range = max - min;
      const sensitivity = range / 120;
      const delta = dy * sensitivity;
      const next = clamp(startValueRef.current + delta);
      onChange(roundToStep(next, e.shiftKey));
    };
    const onUp = () => setIsDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isDragging, min, max, onChange, clamp, roundToStep]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (disabled) return;
      const fine = e.shiftKey;
      const s = fine ? stepFineResolved : step;
      let next = value;
      if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {
        e.preventDefault();
        next = clamp(value + s);
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {
        e.preventDefault();
        next = clamp(value - s);
      } else return;
      onChange(roundToStep(next, fine));
    },
    [disabled, value, step, stepFineResolved, clamp, roundToStep, onChange]
  );

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled || e.button !== 0) return;
      e.preventDefault();
      startYRef.current = e.clientY;
      startValueRef.current = value;
      setIsDragging(true);
    },
    [disabled, value]
  );

  const angle = valueToAngle();

  return (
    <div
      ref={containerRef}
      className="knob-wrap"
      role="group"
      aria-label={ariaLabel ?? label}
    >
      <div
        className={`knob ${isDragging ? 'knob--active' : ''} ${disabled ? 'knob--disabled' : ''}`}
        tabIndex={disabled ? undefined : 0}
        role="slider"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-disabled={disabled}
        onKeyDown={onKeyDown}
        onMouseDown={onMouseDown}
        style={{ width: SIZE, height: SIZE }}
      >
        {isDragging && (
          <div
            className="knob__arc"
            style={{
              background: `conic-gradient(var(--accent) 0deg ${angle + 135}deg, transparent ${angle + 135}deg 360deg)`,
            }}
          />
        )}
        {!isDragging && (
          <div className="knob__dot-wrap" style={{ transform: `rotate(${angle}deg)` }}>
            <div className="knob__dot" />
          </div>
        )}
      </div>
      {label != null && <span className="knob__label">{label}</span>}
    </div>
  );
}
