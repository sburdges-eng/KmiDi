import { useEffect, useRef, useState } from 'react';

const PEAK_HOLD_MS = 400;
const CLIP_THRESHOLD_LINEAR = 0.98;

/** Linear 0..1 to dB (minDb at 0, 0 dB at 1). */
function linearToDb(linear: number, minDb: number): number {
  if (linear <= 0) return minDb;
  const db = 20 * Math.log10(linear);
  return Math.max(minDb, Math.min(0, db));
}

export type CompactMeterProps = {
  label: string;
  value: number;
  minDb?: number;
  maxDb?: number;
  clip?: boolean;
  disabled?: boolean;
  /** Display value in dB (numeric primary). If false, value is 0..1 linear. */
  valueIsLinear?: boolean;
  'aria-label'?: string;
};

export function CompactMeter({
  label,
  value,
  minDb = -60,
  maxDb = 0,
  clip: clipProp,
  disabled = false,
  valueIsLinear = true,
  'aria-label': ariaLabel,
}: CompactMeterProps) {
  const valueDb = valueIsLinear ? linearToDb(value, minDb) : value;
  const clip = clipProp ?? (valueIsLinear && value >= CLIP_THRESHOLD_LINEAR);
  const [peakDb, setPeakDb] = useState(valueDb);
  const [justClipped, setJustClipped] = useState(false);
  const peakDecayRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevClipRef = useRef(false);

  useEffect(() => {
    if (disabled) return;
    if (valueDb > peakDb) {
      setPeakDb(valueDb);
      if (peakDecayRef.current) clearTimeout(peakDecayRef.current);
      peakDecayRef.current = setTimeout(() => {
        setPeakDb((p) => Math.max(valueDb, p - 1));
        peakDecayRef.current = null;
      }, PEAK_HOLD_MS);
    }
    return () => {
      if (peakDecayRef.current) clearTimeout(peakDecayRef.current);
    };
  }, [valueDb, peakDb, disabled]);

  useEffect(() => {
    if (clip && !prevClipRef.current) setJustClipped(true);
    prevClipRef.current = clip;
    if (!justClipped) return;
    const t = setTimeout(() => setJustClipped(false), 150);
    return () => clearTimeout(t);
  }, [clip, justClipped]);

  const pct = Math.min(100, Math.max(0, ((valueDb - minDb) / (maxDb - minDb)) * 100));
  const displayDb = valueDb <= minDb ? `${minDb}` : valueDb.toFixed(1);

  return (
    <div
      className={`compact-meter ${disabled ? 'compact-meter--disabled' : ''} ${justClipped && clip ? 'compact-meter--clip-pulse' : ''}`}
      role="img"
      aria-label={ariaLabel ?? `${label} ${displayDb} dB`}
      aria-live="polite"
    >
      <div className="compact-meter__head">
        <span className="compact-meter__label">
          {label}
        </span>
        <span className={`compact-meter__value ${clip ? 'peak' : ''}`} aria-hidden>
          {displayDb} dB
        </span>
      </div>
      <div className="compact-meter__bar bar" aria-hidden>
        <div className="compact-meter__track">
          <div
            className="compact-meter__fill"
            style={{
              width: `${pct}%`,
              background: clip ? 'var(--accent)' : 'var(--text-lo)',
            }}
          />
        </div>
      </div>
    </div>
  );
}
