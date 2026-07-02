import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import './EmotionDisc.css';

/**
 * Polar emotion surface. Angle picks the family (1/5 sector),
 * radius picks intensity (0..1 from inner dead-zone to outer rim).
 * Ported from design/prototype-tactile.html. Replaces the three-dropdown
 * form in EmotionWheel when USE_POLAR_EMOTION flag is on.
 *
 * Contract matches EmotionWheel: emits { base, intensity, detail } where
 *   base       — family key (calm | bright | intense | reflective | mystical)
 *   intensity  — 'low' | 'medium' | 'high'  (bucketed from continuous r)
 *   detail     — shade word from the family's shade triplet
 *
 * Interaction:
 *   click/drag anywhere on the disc  — set family + intensity
 *   click a family label             — snap to family, keep intensity
 *   1..5                             — snap to nth family
 *   ↑ or →                           — nudge intensity up
 *   ↓ or ←                           — nudge intensity down
 *   [ / ]                            — prev / next shade within family
 *   Esc                              — clear selection
 */

type EmotionNode = {
  base: string;
  intensity: string;
  detail: string;
};

type Props = {
  selected: EmotionNode | null;
  onSelect: (emotion: EmotionNode | null) => void;
};

const FAMILIES = [
  { key: 'calm',       label: 'CALM',       shades: ['muted', 'hopeful', 'serene'] },
  { key: 'bright',     label: 'BRIGHT',     shades: ['bright', 'playful', 'energetic'] },
  { key: 'intense',    label: 'INTENSE',    shades: ['pensive', 'urgent', 'fierce'] },
  { key: 'reflective', label: 'REFLECTIVE', shades: ['nostalgic', 'grateful', 'wistful'] },
  { key: 'mystical',   label: 'MYSTICAL',   shades: ['ethereal', 'dreamy', 'haunting'] },
] as const;

const C = 200;
const MAX_R = 175;
const INNER_R = 24;
const INTENSITY_STEP = 0.08;

function intensityLabel(i: number): string {
  return i < 0.34 ? 'low' : i < 0.67 ? 'medium' : 'high';
}

function intensityFromLabel(label: string | undefined): number {
  if (label === 'low') return 0.2;
  if (label === 'high') return 0.85;
  return 0.55; // medium or unknown
}

function shadeIdxForIntensity(intensity: number, shades: readonly string[]): number {
  return Math.min(
    shades.length - 1,
    Math.max(0, Math.floor(intensity * shades.length)),
  );
}

function familyIdxOf(base: string | null | undefined): number {
  if (!base) return 0;
  const i = FAMILIES.findIndex((f) => f.key === base);
  return i >= 0 ? i : 0;
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

export function EmotionDisc({ selected, onSelect }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const draggingRef = useRef(false);

  // Internal continuous state — authoritative during the component's life.
  // External `selected` only seeds the initial value and triggers a reset
  // when it becomes null. This prevents the 3-bucket intensity string from
  // quantizing the puck mid-drag.
  const initialFamilyIdx = familyIdxOf(selected?.base);
  const initialIntensity = intensityFromLabel(selected?.intensity);
  const initialShadeIdx = (() => {
    if (!selected) return shadeIdxForIntensity(initialIntensity, FAMILIES[initialFamilyIdx].shades);
    const shades = FAMILIES[initialFamilyIdx].shades as readonly string[];
    const si = shades.indexOf(selected.detail);
    return si >= 0 ? si : shadeIdxForIntensity(initialIntensity, FAMILIES[initialFamilyIdx].shades);
  })();

  const [familyIdx, setFamilyIdx] = useState(initialFamilyIdx);
  const [intensity, setIntensity] = useState(initialIntensity);
  const [shadeIdx, setShadeIdx] = useState(initialShadeIdx);

  // External reset → local reset.
  useEffect(() => {
    if (selected === null) {
      setFamilyIdx(0);
      setIntensity(0.55);
      setShadeIdx(0);
    }
  }, [selected]);

  const family = FAMILIES[familyIdx];

  const emit = useCallback(
    (fi: number, i: number, si?: number) => {
      const fam = FAMILIES[fi];
      const sIdx = si ?? shadeIdxForIntensity(i, fam.shades);
      onSelect({
        base: fam.key,
        intensity: intensityLabel(i),
        detail: fam.shades[sIdx],
      });
    },
    [onSelect],
  );

  // Puck position and sector-highlight path.
  const aMid = useMemo(
    () => (((familyIdx + 0.5) / FAMILIES.length) * 360 - 90) * (Math.PI / 180),
    [familyIdx],
  );
  const puck = useMemo(() => {
    const r = INNER_R + (MAX_R - INNER_R) * intensity;
    return {
      x: C + Math.cos(aMid) * r,
      y: C + Math.sin(aMid) * r,
    };
  }, [aMid, intensity]);

  const sectorPath = useMemo(() => {
    const aStart = ((familyIdx / FAMILIES.length) * 360 - 90) * (Math.PI / 180);
    const aEnd = (((familyIdx + 1) / FAMILIES.length) * 360 - 90) * (Math.PI / 180);
    const x1i = C + Math.cos(aStart) * INNER_R;
    const y1i = C + Math.sin(aStart) * INNER_R;
    const x2i = C + Math.cos(aEnd) * INNER_R;
    const y2i = C + Math.sin(aEnd) * INNER_R;
    const x1o = C + Math.cos(aStart) * MAX_R;
    const y1o = C + Math.sin(aStart) * MAX_R;
    const x2o = C + Math.cos(aEnd) * MAX_R;
    const y2o = C + Math.sin(aEnd) * MAX_R;
    return `M ${x1i} ${y1i} L ${x1o} ${y1o} A ${MAX_R} ${MAX_R} 0 0 1 ${x2o} ${y2o} L ${x2i} ${y2i} A ${INNER_R} ${INNER_R} 0 0 0 ${x1i} ${y1i} Z`;
  }, [familyIdx]);

  // Spokes between sectors.
  const spokes = useMemo(
    () =>
      FAMILIES.map((_, i) => {
        const a = ((i / FAMILIES.length) * 360 - 90) * (Math.PI / 180);
        return {
          x1: C + Math.cos(a) * INNER_R,
          y1: C + Math.sin(a) * INNER_R,
          x2: C + Math.cos(a) * MAX_R,
          y2: C + Math.sin(a) * MAX_R,
        };
      }),
    [],
  );

  // Family labels.
  const labels = useMemo(
    () =>
      FAMILIES.map((f, i) => {
        const a = (((i + 0.5) / FAMILIES.length) * 360 - 90) * (Math.PI / 180);
        const lr = 188;
        return {
          x: C + Math.cos(a) * lr,
          y: C + Math.sin(a) * lr,
          label: f.label,
          idx: i,
        };
      }),
    [],
  );

  // Pointer → polar coords.
  const pointerToPolar = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    const svgX = ((clientX - rect.left) / rect.width) * 400 - C;
    const svgY = ((clientY - rect.top) / rect.height) * 400 - C;
    const dist = Math.sqrt(svgX * svgX + svgY * svgY);
    // angle = 0 at 12 o'clock, increasing clockwise
    let angle = Math.atan2(svgY, svgX) + Math.PI / 2;
    if (angle < 0) angle += Math.PI * 2;
    const fi = Math.min(
      FAMILIES.length - 1,
      Math.floor((angle / (Math.PI * 2)) * FAMILIES.length),
    );
    const rawI = (dist - INNER_R) / (MAX_R - INNER_R);
    return { familyIdx: fi, intensity: clamp(rawI, 0.05, 1) };
  }, []);

  const applyPointer = useCallback(
    (clientX: number, clientY: number) => {
      const p = pointerToPolar(clientX, clientY);
      if (!p) return;
      const si = shadeIdxForIntensity(p.intensity, FAMILIES[p.familyIdx].shades);
      setFamilyIdx(p.familyIdx);
      setIntensity(p.intensity);
      setShadeIdx(si);
      emit(p.familyIdx, p.intensity, si);
    },
    [pointerToPolar, emit],
  );

  const onPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    draggingRef.current = true;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* not all platforms support this */
    }
    applyPointer(e.clientX, e.clientY);
  };
  const onPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!draggingRef.current) return;
    applyPointer(e.clientX, e.clientY);
  };
  const onPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    draggingRef.current = false;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  };

  const snapFamily = useCallback(
    (idx: number) => {
      const si = shadeIdxForIntensity(intensity, FAMILIES[idx].shades);
      setFamilyIdx(idx);
      setShadeIdx(si);
      emit(idx, intensity, si);
    },
    [intensity, emit],
  );

  const onKeyDown = (e: ReactKeyboardEvent<SVGSVGElement>) => {
    // Number keys 1-5 snap to family.
    if (e.key >= '1' && e.key <= '5') {
      const idx = parseInt(e.key, 10) - 1;
      if (idx < FAMILIES.length) {
        e.preventDefault();
        snapFamily(idx);
        return;
      }
    }
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {
      e.preventDefault();
      const next = clamp(intensity + INTENSITY_STEP, 0.05, 1);
      const si = shadeIdxForIntensity(next, family.shades);
      setIntensity(next);
      setShadeIdx(si);
      emit(familyIdx, next, si);
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const next = clamp(intensity - INTENSITY_STEP, 0.05, 1);
      const si = shadeIdxForIntensity(next, family.shades);
      setIntensity(next);
      setShadeIdx(si);
      emit(familyIdx, next, si);
    } else if (e.key === '[') {
      e.preventDefault();
      const next = Math.max(0, shadeIdx - 1);
      if (next !== shadeIdx) {
        setShadeIdx(next);
        emit(familyIdx, intensity, next);
      }
    } else if (e.key === ']') {
      e.preventDefault();
      const next = Math.min(family.shades.length - 1, shadeIdx + 1);
      if (next !== shadeIdx) {
        setShadeIdx(next);
        emit(familyIdx, intensity, next);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onSelect(null);
    }
  };

  // Unique SVG defs ids so multiple instances don't collide.
  const rawId = useId();
  const uid = rawId.replace(/:/g, '');
  const ringGlowId = `emo-ring-${uid}`;
  const puckGlowId = `emo-puck-${uid}`;

  const angleDeg = Math.round((aMid * 180) / Math.PI + 90);
  const ariaText = `${family.label.toLowerCase()} ${intensityLabel(intensity)} ${family.shades[shadeIdx]}`;

  return (
    <div className="emotion-disc">
      <div className="emotion-disc__head">
        <span className="emotion-disc__title">Emotion · Polar Surface</span>
        <span className="emotion-disc__meta">
          ∠ {angleDeg}° · r {intensity.toFixed(2)}
        </span>
      </div>

      <div className="emotion-disc__wrap">
        <svg
          ref={svgRef}
          viewBox="0 0 400 400"
          className="emotion-disc__svg"
          role="slider"
          aria-label="Emotion polar surface"
          aria-valuetext={ariaText}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onKeyDown={onKeyDown}
        >
          <defs>
            <radialGradient id={ringGlowId} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#c9a227" stopOpacity="0.04" />
              <stop offset="70%" stopColor="#c9a227" stopOpacity="0.02" />
              <stop offset="100%" stopColor="#c9a227" stopOpacity="0" />
            </radialGradient>
            <filter id={puckGlowId} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <circle cx={C} cy={C} r={190} fill={`url(#${ringGlowId})`} />

          {/* intensity rings */}
          <g stroke="rgba(255,255,255,0.06)" fill="none" strokeWidth="1">
            <circle cx={C} cy={C} r={60} />
            <circle cx={C} cy={C} r={115} />
            <circle cx={C} cy={C} r={170} />
          </g>

          {/* sector spokes */}
          <g stroke="rgba(255,255,255,0.05)" strokeWidth="1">
            {spokes.map((s, i) => (
              <line key={i} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} />
            ))}
          </g>

          {/* active sector highlight */}
          <path
            d={sectorPath}
            fill="rgba(201,162,39,0.10)"
            stroke="rgba(201,162,39,0.35)"
            strokeWidth="1"
          />

          {/* dead zone */}
          <circle
            cx={C}
            cy={C}
            r={22}
            fill="#0c0c0e"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth="1"
          />
          <text
            x={C}
            y={C + 3}
            textAnchor="middle"
            fill="#6b6b69"
            fontFamily="IBM Plex Mono, monospace"
            fontSize="8"
            letterSpacing="2"
            pointerEvents="none"
          >
            FEEL
          </text>

          {/* family labels */}
          <g>
            {labels.map((l) => (
              <text
                key={l.idx}
                x={l.x}
                y={l.y}
                textAnchor="middle"
                dominantBaseline="middle"
                fontFamily="IBM Plex Mono, monospace"
                fontSize="9"
                letterSpacing="1.5"
                fill={l.idx === familyIdx ? 'var(--accent, #c9a227)' : '#8a8a86'}
                className="emotion-disc__label"
                onPointerDown={(e) => {
                  // Prevent the parent SVG from treating this as a disc drag.
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  snapFamily(l.idx);
                }}
              >
                {l.label}
              </text>
            ))}
          </g>

          {/* puck */}
          <g filter={`url(#${puckGlowId})`} pointerEvents="none">
            <circle
              cx={puck.x}
              cy={puck.y}
              r={6}
              fill="var(--accent, #c9a227)"
              stroke="#fff8e0"
              strokeWidth="1"
            />
          </g>
        </svg>
      </div>

      <div className="emotion-disc__readout" aria-live="polite">
        <span>{family.label}</span>
        <span className="emotion-disc__sep" aria-hidden="true">·</span>
        <span>{intensityLabel(intensity).toUpperCase()}</span>
        <span className="emotion-disc__sep" aria-hidden="true">·</span>
        <span>{family.shades[shadeIdx].toUpperCase()}</span>
      </div>

      <p className="emotion-disc__hint" aria-hidden="true">
        drag to set · 1–5 family · ↑↓ intensity · [ ] shade · esc clear
      </p>
    </div>
  );
}

export default EmotionDisc;
