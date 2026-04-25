import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from 'react';
import './RotaryModeSelector.css';

/**
 * Tactile rotary replacement for the vertical NavRail. Same contract as NavRail
 * (activeMode + onModeChange) so the call-site in AppConsole only swaps which
 * component gets rendered. Ported from design/prototype-tactile.html.
 *
 * The disc rotates so its indicator (top of disc) points at the tick for the
 * active item. Ticks are laid out across a 270° arc from -135° to +135° with
 * a 90° gap at the bottom. Each tick is a focusable <button role="tab"> with
 * an accessible label; labels render as sr-only text plus a hover tooltip.
 *
 * Keyboard:
 *   Up / Left      previous mode
 *   Down / Right   next mode
 *   Home / End     first / last mode
 *   Space / Enter  already-standard button activation
 *
 * Scroll wheel over the disc cycles modes. Click on the disc cycles forward.
 * Spring-landing motion is purely a CSS transition on `transform`, so
 * prefers-reduced-motion users get an instant snap.
 */

export type RotaryItem<T extends string> = {
  id: T;
  label: string;
  icon?: ReactNode;
};

type Props<T extends string> = {
  items: readonly RotaryItem<T>[];
  activeMode: T;
  onModeChange: (mode: T) => void;
  /** Passed through as aria-label on the tablist root. */
  ariaLabel?: string;
  /** The id of the tab panel this rail controls. */
  panelId?: string;
};

// Geometry — keep in sync with the CSS so the disc stays centered in its wrap.
const DISC_SIZE = 72;
const WRAP_SIZE = 112; // leaves 20px ring outside the 72px disc for dots + hover padding
const CENTER = WRAP_SIZE / 2;
const DOT_RADIUS = DISC_SIZE / 2 + 14; // just outside the rim
const TICK_HIT = 20; // square hit target around the dot for the tab button

function angleForIndex(index: number, count: number): number {
  if (count <= 1) return 0;
  return -135 + (index / (count - 1)) * 270;
}

function tickPosition(index: number, count: number): { x: number; y: number } {
  const rad = (angleForIndex(index, count) * Math.PI) / 180;
  // Top of the disc is the indicator (angle = 0). Moving clockwise in CSS
  // rotate() is positive, so in screen space x = +sin, y = -cos.
  const x = CENTER + Math.sin(rad) * DOT_RADIUS;
  const y = CENTER - Math.cos(rad) * DOT_RADIUS;
  return { x, y };
}

export function RotaryModeSelector<T extends string>({
  items,
  activeMode,
  onModeChange,
  ariaLabel = 'Studio mode',
  panelId,
}: Props<T>) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const discRef = useRef<HTMLDivElement | null>(null);
  const activeButtonRef = useRef<HTMLButtonElement | null>(null);

  const count = items.length;
  const rawIndex = items.findIndex((item) => item.id === activeMode);
  const activeIndex = rawIndex >= 0 ? rawIndex : 0;
  const active = items[activeIndex];

  const ticks = useMemo(
    () => items.map((_, i) => tickPosition(i, count)),
    [items, count],
  );

  const cycle = useCallback(
    (dir: 1 | -1) => {
      if (count === 0) return;
      const next = (activeIndex + dir + count) % count;
      onModeChange(items[next].id);
    },
    [activeIndex, count, items, onModeChange],
  );

  // Non-passive wheel listener so we can call preventDefault.
  useEffect(() => {
    const el = discRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      // Only consume the scroll when it's a meaningful vertical gesture.
      if (Math.abs(e.deltaY) < 1) return;
      e.preventDefault();
      cycle(e.deltaY > 0 ? 1 : -1);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [cycle]);

  // When activeMode changes via keyboard, move focus to the new active tab so
  // arrow navigation keeps working. Don't steal focus on click from elsewhere.
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    if (root.contains(document.activeElement)) {
      activeButtonRef.current?.focus();
    }
  }, [activeIndex]);

  const onKeyDown = (e: ReactKeyboardEvent<HTMLElement>) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      e.preventDefault();
      cycle(1);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault();
      cycle(-1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      if (count > 0) onModeChange(items[0].id);
    } else if (e.key === 'End') {
      e.preventDefault();
      if (count > 0) onModeChange(items[count - 1].id);
    }
  };

  const discAngle = angleForIndex(activeIndex, count);

  return (
    <div
      ref={rootRef}
      className="rotary-rail"
      role="tablist"
      aria-label={ariaLabel}
      aria-orientation="vertical"
      onKeyDown={onKeyDown}
    >
      <div
        className="rotary-rail__wrap"
        style={{ width: WRAP_SIZE, height: WRAP_SIZE }}
      >
        {/* Static ticks (dots) — laid out in screen space, don't rotate with disc. */}
        <div className="rotary-rail__ticks" aria-hidden="false">
          {items.map((item, i) => {
            const { x, y } = ticks[i];
            const isActive = i === activeIndex;
            return (
              <button
                key={item.id}
                type="button"
                role="tab"
                id={`rotary-tab-${item.id}`}
                aria-selected={isActive}
                aria-controls={panelId}
                tabIndex={isActive ? 0 : -1}
                ref={isActive ? activeButtonRef : undefined}
                className={`rotary-rail__tick ${isActive ? 'is-active' : ''}`}
                style={{ left: x - TICK_HIT / 2, top: y - TICK_HIT / 2 }}
                onClick={() => onModeChange(item.id)}
                title={item.label}
              >
                <span className="rotary-rail__dot" aria-hidden="true" />
                <span className="rotary-rail__sr">{item.label}</span>
              </button>
            );
          })}
        </div>

        {/* The rotating disc — indicator + hub. Purely decorative. */}
        <div
          ref={discRef}
          className="rotary-rail__disc"
          style={{
            transform: `rotate(${discAngle}deg)`,
            left: (WRAP_SIZE - DISC_SIZE) / 2,
            top: (WRAP_SIZE - DISC_SIZE) / 2,
            width: DISC_SIZE,
            height: DISC_SIZE,
          }}
          onClick={() => cycle(1)}
          aria-hidden="true"
        >
          <div className="rotary-rail__indicator" />
          <div className="rotary-rail__hub" />
        </div>
      </div>

      <div className="rotary-rail__readout" aria-live="polite">
        <span className="rotary-rail__readout-name">
          {active ? active.label : ''}
        </span>
        <span className="rotary-rail__readout-count">
          <strong>{String(activeIndex + 1).padStart(2, '0')}</strong>
          <span aria-hidden="true"> / </span>
          <span className="rotary-rail__sr">of </span>
          {String(count).padStart(2, '0')}
        </span>
      </div>
    </div>
  );
}

export default RotaryModeSelector;
