import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import type { CompleteSongIntentRequest } from '../types/Intent';
import { useMusicBrain } from '../hooks/useMusicBrain';
import { emotionColor, SECTION_COLORS } from '../constants/emotions';

async function safeInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import('@tauri-apps/api/core');
  return invoke(cmd, args) as Promise<T>;
}
async function safeListen(
  event: string,
  handler: (payload: unknown) => void
): Promise<(() => void) | undefined> {
  try {
    const { listen } = await import('@tauri-apps/api/event');
    return await listen(event, (e: { payload: unknown }) => handler(e.payload));
  } catch {
    return undefined;
  }
}

const IS_TAURI = Boolean(
  typeof window !== 'undefined' && (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__
);

const SECTION_NAMES = ["intro", "verse", "chorus", "bridge", "outro", "build", "drop"];
const KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"];
const MODES = ["major", "minor", "dorian", "mixolydian", "lydian", "phrygian", "aeolian", "locrian"];

// Colors come from the shared design-system palette so the mood ring
// matches the Sound Palette and structure timeline (src/constants/emotions.ts).
const MOODS = [
  { id: 'nostalgic', label: 'Nostalgic', color: emotionColor('nostalgic') },
  { id: 'melancholic', label: 'Melancholic', color: emotionColor('melancholic') },
  { id: 'grief', label: 'Grief', color: emotionColor('grief') },
  { id: 'tender', label: 'Tender', color: emotionColor('tender') },
  { id: 'romantic', label: 'Romantic', color: emotionColor('romantic') },
  { id: 'hopeful', label: 'Hopeful', color: emotionColor('hopeful') },
  { id: 'joyful', label: 'Joyful', color: emotionColor('joyful') },
  { id: 'energetic', label: 'Energetic', color: emotionColor('energetic') },
  { id: 'fierce', label: 'Fierce', color: emotionColor('fierce') },
  { id: 'mysterious', label: 'Mysterious', color: emotionColor('mysterious') },
  { id: 'ethereal', label: 'Ethereal', color: emotionColor('ethereal') },
  { id: 'calm', label: 'Calm', color: emotionColor('calm') },
];

const DEMO_INTENT: CompleteSongIntentRequest = {
  core_desire:
    'A driving synthwave track that evokes a nighttime highway drive; resolve internal tension with hopeful momentum.',
  mood_primary: 'Nostalgic',
  genre: 'Synthwave',
  tempo: 120,
  key_mode: 'A minor',
  structure: [
    { name: 'intro', bars: 4, repetitions: 1 },
    { name: 'verse', bars: 8, repetitions: 2 },
    { name: 'chorus', bars: 8, repetitions: 1 },
    { name: 'bridge', bars: 2, repetitions: 1 },
  ],
  instruments: [
    { instrument: 'synth_lead', techniques: ['portamento', 'sustain'] },
    { instrument: 'synth_bass', techniques: [] },
    { instrument: 'piano', techniques: ['staccato'] },
  ],
  allow_legacy_fallback: false,
  groove_feel: 'Straight/Driving',
  narrative_arc: 'Climb-to-Climax',
  rule_to_break: 'parallel fifths',
  rule_justification: 'Intentional tension for emotional payoff in the chorus',
};

export default function IntentBuilder() {
  const { generateFromIntentAsTtg } = useMusicBrain();

  const [intent, setIntent] = useState<CompleteSongIntentRequest>({
    core_desire: '', mood_primary: '', genre: '', tempo: 120,
    key_mode: 'C major',
    structure: [{ name: 'intro', bars: 8, repetitions: 1 }],
    instruments: [{ instrument: 'piano', techniques: [] }],
    allow_legacy_fallback: false,
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState('');
  const [activeJobId, setActiveJobId] = useState<number | null>(null);
  const [meterLevel, setMeterLevel] = useState(0);
  const [selectedSection, setSelectedSection] = useState<number | null>(null);
  const [showLengthWarning, setShowLengthWarning] = useState(false);
  const [showRuleFields, setShowRuleFields] = useState(false);
  const tapsRef = useRef<number[]>([]);
  const unlistenRef = useRef<{ progress?: () => void; result?: () => void }>({});
  const tempoWheelRef = useRef<HTMLDivElement>(null);
  const tempoValueRef = useRef(120);
  const moodRingRef = useRef<HTMLDivElement>(null);
  const moodDragRef = useRef(false);

  useEffect(() => {
    let mounted = true;
    const setup = async () => {
      try {
        const unProg = await safeListen('gen-progress', (payload: { status?: string; percent?: number }) => {
          if (!mounted) return;
          if (payload?.status != null) setJobStatus(payload.status);
          if (typeof payload?.percent === 'number') setMeterLevel(payload.percent / 100);
        });
        if (!mounted) return;
        unlistenRef.current.progress = unProg ?? undefined;
        const unRes = await safeListen('gen-result', (payload: { ok?: boolean; error?: string }) => {
          if (!mounted) return;
          setIsGenerating(false);
          setActiveJobId(null);
          if (payload?.ok) { setJobStatus('COMPLETE'); setMeterLevel(0); setError(null); }
          else { setError(payload?.error ?? 'Engine error'); setJobStatus('FAILED'); }
        });
        if (!mounted) return;
        unlistenRef.current.result = unRes ?? undefined;
      } catch { /* expected in browser */ }
    };
    setup();
    return () => { mounted = false; unlistenRef.current.progress?.(); unlistenRef.current.result?.(); unlistenRef.current = {}; };
  }, []);

  tempoValueRef.current = intent.tempo ?? 120;
  useEffect(() => {
    const el = tempoWheelRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -1 : 1;
      const next = Math.max(40, Math.min(300, tempoValueRef.current + delta));
      tempoValueRef.current = next;
      setIntent(prev => ({ ...prev, tempo: next }));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const totalBars = useMemo(() =>
    intent.structure.reduce((acc, s) => acc + s.bars * (s.repetitions ?? 1), 0),
  [intent.structure]);

  const validation = useMemo(() => ({
    core_desire: intent.core_desire.trim().length >= 1 && intent.core_desire.length <= 1000,
    mood_primary: intent.mood_primary.trim().length >= 1 && intent.mood_primary.length <= 100,
    genre: intent.genre.trim().length >= 1 && intent.genre.length <= 100,
    tempo: (intent.tempo ?? 120) >= 40 && (intent.tempo ?? 120) <= 300,
    totalBars: totalBars <= 1000,
    structureLen: intent.structure.length >= 1,
    instrumentsLen: intent.instruments.length >= 1,
    instrumentsValid: intent.instruments.every(i => i.instrument.trim().length >= 1 && i.instrument.length <= 64),
  }), [intent, totalBars]);

  const isFormValid = Object.values(validation).every(Boolean);
  /** True when every field is valid except totalBars (so Generate can be clicked to show length modal). */
  const isFormValidExceptLength = useMemo(
    () =>
      validation.core_desire &&
      validation.mood_primary &&
      validation.genre &&
      validation.tempo &&
      validation.structureLen &&
      validation.instrumentsLen &&
      validation.instrumentsValid,
    [validation]
  );
  /** When totalBars > 1000 we still allow click so handleGenerate can show the length-warning modal. */
  const canClickGenerate =
    !isGenerating &&
    (totalBars > 1000 ? isFormValidExceptLength : isFormValid);

  const set = useCallback((field: keyof CompleteSongIntentRequest, value: unknown) => {
    setIntent(prev => ({ ...prev, [field]: value }));
  }, []);

  const currentKey = intent.key_mode.split(' ')[0] || 'C';
  const currentMode = intent.key_mode.split(' ')[1] || 'major';

  const handleGenerate = async () => {
    if (!validation.totalBars) {
      setShowLengthWarning(true);
      return;
    }
    if (!isFormValid) return;
    setIsGenerating(true);
    setError(null);
    setJobStatus('COMPILING...');

    try {
      const tempo = Math.round(Math.max(40, Math.min(300, intent.tempo ?? 120)));
      const keyMode = intent.key_mode.trim();
      const keyModeValid = /^[A-G][#b]?\s(major|minor|dorian|mixolydian|lydian|phrygian|locrian)$/.test(keyMode);

      const payload = {
        core_desire: intent.core_desire.trim().substring(0, 1000),
        mood_primary: intent.mood_primary.trim().substring(0, 100),
        genre: intent.genre.trim().substring(0, 100),
        tempo,
        key_mode: keyModeValid ? keyMode : 'C major',
        structure: intent.structure.map(s => ({ name: s.name, bars: s.bars, repetitions: s.repetitions ?? 1 })),
        instruments: intent.instruments.map(i => ({
          instrument: i.instrument.trim().toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_').substring(0, 64),
          techniques: i.techniques ?? [],
        })),
        allow_legacy_fallback: intent.allow_legacy_fallback ?? false,
        groove_feel: intent.groove_feel?.trim() || 'Straight/Driving',
        narrative_arc: intent.narrative_arc?.trim() || 'Climb-to-Climax',
        rule_to_break: intent.rule_to_break?.trim() || null,
        rule_justification: intent.rule_justification?.trim() || null,
      };

      if (IS_TAURI) {
        const jobId = await safeInvoke<number>('start_generation', { intentJson: JSON.stringify(payload) });
        setActiveJobId(jobId);
        setJobStatus('GENERATING...');
      } else {
        setJobStatus('GENERATING...');
        await generateFromIntentAsTtg(intent);
        setIsGenerating(false);
        setJobStatus('COMPLETE');
        setMeterLevel(0);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
      setIsGenerating(false);
      setJobStatus('');
    }
  };

  const handleCancel = async () => {
    if (activeJobId == null) return;
    try { await safeInvoke('cancel_generation', { jobId: activeJobId }); setJobStatus('CANCELLING...'); }
    catch { /* ignore */ }
  };

  const handleTap = useCallback(() => {
    const now = performance.now();
    tapsRef.current.push(now);
    if (tapsRef.current.length > 6) tapsRef.current.shift();
    if (tapsRef.current.length >= 2) {
      const intervals = [];
      for (let i = 1; i < tapsRef.current.length; i++) intervals.push(tapsRef.current[i] - tapsRef.current[i - 1]);
      const avg = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const bpm = Math.round(60000 / avg);
      if (bpm >= 40 && bpm <= 300) set('tempo', bpm);
    }
  }, [set]);

  const updateSection = (idx: number, field: string, value: unknown) => {
    const next = [...intent.structure];
    (next[idx] as unknown as Record<string, unknown>)[field] = value;
    set('structure', next);
  };

  const addSection = () => set('structure', [...intent.structure, { name: 'verse', bars: 8, repetitions: 1 }]);
  // Removal is recoverable via an inline Undo chip — a blocking confirm dialog
  // would add friction to every edit, but accidental deletes still lose work.
  const [lastRemoved, setLastRemoved] =
    useState<{ section: CompleteSongIntentRequest['structure'][number]; idx: number } | null>(null);
  const removeSection = (idx: number) => {
    setLastRemoved({ section: intent.structure[idx], idx });
    set('structure', intent.structure.filter((_, i) => i !== idx));
    setSelectedSection(null);
  };
  const undoRemoveSection = () => {
    if (!lastRemoved) return;
    const next = [...intent.structure];
    next.splice(Math.min(lastRemoved.idx, next.length), 0, lastRemoved.section);
    set('structure', next);
    setLastRemoved(null);
  };

  const updateInstrument = (idx: number, field: string, value: unknown) => {
    const next = [...intent.instruments];
    (next[idx] as unknown as Record<string, unknown>)[field] = value;
    set('instruments', next);
  };

  const addInstrument = () => set('instruments', [...intent.instruments, { instrument: '', techniques: [] }]);
  const removeInstrument = (idx: number) => set('instruments', intent.instruments.filter((_, i) => i !== idx));

  const handleLoadDemo = () => {
    setIntent({ ...DEMO_INTENT });
    setError(null);
    setJobStatus('');
    setShowRuleFields(true);
  };

  const activeMoodColor = MOODS.find(m => m.label.toLowerCase() === intent.mood_primary.toLowerCase())?.color;

  // Drag-to-sweep mood selection: the ring reads as a rotary control, so let
  // it behave like one — dragging around the ring sweeps through moods.
  // Button clicks still work; a sweep just selects the nearest node live.
  const selectMoodFromPoint = useCallback((clientX: number, clientY: number) => {
    const el = moodRingRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const dx = clientX - (rect.left + rect.width / 2);
    const dy = clientY - (rect.top + rect.height / 2);
    if (Math.hypot(dx, dy) < 36) return; // dead zone over the vinyl center
    // Nodes are laid out from the top, clockwise — mirror that mapping.
    const deg = (Math.atan2(dy, dx) * 180 / Math.PI + 90 + 360) % 360;
    const idx = Math.round(deg / (360 / MOODS.length)) % MOODS.length;
    const label = MOODS[idx].label;
    setIntent(prev => (prev.mood_primary === label ? prev : { ...prev, mood_primary: label }));
  }, []);

  return (
    <div className="gen-workspace">

      {/* ── Top bar ── */}
      <div className="gen-topbar">
        <button type="button" className="gen-demo-btn" onClick={handleLoadDemo}>Load Demo</button>
        <div className="gen-topbar-right">
          {isGenerating && <button type="button" className="gen-cancel-btn" onClick={handleCancel}>Cancel</button>}
          <button
            type="button"
            className={`gen-go-btn ${!canClickGenerate ? 'gen-go-disabled' : ''}`}
            disabled={!canClickGenerate}
            onClick={handleGenerate}
          >
            {isGenerating ? jobStatus : 'Generate'}
          </button>
        </div>
      </div>

      {/* ── Progress ── */}
      {isGenerating && meterLevel > 0 && (
        <div className="gen-progress-track">
          <div className="gen-progress-fill" style={{ width: `${Math.min(100, meterLevel * 100)}%` }} />
        </div>
      )}

      {/* ── Three-column workspace ── */}
      <div className="gen-columns">

        {/* ─ Left: Emotion ─ */}
        <section className="gen-panel gen-panel-emotion">
          <h3 className="gen-panel-label">Mood</h3>
          <div className="vinyl-wrap">
            <div className="vinyl-disc">
              <div className="vinyl-spin" />
              <div className="vinyl-center" style={activeMoodColor ? { borderColor: activeMoodColor, boxShadow: `0 0 20px ${activeMoodColor}40` } : undefined}>
                <span className="vinyl-center-text">{intent.mood_primary || 'Select'}</span>
              </div>
            </div>
            <div
              className="mood-ring"
              ref={moodRingRef}
              style={{ touchAction: 'none' }}
              onPointerDown={(e) => {
                moodDragRef.current = true;
                e.currentTarget.setPointerCapture(e.pointerId);
                selectMoodFromPoint(e.clientX, e.clientY);
              }}
              onPointerMove={(e) => {
                if (moodDragRef.current) selectMoodFromPoint(e.clientX, e.clientY);
              }}
              onPointerUp={() => { moodDragRef.current = false; }}
              onPointerCancel={() => { moodDragRef.current = false; }}
            >
              {MOODS.map((m, i) => {
                const angle = (i * 360 / MOODS.length) - 90;
                const rad = angle * Math.PI / 180;
                const R = 118;
                const x = Math.cos(rad) * R;
                const y = Math.sin(rad) * R;
                const active = intent.mood_primary.toLowerCase() === m.label.toLowerCase();
                return (
                  <button
                    key={m.id} type="button"
                    className={`mood-node ${active ? 'mood-node-active' : ''}`}
                    style={{ transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`, '--mc': m.color } as React.CSSProperties}
                    onClick={() => set('mood_primary', m.label)}
                  >{m.label}</button>
                );
              })}
            </div>
          </div>
        </section>

        {/* ─ Center: Song Details ─ */}
        <section className="gen-panel gen-panel-details">
          <h3 className="gen-panel-label">Song Details</h3>
          <div className="gf-group">
            <label className="gf-label">Describe your song</label>
            <textarea
              className="gf-textarea"
              value={intent.core_desire}
              onChange={e => set('core_desire', e.target.value)}
              placeholder="A driving synthwave track that evokes a nighttime highway drive..."
              rows={3} maxLength={1000}
            />
          </div>
          <div className="gf-row">
            <div className="gf-group gf-grow">
              <label className="gf-label">Genre</label>
              <input className="gf-input" value={intent.genre} onChange={e => set('genre', e.target.value)} placeholder="Synthwave, Jazz, Orchestral..." maxLength={100} />
            </div>
            <div className="gf-group">
              <label className="gf-label">Key</label>
              <div className="gf-key-row">
                <select className="gf-select" value={currentKey} onChange={e => set('key_mode', `${e.target.value} ${currentMode}`)}>
                  {KEYS.map(k => <option key={k}>{k}</option>)}
                </select>
                <select className="gf-select" value={currentMode} onChange={e => set('key_mode', `${currentKey} ${e.target.value}`)}>
                  {MODES.map(m => <option key={m}>{m}</option>)}
                </select>
              </div>
            </div>
          </div>
          <div className="gf-row">
            <div className="gf-group gf-grow">
              <label className="gf-label">Groove / Feel</label>
              <input className="gf-input" value={intent.groove_feel ?? ''} onChange={e => set('groove_feel', e.target.value || undefined)} placeholder="Straight/Driving, Laid-back..." />
            </div>
            <div className="gf-group gf-grow">
              <label className="gf-label">Narrative Arc</label>
              <input className="gf-input" value={intent.narrative_arc ?? ''} onChange={e => set('narrative_arc', e.target.value || undefined)} placeholder="Climb-to-Climax, Plateau..." />
            </div>
          </div>
          <button type="button" className="gf-toggle-link" onClick={() => setShowRuleFields(!showRuleFields)}>
            {showRuleFields ? '− Hide rule break' : '+ Rule break (optional)'}
          </button>
          {showRuleFields && (
            <div className="gf-row gf-rule-row">
              <div className="gf-group gf-grow">
                <label className="gf-label">Rule to break</label>
                <input className="gf-input" value={intent.rule_to_break ?? ''} onChange={e => set('rule_to_break', e.target.value || null)} placeholder="e.g. parallel fifths" />
              </div>
              <div className="gf-group gf-grow">
                <label className="gf-label">Justification</label>
                <input className="gf-input" value={intent.rule_justification ?? ''} onChange={e => set('rule_justification', e.target.value || null)} placeholder="Narrative reason..." />
              </div>
            </div>
          )}
        </section>

        {/* ─ Right: Structure + Instruments ─ */}
        <section className="gen-panel gen-panel-arrange">
          <h3 className="gen-panel-label">Arrangement</h3>

          {/* Structure timeline */}
          <div className="stl-track">
            {intent.structure.map((sec, i) => {
              const pct = Math.max(8, (sec.bars * (sec.repetitions ?? 1)) / Math.max(1, totalBars) * 100);
              return (
                <button
                  key={i} type="button"
                  className={`stl-block ${selectedSection === i ? 'stl-block-sel' : ''}`}
                  style={{ flex: `${pct} 0 0%`, '--bc': SECTION_COLORS[sec.name] || '#5c7c8a' } as React.CSSProperties}
                  onClick={() => setSelectedSection(selectedSection === i ? null : i)}
                >
                  <span className="stl-name">{sec.name}</span>
                  <span className="stl-bars">{sec.bars}{(sec.repetitions ?? 1) > 1 ? `×${sec.repetitions}` : ''}</span>
                </button>
              );
            })}
            <button type="button" className="stl-add" onClick={addSection}>+</button>
          </div>

          {/* Section editor */}
          {selectedSection !== null && selectedSection < intent.structure.length && (
            <div className="stl-editor">
              <select className="gf-select" value={intent.structure[selectedSection].name} onChange={e => updateSection(selectedSection, 'name', e.target.value)}>
                {SECTION_NAMES.map(n => <option key={n}>{n}</option>)}
              </select>
              <label className="stl-ed-label">Bars
                <input type="number" className="stl-ed-num" min={1} max={128} value={intent.structure[selectedSection].bars} onChange={e => updateSection(selectedSection, 'bars', parseInt(e.target.value) || 1)} />
              </label>
              <label className="stl-ed-label">Reps
                <input type="number" className="stl-ed-num" min={1} max={16} value={intent.structure[selectedSection].repetitions ?? 1} onChange={e => updateSection(selectedSection, 'repetitions', parseInt(e.target.value) || 1)} />
              </label>
              <button type="button" className="stl-remove" onClick={() => removeSection(selectedSection)} disabled={intent.structure.length <= 1}>Remove</button>
            </div>
          )}

          {/* Undo chip for the last removed section */}
          {lastRemoved && (
            <div
              role="status"
              style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, fontSize: 12, opacity: 0.85 }}
            >
              <span>
                Removed {lastRemoved.section.name} ({lastRemoved.section.bars} bars)
              </span>
              <button type="button" className="gf-toggle-link" onClick={undoRemoveSection}>
                Undo
              </button>
              <button
                type="button"
                className="gf-toggle-link"
                onClick={() => setLastRemoved(null)}
                aria-label="Dismiss undo"
              >
                ×
              </button>
            </div>
          )}

          {/* Instruments */}
          <h3 className="gen-panel-label" style={{ marginTop: 16 }}>Instruments</h3>
          <div className="inst-rack">
            {intent.instruments.map((inst, i) => (
              <div key={i} className="inst-module">
                <div className="inst-knob">
                  <span className="inst-knob-label">{inst.instrument ? inst.instrument.charAt(0).toUpperCase() : '?'}</span>
                </div>
                <input className="inst-name" value={inst.instrument} onChange={e => updateInstrument(i, 'instrument', e.target.value)} placeholder="instrument" maxLength={64} />
                <input className="inst-tech" value={inst.techniques?.join(', ') || ''} onChange={e => updateInstrument(i, 'techniques', e.target.value.split(',').map(s => s.trim()).filter(Boolean))} placeholder="techniques" />
                <button type="button" className="inst-remove" onClick={() => removeInstrument(i)} disabled={intent.instruments.length <= 1}>×</button>
              </div>
            ))}
            <button type="button" className="inst-add" onClick={addInstrument}>+ Add</button>
          </div>
        </section>
      </div>

      {/* ── Tempo bar ── */}
      <div className="gen-tempo-bar">
        <div className="tempo-bpm" ref={tempoWheelRef}>
          <span className="tempo-num">{intent.tempo}</span>
          <span className="tempo-unit">BPM</span>
        </div>
        <button type="button" className="tempo-tap" onClick={handleTap}>TAP</button>
        <input type="range" className="tempo-slider" min={40} max={300} value={intent.tempo} onChange={e => set('tempo', parseInt(e.target.value))} />
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="gen-error">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* ── Length warning modal ── */}
      {showLengthWarning && (
        <div
          className="gen-modal-overlay"
          onClick={() => setShowLengthWarning(false)}
          onKeyDown={e => { if (e.key === 'Escape') setShowLengthWarning(false); }}
        >
          <div
            className="gen-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Song length warning"
            onClick={e => e.stopPropagation()}
          >
            <p>Song length exceeded ({totalBars} bars — the limit is 1000). Shorten the structure to generate.</p>
            <button type="button" autoFocus onClick={() => setShowLengthWarning(false)}>OK</button>
          </div>
        </div>
      )}
    </div>
  );
}
