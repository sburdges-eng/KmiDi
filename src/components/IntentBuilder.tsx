import { useState, useMemo, useEffect, useRef } from 'react';
import type { CompleteSongIntentRequest } from '../types/Intent';
import { useMusicBrain, buildGeneratePayload } from '../hooks/useMusicBrain';

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

// Constants strictly mapped to Pydantic (music_brain/engine_api/schema.py):
// - key_mode: ^[A-G][#b]?\s(major|minor|...) → two <select>s (Root + Mode), never free text
// - structure.name: ^(intro|verse|chorus|bridge|outro|build|drop)$ → SECTION_NAMES dropdown only
// - total_bars: computed like Python validate_total_duration, shown live, submit disabled if > 1000
// - bars 1–128, repetitions 1–16, tempo 40–300, maxLength on text fields per schema
const SECTION_NAMES = ["intro", "verse", "chorus", "bridge", "outro", "build", "drop"];
const KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"];
const MODES = ["major", "minor", "dorian", "mixolydian", "lydian", "phrygian", "aeolian", "locrian"];

/** Demo intent: all interactive features, ~1 min at 120 BPM (30 bars). Matches scripts/demo_run.py. */
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
  const { generateMusic } = useMusicBrain();

  const [intent, setIntent] = useState<CompleteSongIntentRequest>({
    core_desire: '',
    mood_primary: '',
    genre: '',
    tempo: 120,
    key_mode: 'C major',
    structure: [{ name: 'intro', bars: 8, repetitions: 1 }],
    instruments: [{ instrument: 'piano', techniques: [] }],
    allow_legacy_fallback: false,
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState('AWAITING INTENT...');
  const [activeJobId, setActiveJobId] = useState<number | null>(null);
  const [meterLevel, setMeterLevel] = useState(0);

  // Refs so cleanup can run unlisten even when setup() is still async
  const unlistenRef = useRef<{ progress?: () => void; result?: () => void }>({});

  // ── Real Tauri event listeners (gen-progress, gen-result) ──
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

        const unRes = await safeListen('gen-result', (payload: { ok?: boolean; error?: string; midi?: unknown }) => {
          if (!mounted) return;
          setIsGenerating(false);
          setActiveJobId(null);
          if (payload?.ok) {
            setJobStatus('GENERATION COMPLETE');
            setMeterLevel(0);
            setError(null);
          } else {
            setError(payload?.error ?? 'Engine error');
            setJobStatus('FAILED');
          }
        });
        if (!mounted) return;
        unlistenRef.current.result = unRes ?? undefined;
      } catch (e) {
        if (mounted) console.warn('Tauri event setup failed (expected in browser):', e);
      }
    };
    setup();

    return () => {
      mounted = false;
      unlistenRef.current.progress?.();
      unlistenRef.current.result?.();
      unlistenRef.current = {};
    };
  }, []);

  // 2. Real-time Validations (Mirroring Pydantic Exactly)
  const totalBars = useMemo(() => {
    return intent.structure.reduce((acc, sec) => acc + (sec.bars * (sec.repetitions ?? 1)), 0);
  }, [intent.structure]);

  const validation = useMemo(() => {
    return {
      core_desire: intent.core_desire.trim().length >= 1 && intent.core_desire.length <= 1000,
      mood_primary: intent.mood_primary.trim().length >= 1 && intent.mood_primary.length <= 100,
      genre: intent.genre.trim().length >= 1 && intent.genre.length <= 100,
      tempo: (intent.tempo ?? 120) >= 40 && (intent.tempo ?? 120) <= 300,
      totalBars: totalBars <= 1000,
      structureLen: intent.structure.length >= 1,
      instrumentsLen: intent.instruments.length >= 1,
      instrumentsValid: intent.instruments.every(i => i.instrument.trim().length >= 1 && i.instrument.length <= 64)
    };
  }, [intent, totalBars]);

  const isFormValid = Object.values(validation).every(Boolean);

  // 3. Update Helpers
  const updateBase = (field: keyof CompleteSongIntentRequest, value: any) => {
    setIntent(prev => ({ ...prev, [field]: value }));
  };

  // Helper to split and update the strict regex key_mode
  const currentKey = intent.key_mode.split(' ')[0] || 'C';
  const currentMode = intent.key_mode.split(' ')[1] || 'major';

  // ── Strict Pydantic contract adapter: flat shape for Rust → Python ──
  const handleGenerate = async () => {
    if (!isFormValid) return;
    setIsGenerating(true);
    setError(null);
    setJobStatus('COMPILING INTENT...');

    try {
      const tempoResolved = Math.round(
        (intent.tempo ?? 120) >= 40 && (intent.tempo ?? 120) <= 300
          ? (intent.tempo ?? 120)
          : 120
      );
      const keyMode = intent.key_mode.trim();
      const keyModeValid = /^[A-G][#b]?\s(major|minor|dorian|mixolydian|lydian|phrygian|locrian)$/.test(keyMode);

      const pydanticPayload = {
        core_desire: intent.core_desire.trim().substring(0, 1000),
        mood_primary: intent.mood_primary.trim().substring(0, 100),
        genre: intent.genre.trim().substring(0, 100),
        tempo: tempoResolved,
        key_mode: keyModeValid ? keyMode : 'C major',
        structure: intent.structure.map((s) => ({
          name: s.name,
          bars: s.bars,
          repetitions: s.repetitions ?? 1,
        })),
        instruments: intent.instruments.map((i) => ({
          instrument: i.instrument.trim().toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_').substring(0, 64),
          techniques: i.techniques ?? [],
        })),
        allow_legacy_fallback: intent.allow_legacy_fallback ?? false,
        groove_feel: intent.groove_feel?.trim() || 'Straight/Driving',
        narrative_arc: intent.narrative_arc?.trim() || 'Climb-to-Climax',
        rule_to_break: intent.rule_to_break?.trim() || null,
        rule_justification: intent.rule_justification?.trim() || null,
      };

      console.log('Dispatching valid Pydantic payload:', pydanticPayload);

      if (IS_TAURI) {
        const jobId = await safeInvoke<number>('start_generation', {
          intentJson: JSON.stringify(pydanticPayload),
        });
        setActiveJobId(jobId);
        setJobStatus('GENERATING...');
      } else {
        setJobStatus('GENERATING VIA HTTP...');
        const apiPayload = buildGeneratePayload(intent);
        const result = await generateMusic(apiPayload);
        console.log('Generation result:', result);
        setIsGenerating(false);
        setJobStatus('GENERATION COMPLETE');
        setMeterLevel(0);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[IntentBuilder] handleGenerate failed:', err);
      setError(message);
      setIsGenerating(false);
      setJobStatus('FAILED TO DISPATCH');
    }
  };

  const handleCancel = async () => {
    if (activeJobId == null) return;
    try {
      await safeInvoke('cancel_generation', { jobId: activeJobId });
      setJobStatus('CANCELLING...');
    } catch (err) {
      console.error('Failed to cancel:', err);
    }
  };

  const handleLoadDemo = () => {
    setIntent({ ...DEMO_INTENT });
    setError(null);
    setJobStatus('AWAITING INTENT...');
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8 bg-white text-slate-900 rounded-xl shadow-sm border border-slate-200">
      <header className="border-b pb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">LASKmiDi Intent Builder</h1>
          <p className="text-slate-500 mt-1">Design your generation intent. Constraints are strictly enforced by the API schema.</p>
        </div>
        <button
          type="button"
          onClick={handleLoadDemo}
          className="shrink-0 py-2 px-4 rounded-md font-medium border border-slate-300 bg-white text-slate-700 hover:bg-slate-50"
        >
          Load demo (1 min)
        </button>
      </header>

      {/* GLOBAL SETTINGS */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold border-b pb-2">1. Global Metadata</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="col-span-1 md:col-span-2">
            <label className="block text-sm font-medium mb-1">
              Core Desire <span className={!validation.core_desire ? 'text-red-500' : ''}>*</span>
            </label>
            <textarea
              value={intent.core_desire}
              onChange={e => updateBase('core_desire', e.target.value)}
              className="w-full border border-slate-300 p-2 rounded-md focus:ring-2 focus:ring-blue-500 focus:outline-none"
              placeholder="A driving synthwave track that evokes a nighttime highway drive..."
              rows={3}
              maxLength={1000}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Primary Mood <span className={!validation.mood_primary ? 'text-red-500' : ''}>*</span>
            </label>
            <input
              type="text" value={intent.mood_primary}
              onChange={e => updateBase('mood_primary', e.target.value)}
              className="w-full border border-slate-300 p-2 rounded-md" placeholder="e.g., Melancholic, Energetic"
              maxLength={100}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Genre <span className={!validation.genre ? 'text-red-500' : ''}>*</span>
            </label>
            <input
              type="text" value={intent.genre}
              onChange={e => updateBase('genre', e.target.value)}
              className="w-full border border-slate-300 p-2 rounded-md" placeholder="e.g., Synthwave, Orchestral"
              maxLength={100}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1 flex justify-between">
              <span>Tempo (BPM)</span>
              <span className={!validation.tempo ? "text-red-500 font-bold" : "text-slate-500"}>{intent.tempo}</span>
            </label>
            <input
              type="range" min="40" max="300" value={intent.tempo}
              onChange={e => updateBase('tempo', parseInt(e.target.value))}
              className="w-full accent-blue-600"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Key & Mode</label>
            <div className="flex gap-2">
              <select
                className="w-1/2 border border-slate-300 p-2 rounded-md bg-white outline-none"
                value={currentKey}
                onChange={e => updateBase('key_mode', `${e.target.value} ${currentMode}`)}
              >
                {KEYS.map(k => <option key={k} value={k}>{k}</option>)}
              </select>
              <select
                className="w-1/2 border border-slate-300 p-2 rounded-md bg-white outline-none"
                value={currentMode}
                onChange={e => updateBase('key_mode', `${currentKey} ${e.target.value}`)}
              >
                {MODES.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-200">
          <div>
            <label className="block text-sm font-medium mb-1 text-slate-600">Groove / Feel</label>
            <input
              type="text"
              value={intent.groove_feel ?? ''}
              onChange={e => updateBase('groove_feel', e.target.value || undefined)}
              className="w-full border border-slate-300 p-2 rounded-md"
              placeholder="e.g. Straight/Driving, Laid-back"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1 text-slate-600">Narrative arc</label>
            <input
              type="text"
              value={intent.narrative_arc ?? ''}
              onChange={e => updateBase('narrative_arc', e.target.value || undefined)}
              className="w-full border border-slate-300 p-2 rounded-md"
              placeholder="e.g. Climb-to-Climax, Plateau"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-1 text-slate-600">Rule to break (optional)</label>
            <input
              type="text"
              value={intent.rule_to_break ?? ''}
              onChange={e => updateBase('rule_to_break', e.target.value || null)}
              className="w-full border border-slate-300 p-2 rounded-md"
              placeholder="Intentional theory violation for effect"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-1 text-slate-600">Rule justification (optional)</label>
            <input
              type="text"
              value={intent.rule_justification ?? ''}
              onChange={e => updateBase('rule_justification', e.target.value || null)}
              className="w-full border border-slate-300 p-2 rounded-md"
              placeholder="Narrative reason for the rule break"
            />
          </div>
        </div>
      </section>

      {/* STRUCTURE BUILDER */}
      <section className="space-y-4">
        <div className="flex justify-between items-end border-b pb-2">
          <h2 className="text-xl font-semibold">2. Structure</h2>
          <span className={`text-sm font-bold ${validation.totalBars ? 'text-slate-600' : 'text-red-600'}`}>
            Total Bars: {totalBars} / 1000
          </span>
        </div>

        <div className="space-y-3">
          {intent.structure.map((sec, idx) => (
            <div key={idx} className="flex flex-wrap md:flex-nowrap gap-3 items-center bg-slate-50 p-3 rounded-md border border-slate-200">
              <select
                value={sec.name}
                onChange={e => {
                  const newStruct = [...intent.structure];
                  newStruct[idx].name = e.target.value;
                  updateBase('structure', newStruct);
                }}
                className="border border-slate-300 p-2 rounded-md uppercase text-sm bg-white flex-1 min-w-[120px] outline-none"
              >
                {SECTION_NAMES.map(name => <option key={name} value={name}>{name}</option>)}
              </select>

              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-500">Bars:</label>
                <input
                  type="number" min="1" max="128" value={sec.bars}
                  onChange={e => {
                    const newStruct = [...intent.structure];
                    newStruct[idx].bars = parseInt(e.target.value) || 1;
                    updateBase('structure', newStruct);
                  }}
                  className="border border-slate-300 p-2 rounded-md w-20 text-center outline-none"
                />
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-500">Reps:</label>
                <input
                  type="number" min="1" max="16" value={sec.repetitions || 1}
                  onChange={e => {
                    const newStruct = [...intent.structure];
                    newStruct[idx].repetitions = parseInt(e.target.value) || 1;
                    updateBase('structure', newStruct);
                  }}
                  className="border border-slate-300 p-2 rounded-md w-20 text-center outline-none"
                />
              </div>

              <button
                onClick={() => {
                  const newStruct = intent.structure.filter((_, i) => i !== idx);
                  updateBase('structure', newStruct);
                }}
                disabled={intent.structure.length <= 1}
                className="ml-auto text-red-500 hover:text-red-700 font-bold px-2 disabled:opacity-30"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
        <button
          onClick={() => updateBase('structure', [...intent.structure, { name: 'verse', bars: 8, repetitions: 1 }])}
          className="text-sm font-medium text-blue-600 hover:text-blue-800 flex items-center gap-1"
        >
          <span>+</span> Add Section
        </button>
      </section>

      {/* INSTRUMENTS MANAGER */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold border-b pb-2">3. Instruments (Tracks)</h2>
        <div className="space-y-3">
          {intent.instruments.map((inst, idx) => (
            <div key={idx} className="flex flex-wrap md:flex-nowrap gap-3 items-center bg-slate-50 p-3 rounded-md border border-slate-200">
              <input
                type="text" value={inst.instrument}
                onChange={e => {
                  const newInst = [...intent.instruments];
                  newInst[idx].instrument = e.target.value;
                  updateBase('instruments', newInst);
                }}
                placeholder="Instrument (e.g., piano, synth_bass)"
                className="border border-slate-300 p-2 rounded-md flex-1 min-w-[200px] outline-none"
                maxLength={64}
              />
              <input
                type="text" value={inst.techniques?.join(', ') || ''}
                onChange={e => {
                  const newInst = [...intent.instruments];
                  // Automatically parse comma-separated strings to an Array of strings
                  newInst[idx].techniques = e.target.value.split(',').map(s => s.trim()).filter(Boolean);
                  updateBase('instruments', newInst);
                }}
                placeholder="Techniques (comma separated)"
                className="border border-slate-300 p-2 rounded-md flex-1 min-w-[200px] outline-none"
              />
              <button
                onClick={() => {
                  const newInst = intent.instruments.filter((_, i) => i !== idx);
                  updateBase('instruments', newInst);
                }}
                disabled={intent.instruments.length <= 1}
                className="ml-auto text-red-500 hover:text-red-700 font-bold px-2 disabled:opacity-30"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
        <button
          onClick={() => updateBase('instruments', [...intent.instruments, { instrument: '', techniques: [] }])}
          className="text-sm font-medium text-blue-600 hover:text-blue-800 flex items-center gap-1"
        >
          <span>+</span> Add Instrument
        </button>
      </section>

      {/* ERROR REPORTING */}
      {error && (
        <div className="p-4 bg-red-50 text-red-700 border border-red-200 rounded-md text-sm">
          <strong>Backend Error:</strong> {error}
        </div>
      )}

      {/* SUBMISSION */}
      <div className="pt-6 border-t border-slate-200 flex flex-col sm:flex-row justify-between items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
          <input
            type="checkbox"
            checked={intent.allow_legacy_fallback}
            onChange={e => updateBase('allow_legacy_fallback', e.target.checked)}
            className="rounded text-blue-600 focus:ring-blue-500"
          />
          Allow Legacy Generation Fallback
        </label>

        <div className="flex flex-wrap items-center gap-3">
          {isGenerating && (
            <button
              type="button"
              onClick={handleCancel}
              className="py-2 px-4 rounded-md font-medium bg-slate-600 text-white hover:bg-slate-700"
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={!isFormValid || isGenerating}
            className={`py-3 px-8 rounded-md font-bold text-white transition-all
              ${isGenerating || !isFormValid ? 'bg-slate-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-sm'}`}
          >
            {isGenerating ? jobStatus : '▶ GENERATE'}
          </button>
        </div>
        {meterLevel > 0 && isGenerating && (
          <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${Math.min(100, meterLevel * 100)}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
