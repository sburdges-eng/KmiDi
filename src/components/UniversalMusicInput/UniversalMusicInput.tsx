import { useState, useCallback, useMemo, useEffect } from 'react';
import { SideAPane } from './SideAPane';
import { SideBPane } from './SideBPane';
import { IntentSummaryBar } from './IntentSummaryBar';
import { ToolDrawer } from './ToolDrawer';
import { useTextParse } from '../../hooks/useTextParse';
import { useMusicBrain, buildGeneratePayload } from '../../hooks/useMusicBrain';
import type { CompleteSongIntentRequest } from '../../types/Intent';
import type { ParseTextResponse } from '../../types/Interpretation';

// Minimal placeholder taxonomy until the full data layer is loaded
const PLACEHOLDER_TAXONOMY = [
  {
    id: 'harmony', label: 'Harmony', keywords: ['harmony', 'chords', 'keys'],
    children: [
      {
        id: 'harmony.scales', label: 'Scales & Modes', keywords: ['scales', 'modes'],
        children: [
          { id: 'harmony.scales.major', label: 'Major (Ionian)', keywords: ['major', 'ionian', 'happy', 'bright'] },
          { id: 'harmony.scales.minor', label: 'Minor (Aeolian)', keywords: ['minor', 'aeolian', 'sad', 'dark'] },
          { id: 'harmony.scales.dorian', label: 'Dorian', keywords: ['dorian', 'jazzy', 'soulful'] },
          { id: 'harmony.scales.phrygian', label: 'Phrygian', keywords: ['phrygian', 'spanish', 'metal', 'dark'] },
          { id: 'harmony.scales.lydian', label: 'Lydian', keywords: ['lydian', 'dreamy', 'bright', 'floating'] },
          { id: 'harmony.scales.mixolydian', label: 'Mixolydian', keywords: ['mixolydian', 'bluesy', 'dominant'] },
          { id: 'harmony.scales.locrian', label: 'Locrian', keywords: ['locrian', 'diminished', 'unstable'] },
          { id: 'harmony.scales.pentatonic-minor', label: 'Pentatonic Minor', keywords: ['pentatonic', 'blues', 'rock'] },
          { id: 'harmony.scales.blues', label: 'Blues Scale', keywords: ['blues', 'bluesy'] },
        ],
      },
    ],
  },
  {
    id: 'rhythm', label: 'Rhythm', keywords: ['rhythm', 'beat', 'groove'],
    children: [
      {
        id: 'rhythm.density', label: 'Density', keywords: ['density', 'busy', 'sparse'],
        children: [
          { id: 'rhythm.density.sparse', label: 'Sparse', keywords: ['sparse', 'minimal', 'space'] },
          { id: 'rhythm.density.medium', label: 'Medium', keywords: ['medium', 'balanced'] },
          { id: 'rhythm.density.dense', label: 'Dense', keywords: ['dense', 'busy', 'fast', 'complex'] },
        ],
      },
      {
        id: 'rhythm.groove-feel', label: 'Groove Feel', keywords: ['groove', 'feel'],
        children: [
          { id: 'rhythm.groove-feel.straight-driving', label: 'Straight/Driving', keywords: ['straight', 'driving', 'tight'] },
          { id: 'rhythm.groove-feel.laid-back', label: 'Laid Back', keywords: ['laid-back', 'relaxed', 'chill'] },
          { id: 'rhythm.groove-feel.swung', label: 'Swung', keywords: ['swing', 'swung', 'shuffle'] },
          { id: 'rhythm.groove-feel.syncopated', label: 'Syncopated', keywords: ['syncopated', 'funky', 'offbeat'] },
        ],
      },
    ],
  },
  {
    id: 'dynamics', label: 'Dynamics', keywords: ['dynamics', 'volume', 'loud', 'quiet'],
    children: [
      { id: 'dynamics.levels.pp', label: 'Pianissimo (pp)', keywords: ['very quiet', 'whisper', 'pp'] },
      { id: 'dynamics.levels.p', label: 'Piano (p)', keywords: ['quiet', 'soft', 'gentle'] },
      { id: 'dynamics.levels.mf', label: 'Mezzo Forte (mf)', keywords: ['moderate', 'medium'] },
      { id: 'dynamics.levels.f', label: 'Forte (f)', keywords: ['loud', 'strong'] },
      { id: 'dynamics.levels.ff', label: 'Fortissimo (ff)', keywords: ['very loud', 'powerful', 'aggressive'] },
    ],
  },
  {
    id: 'articulation', label: 'Articulation', keywords: ['articulation', 'technique'],
    children: [
      { id: 'articulation.staccato', label: 'Staccato', keywords: ['staccato', 'short', 'detached'] },
      { id: 'articulation.legato', label: 'Legato', keywords: ['legato', 'smooth', 'connected'] },
      { id: 'articulation.vibrato', label: 'Vibrato', keywords: ['vibrato', 'wobble'] },
      { id: 'articulation.portamento', label: 'Portamento', keywords: ['portamento', 'slide', 'glide'] },
      { id: 'articulation.tremolo', label: 'Tremolo', keywords: ['tremolo', 'rapid', 'shake'] },
    ],
  },
  {
    id: 'timbre', label: 'Timbre', keywords: ['timbre', 'instrument', 'sound'],
    children: [
      {
        id: 'timbre.instruments', label: 'Instruments', keywords: ['instruments'],
        children: [
          { id: 'timbre.instruments.piano', label: 'Piano', keywords: ['piano', 'keys', 'keyboard'] },
          { id: 'timbre.instruments.electric-guitar', label: 'Electric Guitar', keywords: ['guitar', 'electric guitar'] },
          { id: 'timbre.instruments.acoustic-guitar', label: 'Acoustic Guitar', keywords: ['acoustic', 'acoustic guitar'] },
          { id: 'timbre.instruments.bass-guitar', label: 'Bass Guitar', keywords: ['bass', 'bass guitar'] },
          { id: 'timbre.instruments.synth-lead', label: 'Synth Lead', keywords: ['synth', 'synthesizer', 'lead'] },
          { id: 'timbre.instruments.synth-pad', label: 'Synth Pad', keywords: ['pad', 'synth pad', 'ambient'] },
          { id: 'timbre.instruments.strings', label: 'Strings', keywords: ['strings', 'orchestral', 'violin', 'cello'] },
          { id: 'timbre.instruments.brass', label: 'Brass', keywords: ['brass', 'trumpet', 'horn'] },
          { id: 'timbre.instruments.drums', label: 'Drums', keywords: ['drums', 'percussion', 'beat'] },
          { id: 'timbre.instruments.sax', label: 'Saxophone', keywords: ['sax', 'saxophone'] },
          { id: 'timbre.instruments.808', label: '808', keywords: ['808', 'trap', 'sub bass'] },
        ],
      },
      {
        id: 'timbre.effects', label: 'Effects', keywords: ['effects', 'fx'],
        children: [
          { id: 'timbre.effects.distortion', label: 'Distortion', keywords: ['distortion', 'distorted', 'heavy'] },
          { id: 'timbre.effects.overdrive', label: 'Overdrive', keywords: ['overdrive', 'drive', 'crunch'] },
          { id: 'timbre.effects.clean', label: 'Clean', keywords: ['clean', 'pristine', 'pure'] },
          { id: 'timbre.effects.reverb', label: 'Reverb', keywords: ['reverb', 'spacious', 'echo'] },
          { id: 'timbre.effects.delay', label: 'Delay', keywords: ['delay', 'echo', 'repeat'] },
          { id: 'timbre.effects.lo-fi', label: 'Lo-Fi', keywords: ['lo-fi', 'lofi', 'tape', 'vinyl'] },
        ],
      },
    ],
  },
  {
    id: 'expression', label: 'Expression', keywords: ['expression', 'mood', 'emotion'],
    children: [
      { id: 'expression.mood.nostalgic', label: 'Nostalgic', keywords: ['nostalgic', 'nostalgia', 'memories'] },
      { id: 'expression.mood.melancholic', label: 'Melancholic', keywords: ['melancholic', 'sad', 'somber'] },
      { id: 'expression.mood.joyful', label: 'Joyful', keywords: ['joyful', 'happy', 'upbeat'] },
      { id: 'expression.mood.fierce', label: 'Fierce', keywords: ['fierce', 'intense', 'aggressive'] },
      { id: 'expression.mood.ethereal', label: 'Ethereal', keywords: ['ethereal', 'dreamy', 'otherworldly'] },
      { id: 'expression.mood.calm', label: 'Calm', keywords: ['calm', 'peaceful', 'serene'] },
    ],
  },
  {
    id: 'structure', label: 'Structure', keywords: ['structure', 'form', 'arrangement'],
    children: [
      { id: 'structure.narrative-arc.climb-to-climax', label: 'Climb to Climax', keywords: ['build', 'climax', 'crescendo'] },
      { id: 'structure.narrative-arc.slow-reveal', label: 'Slow Reveal', keywords: ['reveal', 'gradual', 'unfolding'] },
      { id: 'structure.narrative-arc.rise-and-fall', label: 'Rise and Fall', keywords: ['rise', 'fall', 'arc'] },
      { id: 'structure.narrative-arc.sudden-shift', label: 'Sudden Shift', keywords: ['sudden', 'shift', 'surprise'] },
    ],
  },
  {
    id: 'production', label: 'Production', keywords: ['production', 'mix', 'spatial'],
    children: [
      { id: 'production.register.bass', label: 'Bass Register', keywords: ['bass', 'low', 'sub'] },
      { id: 'production.register.mid', label: 'Mid Register', keywords: ['mid', 'middle'] },
      { id: 'production.register.high', label: 'High Register', keywords: ['high', 'treble', 'bright'] },
      { id: 'production.spatial.wide-stereo', label: 'Wide Stereo', keywords: ['wide', 'stereo', 'spacious'] },
      { id: 'production.mix-style.lo-fi', label: 'Lo-Fi Mix', keywords: ['lo-fi', 'lofi', 'raw', 'tape'] },
      { id: 'production.mix-style.polished', label: 'Polished Mix', keywords: ['polished', 'clean', 'professional'] },
    ],
  },
];

// Try to load the full taxonomy tree (will be available after agents complete)
let taxonomyTree = PLACEHOLDER_TAXONOMY;
try {
  // Dynamic import will work once the file exists
  // const { TAXONOMY_TREE } = await import('../../data/taxonomyTree');
  // taxonomyTree = TAXONOMY_TREE;
} catch {
  // Use placeholder
}

interface UniversalMusicInputProps {
  apiStatus: 'checking' | 'online' | 'offline';
  /** Existing components to place in the ToolDrawer */
  toolDrawerContent?: React.ReactNode;
}

export default function UniversalMusicInput({ apiStatus, toolDrawerContent }: UniversalMusicInputProps) {
  // --- State ---
  const [searchQuery, setSearchQuery] = useState('');
  const [naturalText, setNaturalText] = useState('');
  const [activeNodes, setActiveNodes] = useState<Set<string>>(new Set());
  const [pinnedNodes, setPinnedNodes] = useState<Set<string>>(new Set());
  const [interpretiveLevel, setInterpretiveLevel] = useState(0.5);
  const [isGenerating, setIsGenerating] = useState(false);

  const brain = useMusicBrain();
  const { parseResult, isParsing, parseDebounced } = useTextParse();

  // NLP-detected nodes from parse result
  const nlpDetectedNodes = useMemo(() => {
    if (!parseResult) return new Set<string>();
    return new Set(parseResult.activated_taxonomy_ids);
  }, [parseResult]);

  // Combined node weights: manual nodes get weight 1.0, NLP nodes get their confidence
  const nodeWeights = useMemo(() => {
    const weights = new Map<string, number>();
    // Manual selections at full weight
    for (const id of activeNodes) {
      weights.set(id, 1.0);
    }
    // NLP detections at confidence-scaled weight (won't override manual)
    if (parseResult) {
      const conf = parseResult.confidence;
      for (const id of parseResult.activated_taxonomy_ids) {
        if (!weights.has(id)) {
          weights.set(id, conf * 0.85);
        }
      }
    }
    return weights;
  }, [activeNodes, parseResult]);

  // --- Handlers ---
  const handleTextChange = useCallback((text: string) => {
    setNaturalText(text);
    parseDebounced(text);
  }, [parseDebounced]);

  const handleToggleNode = useCallback((id: string) => {
    setActiveNodes(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const handlePinNode = useCallback((id: string) => {
    setPinnedNodes(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const handleReinterpret = useCallback(() => {
    // Re-parse the same text (will get different results if backend has randomness)
    if (naturalText.trim()) {
      parseDebounced(naturalText + ' '); // tiny change to force re-parse
      setTimeout(() => parseDebounced(naturalText), 50);
    }
  }, [naturalText, parseDebounced]);

  // --- Summary ---
  const summary = useMemo(() => {
    const parts: string[] = [];

    if (parseResult?.activated_clusters[0]) {
      parts.push(parseResult.activated_clusters[0].label);
    }

    const modeWeights = parseResult?.param_distributions?.mode_weights?.weights;
    if (modeWeights) {
      const topMode = Object.entries(modeWeights).sort(([, a], [, b]) => b - a)[0];
      if (topMode) {
        parts.push(`${topMode[0]} ${Math.round(topMode[1] * 100)}%`);
      }
    }

    const tempo = parseResult?.param_distributions?.tempo;
    if (tempo?.center) {
      const spread = tempo.spread ?? 10;
      parts.push(`~${Math.round(tempo.center - spread)}–${Math.round(tempo.center + spread)} BPM`);
    }

    if (activeNodes.size > 0) {
      parts.push(`${activeNodes.size} selected`);
    }

    return parts.join(' · ') || '';
  }, [parseResult, activeNodes]);

  const isValid = activeNodes.size > 0 || (parseResult?.detected_keywords?.length ?? 0) > 0;

  // --- Generate ---
  const handleGenerate = useCallback(async () => {
    if (!isValid || isGenerating) return;
    setIsGenerating(true);

    try {
      // Build a CompleteSongIntentRequest from the current interpretation
      const intent: CompleteSongIntentRequest = {
        core_desire: naturalText || 'Generated from Universal Music Input',
        mood_primary: parseResult?.activated_clusters[0]?.label ?? 'Neutral',
        genre: parseResult?.activated_clusters[0]?.label ?? '',
        tempo: parseResult?.param_distributions?.tempo?.center
          ? Math.round(parseResult.param_distributions.tempo.center)
          : 120,
        key_mode: buildKeyModeFromWeights(
          parseResult?.param_distributions?.mode_weights?.weights
        ),
        structure: [
          { name: 'intro', bars: 4, repetitions: 1 },
          { name: 'verse', bars: 8, repetitions: 2 },
          { name: 'chorus', bars: 8, repetitions: 1 },
        ],
        instruments: buildInstrumentsFromNodes(activeNodes, nlpDetectedNodes),
        groove_feel: getGrooveFeel(activeNodes, parseResult),
        narrative_arc: getNarrativeArc(activeNodes),
      };

      const payload = buildGeneratePayload(intent);
      await brain.generateMusic(payload);
    } catch (err) {
      console.error('Generation failed:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [isValid, isGenerating, naturalText, parseResult, activeNodes, nlpDetectedNodes, brain]);

  return (
    <div className="umi-container">
      <SideAPane
        taxonomyNodes={taxonomyTree}
        nodeWeights={nodeWeights}
        nlpDetectedNodes={nlpDetectedNodes}
        pinnedNodes={pinnedNodes}
        searchQuery={searchQuery}
        interpretiveLevel={interpretiveLevel}
        onSearchChange={setSearchQuery}
        onToggleNode={handleToggleNode}
        onPinNode={handlePinNode}
        onInterpretiveLevelChange={setInterpretiveLevel}
      />
      <SideBPane
        text={naturalText}
        isParsing={isParsing}
        parseResult={parseResult}
        onTextChange={handleTextChange}
        onReinterpret={handleReinterpret}
      />
      <IntentSummaryBar
        summary={summary}
        isValid={isValid}
        isGenerating={isGenerating}
        apiStatus={apiStatus}
        onGenerate={handleGenerate}
        onReinterpret={handleReinterpret}
      />
      {toolDrawerContent && (
        <ToolDrawer>{toolDrawerContent}</ToolDrawer>
      )}
    </div>
  );
}

// --- Helpers ---

function buildKeyModeFromWeights(weights?: Record<string, number>): string {
  if (!weights) return 'C major';
  const topMode = Object.entries(weights).sort(([, a], [, b]) => b - a)[0];
  return topMode ? `C ${topMode[0]}` : 'C major';
}

function buildInstrumentsFromNodes(manual: Set<string>, nlp: Set<string>) {
  const all = new Set([...manual, ...nlp]);
  const instruments: Array<{ instrument: string; techniques: string[] }> = [];

  const instrumentMap: Record<string, string> = {
    'timbre.instruments.piano': 'piano',
    'timbre.instruments.electric-guitar': 'electric_guitar',
    'timbre.instruments.acoustic-guitar': 'acoustic_guitar',
    'timbre.instruments.bass-guitar': 'bass_guitar',
    'timbre.instruments.synth-lead': 'synth_lead',
    'timbre.instruments.synth-pad': 'synth_pad',
    'timbre.instruments.strings': 'strings',
    'timbre.instruments.brass': 'brass',
    'timbre.instruments.drums': 'drums',
    'timbre.instruments.sax': 'saxophone',
    'timbre.instruments.808': '808_bass',
  };

  for (const [nodeId, instrumentName] of Object.entries(instrumentMap)) {
    if (all.has(nodeId)) {
      instruments.push({ instrument: instrumentName, techniques: [] });
    }
  }

  // Default to piano if nothing selected
  if (instruments.length === 0) {
    instruments.push({ instrument: 'piano', techniques: [] });
  }

  return instruments;
}

function getGrooveFeel(nodes: Set<string>, parseResult: ParseTextResponse | null): string {
  const grooveNodes: Record<string, string> = {
    'rhythm.groove-feel.straight-driving': 'Straight/Driving',
    'rhythm.groove-feel.laid-back': 'Laid Back',
    'rhythm.groove-feel.swung': 'Swung',
    'rhythm.groove-feel.syncopated': 'Syncopated',
  };
  for (const [nodeId, feel] of Object.entries(grooveNodes)) {
    if (nodes.has(nodeId)) return feel;
  }
  return 'Straight/Driving';
}

function getNarrativeArc(nodes: Set<string>): string {
  const arcNodes: Record<string, string> = {
    'structure.narrative-arc.climb-to-climax': 'Climb-to-Climax',
    'structure.narrative-arc.slow-reveal': 'Slow Reveal',
    'structure.narrative-arc.rise-and-fall': 'Rise and Fall',
    'structure.narrative-arc.sudden-shift': 'Sudden Shift',
  };
  for (const [nodeId, arc] of Object.entries(arcNodes)) {
    if (nodes.has(nodeId)) return arc;
  }
  return 'Climb-to-Climax';
}
