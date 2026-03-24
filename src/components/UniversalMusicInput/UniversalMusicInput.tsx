import { useState, useCallback, useMemo, useEffect } from 'react';
import { SideAPane } from './SideAPane';
import { SideBPane } from './SideBPane';
import { IntentSummaryBar } from './IntentSummaryBar';
import { ToolDrawer } from './ToolDrawer';
import { useTextParse } from '../../hooks/useTextParse';
import { useMusicBrain, buildGeneratePayload } from '../../hooks/useMusicBrain';
import type { CompleteSongIntentRequest } from '../../types/Intent';
import type { ParseTextResponse } from '../../types/Interpretation';
import { TAXONOMY_TREE } from '../../data/taxonomyTree';

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
  const { parseResult, isParsing, parseDebounced, forceReparse } = useTextParse();

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
    if (naturalText.trim()) {
      forceReparse();
    }
  }, [naturalText, forceReparse]);

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
          parseResult?.param_distributions?.mode_weights?.weights,
          parseResult?.param_distributions?.key_weights?.weights,
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
        taxonomyNodes={TAXONOMY_TREE}
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

function buildKeyModeFromWeights(
  modeWeights?: Record<string, number>,
  keyWeights?: Record<string, number>,
): string {
  const mode = modeWeights
    ? Object.entries(modeWeights).sort(([, a], [, b]) => b - a)[0]?.[0] ?? 'major'
    : 'major';
  const key = keyWeights
    ? Object.entries(keyWeights).sort(([, a], [, b]) => b - a)[0]?.[0] ?? 'C'
    : 'C';
  return `${key} ${mode}`;
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
