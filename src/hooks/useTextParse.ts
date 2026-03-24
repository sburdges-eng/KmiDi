import { useCallback, useEffect, useRef, useState } from 'react';
import { useMusicBrain } from './useMusicBrain';
import type { ParseTextResponse } from '../types/Interpretation';

interface UseTextParseOptions {
  debounceMs?: number;
  locale?: string;
}

interface UseTextParseResult {
  /** Current parse result from the backend */
  parseResult: ParseTextResponse | null;
  /** Whether a parse request is in flight */
  isParsing: boolean;
  /** Parse text immediately (bypasses debounce) */
  parseNow: (text: string) => void;
  /** Parse text with debounce */
  parseDebounced: (text: string) => void;
  /** Force a reparse of the latest text */
  forceReparse: () => void;
  /** Last error, if any */
  error: string | null;
}

/**
 * Hook that debounces text input and sends it to POST /parse-text.
 *
 * Returns probabilistic distributions, activated context clusters,
 * and taxonomy node IDs from the backend NLP service.
 */
export function useTextParse(options?: UseTextParseOptions): UseTextParseResult {
  const { debounceMs = 300, locale = 'en' } = options ?? {};

  const brain = useMusicBrain();
  const [parseResult, setParseResult] = useState<ParseTextResponse | null>(null);
  const [isParsing, setIsParsing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestTextRef = useRef('');

  const doParseRequest = useCallback(async (text: string) => {
    if (!text.trim()) {
      setParseResult(null);
      setIsParsing(false);
      return;
    }

    // Cancel any in-flight request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setIsParsing(true);
    setError(null);

    try {
      const result = await brain.parseText(text, locale, undefined, abortRef.current?.signal);
      // Only update if this is still the latest request
      if (text === latestTextRef.current) {
        setParseResult(result);
        setIsParsing(false);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (text === latestTextRef.current) {
        // On API failure, try offline keyword matching
        setParseResult(offlineFallback(text));
        setError(err instanceof Error ? err.message : 'Parse failed');
        setIsParsing(false);
      }
    }
  }, [brain, locale]);

  const parseDebounced = useCallback((text: string) => {
    latestTextRef.current = text;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => doParseRequest(text), debounceMs);
  }, [doParseRequest, debounceMs]);

  const parseNow = useCallback((text: string) => {
    latestTextRef.current = text;
    if (timerRef.current) clearTimeout(timerRef.current);
    doParseRequest(text);
  }, [doParseRequest]);

  const forceReparse = useCallback(() => {
    if (latestTextRef.current.trim()) {
      if (timerRef.current) clearTimeout(timerRef.current);
      doParseRequest(latestTextRef.current);
    }
  }, [doParseRequest]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, []);

  return { parseResult, isParsing, parseNow, parseDebounced, forceReparse, error };
}

/**
 * Client-side keyword matching fallback when the API is offline.
 *
 * Scans text for known musical terms and returns basic taxonomy hints.
 * Much simpler than the backend NLP but provides usable feedback.
 */
function offlineFallback(text: string): ParseTextResponse {
  const lower = text.toLowerCase();
  const keywords: string[] = [];
  const taxonomyIds: string[] = [];

  const KEYWORD_TO_TAXONOMY: Record<string, string> = {
    'dorian': 'harmony.scales.dorian',
    'phrygian': 'harmony.scales.phrygian',
    'lydian': 'harmony.scales.lydian',
    'mixolydian': 'harmony.scales.mixolydian',
    'minor': 'harmony.scales.minor',
    'major': 'harmony.scales.major',
    'pentatonic': 'harmony.scales.pentatonic-minor',
    'blues': 'harmony.scales.blues',
    'guitar': 'timbre.instruments.electric-guitar',
    'bass': 'timbre.instruments.bass-guitar',
    'piano': 'timbre.instruments.piano',
    'synth': 'timbre.instruments.synth-lead',
    'drums': 'timbre.instruments.drums',
    'strings': 'timbre.instruments.strings',
    'overdrive': 'timbre.effects.overdrive',
    'distortion': 'timbre.effects.distortion',
    'clean': 'timbre.effects.clean',
    'reverb': 'timbre.effects.reverb',
    'fast': 'rhythm.density.dense',
    'slow': 'rhythm.density.sparse',
    'staccato': 'articulation.staccato',
    'legato': 'articulation.legato',
    'loud': 'dynamics.levels.f',
    'quiet': 'dynamics.levels.p',
    'sparse': 'rhythm.density.sparse',
    'dense': 'rhythm.density.dense',
  };

  for (const [word, taxId] of Object.entries(KEYWORD_TO_TAXONOMY)) {
    if (lower.includes(word)) {
      keywords.push(word);
      taxonomyIds.push(taxId);
    }
  }

  return {
    param_distributions: {},
    activated_clusters: [],
    activated_taxonomy_ids: taxonomyIds,
    detected_keywords: keywords,
    confidence: Math.min(1, keywords.length / 5) * 0.5, // lower confidence for offline
  };
}
