/**
 * Probabilistic Interpretation Engine for KmiDi Universal Music Input.
 *
 * Takes active taxonomy nodes + NLP keywords, resolves context clusters,
 * merges probability distributions, and samples concrete values.
 *
 * All functions are pure — no side effects, runs entirely client-side.
 */

import type { CompleteSongIntentRequest } from '../types/Intent';
import type { ParamDistribution } from '../types/Interpretation';
import { TAXONOMY_TREE, type TaxonomyNode } from './taxonomyTree';

// ─── Types ───────────────────────────────────────

export interface ContextCluster {
  id: string;
  label: string;
  genreHints: string[];
  triggerKeywords: Set<string>;
  triggerNodes: Set<string>;
  minMatchCount: number;
  parameterOverrides: Record<string, ParamDistribution>;
}

export interface ScoredCluster {
  cluster: ContextCluster;
  score: number;
  matchedNodes: string[];
  matchedKeywords: string[];
}

export interface InterpretationResult {
  distributions: Record<string, ParamDistribution>;
  activeClusters: ScoredCluster[];
  summary: string;
  confidence: number;
}

export interface UserPriors {
  tempoShift?: number;
  modeWeightAdjustments?: Record<string, number>;
  preferredDensity?: number;
}

// ─── Seedable PRNG ──────────────────────────────

function createRng(seed: number): () => number {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─── Cluster Resolution ─────────────────────────

export function resolveContextClusters(
  activeNodeIds: Set<string>,
  keywords: string[],
  clusters: ContextCluster[],
): ScoredCluster[] {
  const keywordSet = new Set(keywords.map((k) => k.toLowerCase()));
  const scored: ScoredCluster[] = [];

  for (const cluster of clusters) {
    const matchedNodes: string[] = [];
    const matchedKeywords: string[] = [];

    for (const nodeId of cluster.triggerNodes) {
      if (activeNodeIds.has(nodeId)) matchedNodes.push(nodeId);
    }
    for (const kw of cluster.triggerKeywords) {
      if (keywordSet.has(kw)) matchedKeywords.push(kw);
    }

    const totalMatches = matchedNodes.length + matchedKeywords.length;
    if (totalMatches >= cluster.minMatchCount) {
      const score = Math.min(1.0, totalMatches / (cluster.minMatchCount + 1));
      scored.push({ cluster, score, matchedNodes, matchedKeywords });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 5);
}

// ─── Distribution Merging ───────────────────────

function blendGaussian(
  a: ParamDistribution,
  b: ParamDistribution,
  bWeight: number,
): ParamDistribution {
  const aWeight = 1 - bWeight;
  return {
    type: 'gaussian',
    center: (a.center ?? 0) * aWeight + (b.center ?? 0) * bWeight,
    spread: Math.max(a.spread ?? 0, b.spread ?? 0),
  };
}

function blendDiscrete(
  a: ParamDistribution,
  b: ParamDistribution,
  bWeight: number,
): ParamDistribution {
  const aWeight = 1 - bWeight;
  const merged: Record<string, number> = {};
  const allKeys = new Set([
    ...Object.keys(a.weights ?? {}),
    ...Object.keys(b.weights ?? {}),
  ]);
  for (const k of allKeys) {
    const av = (a.weights ?? {})[k] ?? 0;
    const bv = (b.weights ?? {})[k] ?? 0;
    merged[k] = av * aWeight + bv * bWeight;
  }
  const total = Object.values(merged).reduce((s, v) => s + v, 0);
  if (total > 0) {
    for (const k of Object.keys(merged)) merged[k] /= total;
  }
  return { type: 'discrete', weights: merged };
}

function blendRange(
  a: ParamDistribution,
  b: ParamDistribution,
  bWeight: number,
): ParamDistribution {
  const aWeight = 1 - bWeight;
  return {
    type: 'range',
    min: (a.min ?? 0) * aWeight + (b.min ?? 0) * bWeight,
    max: (a.max ?? 1) * aWeight + (b.max ?? 1) * bWeight,
    bias: (a.bias ?? 0.5) * aWeight + (b.bias ?? 0.5) * bWeight,
  };
}

function blendDistributions(
  a: ParamDistribution,
  b: ParamDistribution,
  bWeight: number,
): ParamDistribution {
  if (a.type === b.type) {
    if (a.type === 'gaussian') return blendGaussian(a, b, bWeight);
    if (a.type === 'discrete') return blendDiscrete(a, b, bWeight);
    if (a.type === 'range') return blendRange(a, b, bWeight);
  }
  return bWeight > 0.5 ? b : a;
}

export function mergeDistributions(
  contributions: Record<string, ParamDistribution>[],
  clusterOverrides: Record<string, ParamDistribution>[],
  clusterWeights: number[],
): Record<string, ParamDistribution> {
  const merged: Record<string, ParamDistribution> = {};

  // Merge node contributions (equal weight)
  for (const contrib of contributions) {
    for (const [param, dist] of Object.entries(contrib)) {
      if (!merged[param]) {
        merged[param] = dist;
      } else {
        merged[param] = blendDistributions(merged[param], dist, 0.5);
      }
    }
  }

  // Apply cluster overrides weighted by score
  for (let i = 0; i < clusterOverrides.length; i++) {
    const weight = clusterWeights[i] ?? 0.5;
    for (const [param, dist] of Object.entries(clusterOverrides[i])) {
      if (!merged[param]) {
        merged[param] = dist;
      } else {
        merged[param] = blendDistributions(merged[param], dist, weight);
      }
    }
  }

  return merged;
}

// ─── Node Contributions ─────────────────────────

/** Walk taxonomy tree and collect paramContributions for all active node IDs. */
function collectNodeContributions(
  activeNodeIds: Set<string>,
): Record<string, ParamDistribution>[] {
  const contributions: Record<string, ParamDistribution>[] = [];

  function walk(nodes: TaxonomyNode[]) {
    for (const node of nodes) {
      if (activeNodeIds.has(node.id) && node.paramContributions) {
        contributions.push(node.paramContributions);
      }
      if (node.children) walk(node.children);
    }
  }

  walk(TAXONOMY_TREE);
  return contributions;
}

// ─── Interpret ──────────────────────────────────

export function interpret(
  activeNodeIds: Set<string>,
  keywords: string[],
  clusters: ContextCluster[],
  pinnedParams: Map<string, unknown>,
  interpretiveLevel: number,
  userPriors?: UserPriors,
): InterpretationResult {
  // 1. Resolve clusters
  const scoredClusters = resolveContextClusters(activeNodeIds, keywords, clusters);

  // 2. Collect paramContributions from active taxonomy nodes
  const contributions = collectNodeContributions(activeNodeIds);

  // 3. Merge node contributions + cluster overrides
  const overrides = scoredClusters.map((sc) => sc.cluster.parameterOverrides);
  const weights = scoredClusters.map((sc) => sc.score);
  const distributions = mergeDistributions(contributions, overrides, weights);

  // 4. Apply interpretive level (scale spreads)
  const spreadScale = 0.1 + interpretiveLevel * 0.9;
  for (const [param, dist] of Object.entries(distributions)) {
    if (dist.type === 'gaussian' && dist.spread !== undefined) {
      distributions[param] = { ...dist, spread: dist.spread * spreadScale };
    }
  }

  // 5. Apply pinned params
  for (const [param, value] of pinnedParams) {
    if (typeof value === 'number') {
      distributions[param] = { type: 'gaussian', center: value, spread: 0 };
    }
  }

  // 5. Apply user priors
  if (userPriors?.tempoShift && distributions.tempo?.center !== undefined) {
    distributions.tempo = {
      ...distributions.tempo,
      center: distributions.tempo.center + userPriors.tempoShift,
    };
  }

  // 6. Build summary
  const summary = generateSummary(distributions, scoredClusters);

  // 7. Confidence
  const confidence =
    scoredClusters.length > 0
      ? scoredClusters.reduce((s, c) => s + c.score, 0) / scoredClusters.length
      : 0.3;

  return { distributions, activeClusters: scoredClusters, summary, confidence };
}

// ─── Sample ─────────────────────────────────────

export function sample(
  result: InterpretationResult,
  seed?: number,
): CompleteSongIntentRequest {
  const rng = createRng(seed ?? Date.now());

  const sampleGaussian = (dist: ParamDistribution): number => {
    const center = dist.center ?? 120;
    const spread = dist.spread ?? 10;
    // Box-Muller transform
    const u1 = rng();
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(Math.max(0.0001, u1))) * Math.cos(2 * Math.PI * u2);
    return center + z * spread;
  };

  const sampleDiscrete = (dist: ParamDistribution): string => {
    const weights = dist.weights ?? {};
    const entries = Object.entries(weights);
    if (entries.length === 0) return 'major';
    const total = entries.reduce((s, [, w]) => s + w, 0);
    let r = rng() * total;
    for (const [key, weight] of entries) {
      r -= weight;
      if (r <= 0) return key;
    }
    return entries[entries.length - 1][0];
  };

  const sampleRange = (dist: ParamDistribution): number => {
    const min = dist.min ?? 0;
    const max = dist.max ?? 1;
    const bias = dist.bias ?? (min + max) / 2;
    // Bias toward the bias point
    const r = rng();
    const biasNorm = (bias - min) / (max - min || 1);
    const skewed = Math.pow(r, 1 / (biasNorm + 0.5));
    return min + skewed * (max - min);
  };

  // Sample tempo
  const tempo = result.distributions.tempo
    ? Math.round(Math.max(40, Math.min(300, sampleGaussian(result.distributions.tempo))))
    : 120;

  // Sample mode
  const mode = result.distributions.mode_weights
    ? sampleDiscrete(result.distributions.mode_weights)
    : 'major';

  // Sample key
  const KEY_DEFAULT: ParamDistribution = {
    type: 'discrete',
    weights: { C: 0.18, G: 0.14, D: 0.12, A: 0.10, E: 0.08, F: 0.10, Bb: 0.08, Eb: 0.06, Ab: 0.05, B: 0.04, Db: 0.03, Gb: 0.02 },
  };
  const key = result.distributions.key_weights
    ? sampleDiscrete(result.distributions.key_weights)
    : sampleDiscrete(KEY_DEFAULT);

  // Sample groove feel
  const grooveFeel = result.distributions.groove_feel
    ? sampleDiscrete(result.distributions.groove_feel)
    : 'Straight/Driving';

  // Top cluster label as genre
  const genre = result.activeClusters[0]?.cluster.label ?? '';

  return {
    core_desire: '',
    mood_primary: genre,
    genre,
    tempo,
    key_mode: `${key} ${mode}`,
    structure: [
      { name: 'intro', bars: 4, repetitions: 1 },
      { name: 'verse', bars: 8, repetitions: 2 },
      { name: 'chorus', bars: 8, repetitions: 1 },
    ],
    instruments: [{ instrument: 'piano', techniques: [] }],
    groove_feel: grooveFeel,
    narrative_arc: 'Climb-to-Climax',
  };
}

// ─── Summary ────────────────────────────────────

export function generateSummary(
  distributions: Record<string, ParamDistribution>,
  clusters: ScoredCluster[],
): string {
  const parts: string[] = [];

  if (clusters[0]) {
    parts.push(clusters[0].cluster.label);
  }

  const modeWeights = distributions.mode_weights?.weights;
  if (modeWeights) {
    const top = Object.entries(modeWeights).sort(([, a], [, b]) => b - a)[0];
    if (top) parts.push(`${top[0]} ${Math.round(top[1] * 100)}%`);
  }

  const tempo = distributions.tempo;
  if (tempo?.center !== undefined) {
    const spread = tempo.spread ?? 10;
    parts.push(`~${Math.round(tempo.center - spread)}–${Math.round(tempo.center + spread)} BPM`);
  }

  return parts.join(' · ');
}
