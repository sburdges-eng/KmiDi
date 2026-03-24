import { describe, it, expect } from 'vitest';
import {
  resolveContextClusters,
  mergeDistributions,
  sample,
  interpret,
  type ContextCluster,
} from '../../src/data/interpretationEngine';
import type { ParamDistribution } from '../../src/types/Interpretation';

// ─── Test Clusters ──────────────────────────────

const makeCluster = (
  overrides: Partial<ContextCluster> & Pick<ContextCluster, 'id' | 'label'>,
): ContextCluster => ({
  genreHints: [],
  triggerKeywords: new Set<string>(),
  triggerNodes: new Set<string>(),
  minMatchCount: 1,
  parameterOverrides: {},
  ...overrides,
});

const BLUES_CLUSTER = makeCluster({
  id: 'blues',
  label: 'Blues',
  triggerKeywords: new Set(['blues', 'bluesy']),
  triggerNodes: new Set(['harmony.scales.blues', 'rhythm.groove-feel.swung']),
  minMatchCount: 1,
  parameterOverrides: {
    tempo: { type: 'gaussian', center: 90, spread: 15 },
    mode_weights: { type: 'discrete', weights: { minor: 0.6, mixolydian: 0.3, major: 0.1 } },
  },
});

const AMBIENT_CLUSTER = makeCluster({
  id: 'ambient',
  label: 'Ambient',
  triggerKeywords: new Set(['ambient', 'atmospheric', 'ethereal']),
  triggerNodes: new Set(['timbre.instruments.synth-pad', 'expression.mood.ethereal']),
  minMatchCount: 2,
  parameterOverrides: {
    tempo: { type: 'gaussian', center: 70, spread: 20 },
  },
});

const TEST_CLUSTERS = [BLUES_CLUSTER, AMBIENT_CLUSTER];

// ─── resolveContextClusters ─────────────────────

describe('resolveContextClusters', () => {
  it('returns scored clusters when nodes match', () => {
    const nodes = new Set(['harmony.scales.blues']);
    const result = resolveContextClusters(nodes, [], TEST_CLUSTERS);

    expect(result).toHaveLength(1);
    expect(result[0].cluster.id).toBe('blues');
    expect(result[0].score).toBeGreaterThan(0);
    expect(result[0].matchedNodes).toContain('harmony.scales.blues');
  });

  it('returns scored clusters when keywords match', () => {
    const result = resolveContextClusters(new Set(), ['blues'], TEST_CLUSTERS);

    expect(result).toHaveLength(1);
    expect(result[0].cluster.id).toBe('blues');
    expect(result[0].matchedKeywords).toContain('blues');
  });

  it('returns empty when below minMatchCount', () => {
    // ambient requires minMatchCount=2, provide only 1
    const result = resolveContextClusters(
      new Set(['timbre.instruments.synth-pad']),
      [],
      TEST_CLUSTERS,
    );

    const ambientResults = result.filter((r) => r.cluster.id === 'ambient');
    expect(ambientResults).toHaveLength(0);
  });

  it('respects minMatchCount threshold', () => {
    // ambient requires 2, provide 2
    const result = resolveContextClusters(
      new Set(['timbre.instruments.synth-pad', 'expression.mood.ethereal']),
      [],
      TEST_CLUSTERS,
    );

    const ambientResults = result.filter((r) => r.cluster.id === 'ambient');
    expect(ambientResults).toHaveLength(1);
  });

  it('caps results at 5', () => {
    const manyClusters = Array.from({ length: 10 }, (_, i) =>
      makeCluster({
        id: `cluster-${i}`,
        label: `Cluster ${i}`,
        triggerKeywords: new Set(['common']),
        minMatchCount: 1,
      }),
    );

    const result = resolveContextClusters(new Set(), ['common'], manyClusters);
    expect(result.length).toBeLessThanOrEqual(5);
  });
});

// ─── mergeDistributions ─────────────────────────

describe('mergeDistributions', () => {
  it('merges gaussian distributions', () => {
    const overrides: Record<string, ParamDistribution>[] = [
      { tempo: { type: 'gaussian', center: 100, spread: 10 } },
      { tempo: { type: 'gaussian', center: 120, spread: 20 } },
    ];

    const result = mergeDistributions([], overrides, [0.5, 0.5]);

    expect(result.tempo).toBeDefined();
    expect(result.tempo.type).toBe('gaussian');
    expect(result.tempo.center).toBeGreaterThan(0);
  });

  it('merges discrete distributions and normalizes weights', () => {
    const overrides: Record<string, ParamDistribution>[] = [
      { mode_weights: { type: 'discrete', weights: { minor: 0.8, major: 0.2 } } },
      { mode_weights: { type: 'discrete', weights: { major: 0.7, dorian: 0.3 } } },
    ];

    const result = mergeDistributions([], overrides, [0.5, 0.5]);

    expect(result.mode_weights).toBeDefined();
    expect(result.mode_weights.weights).toBeDefined();
    const totalWeight = Object.values(result.mode_weights.weights!).reduce((s, v) => s + v, 0);
    expect(totalWeight).toBeCloseTo(1.0, 5);
  });

  it('preserves non-overlapping params', () => {
    const overrides: Record<string, ParamDistribution>[] = [
      { tempo: { type: 'gaussian', center: 100, spread: 10 } },
      { groove_feel: { type: 'discrete', weights: { swung: 1 } } },
    ];

    const result = mergeDistributions([], overrides, [0.5, 0.5]);

    expect(result.tempo).toBeDefined();
    expect(result.groove_feel).toBeDefined();
  });
});

// ─── sample ─────────────────────────────────────

describe('sample', () => {
  it('produces deterministic output with fixed seed', () => {
    const interpretResult = interpret(
      new Set(['harmony.scales.blues']),
      ['blues'],
      TEST_CLUSTERS,
      new Map(),
      0.5,
    );

    const a = sample(interpretResult, 42);
    const b = sample(interpretResult, 42);

    expect(a.tempo).toBe(b.tempo);
    expect(a.key_mode).toBe(b.key_mode);
    expect(a.groove_feel).toBe(b.groove_feel);
  });

  it('produces tempo within valid range', () => {
    const interpretResult = interpret(
      new Set(['harmony.scales.blues']),
      ['blues'],
      TEST_CLUSTERS,
      new Map(),
      0.5,
    );

    for (let seed = 0; seed < 50; seed++) {
      const result = sample(interpretResult, seed);
      expect(result.tempo).toBeGreaterThanOrEqual(40);
      expect(result.tempo).toBeLessThanOrEqual(300);
    }
  });

  it('returns a valid key_mode string', () => {
    const interpretResult = interpret(
      new Set(['harmony.scales.blues']),
      ['blues'],
      TEST_CLUSTERS,
      new Map(),
      0.5,
    );

    const result = sample(interpretResult, 42);
    expect(result.key_mode).toMatch(/^[A-G][b#]?\s\w+$/);
  });

  it('returns structure array', () => {
    const interpretResult = interpret(
      new Set(),
      [],
      TEST_CLUSTERS,
      new Map(),
      0.5,
    );

    const result = sample(interpretResult, 1);
    expect(result.structure).toBeDefined();
    expect(result.structure.length).toBeGreaterThan(0);
  });
});

// ─── interpret (end-to-end) ─────────────────────

describe('interpret', () => {
  it('returns distributions from matched clusters', () => {
    const result = interpret(
      new Set(['harmony.scales.blues']),
      ['blues'],
      TEST_CLUSTERS,
      new Map(),
      0.5,
    );

    expect(result.activeClusters.length).toBeGreaterThan(0);
    expect(result.distributions.tempo).toBeDefined();
    expect(result.confidence).toBeGreaterThan(0);
  });

  it('applies pinned params as zero-spread gaussian', () => {
    const pinned = new Map<string, unknown>([['tempo', 140]]);

    const result = interpret(
      new Set(['harmony.scales.blues']),
      ['blues'],
      TEST_CLUSTERS,
      pinned,
      0.5,
    );

    expect(result.distributions.tempo.center).toBe(140);
    expect(result.distributions.tempo.spread).toBe(0);
  });

  it('applies user priors tempoShift', () => {
    const result = interpret(
      new Set(['harmony.scales.blues']),
      ['blues'],
      TEST_CLUSTERS,
      new Map(),
      0.5,
      { tempoShift: 20 },
    );

    // Blues cluster has center 90, +20 shift = 110
    expect(result.distributions.tempo.center).toBeCloseTo(110, 0);
  });
});
