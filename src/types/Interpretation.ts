/**
 * Types for the Universal Music Input probabilistic interpretation system.
 */

/** A probability distribution over possible values */
export interface ParamDistribution {
  type: 'gaussian' | 'discrete' | 'range';
  center?: number;
  spread?: number;
  weights?: Record<string, number>;
  min?: number;
  max?: number;
  bias?: number;
}

/** Response from POST /parse-text backend endpoint */
export interface ParseTextResponse {
  param_distributions: Record<string, ParamDistribution>;
  activated_clusters: Array<{
    id: string;
    confidence: number;
    label: string;
  }>;
  activated_taxonomy_ids: string[];
  detected_keywords: string[];
  confidence: number;
}
