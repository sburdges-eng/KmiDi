import type { ParseTextResponse } from '../../types/Interpretation';

interface ContextFeedbackProps {
  parseResult: ParseTextResponse | null;
  onReinterpret: () => void;
}

export function ContextFeedback({ parseResult, onReinterpret }: ContextFeedbackProps) {
  if (!parseResult) return null;

  const topCluster = parseResult.activated_clusters[0];
  const modeWeights = parseResult.param_distributions.mode_weights;
  const tempoDistribution = parseResult.param_distributions.tempo;

  return (
    <div className="umi-context-feedback">
      {/* Active cluster */}
      {topCluster && (
        <div className="umi-cluster-badge">
          <span className="umi-cluster-label">{topCluster.label}</span>
          <span className="umi-cluster-confidence">
            {Math.round(topCluster.confidence * 100)}%
          </span>
        </div>
      )}

      {/* Detected keywords */}
      {parseResult.detected_keywords.length > 0 && (
        <div className="umi-detected-keywords">
          {parseResult.detected_keywords.map(kw => (
            <span key={kw} className="umi-keyword-tag">{kw}</span>
          ))}
        </div>
      )}

      {/* Mode distribution bar */}
      {modeWeights?.weights && (
        <div className="umi-distribution-bar">
          <span className="umi-dist-label">Mode:</span>
          <div className="umi-dist-segments">
            {Object.entries(modeWeights.weights)
              .filter(([, w]) => w > 0.05)
              .sort(([, a], [, b]) => b - a)
              .map(([mode, weight]) => (
                <div
                  key={mode}
                  className="umi-dist-segment"
                  style={{ flex: weight }}
                  title={`${mode}: ${Math.round(weight * 100)}%`}
                >
                  <span className="umi-dist-segment-label">
                    {mode} {Math.round(weight * 100)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Tempo range */}
      {tempoDistribution?.center && (
        <div className="umi-tempo-range">
          <span className="umi-dist-label">Tempo:</span>
          <span className="umi-tempo-value">
            ~{Math.round(tempoDistribution.center - (tempoDistribution.spread ?? 10))}
            –{Math.round(tempoDistribution.center + (tempoDistribution.spread ?? 10))} BPM
          </span>
        </div>
      )}

      {/* Overall confidence */}
      <div className="umi-confidence-row">
        <span className="umi-confidence-label">
          Confidence: {Math.round(parseResult.confidence * 100)}%
        </span>
        <button
          type="button"
          className="umi-reinterpret-btn"
          onClick={onReinterpret}
          title="Re-sample from same distributions"
        >
          🎲 Reinterpret
        </button>
      </div>
    </div>
  );
}
