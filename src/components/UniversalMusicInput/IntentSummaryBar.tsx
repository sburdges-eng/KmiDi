interface IntentSummaryBarProps {
  summary: string;
  isValid: boolean;
  isGenerating: boolean;
  apiStatus: 'checking' | 'online' | 'offline';
  onGenerate: () => void;
  onReinterpret: () => void;
}

export function IntentSummaryBar({
  summary, isValid, isGenerating, apiStatus, onGenerate, onReinterpret,
}: IntentSummaryBarProps) {
  return (
    <div className="umi-summary-bar">
      <div className="umi-summary-text">
        {summary || 'Select commands or describe your music to begin...'}
      </div>
      <div className="umi-summary-actions">
        <button
          type="button"
          className="umi-reinterpret-btn-sm"
          onClick={onReinterpret}
          title="Re-sample from current distributions"
        >
          🎲
        </button>
        <button
          type="button"
          className="umi-generate-btn"
          onClick={onGenerate}
          disabled={!isValid || isGenerating || apiStatus !== 'online'}
        >
          {isGenerating ? 'Generating...' : '▶ Generate'}
        </button>
        <span className={`umi-api-status ${apiStatus}`}>
          <span className="umi-status-dot" />
          {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : '...'}
        </span>
      </div>
    </div>
  );
}
