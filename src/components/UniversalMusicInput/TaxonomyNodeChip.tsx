import { useCallback } from 'react';

interface TaxonomyNodeChipProps {
  id: string;
  label: string;
  /** 0-1 probability weight (undefined = not active) */
  weight?: number;
  /** Whether this node was detected via NLP (vs manually selected) */
  isNlpDetected?: boolean;
  /** Whether this node's parameter is pinned */
  isPinned?: boolean;
  onToggle: (id: string) => void;
  onPin?: (id: string) => void;
}

export function TaxonomyNodeChip({
  id, label, weight, isNlpDetected, isPinned, onToggle, onPin,
}: TaxonomyNodeChipProps) {
  const isActive = weight !== undefined && weight > 0;
  const handleClick = useCallback(() => onToggle(id), [id, onToggle]);
  const handlePin = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onPin?.(id);
  }, [id, onPin]);

  const weightPercent = weight !== undefined ? Math.round(weight * 100) : 0;
  const fillWidth = isActive ? `${Math.max(8, weightPercent)}%` : '0%';

  return (
    <button
      type="button"
      className={`umi-node-chip${isActive ? ' active' : ''}${isNlpDetected ? ' nlp-detected' : ''}`}
      onClick={handleClick}
      title={`${label}${isActive ? ` (${weightPercent}%)` : ''}`}
    >
      <span className="umi-node-chip-fill" style={{ width: fillWidth }} />
      <span className="umi-node-chip-label">{label}</span>
      {isActive && (
        <span className="umi-node-chip-weight">{weightPercent}%</span>
      )}
      {isActive && onPin && (
        <span
          className={`umi-node-chip-pin${isPinned ? ' pinned' : ''}`}
          onClick={handlePin}
          role="button"
          tabIndex={0}
          title={isPinned ? 'Unpin' : 'Pin this value'}
        >
          {isPinned ? '📌' : '📍'}
        </span>
      )}
    </button>
  );
}
