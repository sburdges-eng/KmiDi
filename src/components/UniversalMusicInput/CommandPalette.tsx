import { useState, useCallback, useRef, useEffect } from 'react';

interface CommandPaletteProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  /** Total active nodes for display */
  activeCount: number;
}

export function CommandPalette({ searchQuery, onSearchChange, activeCount }: CommandPaletteProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  // Cmd+K / Ctrl+K to focus
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === 'Escape' && document.activeElement === inputRef.current) {
        onSearchChange('');
        inputRef.current?.blur();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onSearchChange]);

  return (
    <div className="umi-command-palette">
      <div className="umi-search-box">
        <span className="umi-search-icon">⌕</span>
        <input
          ref={inputRef}
          type="text"
          className="umi-search-input"
          placeholder="Search commands... (⌘K)"
          value={searchQuery}
          onChange={e => onSearchChange(e.target.value)}
        />
        {searchQuery && (
          <button
            type="button"
            className="umi-search-clear"
            onClick={() => onSearchChange('')}
          >
            ✕
          </button>
        )}
      </div>
      {activeCount > 0 && (
        <span className="umi-active-count">{activeCount} active</span>
      )}
    </div>
  );
}
