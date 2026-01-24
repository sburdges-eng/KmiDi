/**
 * BrowserPanel - Right panel for file/media browser (utility)
 *
 * Per Spec 02: Layout & Navigation
 * - Right panel, utility
 * - File/media browser
 * - Follows visual system
 * - Implements persistence (save/restore state)
 */

import React, { useState, useEffect } from "react";

export interface BrowserItem {
  id: string;
  name: string;
  type: "file" | "folder" | "preset" | "sample" | "plugin";
  path?: string;
  size?: number;
  modified?: Date;
  metadata?: Record<string, any>;
}

interface BrowserPanelProps {
  currentPath?: string;
  items?: BrowserItem[];
  onItemSelect?: (item: BrowserItem) => void;
  onPathChange?: (path: string) => void;
  isVisible?: boolean;
  onVisibilityChange?: (visible: boolean) => void;
  filter?: string;
  onFilterChange?: (filter: string) => void;
}

/**
 * BrowserPanel component
 *
 * Provides a file and media browser for navigating project assets.
 * Follows Spec 02 requirements for the three-panel layout.
 */
export const BrowserPanel: React.FC<BrowserPanelProps> = ({
  currentPath = "/",
  items = [],
  onItemSelect,
  onPathChange,
  isVisible = true,
  onVisibilityChange,
  filter = "",
  onFilterChange,
}) => {
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");

  // Load persisted state on mount
  useEffect(() => {
    // TODO: Load from Tauri store
    const savedState = localStorage.getItem("browser-panel-state");
    if (savedState) {
      try {
        const state = JSON.parse(savedState);
        setViewMode(state.viewMode || "list");
        setSelectedItemId(state.selectedItemId || null);
      } catch (e) {
        console.warn("Failed to load browser panel state:", e);
      }
    }
  }, []);

  // Save state when it changes
  useEffect(() => {
    const state = {
      viewMode,
      selectedItemId,
      currentPath,
      filter,
    };
    localStorage.setItem("browser-panel-state", JSON.stringify(state));
    // TODO: Save to Tauri store
  }, [viewMode, selectedItemId, currentPath, filter]);

  const handleItemClick = (item: BrowserItem) => {
    setSelectedItemId(item.id);
    if (onItemSelect) {
      onItemSelect(item);
    }
  };

  const handleItemDoubleClick = (item: BrowserItem) => {
    if (item.type === "folder" && onPathChange) {
      onPathChange(item.path || currentPath);
    } else if (onItemSelect) {
      onItemSelect(item);
    }
  };

  const filteredItems = items.filter((item) => {
    if (!filter) return true;
    const searchTerm = filter.toLowerCase();
    return (
      item.name.toLowerCase().includes(searchTerm) ||
      item.type.toLowerCase().includes(searchTerm) ||
      (item.path && item.path.toLowerCase().includes(searchTerm))
    );
  });

  const getTypeIcon = (type: BrowserItem["type"]): string => {
    switch (type) {
      case "folder":
        return "📁";
      case "file":
        return "📄";
      case "preset":
        return "🎛️";
      case "sample":
        return "🎵";
      case "plugin":
        return "🔌";
      default:
        return "📄";
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div
      className="browser-panel bg-bg-secondary border-l border-border-light p-4 h-full overflow-y-auto flex flex-col"
      role="complementary"
      aria-label="Browser panel"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-text-primary">Browser</h2>
        {onVisibilityChange && (
          <button
            onClick={() => onVisibilityChange(false)}
            className="text-text-tertiary hover:text-text-primary min-h-touch min-w-touch"
            aria-label="Hide browser panel"
          >
            ✕
          </button>
        )}
      </div>

      {/* Path Navigation */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <button
            onClick={() => onPathChange && onPathChange("/")}
            className="text-sm text-accent-primary hover:text-accent-secondary min-h-touch min-w-touch"
            aria-label="Go to root directory"
          >
            🏠 Root
          </button>
          {currentPath !== "/" && (
            <button
              onClick={() => {
                const parentPath =
                  currentPath.split("/").slice(0, -1).join("/") || "/";
                onPathChange && onPathChange(parentPath);
              }}
              className="text-sm text-accent-primary hover:text-accent-secondary min-h-touch min-w-touch"
              aria-label="Go to parent directory"
            >
              ⬆ Up
            </button>
          )}
        </div>
        <div
          className="text-xs text-text-tertiary truncate"
          title={currentPath}
        >
          {currentPath}
        </div>
      </div>

      {/* Search/Filter */}
      {onFilterChange && (
        <div className="mb-4">
          <input
            type="text"
            value={filter}
            onChange={(e) => onFilterChange(e.target.value)}
            placeholder="Search..."
            className="w-full px-3 py-2 bg-bg-tertiary border border-border-light rounded text-text-primary text-sm min-h-touch"
            aria-label="Search browser items"
          />
        </div>
      )}

      {/* View Mode Toggle */}
      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={() => setViewMode("list")}
          className={`px-3 py-1 rounded text-sm min-h-touch min-w-touch ${
            viewMode === "list"
              ? "bg-accent-primary text-white"
              : "bg-bg-tertiary text-text-secondary hover:bg-bg-primary"
          }`}
          aria-label="List view"
          aria-pressed={viewMode === "list"}
        >
          List
        </button>
        <button
          onClick={() => setViewMode("grid")}
          className={`px-3 py-1 rounded text-sm min-h-touch min-w-touch ${
            viewMode === "grid"
              ? "bg-accent-primary text-white"
              : "bg-bg-tertiary text-text-secondary hover:bg-bg-primary"
          }`}
          aria-label="Grid view"
          aria-pressed={viewMode === "grid"}
        >
          Grid
        </button>
      </div>

      {/* Items List */}
      <div className="flex-1 overflow-y-auto">
        {filteredItems.length === 0 ? (
          <div className="text-center text-text-tertiary text-sm py-8">
            {filter
              ? "No items match your search"
              : "No items in this location"}
          </div>
        ) : (
          <div
            className={
              viewMode === "grid" ? "grid grid-cols-2 gap-2" : "space-y-1"
            }
          >
            {filteredItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleItemClick(item)}
                onDoubleClick={() => handleItemDoubleClick(item)}
                className={`w-full text-left p-2 rounded transition-colors min-h-touch ${
                  selectedItemId === item.id
                    ? "bg-accent-primary/20 border border-accent-primary"
                    : "bg-bg-tertiary hover:bg-bg-primary border border-transparent"
                }`}
                aria-label={`${item.type} ${item.name}`}
                aria-selected={selectedItemId === item.id}
              >
                <div className="flex items-center gap-2">
                  <span className="text-base" aria-hidden="true">
                    {getTypeIcon(item.type)}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-text-primary truncate">
                      {item.name}
                    </div>
                    {item.size && (
                      <div className="text-xs text-text-tertiary">
                        {(item.size / 1024).toFixed(1)} KB
                      </div>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="mt-4 pt-4 border-t border-border-light text-xs text-text-tertiary">
        {filteredItems.length} item{filteredItems.length !== 1 ? "s" : ""}
        {filter && ` matching "${filter}"`}
      </div>
    </div>
  );
};

export default BrowserPanel;
