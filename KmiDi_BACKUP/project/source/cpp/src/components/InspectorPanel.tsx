/**
 * InspectorPanel - Left panel for displaying selected item details (read-only)
 *
 * Per Spec 02: Layout & Navigation
 * - Left panel, read-only
 * - Displays selected item details
 * - Follows visual system (semantic tokens, 4pt grid)
 * - Implements persistence (save/restore state)
 */

import React, { useState, useEffect } from "react";

export interface InspectorItem {
  id: string;
  type: "track" | "clip" | "plugin" | "parameter" | "region";
  name: string;
  properties: Record<string, any>;
  metadata?: Record<string, any>;
}

interface InspectorPanelProps {
  selectedItem?: InspectorItem | null;
  isVisible?: boolean;
  onVisibilityChange?: (visible: boolean) => void;
}

/**
 * InspectorPanel component
 *
 * Displays detailed information about the currently selected item in a read-only format.
 * Follows Spec 02 requirements for the three-panel layout.
 */
export const InspectorPanel: React.FC<InspectorPanelProps> = ({
  selectedItem,
  isVisible = true,
  onVisibilityChange,
}) => {
  const [isExpanded, setIsExpanded] = useState<Record<string, boolean>>({});

  // Load persisted state on mount
  useEffect(() => {
    // TODO: Load from Tauri store
    const savedState = localStorage.getItem("inspector-panel-state");
    if (savedState) {
      try {
        const state = JSON.parse(savedState);
        setIsExpanded(state.expanded || {});
      } catch (e) {
        console.warn("Failed to load inspector panel state:", e);
      }
    }
  }, []);

  // Save state when it changes
  useEffect(() => {
    const state = {
      expanded: isExpanded,
      visible: isVisible,
    };
    localStorage.setItem("inspector-panel-state", JSON.stringify(state));
    // TODO: Save to Tauri store
  }, [isExpanded, isVisible]);

  const toggleSection = (section: string) => {
    setIsExpanded((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  if (!isVisible) {
    return null;
  }

  if (!selectedItem) {
    return (
      <div
        className="inspector-panel bg-bg-secondary border-r border-border-light p-4 h-full overflow-y-auto"
        role="complementary"
        aria-label="Inspector panel"
      >
        <h2 className="text-lg font-semibold text-text-primary mb-4">
          Inspector
        </h2>
        <div className="text-text-tertiary text-sm">
          No item selected. Select an item to view its details.
        </div>
      </div>
    );
  }

  return (
    <div
      className="inspector-panel bg-bg-secondary border-r border-border-light p-4 h-full overflow-y-auto"
      role="complementary"
      aria-label={`Inspector panel: ${selectedItem.name}`}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-text-primary">Inspector</h2>
        {onVisibilityChange && (
          <button
            onClick={() => onVisibilityChange(false)}
            className="text-text-tertiary hover:text-text-primary min-h-touch min-w-touch"
            aria-label="Hide inspector panel"
          >
            ✕
          </button>
        )}
      </div>

      <div className="inspector-content space-y-2">
        {/* Item Header */}
        <div className="inspector-section bg-bg-tertiary border border-border-light rounded p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-semibold text-text-secondary uppercase">
              {selectedItem.type}
            </span>
          </div>
          <h3 className="text-base font-semibold text-text-primary">
            {selectedItem.name}
          </h3>
          {selectedItem.id && (
            <div className="text-xs text-text-tertiary mt-1">
              ID: {selectedItem.id}
            </div>
          )}
        </div>

        {/* Properties Section */}
        <div className="inspector-section bg-bg-tertiary border border-border-light rounded">
          <button
            onClick={() => toggleSection("properties")}
            className="w-full flex items-center justify-between p-3 text-left hover:bg-bg-secondary transition-colors"
            aria-expanded={isExpanded["properties"] || false}
            aria-controls="inspector-properties"
          >
            <span className="font-semibold text-text-primary">Properties</span>
            <span className="text-text-tertiary">
              {isExpanded["properties"] ? "−" : "+"}
            </span>
          </button>
          {isExpanded["properties"] && (
            <div
              id="inspector-properties"
              className="p-3 border-t border-border-light space-y-2"
            >
              {Object.entries(selectedItem.properties).map(([key, value]) => (
                <div key={key} className="flex justify-between items-start">
                  <span className="text-sm text-text-secondary font-medium">
                    {key}:
                  </span>
                  <span className="text-sm text-text-primary text-right ml-4">
                    {typeof value === "object"
                      ? JSON.stringify(value, null, 2)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Metadata Section */}
        {selectedItem.metadata &&
          Object.keys(selectedItem.metadata).length > 0 && (
            <div className="inspector-section bg-bg-tertiary border border-border-light rounded">
              <button
                onClick={() => toggleSection("metadata")}
                className="w-full flex items-center justify-between p-3 text-left hover:bg-bg-secondary transition-colors"
                aria-expanded={isExpanded["metadata"] || false}
                aria-controls="inspector-metadata"
              >
                <span className="font-semibold text-text-primary">
                  Metadata
                </span>
                <span className="text-text-tertiary">
                  {isExpanded["metadata"] ? "−" : "+"}
                </span>
              </button>
              {isExpanded["metadata"] && (
                <div
                  id="inspector-metadata"
                  className="p-3 border-t border-border-light space-y-2"
                >
                  {Object.entries(selectedItem.metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-start">
                      <span className="text-sm text-text-secondary font-medium">
                        {key}:
                      </span>
                      <span className="text-sm text-text-primary text-right ml-4">
                        {typeof value === "object"
                          ? JSON.stringify(value, null, 2)
                          : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
      </div>
    </div>
  );
};

export default InspectorPanel;
