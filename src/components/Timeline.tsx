/**
 * Timeline - Center panel, primary interface for musical timeline
 *
 * Per Spec 04: Core Musical UI
 * - Center panel, primary interface
 * - Grid rules and zoom behavior
 * - Track scaling
 * - Follows Spec 04 requirements
 */

import React, { useState, useEffect, useRef, useCallback } from "react";

export interface TimelineTrack {
  id: string;
  name: string;
  height: number; // in pixels
  clips: TimelineClip[];
  muted?: boolean;
  solo?: boolean;
  volume?: number;
}

export interface TimelineClip {
  id: string;
  trackId: string;
  startTime: number; // in beats
  duration: number; // in beats
  name: string;
  color?: string;
}

interface TimelineProps {
  tracks?: TimelineTrack[];
  timeSignature?: { numerator: number; denominator: number };
  tempo?: number; // BPM
  zoomLevel?: number; // pixels per beat
  onZoomChange?: (zoom: number) => void;
  onClipSelect?: (clip: TimelineClip) => void;
  onClipMove?: (clipId: string, newStartTime: number) => void;
  onClipResize?: (clipId: string, newDuration: number) => void;
  playheadPosition?: number; // in beats
  onPlayheadChange?: (position: number) => void;
  isPlaying?: boolean;
}

/**
 * Timeline component
 *
 * Primary musical interface for arranging and editing timeline-based content.
 * Follows Spec 04 requirements for core musical UI.
 */
export const Timeline: React.FC<TimelineProps> = ({
  tracks = [],
  timeSignature = { numerator: 4, denominator: 4 },
  tempo = 120,
  zoomLevel: initialZoom = 50,
  onZoomChange,
  onClipSelect,
  onClipMove,
  onClipResize,
  playheadPosition = 0,
  onPlayheadChange,
  isPlaying = false,
}) => {
  const [zoomLevel, setZoomLevel] = useState(initialZoom);
  const [scrollPosition, setScrollPosition] = useState(0);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const dragStartX = useRef(0);

  // Grid calculations
  const beatsPerBar = timeSignature.numerator;
  const gridSubdivision = 16; // 16th notes
  const pixelsPerBeat = zoomLevel;
  const pixelsPerBar = pixelsPerBeat * beatsPerBar;
  const pixelsPerGrid = pixelsPerBeat / (gridSubdivision / 4); // 16th note grid

  const handleZoom = useCallback(
    (delta: number) => {
      const newZoom = Math.max(10, Math.min(500, zoomLevel + delta));
      setZoomLevel(newZoom);
      if (onZoomChange) {
        onZoomChange(newZoom);
      }
    },
    [zoomLevel, onZoomChange],
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        // Zoom with Ctrl/Cmd + wheel
        e.preventDefault();
        const delta = e.deltaY > 0 ? -5 : 5;
        handleZoom(delta);
      } else {
        // Scroll horizontally
        e.preventDefault();
        setScrollPosition((prev) => Math.max(0, prev - e.deltaX));
      }
    },
    [handleZoom],
  );

  const snapToGrid = (position: number): number => {
    return Math.round(position / pixelsPerGrid) * pixelsPerGrid;
  };

  const beatsToPixels = (beats: number): number => {
    return beats * pixelsPerBeat;
  };

  const pixelsToBeats = (pixels: number): number => {
    return pixels / pixelsPerBeat;
  };

  const handleClipClick = (clip: TimelineClip) => {
    setSelectedClipId(clip.id);
    if (onClipSelect) {
      onClipSelect(clip);
    }
  };

  const handleClipDragStart = (e: React.MouseEvent, clip: TimelineClip) => {
    e.stopPropagation();
    isDragging.current = true;
    dragStartX.current = e.clientX;
    setSelectedClipId(clip.id);
  };

  const handleClipDrag = useCallback(
    (e: MouseEvent, clip: TimelineClip) => {
      if (!isDragging.current || !onClipMove) return;

      const deltaX = e.clientX - dragStartX.current;
      const deltaBeats = pixelsToBeats(deltaX);
      const newStartTime = Math.max(0, clip.startTime + deltaBeats);
      const snappedTime =
        Math.round(newStartTime * gridSubdivision) / gridSubdivision;

      onClipMove(clip.id, snappedTime);
      dragStartX.current = e.clientX;
    },
    [onClipMove, pixelsToBeats, gridSubdivision],
  );

  useEffect(() => {
    if (!isDragging.current) return;

    const handleMouseMove = (e: MouseEvent) => {
      const selectedClip = tracks
        .flatMap((t) => t.clips)
        .find((c) => c.id === selectedClipId);
      if (selectedClip) {
        handleClipDrag(e, selectedClip);
      }
    };

    const handleMouseUp = () => {
      isDragging.current = false;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [selectedClipId, tracks, handleClipDrag]);

  // Render grid lines
  const renderGrid = () => {
    const bars = 32; // Show 32 bars
    const lines: JSX.Element[] = [];

    for (let bar = 0; bar < bars; bar++) {
      const x = beatsToPixels(bar * beatsPerBar) - scrollPosition;
      if (x < -100 || x > (timelineRef.current?.clientWidth || 0) + 100)
        continue;

      // Bar line (thicker)
      lines.push(
        <line
          key={`bar-${bar}`}
          x1={x}
          y1={0}
          x2={x}
          y2="100%"
          stroke="currentColor"
          strokeWidth={1}
          className="text-border-medium"
        />,
      );

      // Beat lines within bar
      for (let beat = 1; beat < beatsPerBar; beat++) {
        const beatX = x + beatsToPixels(beat);
        lines.push(
          <line
            key={`beat-${bar}-${beat}`}
            x1={beatX}
            y1={0}
            x2={beatX}
            y2="100%"
            stroke="currentColor"
            strokeWidth={0.5}
            strokeDasharray="2,2"
            className="text-border-light opacity-50"
          />,
        );
      }
    }

    return lines;
  };

  const totalHeight = tracks.reduce((sum, track) => sum + track.height, 0);
  const timelineWidth = Math.max(2000, beatsToPixels(32 * beatsPerBar));

  return (
    <div
      className="timeline bg-bg-primary flex-1 overflow-hidden flex flex-col"
      role="main"
      aria-label="Timeline"
      onWheel={handleWheel}
    >
      {/* Timeline Header */}
      <div className="timeline-header bg-bg-secondary border-b border-border-light p-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold text-text-primary">Timeline</h2>
          <div className="text-sm text-text-secondary">
            {timeSignature.numerator}/{timeSignature.denominator} @ {tempo} BPM
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleZoom(-10)}
            className="px-3 py-1 bg-bg-tertiary text-text-primary rounded text-sm min-h-touch min-w-touch hover:bg-bg-primary"
            aria-label="Zoom out"
          >
            −
          </button>
          <span className="text-sm text-text-secondary min-w-[60px] text-center">
            {Math.round(zoomLevel)}px/beat
          </span>
          <button
            onClick={() => handleZoom(10)}
            className="px-3 py-1 bg-bg-tertiary text-text-primary rounded text-sm min-h-touch min-w-touch hover:bg-bg-primary"
            aria-label="Zoom in"
          >
            +
          </button>
        </div>
      </div>

      {/* Timeline Content */}
      <div
        className="timeline-content flex-1 overflow-auto relative"
        ref={timelineRef}
      >
        <svg
          width={timelineWidth}
          height={totalHeight}
          className="absolute top-0 left-0"
          style={{ transform: `translateX(${scrollPosition}px)` }}
        >
          <defs>
            <pattern
              id="grid-pattern"
              x={0}
              y={0}
              width={pixelsPerGrid}
              height="100%"
              patternUnits="userSpaceOnUse"
            >
              <line
                x1={0}
                y1={0}
                x2={0}
                y2="100%"
                stroke="currentColor"
                strokeWidth={0.5}
                className="text-border-light opacity-25"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid-pattern)" />
          <g className="grid-lines">{renderGrid()}</g>

          {/* Playhead */}
          <line
            x1={beatsToPixels(playheadPosition) - scrollPosition}
            y1={0}
            x2={beatsToPixels(playheadPosition) - scrollPosition}
            y2="100%"
            stroke="rgb(255, 59, 48)"
            strokeWidth={2}
            className={isPlaying ? "animate-pulse" : ""}
          />

          {/* Tracks and Clips */}
          {tracks.map((track, trackIndex) => {
            let trackY = tracks
              .slice(0, trackIndex)
              .reduce((sum, t) => sum + t.height, 0);

            return (
              <g key={track.id}>
                {/* Track background */}
                <rect
                  x={0}
                  y={trackY}
                  width={timelineWidth}
                  height={track.height}
                  fill={
                    trackIndex % 2 === 0
                      ? "transparent"
                      : "rgba(255, 255, 255, 0.02)"
                  }
                  className="track-background"
                />

                {/* Clips */}
                {track.clips.map((clip) => {
                  const clipX = beatsToPixels(clip.startTime) - scrollPosition;
                  const clipWidth = beatsToPixels(clip.duration);
                  const isSelected = selectedClipId === clip.id;

                  return (
                    <g key={clip.id}>
                      <rect
                        x={clipX}
                        y={trackY + 4}
                        width={clipWidth}
                        height={track.height - 8}
                        fill={clip.color || "rgb(0, 122, 255)"}
                        stroke={
                          isSelected ? "rgb(255, 255, 255)" : "transparent"
                        }
                        strokeWidth={2}
                        rx={2}
                        className="cursor-move"
                        onMouseDown={(e) => handleClipDragStart(e, clip)}
                        onClick={() => handleClipClick(clip)}
                        aria-label={`Clip ${clip.name} on track ${track.name}`}
                      />
                      <text
                        x={clipX + 4}
                        y={trackY + track.height / 2}
                        fill="white"
                        fontSize={12}
                        className="pointer-events-none"
                      >
                        {clip.name}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>

        {/* Track Labels */}
        <div className="track-labels absolute left-0 top-0 w-32 bg-bg-secondary border-r border-border-light">
          {tracks.map((track, trackIndex) => {
            let trackY = tracks
              .slice(0, trackIndex)
              .reduce((sum, t) => sum + t.height, 0);

            return (
              <div
                key={track.id}
                className="track-label border-b border-border-light p-2"
                style={{ height: track.height }}
              >
                <div className="text-sm font-medium text-text-primary truncate">
                  {track.name}
                </div>
                {track.muted && (
                  <div className="text-xs text-text-tertiary">Muted</div>
                )}
                {track.solo && (
                  <div className="text-xs text-accent-warning">Solo</div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Timeline;
