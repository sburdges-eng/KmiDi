/**
 * iDAWi Unified Application
 *
 * COMBINED SIDES A + B - No longer separate entities
 * The professional DAW (Work State) and emotional intelligence (Dream State)
 * are now integrated into a single cohesive workspace.
 *
 * Philosophy: "Interrogate Before Generate" - Emotion drives technical choices
 */

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useUnifiedStore } from './store/useUnifiedStore';

// Side A (DAW) Components
import { Timeline } from './components/SideA/Timeline';
import { Mixer } from './components/SideA/Mixer';
import { Transport } from './components/SideA/Transport';

// Side B (Emotion) Components
import { EmotionWheel } from './components/SideB/EmotionWheel';
import { Interrogator } from './components/SideB/Interrogator';
import { GhostWriter } from './components/SideB/GhostWriter';
import { RuleBreaker } from './components/SideB/RuleBreaker';

// Unified Bridge Components
import { UnifiedBridge } from './components/Unified/UnifiedBridge';
import { EmotionToMixerBridge } from './components/Unified/EmotionToMixerBridge';
import { LogicProExport } from './components/Unified/LogicProExport';

// Panel state type
type PanelState = 'collapsed' | 'minimal' | 'expanded';

function UnifiedApp() {
  const {
    // Playback
    isPlaying,
    setPlaying,
    setCurrentTime,
    tempo,
    setTempo,

    // Emotion/Intent (Side B state now unified)
    selectedEmotion,
    setSelectedEmotion,
    completedIntent,
    setCompletedIntent,
    songIntent,
    updateSongIntent,

    // Tracks
    tracks,
    updateTrack,

    // UI Layout
    emotionPanelState,
    setEmotionPanelState,
    mixerPanelState,
    setMixerPanelState,
  } = useUnifiedStore();

  const lastTimeRef = useRef<number>(0);
  const wasPlayingRef = useRef<boolean>(false);
  const animationFrameRef = useRef<number>();

  // Track whether emotion has been set (Phase 0 complete)
  const [emotionLocked, setEmotionLocked] = useState(false);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+E or Ctrl+E to toggle emotion panel
      if ((e.metaKey || e.ctrlKey) && e.key === 'e') {
        e.preventDefault();
        setEmotionPanelState(
          emotionPanelState === 'expanded' ? 'minimal' : 'expanded'
        );
      }
      // Cmd+M or Ctrl+M to toggle mixer
      if ((e.metaKey || e.ctrlKey) && e.key === 'm') {
        e.preventDefault();
        setMixerPanelState(
          mixerPanelState === 'expanded' ? 'collapsed' : 'expanded'
        );
      }
      // Space to play/pause
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        setPlaying(!isPlaying);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, setPlaying, emotionPanelState, setEmotionPanelState, mixerPanelState, setMixerPanelState]);

  // Playback timer
  useEffect(() => {
    if (isPlaying) {
      if (!wasPlayingRef.current) {
        lastTimeRef.current = performance.now();
        wasPlayingRef.current = true;
      }

      const tick = () => {
        const now = performance.now();
        const delta = (now - lastTimeRef.current) / 1000;
        lastTimeRef.current = now;
        setCurrentTime((prevTime: number) => prevTime + delta);
        animationFrameRef.current = requestAnimationFrame(tick);
      };

      animationFrameRef.current = requestAnimationFrame(tick);
    } else {
      wasPlayingRef.current = false;
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = undefined;
      }
    };
  }, [isPlaying, setCurrentTime]);

  // Handle emotion selection - flows from Side B to Side A
  const handleEmotionSelect = (emotion: string) => {
    setSelectedEmotion(emotion);
    updateSongIntent({ coreEmotion: emotion });
    setCompletedIntent(null);
  };

  // Handle intent completion - triggers music generation
  const handleIntentComplete = (intent: Record<string, unknown>) => {
    setCompletedIntent(intent);
    setEmotionLocked(true);

    // Auto-apply emotion-based mixer settings via bridge
    // This is the key integration point between Side B emotion and Side A DAW
  };

  // Panel width calculations
  const emotionPanelWidth = {
    collapsed: 'w-0',
    minimal: 'w-64',
    expanded: 'w-96',
  }[emotionPanelState];

  const mixerPanelWidth = {
    collapsed: 'w-6',
    minimal: 'w-48',
    expanded: 'w-80',
  }[mixerPanelState];

  return (
    <div className="w-screen h-screen bg-ableton-bg overflow-hidden flex flex-col">
      {/* Unified Header - Shows both emotion state and DAW info */}
      <header className="h-12 bg-ableton-surface border-b border-ableton-border flex items-center px-4 justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold">iDAWi</h1>
          <span className="text-ableton-text-dim text-sm">Unified Workspace</span>
        </div>

        {/* Emotion State Indicator - Always visible */}
        <div className="flex items-center gap-4">
          {selectedEmotion ? (
            <div className="flex items-center gap-2 bg-ableton-surface-light px-3 py-1 rounded">
              <div className="w-2 h-2 rounded-full bg-ableton-accent animate-pulse" />
              <span className="text-sm">
                Feeling: <span className="text-ableton-accent font-medium">{selectedEmotion}</span>
              </span>
              {emotionLocked && (
                <span className="text-xs text-ableton-text-dim">(locked)</span>
              )}
            </div>
          ) : (
            <span className="text-sm text-ableton-text-dim">
              Select an emotion to begin
            </span>
          )}

          {/* Panel Toggle Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setEmotionPanelState(
                emotionPanelState === 'expanded' ? 'collapsed' : 'expanded'
              )}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                emotionPanelState === 'expanded'
                  ? 'bg-ableton-accent text-black'
                  : 'bg-ableton-surface-light text-ableton-text-dim hover:bg-ableton-border'
              }`}
              title="Toggle Emotion Panel (⌘E)"
            >
              Emotion
            </button>
            <button
              onClick={() => setMixerPanelState(
                mixerPanelState === 'expanded' ? 'collapsed' : 'expanded'
              )}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                mixerPanelState === 'expanded'
                  ? 'bg-ableton-accent text-black'
                  : 'bg-ableton-surface-light text-ableton-text-dim hover:bg-ableton-border'
              }`}
              title="Toggle Mixer (⌘M)"
            >
              Mixer
            </button>
          </div>
        </div>
      </header>

      {/* Main Content - Unified Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT: Emotion Panel (Side B elements) */}
        <AnimatePresence mode="wait">
          {emotionPanelState !== 'collapsed' && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{
                width: emotionPanelState === 'expanded' ? 384 : 256,
                opacity: 1
              }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="bg-ableton-surface border-r border-ableton-border overflow-hidden flex flex-col"
            >
              {/* Emotion Selection */}
              <div className="flex-shrink-0 border-b border-ableton-border">
                <EmotionWheel
                  onSelectEmotion={handleEmotionSelect}
                  selectedEmotion={selectedEmotion}
                />
              </div>

              {/* Interrogator */}
              <div className="flex-shrink-0 border-b border-ableton-border max-h-80 overflow-auto">
                <Interrogator
                  emotion={selectedEmotion}
                  onComplete={handleIntentComplete}
                />
              </div>

              {/* Rule Breaker - Compact view */}
              {emotionPanelState === 'expanded' && (
                <div className="flex-shrink-0 border-b border-ableton-border">
                  <RuleBreaker />
                </div>
              )}

              {/* Ghost Writer - Scrollable */}
              <div className="flex-1 overflow-auto">
                <GhostWriter
                  emotion={selectedEmotion}
                  intent={completedIntent}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* CENTER: Timeline & Transport (Side A main area) */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Timeline */}
          <div className="flex-1 overflow-hidden">
            <Timeline />
          </div>

          {/* Transport - Enhanced with emotion context */}
          <div className="flex-shrink-0">
            <Transport />
          </div>

          {/* Unified Bridge - Emotion-to-DAW integration */}
          <UnifiedBridge
            emotion={selectedEmotion}
            intent={completedIntent}
          />
        </div>

        {/* RIGHT: Mixer Panel (Side A) */}
        <AnimatePresence mode="wait">
          {mixerPanelState !== 'collapsed' && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{
                width: mixerPanelState === 'expanded' ? 320 : 192,
                opacity: 1
              }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="overflow-hidden"
            >
              <Mixer />

              {/* Emotion-to-Mixer Bridge */}
              <EmotionToMixerBridge
                emotion={selectedEmotion}
                intent={completedIntent}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Bar - Logic Pro Export & Status */}
      <footer className="h-8 bg-ableton-surface border-t border-ableton-border flex items-center px-4 justify-between">
        <div className="flex items-center gap-4 text-xs text-ableton-text-dim">
          <span>Tempo: {tempo} BPM</span>
          <span>|</span>
          <span>Tracks: {tracks.length}</span>
          {selectedEmotion && (
            <>
              <span>|</span>
              <span>Mode: {selectedEmotion}</span>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <LogicProExport
            tracks={tracks}
            tempo={tempo}
            emotion={selectedEmotion}
            intent={completedIntent}
          />
          <span className="text-xs text-ableton-text-dim opacity-50">
            iDAWi v1.0.0 Unified
          </span>
        </div>
      </footer>
    </div>
  );
}

export default UnifiedApp;
