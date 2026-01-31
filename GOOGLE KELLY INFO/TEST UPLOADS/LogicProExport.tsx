/**
 * LogicProExport - Export unified project to Logic Pro compatible format
 *
 * Creates a comprehensive JSON file that can be used with Logic Pro's
 * MIDI import functionality, plus additional metadata for the iDAW ecosystem.
 */

import React, { useState } from 'react';
import { Download, FileJson, Check, Music } from 'lucide-react';
import { useUnifiedStore, LogicProProject, Track } from '../../store/useUnifiedStore';
import { harmonyToMidiNotes, chordToMidi } from './UnifiedBridge';

interface LogicProExportProps {
  tracks: Track[];
  tempo: number;
  emotion: string | null;
  intent: Record<string, unknown> | null;
}

export const LogicProExport: React.FC<LogicProExportProps> = ({
  tracks,
  tempo,
  emotion,
  intent,
}) => {
  const [isExporting, setIsExporting] = useState(false);
  const [exported, setExported] = useState(false);
  const { exportToLogicPro, generatedMusic, songIntent, timeSignature, projectName } = useUnifiedStore();

  const handleExport = async () => {
    setIsExporting(true);

    try {
      // Generate the Logic Pro project
      const project = exportToLogicPro();

      // If we have generated harmony, add it to the project
      if (generatedMusic?.harmony) {
        const harmonyNotes = harmonyToMidiNotes(generatedMusic.harmony, 480, 4);

        // Add a harmony track if one doesn't exist
        const harmonyTrack = {
          name: 'Generated Harmony',
          channel: 2,
          instrument: 0, // Piano
          notes: harmonyNotes,
          emotion_params: {
            reverb: 0.5,
            delay: 0.3,
            compression: 0.4,
          },
        };

        project.tracks.push(harmonyTrack);
      }

      // Create downloadable JSON
      const jsonString = JSON.stringify(project, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      // Generate filename with emotion and timestamp
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
      const emotionTag = emotion ? `_${emotion.toLowerCase()}` : '';
      const filename = `${projectName.replace(/\s+/g, '_')}${emotionTag}_${timestamp}.idaw.json`;

      // Download the file
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setExported(true);
      setTimeout(() => setExported(false), 3000);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={isExporting}
      className={`flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors ${
        exported
          ? 'bg-green-600 text-white'
          : 'bg-ableton-surface-light text-ableton-text-dim hover:bg-ableton-border hover:text-ableton-text'
      }`}
      title="Export to Logic Pro JSON"
    >
      {isExporting ? (
        <>
          <FileJson size={12} className="animate-pulse" />
          Exporting...
        </>
      ) : exported ? (
        <>
          <Check size={12} />
          Exported!
        </>
      ) : (
        <>
          <Download size={12} />
          Logic Pro Export
        </>
      )}
    </button>
  );
};

/**
 * Full Logic Pro Project Generator
 *
 * This can be called programmatically to generate a complete
 * Logic Pro compatible project file.
 */
export function generateLogicProProject(
  options: {
    projectName?: string;
    tempo?: number;
    timeSignature?: [number, number];
    key?: string;
    emotion?: string;
    harmony?: string[];
    tracks?: Track[];
    intent?: Record<string, unknown>;
  }
): LogicProProject {
  const PPQ = 480; // Logic Pro standard pulses per quarter note

  const {
    projectName = 'iDAWi Project',
    tempo = 120,
    timeSignature = [4, 4],
    key = 'C',
    emotion = null,
    harmony = [],
    tracks = [],
    intent = null,
  } = options;

  // Convert tracks to Logic Pro format
  const logicTracks = tracks.map((track, index) => {
    const channel = track.name.toLowerCase().includes('drum') ? 10 : index + 1;

    // Generate notes from clips
    const notes = track.clips.flatMap((clip) => {
      const startTick = Math.round(clip.startTime * PPQ);
      const durationTicks = Math.round(clip.duration * PPQ);

      // If clip has an emotion tag that matches a chord name, use that
      if (clip.name && chordToMidi(clip.name)) {
        const midiNotes = chordToMidi(clip.name);
        return midiNotes.map((pitch) => ({
          pitch,
          velocity: 80,
          start_tick: startTick,
          duration_ticks: durationTicks,
        }));
      }

      // Default: single note
      return [{
        pitch: 60,
        velocity: 80,
        start_tick: startTick,
        duration_ticks: durationTicks,
      }];
    });

    return {
      name: track.name,
      channel,
      instrument: track.type === 'midi' ? 0 : null,
      notes,
      emotion_params: track.emotionOverrides ? {
        reverb: track.emotionOverrides.reverb || 0.3,
        delay: track.emotionOverrides.delay || 0.2,
        compression: track.emotionOverrides.compression || 0.4,
      } : undefined,
    };
  });

  // Add harmony track if harmony is provided
  if (harmony.length > 0) {
    const harmonyNotes = harmonyToMidiNotes(harmony, PPQ, 4);
    logicTracks.push({
      name: 'Generated Harmony',
      channel: logicTracks.length + 1,
      instrument: 0,
      notes: harmonyNotes,
      emotion_params: emotion ? {
        reverb: 0.5,
        delay: 0.3,
        compression: 0.4,
      } : undefined,
    });
  }

  return {
    name: projectName,
    tempo_bpm: tempo,
    time_signature: timeSignature,
    ppq: PPQ,
    key,
    mode: emotion?.toLowerCase() || 'neutral',
    tracks: logicTracks,
    metadata: {
      emotion,
      intent,
      generated_at: new Date().toISOString(),
      idawi_version: '1.0.0',
    },
  };
}
