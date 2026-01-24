import React, { useState, useRef, useEffect } from "react";

interface AudioPlayerProps {
  audioPath: string | null;
  title?: string;
}

export function AudioPlayer({
  audioPath,
  title = "Generated Music",
}: AudioPlayerProps) {
  console.log("🎵 AudioPlayer component rendering!", { audioPath, title });

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Convert file path to HTTP URL
  const getAudioUrl = (path: string | null): string | null => {
    if (!path) return null;

    // If it's already a URL, return it
    if (path.startsWith("http://") || path.startsWith("https://")) {
      return path;
    }

    // Convert file path to API endpoint
    // Handle both absolute and relative paths
    let normalizedPath = path;

    // If it's a relative path, it might need to be absolute
    // The API expects the full path to the file
    const encodedPath = encodeURIComponent(normalizedPath);
    const url = `http://127.0.0.1:8000/audio/${encodedPath}`;
    console.log("Audio URL constructed:", {
      originalPath: path,
      normalizedPath,
      encodedPath,
      url,
    });
    return url;
  };

  const audioUrl = getAudioUrl(audioPath);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);
    const handleError = (e: any) => {
      console.error("Audio playback error:", e);
      const audio = audioRef.current;
      if (audio) {
        const errorCode = audio.error?.code;
        const errorMessage = audio.error?.message || "Unknown error";
        let userMessage = `Failed to load audio: ${errorMessage}`;

        if (errorCode === 4) {
          userMessage = `Format error: The audio file format may not be supported. File: ${audioPath}`;
        } else if (errorCode === 2) {
          userMessage = `Network error: Could not reach the audio file. Check if the API is running.`;
        } else if (errorCode === 3) {
          userMessage = `Decode error: The audio file may be corrupted or in an unsupported format.`;
        }

        console.error("Audio error details:", {
          code: errorCode,
          message: errorMessage,
          src: audio.src,
          networkState: audio.networkState,
          readyState: audio.readyState,
        });

        setError(userMessage);
      }
    };
    const handleLoadStart = () => {
      console.log("Audio loading started");
      setError(null);
    };

    audio.addEventListener("timeupdate", updateTime);
    audio.addEventListener("loadedmetadata", updateDuration);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("error", handleError);
    audio.addEventListener("loadstart", handleLoadStart);

    return () => {
      audio.removeEventListener("timeupdate", updateTime);
      audio.removeEventListener("loadedmetadata", updateDuration);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("error", handleError);
      audio.removeEventListener("loadstart", handleLoadStart);
    };
  }, [audioUrl]);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newTime = parseFloat(e.target.value);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newVolume = parseFloat(e.target.value);
    audio.volume = newVolume;
    setVolume(newVolume);
  };

  const formatTime = (seconds: number): string => {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Debug logging
  useEffect(() => {
    console.log("AudioPlayer rendered:", { audioPath, audioUrl });
  }, [audioPath, audioUrl]);

  // Always render the container, show placeholder if no audio
  if (!audioPath || !audioUrl) {
    return (
      <div
        className="audio-player-container"
        style={{
          border: "2px solid rgb(99, 102, 241)",
          background: "rgba(99, 102, 241, 0.1)",
          minHeight: "120px",
        }}
      >
        <div className="audio-player-header">
          <h3 style={{ color: "rgb(224, 224, 228)" }}>{title}</h3>
        </div>
        <div
          className="audio-player-placeholder"
          style={{
            padding: "30px",
            textAlign: "center",
            color: "rgba(224, 224, 228, 0.8)",
          }}
        >
          <p style={{ fontSize: "1.1em", marginBottom: "10px" }}>
            🎵 No audio available yet
          </p>
          <p style={{ fontSize: "0.9em" }}>
            Generate music using the "Generate Music" button above to play it
            here.
          </p>
          <p style={{ fontSize: "0.75em", marginTop: "10px", opacity: 0.6 }}>
            Debug: audioPath = {audioPath ? `"${audioPath}"` : "null"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="audio-player-container"
      style={{
        border: "2px solid rgb(34, 197, 94)",
        background: "rgba(34, 197, 94, 0.1)",
      }}
    >
      <div className="audio-player-header">
        <h3 style={{ color: "rgb(224, 224, 228)" }}>{title}</h3>
        {audioPath && (
          <span className="audio-file-name">{audioPath.split("/").pop()}</span>
        )}
      </div>

      <audio ref={audioRef} src={audioUrl} preload="metadata" />

      {error && (
        <div
          style={{
            padding: "10px",
            background: "rgba(239, 68, 68, 0.1)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: "6px",
            marginBottom: "12px",
            color: "rgb(248, 113, 113)",
            fontSize: "0.9em",
          }}
        >
          {error}
        </div>
      )}

      <div className="audio-player-controls">
        <button
          onClick={togglePlayPause}
          className="audio-play-button"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        <div className="audio-progress-container">
          <span className="audio-time">{formatTime(currentTime)}</span>
          <input
            type="range"
            min="0"
            max={duration || 0}
            value={currentTime}
            onChange={handleSeek}
            className="audio-progress-slider"
          />
          <span className="audio-time">{formatTime(duration)}</span>
        </div>

        <div className="audio-volume-container">
          <span className="audio-volume-icon">🔊</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={handleVolumeChange}
            className="audio-volume-slider"
          />
        </div>
      </div>
    </div>
  );
}

export default AudioPlayer;
