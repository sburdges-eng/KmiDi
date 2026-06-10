/**
 * Single source of truth for emotion and song-section colors across the UI.
 *
 * The palette stays in the muted, no-purple-flood family of the design
 * system (see index.css --accent and the MusicCustomizer "muted palette"
 * note) so the mood ring, sound palette, and structure timeline read as one
 * system instead of three.
 */
export const EMOTION_COLORS: Record<string, string> = {
  happy: '#c9a227',
  joyful: '#c9a227',
  sad: '#5c7c8a',
  angry: '#b54a4a',
  fierce: '#b54a4a',
  peaceful: '#4a7c6b',
  calm: '#4a7c6b',
  energetic: '#d4b03d',
  melancholic: '#6b7c8a',
  grief: '#5c6b7a',
  tender: '#9e8a7c',
  romantic: '#9e6b7c',
  nostalgic: '#8a7c6b',
  hopeful: '#7c9e6b',
  mysterious: '#6b5c8a',
  ethereal: '#5c8a8a',
};

export const emotionColor = (id: string, fallback = '#9c9c9a'): string =>
  EMOTION_COLORS[id.toLowerCase()] ?? fallback;

/** Structure-timeline section colors (same muted family; chorus = accent gold). */
export const SECTION_COLORS: Record<string, string> = {
  intro: '#5c7c8a',
  verse: '#4a7c6b',
  chorus: '#c9a227',
  bridge: '#9e6b7c',
  outro: '#8a7c6b',
  build: '#5c8a8a',
  drop: '#b54a4a',
};
