export type MusicTechnique = {
  id: string;
  name: string;
  icon: string;
  description: string;
};

export type Genre = {
  id: string;
  name: string;
  icon: string;
  description: string;
};

export type QuickEmotion = {
  id: string;
  name: string;
  icon: string;
  color: string;
};

const musicTechniques: MusicTechnique[] = [
  { id: "reverb", name: "Reverb", icon: "🌊", description: "Add spatial depth and atmosphere" },
  { id: "delay", name: "Delay", icon: "🔁", description: "Create echo and rhythmic patterns" },
  { id: "compression", name: "Compression", icon: "📊", description: "Balance dynamics and punch" },
  { id: "distortion", name: "Distortion", icon: "⚡", description: "Add grit and character" },
  { id: "chorus", name: "Chorus", icon: "✨", description: "Widen and thicken the sound" },
  { id: "flanger", name: "Flanger", icon: "🌀", description: "Sweeping, whooshing effects" },
  { id: "phaser", name: "Phaser", icon: "🌊", description: "Smooth, swirling modulation" },
  { id: "tremolo", name: "Tremolo", icon: "💫", description: "Rhythmic volume variation" },
  { id: "wah", name: "Wah-Wah", icon: "🎸", description: "Vocal-like filtering" },
  { id: "vibrato", name: "Vibrato", icon: "🎵", description: "Pitch modulation for expression" },
];

const genres: Genre[] = [
  { id: "rock", name: "Rock", icon: "🎸", description: "Guitar-driven energy" },
  { id: "pop", name: "Pop", icon: "✨", description: "Catchy and commercial" },
  { id: "jazz", name: "Jazz", icon: "🎺", description: "Swinging improvisation" },
  { id: "electronic", name: "Electronic", icon: "🎧", description: "Synthetic and rhythmic" },
  { id: "hip-hop", name: "Hip-Hop", icon: "🎤", description: "Beat-driven and rhythmic" },
  { id: "folk", name: "Folk", icon: "🪕", description: "Acoustic and organic" },
  { id: "classical", name: "Classical", icon: "🎻", description: "Orchestral and refined" },
  { id: "blues", name: "Blues", icon: "🎹", description: "Soulful and expressive" },
  { id: "country", name: "Country", icon: "🤠", description: "Narrative and warm" },
  { id: "r-and-b", name: "R&B", icon: "🎹", description: "Smooth and soulful" },
  { id: "metal", name: "Metal", icon: "🔥", description: "Heavy and intense" },
  { id: "ambient", name: "Ambient", icon: "🌌", description: "Atmospheric and textural" },
];

/* Muted palette aligned with design system (no purple/indigo) */
const quickEmotions: QuickEmotion[] = [
  { id: "happy", name: "Happy", icon: "😊", color: "#c9a227" },
  { id: "sad", name: "Sad", icon: "😢", color: "#5c7c8a" },
  { id: "angry", name: "Angry", icon: "😠", color: "#b54a4a" },
  { id: "peaceful", name: "Peaceful", icon: "😌", color: "#4a7c6b" },
  { id: "energetic", name: "Energetic", icon: "⚡", color: "#c9a227" },
  { id: "melancholic", name: "Melancholic", icon: "🌙", color: "#6b7c8a" },
  { id: "romantic", name: "Romantic", icon: "💕", color: "#9e6b7c" },
  { id: "nostalgic", name: "Nostalgic", icon: "📷", color: "#8a7c6b" },
  { id: "hopeful", name: "Hopeful", icon: "🌅", color: "#7c9e6b" },
  { id: "mysterious", name: "Mysterious", icon: "🔮", color: "#6b5c8a" },
];

type Props = {
  selectedTechniques: string[];
  selectedGenre: string | null;
  selectedEmotion: string | null;
  onTechniquesChange: (techniques: string[]) => void;
  onGenreChange: (genre: string | null) => void;
  onEmotionChange: (emotion: string | null) => void;
};

export function MusicCustomizer({
  selectedTechniques,
  selectedGenre,
  selectedEmotion,
  onTechniquesChange,
  onGenreChange,
  onEmotionChange,
}: Props) {
  const handleTechniqueToggle = (techniqueId: string) => {
    if (selectedTechniques.includes(techniqueId)) {
      onTechniquesChange(selectedTechniques.filter(id => id !== techniqueId));
    } else {
      onTechniquesChange([...selectedTechniques, techniqueId]);
    }
  };

  return (
    <div className="music-customizer">
      <div className="section-header">
        <div className="section-header-inline">
          <h2>Shape the sound</h2>
          <span className="section-badge">Style & Mood</span>
        </div>
        <p className="section-description">
          Pick genre, mood, and techniques. Mix and match.
        </p>
      </div>

      <div className="customizer-section">
        <div className="customizer-header">
          <h3>Genre</h3>
          <p className="customizer-hint">Style of music</p>
        </div>
        <div className="genre-grid">
          {genres.map((genre) => (
            <button
              key={genre.id}
              onClick={() => onGenreChange(selectedGenre === genre.id ? null : genre.id)}
              className={`genre-btn ${selectedGenre === genre.id ? 'genre-btn-selected' : ''}`}
              title={genre.description}
            >
              <span className="genre-icon">{genre.icon}</span>
              <span className="genre-name">{genre.name}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="customizer-section">
        <div className="customizer-header">
          <h3>Mood</h3>
          <p className="customizer-hint">Emotional tone</p>
        </div>
        <div className="emotion-quick-grid">
          {quickEmotions.map((emotion) => (
            <button
              key={emotion.id}
              onClick={() => onEmotionChange(selectedEmotion === emotion.id ? null : emotion.id)}
              className={`emotion-quick-btn ${selectedEmotion === emotion.id ? 'emotion-quick-btn-selected' : ''}`}
              style={{
                borderColor: selectedEmotion === emotion.id ? emotion.color : undefined,
                backgroundColor: selectedEmotion === emotion.id ? `${emotion.color}20` : undefined,
              }}
            >
              <span className="emotion-quick-icon">{emotion.icon}</span>
              <span className="emotion-quick-name">{emotion.name}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="customizer-section">
        <div className="customizer-header">
          <h3>Techniques</h3>
          <p className="customizer-hint">Effects and techniques</p>
        </div>
        <div className="technique-grid">
          {musicTechniques.map((technique) => (
            <button
              key={technique.id}
              onClick={() => handleTechniqueToggle(technique.id)}
              className={`technique-btn ${selectedTechniques.includes(technique.id) ? 'technique-btn-selected' : ''}`}
              title={technique.description}
            >
              <span className="technique-icon">{technique.icon}</span>
              <span className="technique-name">{technique.name}</span>
            </button>
          ))}
        </div>
      </div>

      {(selectedGenre || selectedEmotion || selectedTechniques.length > 0) && (
        <div className="customizer-summary">
          <div className="summary-header">Your choices:</div>
          <div className="summary-tags">
            {selectedGenre && (
              <span className="summary-tag">
                {genres.find(g => g.id === selectedGenre)?.icon} {genres.find(g => g.id === selectedGenre)?.name}
              </span>
            )}
            {selectedEmotion && (
              <span className="summary-tag">
                {quickEmotions.find(e => e.id === selectedEmotion)?.icon} {quickEmotions.find(e => e.id === selectedEmotion)?.name}
              </span>
            )}
            {selectedTechniques.map(techId => {
              const tech = musicTechniques.find(t => t.id === techId);
              return tech ? (
                <span key={techId} className="summary-tag">
                  {tech.icon} {tech.name}
                </span>
              ) : null;
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default MusicCustomizer;
