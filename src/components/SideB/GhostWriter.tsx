type Props = {
  seed: string;
  output: string;
  onGenerate: (value: string) => void;
};

const BASE_TEXTS: Record<string, string> = {
  calm: 'The room is quiet and the harmony drifts forward with patient notes.',
  bright: 'Lifted chords bounce with clear motion and warm momentum.',
  intense: 'A restless pulse pushes forward, with deliberate tension and release.',
  reflective: 'A soft phrase returns, echoing memory with careful restraint.',
  mystical: 'Cloudlike textures blur the line between foreground and atmosphere.',
  default: 'A balanced phrase with breathing silence and clear forward motion.',
};

export function GhostWriter({ seed, output, onGenerate }: Props) {
  const key = seed ? seed.split(' ')[0] : 'default';

  return (
    <div className="ghost">
      <label>
        Starting vibe
        <input
          type="text"
          value={seed}
          placeholder="Pick a mood above to seed the lyrics"
          readOnly
          aria-readonly="true"
        />
      </label>
      <button
        type="button"
        className="outline"
        onClick={() => onGenerate(BASE_TEXTS[key] ?? BASE_TEXTS.default)}
      >
        Spark a lyric
      </button>
      <p className="ghost-output">{output || 'Your lyric spark will appear here.'}</p>
    </div>
  );
}
