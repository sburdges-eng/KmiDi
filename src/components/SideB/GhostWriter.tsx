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
        Seed phrase
        <input
          type="text"
          value={seed}
          placeholder="Optional emotion seed"
          readOnly
          aria-readonly="true"
        />
      </label>
      <button
        type="button"
        className="outline"
        onClick={() => onGenerate(BASE_TEXTS[key] ?? BASE_TEXTS.default)}
      >
        Generate lyric hint
      </button>
      <p className="ghost-output">{output || 'Output will appear here.'}</p>
    </div>
  );
}
