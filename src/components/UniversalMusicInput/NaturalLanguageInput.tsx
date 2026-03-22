import { useCallback, useRef } from 'react';

interface NaturalLanguageInputProps {
  value: string;
  isParsing: boolean;
  onChange: (text: string) => void;
}

export function NaturalLanguageInput({ value, isParsing, onChange }: NaturalLanguageInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  }, [onChange]);

  return (
    <div className="umi-nl-input-wrap">
      <textarea
        ref={textareaRef}
        className="umi-nl-textarea"
        placeholder="Describe the music you hear in your head...&#10;&#10;Examples:&#10;• &quot;dorian with fast guitar licks and lots of overdrive&quot;&#10;• &quot;slow clean bass, pentatonic, bluesy feel&quot;&#10;• &quot;a thunderstorm clearing into sunshine&quot;&#10;• &quot;laid-back neo-soul with warm keys&quot;"
        value={value}
        onChange={handleChange}
        rows={6}
      />
      {isParsing && (
        <div className="umi-parsing-indicator">
          <span className="umi-parsing-dot" />
          interpreting...
        </div>
      )}
    </div>
  );
}
