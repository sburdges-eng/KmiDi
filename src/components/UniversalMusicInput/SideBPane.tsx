import { NaturalLanguageInput } from './NaturalLanguageInput';
import { ContextFeedback } from './ContextFeedback';
import type { ParseTextResponse } from '../../types/Interpretation';

interface SideBPaneProps {
  text: string;
  isParsing: boolean;
  parseResult: ParseTextResponse | null;
  onTextChange: (text: string) => void;
  onReinterpret: () => void;
}

export function SideBPane({
  text, isParsing, parseResult, onTextChange, onReinterpret,
}: SideBPaneProps) {
  return (
    <div className="umi-side-b">
      <div className="umi-side-header">
        <h2 className="umi-side-title">Natural Language</h2>
      </div>
      <NaturalLanguageInput
        value={text}
        isParsing={isParsing}
        onChange={onTextChange}
      />
      <ContextFeedback
        parseResult={parseResult}
        onReinterpret={onReinterpret}
      />
    </div>
  );
}
