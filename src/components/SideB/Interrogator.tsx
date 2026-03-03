import { FormEvent, useState } from 'react';

type Props = {
  starter: string;
  onAsk: (question: string) => void;
};

export function Interrogator({ starter, onAsk }: Props) {
  const [prompt, setPrompt] = useState(starter);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const next = prompt.trim();
    if (!next) {
      return;
    }
    onAsk(next);
    setPrompt('');
  };

  return (
    <form className="interrogator" onSubmit={handleSubmit}>
      <label>
        Prompt
        <input
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Ask about arrangement, harmony, or mood"
        />
      </label>
      <button type="submit" className="outline">Run query</button>
    </form>
  );
}
