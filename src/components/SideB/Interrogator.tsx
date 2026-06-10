import { FormEvent, useEffect, useState } from 'react';

type Props = {
  starter: string;
  onAsk: (question: string) => void;
};

export function Interrogator({ starter, onAsk }: Props) {
  // Start empty (a starter seeded only into useState would go stale when the
  // selected emotion changes, and a placeholder-ish starter could be submitted
  // verbatim as a question). Prefill only when a real starter arrives.
  const [prompt, setPrompt] = useState('');

  useEffect(() => {
    if (starter) setPrompt(starter);
  }, [starter]);

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
        What do you need?
        <input
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="e.g. What chord next? Make the chorus bigger"
        />
      </label>
      <button type="submit" className="outline">Ask</button>
    </form>
  );
}
