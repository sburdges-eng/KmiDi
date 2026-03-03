import { ChangeEvent } from 'react';

type Channel = {
  id: string;
  name: string;
  level: number;
  pan: number;
};

type Props = {
  channels: Channel[];
  onChannelChange: (channelId: string, patch: Partial<Pick<Channel, 'level' | 'pan'>>) => void;
};

export function Mixer({ channels, onChannelChange }: Props) {
  return (
    <div className="mixer-grid">
      {channels.map((channel) => (
        <div key={channel.id} className="mixer-strip">
          <p className="mixer-label">{channel.name}</p>
          <label>
            Vol
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={channel.level}
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                onChannelChange(channel.id, { level: Number(event.target.value) })
              }
            />
          </label>
          <label>
            Pan
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={channel.pan}
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                onChannelChange(channel.id, { pan: Number(event.target.value) })
              }
            />
          </label>
          <div className="meter" aria-hidden="true">
            <span style={{ transform: `scaleY(${channel.level})` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
