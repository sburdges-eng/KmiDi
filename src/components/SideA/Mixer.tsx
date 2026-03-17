import { ChangeEvent } from 'react';
import { CompactMeter } from './CompactMeter';
import { Knob } from '../ui/Knob';

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
          <div className="mixer-strip__vol">
            <Knob
              value={channel.level}
              min={0}
              max={1}
              step={0.05}
              stepFine={0.01}
              onChange={(v) => onChannelChange(channel.id, { level: v })}
              aria-label={`${channel.name} level`}
            />
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
          </div>
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
          <CompactMeter
            label={channel.name}
            value={channel.level}
            valueIsLinear
            aria-label={`${channel.name} level`}
          />
        </div>
      ))}
    </div>
  );
}
