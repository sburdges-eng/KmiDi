# MIDI-CI Microformat-to-SysEx Daemon

Lightweight daemon that reads **token-stingy JSON** from stdin (e.g. from an LLM Orchestrator), wraps it in **MIDI 2.0 / MIDI-CI Property Exchange** SysEx, and sends it to the default MIDI output port via **libremidi**.

## Build

From the repo root, enable the daemon and build:

```bash
cmake -S . -B build -DBUILD_MIDI_CI_DAEMON=ON
cmake --build build --target midi_ci_daemon
```

Binary: `build/tools/midi_ci_daemon/midi_ci_daemon` (or equivalent per generator).

**Note:** Building the daemon uses FetchContent to download libremidi; if `KMIDI_OFFLINE_BUILD=ON`, the daemon is not built.

## Usage

1. Start the daemon (it opens the first available MIDI output port).
2. Feed one JSON object per line on stdin.

Example:

```bash
echo '{"op":"set","target":"cutoff","val":85}' | ./midi_ci_daemon
```

Or run interactively and type lines (one JSON per line), then Ctrl-D.

## Microformat

The daemon validates that each line contains the keys `"op"` and `"target"` before sending. Valid lines are sent as-is as the Property Exchange "Set Property Data" payload. Invalid lines are skipped and a short message is printed to stderr.

**Required shape:** at least `op` and `target`. Example:

- `{"op":"set","target":"cutoff","val":85}`
- `{"op":"set","target":"resonance","val":50}`

The receiving device (synth, MPE controller, etc.) must support MIDI-CI Property Exchange and interpret the JSON resource.

## MIDI 1.0 vs 2.0 (UMP)

- **Current behavior:** The daemon sends **MIDI 1.0 SysEx** (F0 … F7). libremidi’s `send_message()` is used; the payload is the JSON as 7-bit bytes inside the SysEx packet. Devices that accept MIDI-CI Property Exchange over classic SysEx will receive it.
- **MIDI 2.0 UMP:** For strict UMP (64-bit packets, fixed-size, hard real-time), MIDI 2.0 would require packaging the same SysEx into UMP SysEx8/16 messages. That path is not implemented here; if your platform’s libremidi offers a UMP send API, it can be added as an optional code path with SysEx kept as fallback.

## Interop

- **MIDI 2.0 Workbench:** Use it to monitor UMP/SysEx and verify Property Exchange messages.
- **libremidi** uses the platform MIDI API (CoreMIDI on macOS, ALSA/JACK on Linux, etc.), so any port visible in the system is available.

## References

- MIDI-CI Property Exchange: MMA/AMEI MIDI 2.0 specification (JSON over SysEx get/set resources).
- libremidi: https://github.com/celtera/libremidi
