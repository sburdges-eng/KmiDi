# Sensel Morph Profile for KmiDi Affect and MPE

Recommended SenselApp setup so the Sensel Morph can drive KmiDi's affect channel (valence, arousal, dynamics) and optionally MPE for per-note expression. See [docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md](../docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md) section 5 and [README.md](README.md).

## Overview

- **MPE:** Use an MPE-capable overlay/map so per-note expression (pressure, slide) goes to MPE channels.
- **Affect CCs:** Configure three controls (or zones) to send continuous CC values that map to valence, arousal, and dynamics. The Morph bridge (`scripts/morph_affect_bridge.py`) can read these and optionally re-send as UMP to a DAW.

## SenselApp Map Storage

Maps are stored in SenselApp preferences and can be exported/imported:

- **macOS:** `~/Library/Application Support/unity.Sensel.SenselApp/`
- **Windows:** `%appdata%\..\LocalLow\Sensel\SenselApp\`

Use **Export Map** (top menu) to save a `.senselmap` file to share or back up. Use **Import Map** to load a map. After editing, click **Send Map to Morph** so the device uses the new map.

## MPE Setup

1. In SenselApp, select your Morph and open **Morph Settings**.
2. Set **MPE Channel Start** and **MPE Channel End** (e.g. 1–16) for per-note expression.
3. Enable **X/Y/Pitch Bend On** and **Pressure/Velocity On** for MIDI/MPE if you want position and pressure to drive MPE.
4. Add or select an overlay that supports MPE (e.g. Music Production overlay with an MPE variant). Send Map to Morph.

Per-note expression then goes to your DAW on the configured MPE channels; no extra KmiDi config required.

## Affect CC Mapping (Global Valence / Arousal / Dynamics)

KmiDi uses three vendor CC indices for live affect. In SenselApp Overlay Mapper, create **MIDI CC** controls with **After-Pressure** set so values update continuously while you press/slide.

### Option A — MIDI 2.0 / UMP (if host supports UMP)

Configure three controls to send CC numbers that match KmiDi's vendor indices:

| CC (decimal) | Hex  | KmiDi property | Float range |
|--------------|------|-----------------|-------------|
| 40           | 0x28 | valence         | −1..1       |
| 41           | 0x29 | arousal         | 0..1        |
| 42           | 0x2A | dynamics        | 0..1        |

- **Control type:** MIDI CC.
- **After-Pressure:** Set to the same CC number (or a CC message type) so the control sends continuous values while pressed. Valence should use center = neutral: map your zone so that the middle of the range (e.g. CC 64 for 7-bit) corresponds to 0 valence; left = negative, right = positive.
- **Channel:** Pick a channel (e.g. 0) that your DAW or Morph bridge will listen on.

If your SenselApp or host uses 32-bit UMP CC values, the same indices 0x28, 0x29, 0x2A apply; the Morph bridge can forward 7-bit or 14-bit MIDI 1.0 CC as UMP.

### Option B — MIDI 1.0 fallback

Use the same CC numbers **40, 41, 42** (decimal). The Morph bridge in MIDI-in mode will:

- Read CC 40 as valence (0–127 → −1..1, with 64 = 0).
- Read CC 41 as arousal (0–127 → 0..1).
- Read CC 42 as dynamics (0–127 → 0..1).

You can then run the bridge with `--forward-ump` to re-send as UMP to a KmiDi virtual port or DAW.

### Scaling

- **Valence:** CC 0 = −1, CC 64 = 0, CC 127 = 1. (Linear: `(cc - 64) / 64`.)
- **Arousal and dynamics:** CC 0 = 0, CC 127 = 1. (Linear: `cc / 127`.)

Configure your SenselApp control areas (e.g. sliders or pressure zones) so the output CC range matches these semantics.

## Using the Morph Bridge

After the Morph is configured to send these CCs (or you use the Sensel API path):

```bash
# List MIDI ports
python3 scripts/morph_affect_bridge.py --list-ports

# MIDI-in: read from Morph, forward to UMP port
python3 scripts/morph_affect_bridge.py --midi-in "Sensel Morph" --midi-out "KmiDi Virtual" --forward-ump

# Optional: Sensel API direct (requires sensel-api from https://github.com/sensel/sensel-api)
python3 scripts/morph_affect_bridge.py --sensel --midi-out "KmiDi Virtual" --forward-ump
```

See `scripts/morph_affect_bridge.py --help` and [README.md](README.md#expressive-controller-sensel-morph--mpe).
