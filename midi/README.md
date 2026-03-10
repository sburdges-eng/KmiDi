# KmiDi MIDI 2.0 Affect Channel

Official MIDI 2.0 affect channel for KmiDi: Property Exchange (PE) resource declaration and UMP 32-bit controller mapping for live affect lanes. See [docs/research/MULTIMODAL_REPRESENTATIONS_2026.md](../docs/research/MULTIMODAL_REPRESENTATIONS_2026.md) (section 6) and [docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md](../docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md) (section 4).

## PE Resource

- **Resource ID:** `com.sburdges.kmidi/affect.v1`
- **Schema:** [pe_affect_schema.json](pe_affect_schema.json) — JSON Schema for the affect resource (MMA/AMEI PE rules).
- **Properties:** `valence` (−1..1), `arousal` (0..1), `dynamics` (0..1), optional `timestamp`, `mode` (`"ride"` | `"snapshot"`).

A standalone PE responder responds to MIDI-CI Get Resource for this ID with the resource body:

```bash
python3 scripts/pe_affect_responder.py --port "Your MIDI Port" -v
```

Optional: add `midi/pe_affect_instance.json` with default `valence`/`arousal`/`dynamics`/`mode` values. The plugin (or a future build) will advertise and serve this resource when running in a MIDI 2.0 host.

## UMP Channel Voice Mapping

Vendor assignable controller indices for live control (100–250 Hz):

| Index (hex) | Property  | Float range |
|-------------|-----------|-------------|
| 0x28        | valence   | −1..1       |
| 0x29        | arousal   | 0..1        |
| 0x2A        | dynamics  | 0..1        |

Floats are linearly mapped to 32-bit UMP values. Control rate: target 100–250 Hz when driving from the Brain or UI.

## Test Harness

Run the UMP affect test harness to stream synthetic curves to a UMP-capable port:

```bash
python3 scripts/ump_affect_harness.py --port "KmiDi Virtual" --duration 30 --curve sine
```

Use `--list-ports` to see output port names. See script docstring and [scripts/ump_affect_harness.py](../scripts/ump_affect_harness.py). For DAW verification: use a virtual UMP port or loopback if needed; verify in a MIDI 2.0–capable DAW (e.g. Logic Pro X) that 32-bit CC lanes 0x28, 0x29, 0x2A show the curves. If the port or driver does not accept UMP, drive the plugin from the Music Brain valence/arousal API or use a C++ harness that links `src/midi/AffectUMP` and sends to a UMP-capable port.

## Expressive Controller (Sensel Morph / MPE)

Section 5 of the [Multimodal Implementations Plan](../docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md): use Sensel Morph (or similar) to drive affect and/or MPE.

- **SenselApp profile:** [sensel_kmidi_profile.md](sensel_kmidi_profile.md) — recommended map: MPE channels, affect CC 40/41/42 (valence/arousal/dynamics), scaling, and export/import.
- **Morph bridge:** `scripts/morph_affect_bridge.py` — reads MIDI from the Morph (or direct Sensel API), normalizes to valence/arousal/dynamics, optionally forwards as UMP to a DAW.

```bash
python3 scripts/morph_affect_bridge.py --list-ports
python3 scripts/morph_affect_bridge.py --midi-in "Sensel Morph" --midi-out "KmiDi Virtual" --forward-ump
# Or with Sensel API (requires sensel-api from GitHub): --sensel --midi-out "KmiDi Virtual" --forward-ump
```
