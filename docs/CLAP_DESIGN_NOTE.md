# CLAP Design Note: Deep Modulation & Per-Note Expression

**Date:** 2026-02-28  
**Purpose:** Capture CLAP as the target for host-native modulation and per-note expression when implementing KmiDi’s plugin/host layer.  
**Related:** [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md), [HOST_GLUE_ARCHITECTURE.md](HOST_GLUE_ARCHITECTURE.md), [specs/07_PLUGIN_SPECIFIC.md](specs/07_PLUGIN_SPECIFIC.md)

---

## Why CLAP

The **CLAP** (CLever Audio Plug-in) API is an open-source plugin standard that exposes richer host-facing primitives than VST2/VST3/AU:

- **Per-note parameters / note expression** — Different expressive control per key or voice, not only global automation.
- **Parameter metadata & modulation flags** — Hosts can query structure and route non-destructive, per-voice modulation.
- **Modern threading & MIDI 2.0** — Designed for current CPUs and UMP transport.
- **Permissive license** — No proprietary SDK lock-in.

Hosts (e.g. Bitwig Studio, REAPER) and plugins (Surge, u-he) are already using these features. Designing for them from the start avoids the workarounds of older formats and aligns with “emotional latents” or per-voice modulation workflows (e.g. painting expression lanes per note with host automation remaining non-destructive).

---

## Design Target for KmiDi

When implementing **plugin** and **host** support (see `src/plugin/`, JUCE 8 in PROJECT_SUMMARY, and audits that list VST3/CLAP as planned):

1. **CLAP as first-class**  
   Treat CLAP as the primary format for any new host/plugin code that needs deep modulation. VST3 can follow or run in parallel, but CLAP’s primitives should drive the design.

2. **Per-note and per-voice from the start**  
   Expose parameters and metadata so that:
   - Hosts can draw automation per voice and route modulation per note.
   - “Emotional” or expressive targets (e.g. per-note intensity, tension) can be represented as per-note parameters or note expression where the host supports it.

3. **Non-destructive host automation**  
   Use CLAP’s model so host automation and modulation do not overwrite or destroy the plugin’s base parameter state; the host can query and route without hacky workarounds.

---

## Host Responsibilities (When KmiDi Acts as Host)

The host glue layer (see [HOST_GLUE_ARCHITECTURE.md](HOST_GLUE_ARCHITECTURE.md)) should, for CLAP plugins:

| Responsibility | Description |
|----------------|-------------|
| **Parameter metadata** | Query and expose parameter info (ranges, steps, modulation flags) so the host can build modulation UI and automation lanes. |
| **Modulation flags** | Honor CLAP modulation flags so the host knows which parameters accept per-voice or per-note modulation and can route accordingly. |
| **Note expression** | Support CLAP note expression where applicable (e.g. per-note pitch, pressure, timbre) and pass through high-resolution MIDI/UMP when available. |
| **Structure query** | Use CLAP’s discovery/query APIs so the host can present parameter sections and modulation targets without hard-coded hacks. |

Host glue remains “plumbing”: format translation, lifecycle, threading. The above are requirements on what that plumbing must expose for CLAP, not DSP or UI logic.

---

## Plugin Responsibilities (When KmiDi Exposes a CLAP Plugin)

For KmiDi’s own plugins (e.g. ML/intent-driven processors as in [specs/07_PLUGIN_SPECIFIC.md](specs/07_PLUGIN_SPECIFIC.md)):

- **Expose rich parameter metadata** — So any CLAP host can discover modulation targets and build unified modulation (Bitwig-style) or per-note lanes.
- **Declare modulation capability** — Use CLAP flags to indicate which parameters support per-note vs global modulation.
- **Respect note expression** — Where the design has per-voice or per-note meaning (e.g. “emotional” or intensity per note), expose it via CLAP’s per-note/note-expression model so hosts can automate and modulate it natively.

Existing required parameters (`ml_intensity`, `melody_influence`, etc.) should be extended with CLAP-friendly metadata and, where semantically correct, per-note or modulation flags so they fit into modern host workflows.

---

## Connection to Existing Roadmap

- **PROJECT_SUMMARY.md:** JUCE 8, plugin hosting (VST3/CLAP), `src/plugin/` for VST3/CLAP plugin code.
- **Audits:** VST3/CLAP implementation is planned but not yet implemented; no plugin shells in tree yet.
- **HOST_GLUE_ARCHITECTURE.md:** Host glue does format translation and lifecycle; this note adds CLAP-specific requirements to that contract.

Implementing CLAP support (and host support for CLAP) with the above in mind keeps KmiDi aligned with host-native deep modulation and per-note expression as the ecosystem (Bitwig, REAPER, Surge, u-he) adopts them.

---

## References (External)

- CLAP specification and SDK (open-source).
- Bitwig Studio: Unified Modulation System and CLAP support.
- REAPER: Community work on CLAP and parameter sections.
- Surge, u-he: Early CLAP adopters with per-note/polyphonic modulation in practice.
