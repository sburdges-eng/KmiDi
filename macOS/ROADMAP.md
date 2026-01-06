# Roadmap
This module follows the canonical roadmap defined in the root ROADMAP.md (see canonical archive referenced by maintainers).
See TODO.md in this directory for actionable items.

## Sequencer-grade baseline decisions (initial)
- Project time is stored in musical ticks; runtime maps project to/from host/sample via a transport mapper.
- Standalone mode: audio render callback sampleTime is the clock authority; MIDI scheduling uses hostTime derived from sampleTime with output-latency compensation.
- DAW/headless mode: host monotonic time is the authority; playback and record offsets are configurable (calibration required per setup).
- Default monitoring policy: play immediately for feel; record with offset to align to project time. Offer compensated monitoring as an optional mode.
