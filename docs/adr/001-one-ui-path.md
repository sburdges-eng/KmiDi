# ADR 001: One UI Path for V1

## Status

Accepted

## Context

The repository currently contains multiple UI surfaces and integration paths, which creates
non-deterministic builds and unclear ownership boundaries. For v1, we need a single shell,
strict pipeline contracts, and a headless-capable engine build path.

## Decision

- Shell: Tauri + React is the only supported v1 desktop shell.
- Engine: Python `music_brain` intent pipeline plus C++ DSP core exposed through bindings.
- JUCE: restricted to audio/MIDI rendering and DSP support; no v1 standalone JUCE UI commitment.
- Legacy UI stacks (native AppKit paths and external Rust UI layers) are out of the v1 build matrix
  and treated as legacy code paths.

## Consequences

- CMake defaults disable legacy UI surfaces unless explicitly enabled.
- CI validates deterministic bootstrap + build with a headless-leaning configuration.
- API/schema hardening becomes mandatory at the UI-to-engine boundary.
- New feature work must align with this architecture until a future ADR supersedes it.
