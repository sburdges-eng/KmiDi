# TASK-0003: JUCE Plugin Shell
Status: TODO
Owner: local
Epic: EPIC-09
Priority: P0
Estimate: 4d
DependsOn: TASK-0002

## Goal
Create AU/VST3 plugin shell with host transport sync and non-realtime IPC bridge points.

## Acceptance Criteria
- Plugin builds as AU/VST3 target.
- Host tempo/transport values are readable in plugin.
- Audio thread remains non-blocking; sidecar calls are offloaded.

## Notes
Begin with dry-run requests and cached placeholder responses.
