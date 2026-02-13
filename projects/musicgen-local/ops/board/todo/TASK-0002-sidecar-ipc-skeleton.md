# TASK-0002: Sidecar IPC Skeleton
Status: TODO
Owner: local
Epic: EPIC-07
Priority: P0
Estimate: 3d
DependsOn: TASK-0001

## Goal
Scaffold a local sidecar runtime and IPC contract for generation requests and status streaming.

## Acceptance Criteria
- `apps/sidecar-engine` process starts and exposes health endpoint.
- `libs/ipc-protocol` defines request/response + heartbeat messages.
- Timeout and fallback behavior is documented.

## Notes
No inference yet; only protocol, lifecycle, and reliability primitives.
