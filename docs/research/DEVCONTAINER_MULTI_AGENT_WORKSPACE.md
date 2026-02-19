# Devcontainer and Multi-Agent Workspace Setup

## Base Runtime

- Python 3.11+
- Node 20+
- CMake + Ninja
- Optional ONNX/CoreML toolchains

## Workspace Mounts

- Source tree mounted read/write.
- Model cache mounted to `~/Models`.
- Dataset mount at `~/Datasets` (governance-aligned).

## Agent Context Split

- Agent A: API / intent pipeline.
- Agent B: ML training/export.
- Agent C: UI/Tauri.
- Agent D: C++/JUCE and bridge.

## Build Steps

1. Install Python dependencies.
2. Install Node dependencies.
3. Build native targets.
4. Build/rebuild retrieval indexes.

## Caching

- Cache pip/npm layers in container image.
- Persist model/checkpoint outputs outside container lifecycle.
