# KmiDi Workspace Setup

Status: current editor/workspace guide aligned to checked-in workspace files
Last updated: 2026-06-08

This document explains the workspace/editor assets that actually exist in the repo today.
It is not architecture authority.

## 1. What is currently checked in

Workspace/editor configuration currently present:
- `KmiDi.code-workspace`
- `.vscode/settings.json`
- `.vscode/tasks.json`
- `.vscode/launch.json`
- `.vscode/extensions.json`

Use these as convenience tooling around the canonical repo.
Do not infer product architecture from them.

## 2. Opening the workspace

In VS Code or Cursor:
1. File -> Open Workspace from File...
2. choose `KmiDi.code-workspace`

If you prefer opening the repo directly instead of the workspace file, that is also fine; the `.vscode/` folder still applies.

## 3. What the current VS Code tasks actually do

The checked-in tasks are native/debug oriented, not full-stack app launchers.
Current task labels in `.vscode/tasks.json` are:
- `cmake-configure-debug`
- `cmake-build-rt-harness-debug`
- `cmake-configure-asan`
- `cmake-build-rt-harness-asan`
- `cmake-configure-tsan`
- `cmake-build-rt-harness-tsan`
- `build-debug`

What they correspond to:
- `cmake-configure-debug` -> `cmake --preset ninja-debug`
- `cmake-build-rt-harness-debug` -> build RT harness with the debug preset
- `cmake-configure-asan` -> `cmake --preset ninja-asan`
- `cmake-build-rt-harness-asan` -> build RT harness with ASan
- `cmake-configure-tsan` -> `cmake --preset ninja-tsan`
- `cmake-build-rt-harness-tsan` -> build RT harness with TSan
- `build-debug` -> build a scratch single-file C++ program from `.cxx-scratch/`

What is not currently present in checked-in tasks:
- no `Tauri: Dev` task
- no combined React + API launch task
- no generic “build everything” task

For app/service bring-up, use terminal commands from `docs/DEVELOPMENT.md` instead.

## 4. What the current debug launch configs actually do

Current launch entries in `.vscode/launch.json` are intended for C++/RT-harness debugging:
- `Debug rt_harness (clangd)`
- `Debug rt_harness ASan (LLDB)`
- `Debug rt_harness TSan (LLDB)`
- `Debug single-file app (LLDB)`

Operational notes:
- the launch file is native-focused
- it is not a frontend/API debug setup
- if you need browser or API debugging, use the normal web/browser and Python workflows outside these launch configs

Caveat:
- `.vscode/launch.json` appears to contain at least one malformed line near the first configuration (`type` is damaged). Treat the file as partially stale until repaired. This guide documents intent and checked-in contents, not guaranteed editor correctness.

## 5. Extensions currently recommended

`.vscode/extensions.json` currently recommends only:
- `llvm-vs-code-extensions.vscode-clangd`

That means the repo is not presently asserting a broad extension pack.
Install other tools as needed for your workflow, for example:
- Python support
- rust-analyzer
- CMake tooling
- Markdown support

But those are personal/editor choices unless the repo later checks them in explicitly.

## 6. CMake presets that pair with workspace tasks

The editor tasks rely on `CMakePresets.json`.
Current configure presets:
- `xcode-debug`
- `xcode-release`
- `ninja-debug`
- `ninja-asan`
- `ninja-tsan`

Current build presets:
- `xcode-debug`
- `xcode-release`
- `ninja-debug-rt-harness`
- `ninja-asan-rt-harness`
- `ninja-tsan-rt-harness`

If you use the workspace for native work, prefer these presets over ad hoc editor-specific CMake settings.

## 7. Recommended practical usage

### Frontend/API work
Use the terminal, not the checked-in workspace tasks:

```bash
npm run dev:all
```

Or separately:

```bash
npm run dev
npm run dev:python
```

### Native runtime work
Use the checked-in preset/task flow:

```bash
cmake --preset ninja-debug
cmake --build --preset ninja-debug-rt-harness
```

For sanitizer work:

```bash
cmake --preset ninja-asan
cmake --build --preset ninja-asan-rt-harness
```

### Scratch C++ experiments
Put a source file under `.cxx-scratch/` and use the `build-debug` task.
This is an isolated convenience path, not a production target workflow.

## 8. Workspace interpretation rules

When workspace files and docs disagree, prefer:
1. actual commands in `package.json`
2. actual commands in `CMakePresets.json`
3. architecture authority docs for ownership and boundaries
4. older workspace narratives last

Examples of older assumptions that should not be reintroduced automatically:
- “Tauri is the canonical app shell”
- “workspace tasks launch the whole stack”
- “the workspace debug profiles are the main way to run the product”

## 9. Known drift

These facts matter for future cleanup:
- older workspace documentation claimed Tauri tasks and broader extension recommendations that are not present now
- the checked-in launch config is partially malformed and should be repaired before being treated as a dependable default debug surface
- current checked-in workspace assets are much more native/RT-harness oriented than product/app oriented

## 10. Related docs

- `docs/DEVELOPMENT.md`
- `docs/BOOT.md`
- `BUILD.md`
- `AGENTS.md`
