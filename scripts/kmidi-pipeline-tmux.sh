#!/usr/bin/env bash
# KmiDi Hermes Tmux Matrix — canonical bootstrap
# Pipeline: Schemas → Rust IR → C++ Core → Bindings → Music Brain → React
#
# Window indices 0-6 are the contract with scripts/kmidi_swarm.py.
# DO NOT renumber windows without also updating the PANES map in that script.
set -euo pipefail

SESSION="kmidi-pipeline"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Non-interactive env so manual panes match scripts/kmidi_swarm.py executors.
ENV_PREFIX='export CI=true DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true GIT_PAGER=cat PAGER=cat && '

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach-session -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" -n "⓪ overview"

# Apply env globally so panes created later inherit it too.
tmux set-environment -t "$SESSION" CI true
tmux set-environment -t "$SESSION" DEBIAN_FRONTEND noninteractive
tmux set-environment -t "$SESSION" GIT_TERMINAL_PROMPT 0
tmux set-environment -t "$SESSION" GIT_EDITOR true
tmux set-environment -t "$SESSION" GIT_PAGER cat
tmux set-environment -t "$SESSION" PAGER cat

# ── Window 0: Overview ────────────────────────────────
tmux send-keys -t "$SESSION:0" "${ENV_PREFIX}clear && printf '\\n\\033[1;36m  KmiDi Critical Path Pipeline\\033[0m\\n\\n' && \
printf '  \\033[0;90m───────────────────────────────────────────────────\\033[0m\\n' && \
printf '  \\033[1;37m① Shared Schemas\\033[0m  →  shared_schemas/\\n' && \
printf '  \\033[1;32m② Rust Intent IR\\033[0m  →  engine/intent_ir/\\n' && \
printf '  \\033[1;36m③ C++ Core/FFI\\033[0m    →  src/ engine/ include/\\n' && \
printf '  \\033[1;35m④ Python Bindings\\033[0m →  bindings/\\n' && \
printf '  \\033[1;33m⑤ Music Brain\\033[0m     →  music_brain/\\n' && \
printf '  \\033[1;34m⑥ React Frontend\\033[0m  →  src/ (TSX)\\n' && \
printf '  \\033[0;90m───────────────────────────────────────────────────\\033[0m\\n\\n' && \
printf '  \\033[0;90mPrefix: Ctrl-b  |  Windows: 0-6  |  Panes: Ctrl-b %%\\033[0m\\n\\n' && \
git status -sb 2>/dev/null | head -8" Enter

# ── Window 1: Shared Schemas ──────────────────────────
tmux new-window -t "$SESSION" -n "① schemas" -c "$ROOT/shared_schemas"
tmux send-keys -t "$SESSION:1" "${ENV_PREFIX}clear && printf '\\n\\033[1;37m  ① Shared Schemas\\033[0m  ─  source of truth for cross-language contract\\n\\n' && \
ls -la *.json 2>/dev/null && echo && \
printf '\\033[0;90m  Sync command: python3 scripts/sync_entities.py\\033[0m\\n\\n'" Enter

# ── Window 2: Rust Intent IR ─────────────────────────
tmux new-window -t "$SESSION" -n "② rust-ir" -c "$ROOT/engine/intent_ir"
tmux send-keys -t "$SESSION:2" "${ENV_PREFIX}clear && printf '\\n\\033[1;32m  ② Rust Intent IR\\033[0m  ─  IntentFrame validator/builder/types (staticlib)\\n\\n' && \
cargo check 2>&1 | tail -5 && echo && \
printf '\\033[0;90m  Build:  cargo build --release\\033[0m\\n' && \
printf '\\033[0;90m  Test:   cargo test\\033[0m\\n' && \
printf '\\033[0;90m  Lint:   cargo clippy\\033[0m\\n\\n'" Enter

# ── Window 3: C++ Core / KellyFFI ────────────────────
tmux new-window -t "$SESSION" -n "③ cpp-core" -c "$ROOT"
# Split: left = src/ browser, right = build commands
tmux send-keys -t "$SESSION:3" "${ENV_PREFIX}clear && printf '\\n\\033[1;36m  ③ C++ Core / KellyCore / KellyFFI\\033[0m\\n\\n' && \
printf '  src/           — UI, core, DSP, bridge, ML, biometric, PRROT\\n' && \
printf '  engine/        — CoreBridge, IntentPipeline, StateMachineConductor\\n' && \
printf '  include/       — daiw headers, common types\\n\\n' && \
ls build*/libKellyCore* build*/libKellyFFI* 2>/dev/null | head -6; \
printf '\\n\\033[0;90m  Configure: cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON\\033[0m\\n' && \
printf '\\033[0;90m  Build:     cmake --build build --target KellyFFI -j8\\033[0m\\n' && \
printf '\\033[0;90m  Test:      ctest --test-dir build --output-on-failure\\033[0m\\n\\n'" Enter

# ── Window 4: Python Bindings ────────────────────────
tmux new-window -t "$SESSION" -n "④ bindings" -c "$ROOT/bindings"
tmux send-keys -t "$SESSION:4" "${ENV_PREFIX}clear && printf '\\n\\033[1;35m  ④ Python Bindings (pybind11)\\033[0m\\n\\n' && \
ls -la *.cpp 2>/dev/null && echo && \
printf '\\033[0;90m  Modules: diagnostics, groove, harmony, ML, OSC, PRROT\\033[0m\\n' && \
printf '\\033[0;90m  Exposes C++ engine to Python / Music Brain\\033[0m\\n\\n'" Enter

# ── Window 5: Music Brain ────────────────────────────
tmux new-window -t "$SESSION" -n "⑤ brain" -c "$ROOT/music_brain"
# Split: top = main dir, bottom = API server
tmux send-keys -t "$SESSION:5" "${ENV_PREFIX}clear && printf '\\n\\033[1;33m  ⑤ Music Brain  ─  FastAPI backend (172K LOC, 455 files)\\033[0m\\n\\n' && \
ls -d */ 2>/dev/null | head -20 && echo && \
printf '\\033[0;90m  Run API:  python3 -m uvicorn music_brain.api:app --reload --port 8000\\033[0m\\n' && \
printf '\\033[0;90m  Lint:     python3 -m flake8 music_brain/ --max-line-length 100\\033[0m\\n' && \
printf '\\033[0;90m  Test:     python3 -m pytest tests/\\033[0m\\n\\n'" Enter
tmux split-window -v -t "$SESSION:5" -c "$ROOT" -p 30
tmux send-keys -t "$SESSION:5.1" "${ENV_PREFIX}true # API server pane — run: python3 -m uvicorn music_brain.api:app --reload --port 8000" Enter

# ── Window 6: React Frontend ────────────────────────
tmux new-window -t "$SESSION" -n "⑥ react" -c "$ROOT"
tmux send-keys -t "$SESSION:6" "${ENV_PREFIX}clear && printf '\\n\\033[1;34m  ⑥ React Frontend  ─  16 components, Vite + TypeScript\\033[0m\\n\\n' && \
printf '  Entry:     src/main.tsx → AppConsole.tsx\\n' && \
printf '  Hook:      src/hooks/useMusicBrain.ts (14 API methods)\\n' && \
printf '  Components: SideA (DAW), SideB (AI), IntentBuilder, Panels\\n' && \
printf '  CSS:       2533 lines design system\\n\\n' && \
printf '\\033[0;90m  Dev:        npm run dev\\033[0m\\n' && \
printf '\\033[0;90m  Dev+API:    npm run dev:all\\033[0m\\n' && \
printf '\\033[0;90m  Type-check: npx tsc --noEmit\\033[0m\\n' && \
printf '\\033[0;90m  Build:      npm run build\\033[0m\\n\\n'" Enter
tmux split-window -v -t "$SESSION:6" -c "$ROOT" -p 30
tmux send-keys -t "$SESSION:6.1" "${ENV_PREFIX}true # Vite dev server pane — run: npm run dev" Enter

# ── Select overview window and attach ────────────────
tmux select-window -t "$SESSION:0"
tmux attach-session -t "$SESSION"
