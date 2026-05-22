#!/usr/bin/env bash
# Option A (in-repo): copy PRROT, engine/DSP, music_brain learning/voice from KmiDi_FINAL
# into main repo. Excludes all UI (apps, gui, plugin editors, kmidi_gui). Run from repo root.
# See docs/KMIDI_FINAL_MERGE_PLAN.md.
set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
FINAL="${REPO_ROOT}/KmiDi_FINAL"
cd "$REPO_ROOT"

if [[ ! -d "$FINAL" ]]; then
  echo "KmiDi_FINAL not found at $FINAL. Set KMI_DI_FINAL_ROOT or clone into repo."
  exit 1
fi

echo "Source: $FINAL"
echo "Destination: $REPO_ROOT"

# ---- 1. C++ / Engine: include (prrot, penta), src_penta-core, engine/src/dsp (no UI) ----
copy_safe() {
  local src="$1" dest="$2"
  if [[ -d "$FINAL/$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    rsync -a --exclude='.DS_Store' "$FINAL/$src/" "$dest/" 2>/dev/null || cp -R "$FINAL/$src" "$dest"
    echo "Copied $src -> $dest"
  fi
}

# PRROT headers (C++ side)
copy_safe "engine/include/prrot" "include/prrot"

# Penta headers (merge into existing include/penta)
if [[ -d "$FINAL/engine/include/penta" ]]; then
  mkdir -p include/penta
  rsync -a --exclude='.DS_Store' "$FINAL/engine/include/penta/" include/penta/ 2>/dev/null || true
  echo "Merged engine/include/penta -> include/penta"
fi

# Penta-core source (merge; may have overlaps)
if [[ -d "$FINAL/engine/src_penta-core" ]]; then
  for sub in harmony groove diagnostics common ml osc mixer; do
    [[ -d "$FINAL/engine/src_penta-core/$sub" ]] && copy_safe "engine/src_penta-core/$sub" "src_penta-core/$sub"
  done
fi

# Engine DSP only (no ui, gui, components, plugin UI)
if [[ -d "$FINAL/engine/src/dsp" ]]; then
  mkdir -p engine/src
  rsync -a --exclude='.DS_Store' "$FINAL/engine/src/dsp" engine/src/
  echo "Copied engine/src/dsp -> engine/src/dsp"
fi

# Intent IR C++ (for alignment; main has engine/intent_ir intent_ir)
if [[ -d "$FINAL/engine/intent_ir/src" ]]; then
  mkdir -p engine/intent_ir
  rsync -a --exclude='.DS_Store' "$FINAL/engine/intent_ir/" engine/intent_ir/
  echo "Copied engine/intent_ir -> engine/intent_ir"
fi

# ---- 2. Python: prrot, music_brain/learning, music_brain/voice ----
if [[ -d "$FINAL/python/prrot" ]]; then
  mkdir -p music_brain
  rsync -a --exclude='.DS_Store' --exclude='__pycache__' "$FINAL/python/prrot" music_brain/
  echo "Copied python/prrot -> music_brain/prrot"
fi

if [[ -d "$FINAL/python/music_brain/learning" ]]; then
  mkdir -p music_brain
  rsync -a --exclude='.DS_Store' --exclude='__pycache__' "$FINAL/python/music_brain/learning" music_brain/
  echo "Copied python/music_brain/learning -> music_brain/learning"
fi

if [[ -d "$FINAL/python/music_brain/voice" ]]; then
  mkdir -p music_brain
  rsync -a --exclude='.DS_Store' --exclude='__pycache__' "$FINAL/python/music_brain/voice" music_brain/
  echo "Copied python/music_brain/voice -> music_brain/voice"
fi

# ---- 3. Configs (no large binaries) ----
if [[ -d "$FINAL/configs" ]]; then
  shopt -s nullglob
  for f in "$FINAL/configs"/*.yaml "$FINAL/configs"/*.yml; do
    [[ -f "$f" ]] && cp "$f" config/ 2>/dev/null && echo "Copied config $(basename "$f")"
  done
  shopt -u nullglob 2>/dev/null || true
fi

# ---- 4. Scripts (merge; avoid overwriting main entrypoints) ----
if [[ -d "$FINAL/scripts" ]]; then
  shopt -s nullglob
  for name in robustness verify build_intent; do
    for f in "$FINAL/scripts/"*"$name"*; do
      [[ -f "$f" ]] && cp "$f" scripts/ 2>/dev/null && echo "Copied script $(basename "$f")"
    done
  done
  shopt -u nullglob 2>/dev/null || true
fi

echo "Done. Next: resolve includes, CMake targets, Python imports; run build and tests."
echo "Do not commit KmiDi_FINAL UI or duplicate entrypoints."
