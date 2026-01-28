#!/bin/bash
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
log="$repo_root/KmiDi_FINAL/RECOVERY_OPS/logs/disconnect_log.txt"
roots=(KmiDi_BACKUP KmiDi_PROJECT KmiDi_TRAINING assets data docs music_brain python penta_build scripts)
{
  echo "=== disconnect $(date) ==="
  for name in "${roots[@]}"; do
    src="$repo_root/$name"
    dst="${src}_DISCONNECTED"
    if [ -e "$dst" ]; then
      echo "skip: $dst already exists"
      continue
    fi
    if [ -e "$src" ]; then
      mv "$src" "$dst"
      echo "mv $src -> $dst"
    else
      echo "skip: $src not found"
    fi
  done
} >>"$log"
echo "Disconnection complete. See $log"
