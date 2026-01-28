#!/bin/bash
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
log="$repo_root/KmiDi_FINAL/RECOVERY_OPS/logs/disconnect_log.txt"
roots=(KmiDi_BACKUP KmiDi_PROJECT KmiDi_TRAINING assets data docs music_brain python penta_build scripts)
{
  echo "=== restore $(date) ==="
  for name in "${roots[@]}"; do
    src="$repo_root/${name}_DISCONNECTED"
    dst="$repo_root/$name"
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
      mv "$src" "$dst"
      echo "mv $src -> $dst"
    else
      echo "skip: $src missing or $dst already exists"
    fi
  done
} >>"$log"
echo "Restore complete. See $log"
