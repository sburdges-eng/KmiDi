#!/bin/bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
log_dir="$repo_root/KmiDi_FINAL/RECOVERY_OPS/logs"
raw_log="$log_dir/file_access_log_raw.txt"
dtrace_log="$log_dir/file_access_dtrace.txt"
output="$log_dir/external_accesses.txt"
kmidi_path="$repo_root/KmiDi_FINAL"
> "$output"

if [ -f "$raw_log" ]; then
  echo "=== Analyzing fs_usage log ==="
  grep -E "(open|stat|access)" "$raw_log" | \
    awk -v kmidi_path="$kmidi_path" '{ for (i=1; i<=NF; i++) if ($i ~ /^\// && $i !~ kmidi_path) print $i }' | \
    sort -u >> "$output"
fi

if [ -f "$dtrace_log" ]; then
  echo "=== Analyzing dtrace log ==="
  awk -F'|' -v kmidi_path="$kmidi_path" '$2 ~ /^\// && $2 !~ kmidi_path { print $1 "|" $2 }' "$dtrace_log" | \
    sort -u >> "$output"
fi

# Filter out known system paths
if [ -s "$output" ]; then
  grep -vE '^(/System|/usr|/Library|/opt/homebrew|/tmp|/var|/dev)' "$output" | sort -u > "$output.filtered" || true
  mv "$output.filtered" "$output"
fi

if [ -s "$output" ]; then
  echo "WARNING: External file accesses detected:"
  cat "$output"
  exit 1
else
  echo "SUCCESS: No external file accesses detected"
  exit 0
fi
