#!/usr/bin/env bash
set -euo pipefail

EVAL_SUMMARY=""
METRIC_PATH="student.intent.val_macro_f1"
POLL_SECONDS=15
COMPLETE_SIGNAL=""
STATUS_FILE=""
PATIENCE=0
MIN_DELTA=0.0

usage() {
  cat <<'USAGE'
Usage: watchdog.sh [options]

Monitors eval summary progress and reports metric drift while training runs.

Options:
  --eval-summary <path>      Path to eval summary JSON
  --metric-path <dot.path>   Metric path to read (default: student.intent.val_macro_f1)
  --poll-seconds <n>         Poll interval seconds (default: 15)
  --complete-signal <path>   File that indicates training completion
  --status-file <path>       Optional status output file
  --patience <n>             Optional patience counter for staleness reporting
  --min-delta <float>        Improvement threshold for staleness (default: 0.0)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-summary)
      EVAL_SUMMARY="$2"
      shift 2
      ;;
    --metric-path)
      METRIC_PATH="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --complete-signal)
      COMPLETE_SIGNAL="$2"
      shift 2
      ;;
    --status-file)
      STATUS_FILE="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --min-delta)
      MIN_DELTA="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${STATUS_FILE}" ]]; then
  mkdir -p "$(dirname "${STATUS_FILE}")"
fi

BEST=""
STALL_COUNT=0

read_metric() {
  local path="$1"
  local metric_path="$2"
  WATCHDOG_EVAL_PATH="${path}" WATCHDOG_METRIC_PATH="${metric_path}" python3 - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ.get("WATCHDOG_EVAL_PATH", ""))
metric_path = os.environ.get("WATCHDOG_METRIC_PATH", "")
if not path.exists():
    print("")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
cur = payload
for key in metric_path.split('.'):
    if isinstance(cur, dict) and key in cur:
        cur = cur[key]
    else:
        print("")
        raise SystemExit(0)
try:
    print(float(cur))
except Exception:
    print("")
PY
}

while true; do
  if [[ -n "${COMPLETE_SIGNAL}" && -f "${COMPLETE_SIGNAL}" ]]; then
    echo "[watchdog] completion signal detected: ${COMPLETE_SIGNAL}"
    if [[ -n "${STATUS_FILE}" ]]; then
      printf 'state=complete\n' > "${STATUS_FILE}"
    fi
    exit 0
  fi

  metric=""
  if [[ -n "${EVAL_SUMMARY}" ]]; then
    metric="$(read_metric "${EVAL_SUMMARY}" "${METRIC_PATH}")"
  fi

  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -n "${metric}" ]]; then
    if [[ -z "${BEST}" ]]; then
      BEST="${metric}"
      STALL_COUNT=0
    else
      improved="$(python3 - <<PY
best = float(${BEST})
current = float(${metric})
min_delta = float(${MIN_DELTA})
print("1" if current > best + min_delta else "0")
PY
)"
      if [[ "${improved}" == "1" ]]; then
        BEST="${metric}"
        STALL_COUNT=0
      else
        STALL_COUNT=$((STALL_COUNT + 1))
      fi
    fi

    echo "[watchdog] ${now} metric=${metric} best=${BEST} stall=${STALL_COUNT}"

    if [[ -n "${STATUS_FILE}" ]]; then
      printf 'timestamp=%s\nmetric=%s\nbest=%s\nstall=%s\n' "${now}" "${metric}" "${BEST}" "${STALL_COUNT}" > "${STATUS_FILE}"
    fi

    if [[ "${PATIENCE}" -gt 0 && "${STALL_COUNT}" -ge "${PATIENCE}" ]]; then
      echo "[watchdog] metric plateau detected (stall=${STALL_COUNT}, patience=${PATIENCE})"
      if [[ -n "${STATUS_FILE}" ]]; then
        printf 'state=plateau\n' >> "${STATUS_FILE}"
      fi
    fi
  else
    echo "[watchdog] ${now} waiting for metric path ${METRIC_PATH}"
  fi

  sleep "${POLL_SECONDS}"
done
