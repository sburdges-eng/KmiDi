#!/usr/bin/env bash
set -euo pipefail

HOURLY_USD=0
BUDGET_CAP_USD=100
POLL_SECONDS=30
START_EPOCH="$(date +%s)"
COMPLETE_SIGNAL=""

usage() {
  cat <<'USAGE'
Usage: cost_estimate.sh --hourly-usd <rate> [options]

Prints running cost estimate while training is active.

Options:
  --hourly-usd <rate>        Instance hourly cost in USD (required)
  --budget-cap-usd <cap>     Budget cap (default: 100)
  --start-epoch <seconds>    Epoch seconds when run started (default: now)
  --poll-seconds <n>         Poll interval (default: 30)
  --complete-signal <path>   Stop when this file exists
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hourly-usd)
      HOURLY_USD="$2"
      shift 2
      ;;
    --budget-cap-usd)
      BUDGET_CAP_USD="$2"
      shift 2
      ;;
    --start-epoch)
      START_EPOCH="$2"
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

if [[ "${HOURLY_USD}" == "0" || -z "${HOURLY_USD}" ]]; then
  echo "--hourly-usd must be > 0" >&2
  exit 2
fi

while true; do
  if [[ -n "${COMPLETE_SIGNAL}" && -f "${COMPLETE_SIGNAL}" ]]; then
    echo "[cost] completion signal detected: ${COMPLETE_SIGNAL}"
    exit 0
  fi

  now_epoch="$(date +%s)"
  elapsed_sec=$((now_epoch - START_EPOCH))

  cost_info="$(python3 - <<PY
elapsed_sec = int(${elapsed_sec})
hourly = float(${HOURLY_USD})
budget = float(${BUDGET_CAP_USD})
elapsed_hours = elapsed_sec / 3600.0
cost = hourly * elapsed_hours
remaining = budget - cost
pct = (cost / budget * 100.0) if budget > 0 else 0.0
breached = "1" if cost >= budget else "0"
print(f"elapsed_h={elapsed_hours:.3f} est_usd={cost:.2f} remaining_usd={remaining:.2f} used_pct={pct:.1f} breached={breached}")
PY
)"

  echo "[cost] ${cost_info}"

  breached="$(echo "${cost_info}" | sed -n 's/.*breached=\([01]\).*/\1/p')"
  if [[ "${breached}" == "1" ]]; then
    echo "[cost] BUDGET BREACHED — initiating shutdown"
    if [[ -n "${COMPLETE_SIGNAL}" ]]; then
      touch "${COMPLETE_SIGNAL}"
    fi
    shutdown -h +1 "Budget cap reached" 2>/dev/null || true
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done
