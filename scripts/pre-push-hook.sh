#!/usr/bin/env bash
# Pre-push hook: run brain check and stub-creep check.
# Install: cp scripts/pre-push-hook.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/KmiDi_CANON/brain:${PYTHONPATH:-}"
echo "[pre-push] run_brain.py check ..."
python3 run_brain.py check
echo "[pre-push] check_stub_creep.py --allow-docs ..."
python3 scripts/check_stub_creep.py --allow-docs
echo "[pre-push] OK"
exit 0
