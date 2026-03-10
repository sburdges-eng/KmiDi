#!/usr/bin/env bash
# =============================================================================
# Begin KmiDi training (kmidi-100-limit run contract). Uses tmux for long jobs.
# =============================================================================
# Prereqs for real run:
#   - On AWS GPU instance, or pass --allow-non-aws (debug only)
#   - Package: set training.activePackageId in config/run_contract.yaml,
#     or pass --package-id PKG, or --package-s3-uri s3://..., or --package-local-dir /path
#   - Real S3 buckets in run contract (or pass --output-s3-uri)
#
# Usage:
#   ./scripts/begin_training.sh
#   ./scripts/begin_training.sh --package-id my-pkg --run-id run-$(date +%Y%m%d-%H%M)
#   ./scripts/begin_training.sh --package-local-dir /path/to/package --allow-non-aws
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONTRACT="${REPO_ROOT}/config/run_contract.yaml"
SESSION="train"

# If no package specified and no activePackageId, dry-run and exit
if [[ $# -eq 0 ]]; then
  ACTIVE_PKG=""
  if [[ -f "$CONTRACT" ]]; then
    ACTIVE_PKG="$(python3 -c "
import json, sys
from pathlib import Path
p = Path('$CONTRACT')
text = p.read_text()
data = json.loads(text) if p.suffix == '.json' else __import__('yaml').safe_load(text)
print(data.get('training', {}).get('activePackageId', '') or '')
" 2>/dev/null)" || true
  fi
  if [[ -z "$ACTIVE_PKG" ]]; then
    echo "No package set. Running dry-run with a placeholder (use real --package-id / activePackageId for training)."
    exec python3 scripts/aws_train_entrypoint.py --dry-run --package-id "set-me" --run-id "run-dry"
  fi
fi

echo "Starting training in tmux session: $SESSION"
echo "Detach: Ctrl+B then D. Reattach: tmux attach -t $SESSION"
tmux new-session -s "$SESSION" -d "cd $REPO_ROOT && export PYTHONPATH=$REPO_ROOT && ./scripts/run_aws_training.sh $*; echo '---'; echo 'Session ended. Press Enter to close.'; read"
exec tmux attach -t "$SESSION"
