#!/usr/bin/env bash
# Download all datasets from prepare_datasets.py to a local dataset root.
# Uses KMIDI_DATA_ROOT from env, or defaults to repo-local datasets/ directory.
# Run in tmux for long runs: tmux new -s datasets './scripts/download_all_datasets.sh'

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

DEFAULT_ROOT="${REPO_ROOT}/datasets"
export KMIDI_DATA_ROOT="${KMIDI_DATA_ROOT:-$DEFAULT_ROOT}"

if [[ ! -d "$KMIDI_DATA_ROOT" ]]; then
  echo "KMIDI_DATA_ROOT not mounted: $KMIDI_DATA_ROOT" >&2
  exit 1
fi

echo "Downloading all datasets to $KMIDI_DATA_ROOT/Datasets/raw (and downloads/)"
echo "This can take hours and use hundreds of GB. Detach with Ctrl+B then D if using tmux."
echo ""

python3 scripts/utilities/prepare_datasets.py --dataset all --download
