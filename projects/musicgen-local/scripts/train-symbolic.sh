#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT_DIR/scripts/check_training_gate.sh"

echo "Train symbolic placeholder: wire to ml/training/symbolic entrypoint."
