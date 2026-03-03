#!/usr/bin/env bash
set -euo pipefail
# Wrapper for root bootstrap.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/bootstrap.sh" "$@"
