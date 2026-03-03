#!/usr/bin/env bash
# Wrapper for root bootstrap.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$BASH" "$ROOT_DIR/bootstrap.sh" "$@"
