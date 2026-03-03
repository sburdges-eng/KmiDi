#!/usr/bin/env bash
# Wrapper for root bootstrap.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/bootstrap.sh" ]]; then
  exec "$ROOT_DIR/bootstrap.sh" "$@"
else
  exec bash "$ROOT_DIR/bootstrap.sh" "$@"
fi
