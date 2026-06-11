#!/usr/bin/env bash
# scripts/init_matrix.sh — Hermes Tmux Matrix bootstrap (thin wrapper)
#
# Delegates to the canonical pipeline script. Window indices 0-6 are the
# contract with scripts/kmidi_swarm.py — see docs/SWARM_MATRIX.md.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kmidi-pipeline-tmux.sh" "$@"
