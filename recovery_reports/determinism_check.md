# Determinism Check

- input set: `/tmp/kmidi_recovered_paths.txt`
- matcher version: `v1.0.0`
- check result: `pass`
- compared files:
  - `recovery_reports/inventory_canonical.jsonl`
  - `recovery_reports/inventory_recovered.jsonl`
  - `recovery_reports/matches.jsonl`
  - `recovery_reports/decisions.csv`
  - `recovery_reports/review_queue.csv`

All five files were byte-identical across consecutive runs with unchanged inputs.
