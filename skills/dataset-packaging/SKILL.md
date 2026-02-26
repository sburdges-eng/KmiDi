---
name: dataset-packaging
description: Build, split, package, validate, and sync training datasets for the Kelly listening-contract pipeline. Use when tasks involve `scripts/build_training_records.py`, `scripts/split_dataset.py`, `scripts/package_dataset.py`, `scripts/validate_package.py`, or `scripts/sync_package_to_s3.py`, including dry-run destination checks, manifest/checksum integrity troubleshooting, and run-contract package bucket/prefix resolution.
---

# Dataset Packaging

Use this skill for deterministic dataset preparation from transcript JSONL through S3 package sync.

Read `references/dataset-packaging-runbook.md` first when you need exact command templates or recovery steps.

## Workflow Decision Tree

- Creating a new package from transcript data:
Run record build, deterministic split, package dry-run, package write, validate, then sync.
- Revalidating an existing package:
Run validator against `PACKAGES/<package-id>` and inspect manifest/checksum/shard mismatches.
- Syncing or resuming uploads:
Use sync dry-run, then run sync; rely on SHA256 metadata and skip logic for incremental reruns.
- Debugging pipeline failures:
Classify as transcript input/schema, split inputs, shard packaging, schema validation, or S3 destination/auth.

## 1) Build Task Records

Generate records from transcript JSONL before all downstream steps:
- `python3 scripts/build_training_records.py --transcript-jsonl <path/to/transcripts.jsonl>`
Confirm outputs under `training/records`:
- `intent_router_records.jsonl`
- `axis_proposer_records.jsonl`
- `build_summary.json`
If axis extraction looks wrong, inspect `config/axes.yaml` and the candidate source fields.

## 2) Split Deterministically

Split records by `sessionId` hash:
- `python3 scripts/split_dataset.py`
Only adjust `--seed` or ratio flags when requested; default seed and ratios are part of reproducibility.
Confirm outputs under `training/splits/<task>/` for `train.jsonl`, `val.jsonl`, `test.jsonl`.

## 3) Package Shards and Manifests

Run packaging dry-run first:
- `python3 scripts/package_dataset.py --package-id <package-id> --dry-run`
Then package:
- `python3 scripts/package_dataset.py --package-id <package-id>`
Treat dry-run JSON as the source of truth for resolved S3 destination (`packageBucket`, `packagePrefix`, `packageS3Uri`).

## 4) Validate Package Integrity

Validate every package before sync:
- `python3 scripts/validate_package.py --package-dir PACKAGES/<package-id>`
Require successful schema, checksum, and shard record-count validation before upload.

## 5) Sync Incrementally to S3

Run sync dry-run first:
- `python3 scripts/sync_package_to_s3.py --package-dir PACKAGES/<package-id> --dry-run`
Run real sync after dry-run output is correct:
- `python3 scripts/sync_package_to_s3.py --package-dir PACKAGES/<package-id>`
Expect incremental behavior: unchanged objects are skipped by SHA256 comparison.

## 6) Emit Shared Handoff State

After package validation and sync, persist handoff state in this exact JSON shape:

```json
{
  "package-id": "",
  "runId": "",
  "instanceId": "",
  "sessionName": "",
  "outputS3Uri": ""
}
```

Set `package-id` from packaging output and leave unknown runtime keys as empty strings.

## Troubleshooting Rules

Classify failures before changing scripts:
- Input failures: missing transcript JSONL or missing `training/records` task files.
- Packaging failures: missing split files, invalid `--records-per-shard`, or task directory mismatch.
- Validation failures: schema mismatch, missing checksums, checksum mismatch, or decode failures.
- Sync failures: unresolved package bucket/prefix, invalid `--s3-uri`, missing boto3, or AWS auth/profile issues.

## References

- `references/dataset-packaging-runbook.md`: End-to-end command templates and failure playbook for packaging and sync.
