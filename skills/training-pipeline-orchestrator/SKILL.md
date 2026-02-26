---
name: training-pipeline-orchestrator
description: Orchestrate the full Kelly listening-contract training pipeline from transcript ingestion through AWS launch and artifact retrieval with explicit gate checks between phases. Use when a task spans both dataset packaging scripts (`build_training_records.py`, `split_dataset.py`, `package_dataset.py`, `validate_package.py`, `sync_package_to_s3.py`) and training ops scripts (`launch_aws_train.py`, `fetch_aws_artifacts.py`), or when users ask for an end-to-end runbook, execution plan, or failure triage across those phases.
---

# Training Pipeline Orchestrator

Use this skill when the request is cross-phase and needs one coordinated flow.

Read `references/pipeline-orchestration-runbook.md` first for command templates and phase gates.

## Workflow Decision Tree

- End-to-end run from transcript JSONL:
Execute phases 1 through 7 in sequence and stop on any failed gate.
- Resume from known package:
Start at phase 4 (launch preflight and dry-run) using existing `package-id`.
- Resume from launched run:
Start at phase 6 for monitoring and artifact fetch with known `runId`.
- Debugging across phases:
Classify issue by phase boundary first, then apply phase-specific recovery.

## Phase 1) Build Records

Run:
- `python3 scripts/build_training_records.py --transcript-jsonl <path/to/transcripts.jsonl>`

Gate:
- Require `training/records/intent_router_records.jsonl`.
- Require `training/records/axis_proposer_records.jsonl`.
- Require `training/records/build_summary.json`.

## Phase 2) Split Deterministically

Run:
- `python3 scripts/split_dataset.py`

Gate:
- Require `training/splits/intent_router/{train,val,test}.jsonl`.
- Require `training/splits/axis_proposer/{train,val,test}.jsonl`.
- Require `training/splits/split_summary.json`.

## Phase 3) Package, Validate, and Sync

Run package dry-run first:
- `python3 scripts/package_dataset.py --package-id <package-id> --dry-run`

Run package write:
- `python3 scripts/package_dataset.py --package-id <package-id>`

Validate package:
- `python3 scripts/validate_package.py --package-dir PACKAGES/<package-id>`

Run sync dry-run first:
- `python3 scripts/sync_package_to_s3.py --package-dir PACKAGES/<package-id> --dry-run`

Run sync:
- `python3 scripts/sync_package_to_s3.py --package-dir PACKAGES/<package-id>`

Gate:
- Do not launch training unless package validation passes and sync destination is correct.

## Phase 4) Preflight and Launch Dry-Run

Run:
- `./scripts/security_compliance_preflight.sh`
- `python3 scripts/launch_aws_train.py --package-id <package-id> --ami-id <ami-id> --subnet-id <subnet-id> --security-group-id <sg-id> --iam-instance-profile <instance-profile> --dry-run`

Gate:
- Require dry-run output with valid `resolved.package.uri`, `resolved.output.uri`, and `resolved.runner.uri`.
- Require estimated cost under configured budget and hard cap.
- Do not bypass preflight unless explicitly requested.

## Phase 5) Launch and Capture Runtime IDs

Run:
- `python3 scripts/launch_aws_train.py --package-id <package-id> --ami-id <ami-id> --subnet-id <subnet-id> --security-group-id <sg-id> --iam-instance-profile <instance-profile>`

Capture and preserve:
- `runId`
- `instanceId`
- `sessionName`
- `outputS3Uri`
Update shared handoff state using this exact JSON shape:

```json
{
  "package-id": "",
  "runId": "",
  "instanceId": "",
  "sessionName": "",
  "outputS3Uri": ""
}
```

Set known keys from command output and keep unknown keys as empty strings.

## Phase 6) Monitor and Fetch Artifacts

Monitor:
- `tools/open_train_terminal.sh --instance-id <instance-id> --session-name <session-name>`

Fetch student artifacts by default:
- `python3 scripts/fetch_aws_artifacts.py --run-id <run-id> --output-dir artifacts/download`

Fetch teacher artifacts only via break-glass:
- Require `--scope teacher` or `--scope all`.
- Require `--break-glass`.
- Require explicit break-glass `--profile`.
- Require `--break-glass-role-arn`.

## Phase 7) Cross-Phase Failure Triage

Apply this order:
- Check earliest failing phase first; do not debug downstream phases until upstream gates pass.
- For data/package failures, rerun packaging phases and revalidate before re-launch.
- For launch failures, rerun launch dry-run before any new launch attempt.
- For fetch failures, confirm run ID and break-glass policy before changing commands.

## References

- `references/pipeline-orchestration-runbook.md`: Full orchestration command set, handoff contract, and failure matrix.
