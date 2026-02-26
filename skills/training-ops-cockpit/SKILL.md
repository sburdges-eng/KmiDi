---
name: training-ops-cockpit
description: Operate AWS GPU training runs for the Kelly listening-contract pipeline with fail-closed preflight checks, run-contract resolution, tmux cockpit observability, and artifact retrieval controls. Use when work involves launching `scripts/launch_aws_train.py`, validating `config/run_contract.yaml`, attaching or troubleshooting remote tmux training sessions, managing dataset cache and early-stop behavior, or fetching student and teacher artifacts through `scripts/fetch_aws_artifacts.py`.
---

# Training Ops Cockpit

Use this skill to run or troubleshoot the end-to-end training operations flow in this repository.

Read `references/training-ops-runbook.md` first when a task requires exact command templates or failure recovery guidance.

## Workflow Decision Tree

- Launching a new training run:
Run preflight checks, validate contract values, execute launch dry-run, then launch.
- Inspecting active training operations:
Use tmux session details from launch output and inspect TRAIN/GPU/IO/CACHE/CHECKPOINTS/WATCHDOG/COST panes.
- Fetching artifacts:
Fetch student artifacts by default, enforce break-glass policy for teacher artifacts.
- Debugging failures:
Classify as contract/config, permissions/IAM, S3 prefix validation, cache staging, or policy violations, then apply runbook fixes.

## 1) Validate Launch Inputs

Run `./scripts/security_compliance_preflight.sh` unless the user explicitly requests otherwise.
Open `config/run_contract.yaml` and confirm required fields are populated before launch:
- `s3.packageBucket`
- `s3.runBucket`
- `aws.region`
Prefer contract-driven defaults instead of hardcoding launch values where possible.

## 2) Resolve and Dry-Run the Launch

Use `scripts/launch_aws_train.py --dry-run` before starting any new run.
Require these launch arguments unless already provided through automated context:
- `--ami-id`
- `--subnet-id`
- `--security-group-id`
- `--iam-instance-profile`
Treat dry-run JSON as the source of truth for resolved package/output/runner URIs, run ID, session name, and cost estimate.

## 3) Launch and Monitor

Launch without `--dry-run` after validation passes.
Capture these values from launch output and reuse them in follow-on steps:
- `runId`
- `instanceId`
- `sessionName`
- `outputS3Uri`
Persist handoff state in this exact JSON shape:

```json
{
  "package-id": "",
  "runId": "",
  "instanceId": "",
  "sessionName": "",
  "outputS3Uri": ""
}
```

Fill keys only from command output and keep unknown values as empty strings.
Prefer the repository tools for observability:
- `tools/open_train_terminal.sh` for direct remote attach.
- `tools/open_workbench_wezterm.sh` for TRAIN/OPUS/CODEX/SPARK local workbench.
Expect tmux panes to provide TRAIN, GPU, IO/DISK, CACHE, CHECKPOINTS, WATCHDOG, and COST visibility.

## 4) Fetch Artifacts Safely

Default to student-only retrieval:
- `python3 scripts/fetch_aws_artifacts.py --run-id <run-id> --output-dir artifacts/download`
Require explicit break-glass flow for teacher artifact retrieval:
- `--scope teacher` or `--scope all`
- `--break-glass`
- `--profile <break-glass-profile>`
- `--break-glass-role-arn <role-arn>`
If teacher retrieval fails policy checks, inspect `security.teacherFetch` in `config/run_contract.yaml`.

## 5) Troubleshooting Rules

Classify failures before editing code:
- Preflight failures: run the exact preflight command and fix failing checks.
- Prefix validation failures: keep package URIs under `training/packages` and outputs under `training/runs`.
- Budget/cost failures: lower runtime, choose cheaper known instance type, or provide `--hourly-usd`.
- Cache failures: inspect `tools/precache_dataset.sh` mode/root and fallback policy.
- Teacher fetch failures: verify break-glass profile and role ARN match run-contract policy.

## References

- `references/training-ops-runbook.md`: Command templates, guardrail expectations, and recovery playbook for launch/monitor/fetch workflows.
