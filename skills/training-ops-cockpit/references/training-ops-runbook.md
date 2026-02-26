# Training Ops Runbook

## Launch Prerequisites

Validate these before launching:
- `config/run_contract.yaml` has non-empty `s3.packageBucket`, `s3.runBucket`, and `aws.region`.
- Dataset package exists under `s3://<bucket>/training/packages/<package-id>`.
- IAM instance profile has access for EC2 launch plus read/write to package/output buckets.
- Local environment can execute `./scripts/security_compliance_preflight.sh`.

## Shared Handoff State Format

Use this exact JSON shape for cross-skill handoff:

```json
{
  "package-id": "",
  "runId": "",
  "instanceId": "",
  "sessionName": "",
  "outputS3Uri": ""
}
```

Set `package-id` before launch, then populate runtime keys from launch output.

## Preflight and Dry-Run

Run preflight directly:

```bash
./scripts/security_compliance_preflight.sh
```

Run launch dry-run first:

```bash
python3 scripts/launch_aws_train.py \
  --package-id <pkg-id> \
  --ami-id <ami-id> \
  --subnet-id <subnet-id> \
  --security-group-id <sg-id> \
  --iam-instance-profile <instance-profile> \
  --dry-run
```

Inspect dry-run JSON for:
- `resolved.package.uri`
- `resolved.output.uri`
- `resolved.runner.uri`
- `estimatedCostUsd`
- `budgetCapUsd`
- `sessionName`

If any resolved URI is not under expected prefixes, stop and fix contract/arguments.

## Launch and Observe

Launch after dry-run succeeds:

```bash
python3 scripts/launch_aws_train.py \
  --package-id <pkg-id> \
  --ami-id <ami-id> \
  --subnet-id <subnet-id> \
  --security-group-id <sg-id> \
  --iam-instance-profile <instance-profile>
```

Update shared handoff state keys from launch output:
- `runId`
- `instanceId`
- `sessionName`
- `outputS3Uri`

Attach to remote tmux:

```bash
tools/open_train_terminal.sh --instance-id <instance-id> --session-name <session-name>
```

Optional local workbench:

```bash
tools/open_workbench_wezterm.sh --instance-id <instance-id> --run-id <run-id> --session-name <session-name>
```

Expected panes in `tools/tmux_train_layout.sh`:
- `TRAIN_LOG`
- `GPU`
- `IO/DISK`
- `CACHE`
- `CHECKPOINTS`
- `WATCHDOG`
- `COST`

## Artifact Retrieval

Fetch student artifacts (default safe mode):

```bash
python3 scripts/fetch_aws_artifacts.py \
  --run-id <run-id> \
  --output-dir artifacts/download
```

Dry-run fetch resolution:

```bash
python3 scripts/fetch_aws_artifacts.py --run-id <run-id> --dry-run
```

Fetch teacher artifacts only with break-glass:

```bash
python3 scripts/fetch_aws_artifacts.py \
  --run-id <run-id> \
  --scope teacher \
  --break-glass \
  --profile <break-glass-profile> \
  --break-glass-role-arn <role-arn>
```

Teacher fetch enforcement rules come from `security.teacherFetch` in `config/run_contract.yaml`.

## Failure Playbook

Preflight fails:
- Re-run `./scripts/security_compliance_preflight.sh` and fix reported check failures.
- Common blocker: empty run-contract required fields.

Launch fails on cost/budget:
- Keep `--budget-cap-usd <= 100`.
- Reduce `--max-runtime-hours`, use a cheaper known instance type, or pass `--hourly-usd`.

Launch fails on URI prefix validation:
- Package URI must remain under `training/packages`.
- Output URI must remain under `training/runs`.
- Runner URI must stay under output URI prefix and same bucket.

Cache stage fails:
- Inspect dataset cache mode/root from launch dry-run.
- Verify `tools/precache_dataset.sh` fallback behavior and package path accessibility.

Teacher fetch denied:
- Provide `--break-glass`.
- Use explicit `--profile` that differs from default profile.
- Provide `--break-glass-role-arn` matching `security.teacherFetch.requiredRoleArn`.
