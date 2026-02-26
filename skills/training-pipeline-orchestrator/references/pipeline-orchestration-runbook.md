# Pipeline Orchestration Runbook

## Scope

Use this runbook when a task spans packaging and training operations in one flow:
- Transcript JSONL -> records -> splits -> package -> validate -> sync.
- Package ID -> launch dry-run -> launch -> monitor -> artifact fetch.

## Required Inputs

Collect before execution:
- Transcript path for packaging, or existing `package-id`.
- AWS launch values: `ami-id`, `subnet-id`, `security-group-id`, `iam-instance-profile`.
- Optional AWS profile/region overrides when defaults in `config/run_contract.yaml` are empty.

## Shared Handoff State Format

Use this exact JSON shape across every phase handoff:

```json
{
  "package-id": "",
  "runId": "",
  "instanceId": "",
  "sessionName": "",
  "outputS3Uri": ""
}
```

Populate keys only from command output and keep unknown keys as empty strings.

## Phase-by-Phase Commands

### 1) Build Records

```bash
python3 scripts/build_training_records.py \
  --transcript-jsonl /Volumes/<external-ssd>/vault/transcripts.jsonl
```

### 2) Split Records

```bash
python3 scripts/split_dataset.py
```

### 3) Package and Validate

```bash
python3 scripts/package_dataset.py --package-id <package-id> --dry-run
python3 scripts/package_dataset.py --package-id <package-id>
python3 scripts/validate_package.py --package-dir PACKAGES/<package-id>
```

### 4) Sync Package

```bash
python3 scripts/sync_package_to_s3.py --package-dir PACKAGES/<package-id> --dry-run
python3 scripts/sync_package_to_s3.py --package-dir PACKAGES/<package-id>
```

### 5) Training Preflight and Dry-Run

```bash
./scripts/security_compliance_preflight.sh
python3 scripts/launch_aws_train.py \
  --package-id <package-id> \
  --ami-id <ami-id> \
  --subnet-id <subnet-id> \
  --security-group-id <sg-id> \
  --iam-instance-profile <instance-profile> \
  --dry-run
```

### 6) Launch

```bash
python3 scripts/launch_aws_train.py \
  --package-id <package-id> \
  --ami-id <ami-id> \
  --subnet-id <subnet-id> \
  --security-group-id <sg-id> \
  --iam-instance-profile <instance-profile>
```

Capture from launch output:
- `runId`
- `instanceId`
- `sessionName`
- `outputS3Uri`

### 7) Monitor and Fetch

Attach:

```bash
tools/open_train_terminal.sh --instance-id <instance-id> --session-name <session-name>
```

Fetch student artifacts:

```bash
python3 scripts/fetch_aws_artifacts.py --run-id <run-id> --output-dir artifacts/download
```

Teacher artifacts require break-glass:

```bash
python3 scripts/fetch_aws_artifacts.py \
  --run-id <run-id> \
  --scope teacher \
  --break-glass \
  --profile <break-glass-profile> \
  --break-glass-role-arn <role-arn>
```

## Hard Gates

Do not continue to next phase unless current phase gate passes:
- Build gate: record files exist.
- Split gate: all six split files exist.
- Package gate: validator returns success.
- Sync gate: dry-run destination matches expected package bucket/prefix.
- Launch gate: dry-run resolves valid URIs and acceptable cost.
- Fetch gate: student fetch default, teacher fetch only with policy-compliant break-glass flags.

## Handoff Contract Between Phases

Preserve the shared handoff JSON object across phases and update fields in place.
Reuse exact values from script output. Do not infer replacements.

## Failure Matrix

Data prep fails:
- Fix transcript path, axis extraction assumptions, or record generation issues.
- Rerun from phase 1.

Package validate fails:
- Regenerate package from splits.
- Re-run validation before sync.

Sync fails:
- Resolve bucket/prefix/profile/region in `config/run_contract.yaml` or pass explicit CLI overrides.
- Re-run sync; incremental hashing will skip unchanged objects.

Launch dry-run fails:
- Fix missing launch arguments, run-contract placeholders, prefix constraints, or cost settings.
- Re-run dry-run before launching.

Runtime monitoring/fetch fails:
- Verify `instanceId` and `sessionName` for attach.
- Verify `runId` and break-glass policy for fetch.
