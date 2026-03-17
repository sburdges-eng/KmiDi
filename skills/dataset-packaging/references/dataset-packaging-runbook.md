# Dataset Packaging Runbook

## Pipeline Summary

Run the packaging pipeline in this order:
1. Build training records from transcript JSONL.
2. Split records deterministically by `sessionId`.
3. Package split files into shard artifacts plus manifests/checksums.
4. Validate package integrity.
5. Sync package to S3.

## Shared Handoff State Format

Use this exact JSON shape when handing state to other skills:

```json
{
  "package-id": "",
  "runId": "",
  "instanceId": "",
  "sessionName": "",
  "outputS3Uri": ""
}
```

After packaging, set `package-id` and leave other fields as empty strings.

## Step 1: Build Training Records

Generate task records from transcript JSONL:

```bash
python3 scripts/build_training_records.py \
  --transcript-jsonl /Volumes/<external-ssd>/vault/transcripts.jsonl
```

Default outputs:
- `training/records/intent_router_records.jsonl`
- `training/records/axis_proposer_records.jsonl`
- `training/records/build_summary.json`

Notes:
- Sessions with harness violations are rejected.
- `axis_proposer` records are only created for `VALID` intent rows with non-leaky axis features.

## Step 2: Split Records

Split both tasks using deterministic hash routing:

```bash
python3 scripts/split_dataset.py
```

Default outputs:
- `training/splits/intent_router/train.jsonl`
- `training/splits/intent_router/val.jsonl`
- `training/splits/intent_router/test.jsonl`
- `training/splits/axis_proposer/train.jsonl`
- `training/splits/axis_proposer/val.jsonl`
- `training/splits/axis_proposer/test.jsonl`
- `training/splits/split_summary.json`

Keep default seed and ratios unless reproducibility requirements change.

## Step 3: Package Shards

Dry-run destination resolution first:

```bash
python3 scripts/package_dataset.py --package-id <package-id> --dry-run
```

Then write package artifacts:

```bash
python3 scripts/package_dataset.py --package-id <package-id>
```

Expected package layout:
- `PACKAGES/<package-id>/intent_router/manifest.json`
- `PACKAGES/<package-id>/intent_router/checksums.txt`
- `PACKAGES/<package-id>/intent_router/<split>-NNNN.jsonl.zst`
- `PACKAGES/<package-id>/axis_proposer/manifest.json`
- `PACKAGES/<package-id>/axis_proposer/checksums.txt`
- `PACKAGES/<package-id>/axis_proposer/<split>-NNNN.jsonl.zst`
- `PACKAGES/<package-id>/package_summary.json`

Dry-run reads destination defaults from `config/run_contract.yaml`:
- `s3.packageBucket`
- `s3.packagePrefix`

## Step 4: Validate Package

Validate schema, checksums, shard hashes, and record counts:

```bash
python3 scripts/validate_package.py --package-dir PACKAGES/<package-id>
```

Validation must pass before S3 sync.

## Step 5: Sync to S3

Dry-run sync first:

```bash
python3 scripts/sync_package_to_s3.py \
  --package-dir PACKAGES/<package-id> \
  --dry-run
```

Run actual sync:

```bash
python3 scripts/sync_package_to_s3.py \
  --package-dir PACKAGES/<package-id>
```

Behavior notes:
- Sync plans uploads from task manifests plus listed shards.
- Existing objects with matching SHA256 are skipped.
- `--resume` is a no-op compatibility flag; rerunning sync is already safe.

## Failure Playbook

Missing transcript input:
- Confirm `--transcript-jsonl` path exists and is readable.

Split errors:
- Verify `training/records/*_records.jsonl` files were generated.

Packaging errors:
- Ensure `training/splits/<task>/<split>.jsonl` exists for both tasks.
- Ensure `--records-per-shard` is greater than zero.

Validation failures:
- For checksum mismatch, regenerate package from split inputs.
- For schema mismatch, inspect `config/dataset_manifest_schema.json` and task manifests.
- For decode failures, treat shard payload as corrupt and repackage.

Sync failures:
- Provide explicit `--s3-uri` or fill `s3.packageBucket` in `config/run_contract.yaml`.
- Ensure AWS credentials/profile/region are valid for the package bucket.
- Install `boto3` if missing in local environment.

## Cloud training optimization (sharded data)

Training data is sharded so that multi-node or multi-GPU runs can load only a subset of shards per worker, reducing download and memory.

### Sharding for cloud

- **Records per shard**: When packaging, use `--records-per-shard` so the number of **train** shards is at least the number of workers (e.g. 500–2000 records per shard). Example: 50k train records with `--records-per-shard 2000` → 25 train shards, enough for up to 25-way distribution.
- **Split layout**: Only the **train** split is shard-distributed. Val and test are loaded in full on every rank so evaluation is consistent.

### Distributed run (S3)

Set `RANK` and `WORLD_SIZE` (or pass `--rank` and `--world-size`). The AWS training entrypoint will:

- Download only this rank’s train shards from S3 (round-robin: rank `r` gets shards `r`, `r+world_size`, …).
- Load only those train shards; val/test shards are downloaded and loaded in full on every rank.

Example (2 workers):

```bash
# Worker 0
RANK=0 WORLD_SIZE=2 python scripts/aws_train_entrypoint.py --package-s3-uri s3://bucket/prefix ...

# Worker 1
RANK=1 WORLD_SIZE=2 python scripts/aws_train_entrypoint.py --package-s3-uri s3://bucket/prefix ...
```

### Pre-cached package (local)

With a full package already on disk (`tools/precache_dataset.sh` or `scripts/sync_package_to_s3.py` used to download, or a shared volume):

- Use `--package-local-dir /path/to/PACKAGES/<package-id>`.
- All shards must be present. Each rank still only **loads** its share of train shards (filtered by `RANK`/`WORLD_SIZE`), so memory and read I/O scale with world size.
