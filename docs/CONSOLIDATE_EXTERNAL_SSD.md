# Consolidate EXTERNAL folders on SSD

Script **`scripts/consolidate_external_on_ssd.sh`** parses the SSD volume and moves all `*EXTERNAL*` dirs/files into their non-EXTERNAL counterpart names. When a counterpart already exists, it **merges** with `rsync --ignore-existing` (no overwrite = dedupe by path), then removes the EXTERNAL source.

## Usage

```bash
# Dry-run only (default) — shows what would happen
./scripts/consolidate_external_on_ssd.sh

# Apply changes (run from repo with SSD mounted)
./scripts/consolidate_external_on_ssd.sh --execute

# Custom volume path
./scripts/consolidate_external_on_ssd.sh "/Volumes/YourVolume" --execute
```

## Behavior

| Case | Action |
|------|--------|
| No counterpart exists | `mv EXTERNAL -> counterpart` |
| Counterpart exists (dir) | `rsync -a --ignore-existing EXTERNAL/ counterpart/`, then `rm -rf EXTERNAL` |
| File `EXTERNAL_BUILD_REPORT.md` | `mv` to `BUILD_REPORT.md` if it doesn’t exist |

**Dedupe:** Merges use `--ignore-existing` so files already in the counterpart are not overwritten; only new files are added. No content-based dedupe (e.g. same file different path) is done.

## Safety

- Default is **dry-run**; nothing is changed until you pass `--execute`.
- Run in **tmux** if the volume is large (Dev/KmiDi merges can take a while).
- Ensure the SSD is **mounted** and you have **write** permission.
