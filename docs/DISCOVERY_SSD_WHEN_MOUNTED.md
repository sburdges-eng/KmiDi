# Discovery: External SSD (KmiDi-external) — run when mounted

**Date:** 2026-03-10  
**Plan:** KmiDi folders discovery — local drive scan

## Status

- **Volume `/Volumes/KmiDi-external`:** Not mounted at time of discovery (ls/find returned "No such file or directory"). Only `/Volumes/Macintosh HD` was present.
- Per [docs/SSD_WORKDIR_STRUCTURE.md](docs/SSD_WORKDIR_STRUCTURE.md), the external SSD may be named **KmiDi-external** or **Sean's SSD**. When connected, set `KMIDI_DATA_ROOT` to its mount path.

## Commands to run after connecting the SSD

Run from repo root or any directory. Prefer read-only discovery; do not index COLD_STORAGE per ARCHIVE LAW.

```bash
# 1. Confirm volume is mounted (use KmiDi-external or "Sean's SSD" as appropriate)
ls -la "/Volumes/KmiDi-external"
# or
ls -la "/Volumes/Sean's SSD"

# 2. Find directories with kmidi/KmiDi in the name (limit depth to avoid long scan)
VOL=/Volumes/KmiDi-external   # or VOL="/Volumes/Sean's SSD"
find "$VOL" -maxdepth 5 -type d \( -iname "*kmidi*" -o -iname "*KmiDi*" \) 2>/dev/null

# 3. List MASTER_VAULTEXTERNAL and musicgen-local if present (referenced in projects/musicgen-local/ops/runs)
ls -la "$VOL/KmiDi_MASTER_VAULTEXTERNAL" 2>/dev/null
ls -la "$VOL/musicgen-local" 2>/dev/null
```

## Expected locations (from repo references)

- [docs/SSD_WORKDIR_STRUCTURE.md](docs/SSD_WORKDIR_STRUCTURE.md): `$KMIDI_DATA_ROOT` = `/Volumes/KmiDi-external` or `/Volumes/Sean's SSD`; under it: Datasets/, build/, Models/.
- [projects/musicgen-local/ops/runs](projects/musicgen-local/ops/runs): paths under `/Volumes/KmiDi-external/`:
  - `KmiDi_MASTER_VAULTEXTERNAL/KmiDi/projects/musicgen-local/...`
  - `musicgen-local/scripts/build_training_manifests.sh`
  - `musicgen-local/ml/data/manifests/active/...`

---

## Discovery results (2026-03-10 — SSD mounted)

**Volume:** `/Volumes/Sean's SSD` (KmiDi-external was not present; Sean's SSD is the mounted external).

### KmiDi_MASTER_VAULTEXTERNAL vs KmiDi_MASTER_VAULT

- **KmiDi_MASTER_VAULTEXTERNAL:** Not found on this volume.
- **KmiDi_MASTER_VAULT:** Present at `/Volumes/Sean's SSD/KmiDi_MASTER_VAULT`. Contains `KmiDi/` with KmiDi_PROJECT, KmiDi_BACKUP, KmiDi_FINAL, KmiDi, kmidi_gui, music_brain, KmiDi_TRAINING, ML_TRAINED_MODELS, _ARCHIVE_AUDIT, .git/worktrees/kmidi-musicgen-push, etc. Likely the same content the repo referred to as "KmiDi_MASTER_VAULTEXTERNAL" under a different name.

### musicgen-local

- **Present** at `/Volumes/Sean's SSD/musicgen-local`. Top-level contents: README.md, apps/, docs/, infra/, libs/, ml/, ops/, schemas/, scripts/, services/, tests/.

### *kmidi* / *KmiDi* directories (summary, depth ≤ 5)

Find returned 170+ matching directories. Key top-level or high-value locations:

| Path on Sean's SSD | Description |
|--------------------|-------------|
| **KmiDi_MASTER_VAULT/** | Master vault (see above). |
| **KmiDi/** | Large tree: KmiDi-1, RECOVERY_OPS, Desktop, Library, Downloads, KmiDi-Backup-20260108_224837, etc. |
| **Dev/KmiDi** | Clone/copy of repo layout (KmiDi_PROJECT, KmiDi_FINAL, music_brain, KmiDi_TRAINING, include/kmidi, etc.). |
| **Dev/KmiDi MIDI Companion** | KmiDi_CANON, KmiDi, CANONICAL_REBUILD/KmiDi_MASTER_VAULT, ML_TRAINED_MODELS, FINAL_KMIDI, etc. |
| **Dev/KmiDi_recovery_20260218-0329** | Recovery snapshot with same structure as Dev/KmiDi. |
| **Dev/KmiDi-xcode** | Xcode project (KmiDi, KmiDiTests, KmiDiUITests). |
| **Dev/GH_REPOS/KmiDi**, **GH_REPOS/KmiDi-Companion-dev**, **GH_REPOS/KmiDi-MIDI-Companion** | GitHub repo clones. |
| **Datasets/by_source/kmidi** | midi_companion/FINAL_KMIDI, consolidated/from_KmiDi, from_KmiDi_MASTER_VAULTEXTERNAL. |
| **Datasets/_FORENSIC_READONLY_KMIDI** | Forensic read-only dataset area. |
| **CLEANUP_RECOVERY_20260225-032025/backups/KmiDi**, **backups/KmiDi** | Backup copies. |
| **COLD_STORAGE/recovered_intelligence/My_Mac_KmiDi_Companion** | CANONICAL_REBUILD/KmiDi, KmiDi_MASTER_VAULT. |
| **backup/xcode/KmiDi**, **backup/KmiDi**, **backup/workspace-scaffold/apps/kmidi** | Backups of xcode clone, repo, and workspace-scaffold app. |

Use `find "/Volumes/Sean's SSD" -maxdepth 5 -type d \( -iname "*kmidi*" -o -iname "*KmiDi*" \) 2>/dev/null` to regenerate the full list. Prefer read-only discovery; no indexing of COLD_STORAGE per ARCHIVE LAW.
