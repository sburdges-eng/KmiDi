# License Evidence Index

Last Updated: 2026-02-13
Purpose: track concrete license/readme evidence for imported source roots.

## Coverage Summary
| Source Root | Evidence Found | Status | Recommended Classification |
|---|---|---|---|
| `/Volumes/KmiDi-external/DatasetsEXTERNAL/midi` | No local `LICENSE*` or `README*` found under this tree | Missing evidence | `UNKNOWN` |
| `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI` | Multiple license/readme files found | Partial evidence | `INTERNAL/UNKNOWN` |
| `/Volumes/KmiDi-external/_sortedEXTERNAL` | Readme files and one nested license found | Partial evidence | `UNKNOWN` |
| `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi` | Root `LICENSE` and project readmes found | Evidence present (internal scope) | `INTERNAL_APPROVED` pending owner signoff |

## Evidence Paths (Curated)

### `/Volumes/KmiDi-external/DatasetsEXTERNAL/midi`
- No evidence files found by scan:
  - `find /Volumes/KmiDi-external/DatasetsEXTERNAL/midi -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'README*' \)`

### `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI`
- `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain/LICENSE`
- `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain/README.md`
- `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain copy/LICENSE`
- `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI/daiw_complete/README.md`
- `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI/CODE/DOWLOADS/idaw_v1.0.00/README.md`

### `/Volumes/KmiDi-external/_sortedEXTERNAL`
- `/Volumes/KmiDi-external/_sortedEXTERNAL/Docs/My Mac/Desktop/KmiDi-remote/README.md`
- `/Volumes/KmiDi-external/_sortedEXTERNAL/Docs/My Mac/Desktop/KmiDi-remote/datasets/README.md`
- `/Volumes/KmiDi-external/_sortedEXTERNAL/Docs/My Mac/Desktop/KmiDi-remote/models/README.md`
- `/Volumes/KmiDi-external/_sortedEXTERNAL/Docs/My Mac/Desktop/KmiDi-remote/training/README.md`
- `/Volumes/KmiDi-external/_sortedEXTERNAL/Docs/My Mac/Desktop/KmiDi-remote/legacy/super-spork/LICENSE`

### `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi`
- `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/LICENSE`
- `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/datasets/README.md`
- `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/training/README.md`
- `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/ML Kelly Training/backup/configs/README.md`

## Decision Notes
- This index is evidence-only and not legal approval.
- Until explicit signoff, keep `ALLOW_TRAINING=NO` for imported data/artifacts.
- If owner confirms internal rights for vault assets, update classification to `INTERNAL_APPROVED` and record approver/date.

## Approval Workflow
- `/Volumes/KmiDi-external/musicgen-local/docs/ml/APPROVAL_CHECKLIST.md`
