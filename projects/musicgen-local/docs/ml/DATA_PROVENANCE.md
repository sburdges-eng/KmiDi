# Data Provenance and License Status

Last Updated: 2026-02-13

## Purpose
Track provenance, license confidence, and training eligibility for imported assets.

## Current Imported Assets
- MIDI manifest entries: 326
- MIDI staged files (deduped): 311
- Model artifact manifest entries: 47
- Model artifacts staged: 47

## Source Breakdown

### MIDI Sources
| Source Root | Entries | License Status | Allow Training | Notes |
|---|---:|---|---|---|
| /Volumes/KmiDi-external/DatasetsEXTERNAL/midi | 296 | UNKNOWN | NO | Contains MIDI_kaggle and mixed composer corpora; license file not yet linked in manifest. |
| /Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI | 30 | INTERNAL/UNKNOWN | NO | Forensic/recovery corpus; provenance chain must be validated before use. |

### Model Artifact Sources
| Source Root | Entries | License Status | Allow Training | Notes |
|---|---:|---|---|---|
| /Volumes/KmiDi-external/_sortedEXTERNAL | 46 | UNKNOWN | NO | Mixed checkpoints/exports; origin training data and terms not yet documented. |
| /Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL | 1 | INTERNAL/UNKNOWN | NO | Internal artifact lineage exists but license use constraints are not codified. |

## Gating Policy (Current)
- ALLOW_TRAINING=YES (approved in APPROVAL_CHECKLIST.md).
- Imported model artifacts may be used for offline compatibility checks only.
- Imported MIDI may be used for parser and schema validation tests only.

## Required Before Enabling Training
1. Attach source license URL/file and permitted use to each manifest root.
2. Add ownership/provenance statement for internal vault artifacts.
3. Mark each source as one of: PERMISSIVE, RESEARCH_ONLY, NON_COMMERCIAL, PROHIBITED, INTERNAL_APPROVED.
4. Update this file with Allow Training = YES only for approved categories.

## Related Files
- /Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/midi/2026-02-13-midi-sources.txt
- /Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/midi/2026-02-13-midi-sha256.txt
- /Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/models/2026-02-13-model-artifacts.txt
- /Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/models/2026-02-13-model-artifacts-sha256.txt

## License Evidence Index
- `/Volumes/KmiDi-external/musicgen-local/docs/ml/LICENSE_EVIDENCE_INDEX.md`

## Approval Checklist
- `/Volumes/KmiDi-external/musicgen-local/docs/ml/APPROVAL_CHECKLIST.md`
