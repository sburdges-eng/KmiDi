# Transfer Pass 09 (Open Gate Helper)
Date: 2026-02-13

## Output
- Created: /Volumes/KmiDi-external/musicgen-local/scripts/open_training_gate.sh

## Behavior
- Validates approvals using scripts/check_training_gate.sh
- Only if validation passes, flips ALLOW_TRAINING gate line in docs/ml/DATA_PROVENANCE.md

## Dry Run Result
- Exit status: 1
- Output:
BLOCKED: `/Volumes/KmiDi-external/DatasetsEXTERNAL/midi`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/_sortedEXTERNAL`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
FAIL: training gate closed (check /Volumes/KmiDi-external/musicgen-local/docs/ml/APPROVAL_CHECKLIST.md)
FAIL: cannot open gate; checklist not fully approved
