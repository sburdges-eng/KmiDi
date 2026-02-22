# Transfer Pass 07 (Training Gate Script)
Date: 2026-02-13

## Output
- Created: /Volumes/KmiDi-external/musicgen-local/scripts/check_training_gate.sh

## Behavior
- Parses source signoff matrix from docs/ml/APPROVAL_CHECKLIST.md
- Blocks unless each source row has:
  - Owner non-empty
  - Approval Date non-empty
  - Allowed Uses non-empty
  - Commercial Use = YES
  - Training Allowed = YES

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
