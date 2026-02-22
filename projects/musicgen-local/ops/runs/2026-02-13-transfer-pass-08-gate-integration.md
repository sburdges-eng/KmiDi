# Transfer Pass 08 (Gate Integration in Train Scripts)
Date: 2026-02-13

## Updated
- /Volumes/KmiDi-external/musicgen-local/scripts/train-symbolic.sh
- /Volumes/KmiDi-external/musicgen-local/scripts/train-jepa.sh

## Behavior
- Both scripts now execute scripts/check_training_gate.sh before training entrypoint logic.

## Dry Run Results
- train-symbolic.sh exit: 1
- train-jepa.sh exit: 1

### train-symbolic.sh output
BLOCKED: `/Volumes/KmiDi-external/DatasetsEXTERNAL/midi`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/_sortedEXTERNAL`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
FAIL: training gate closed (check /Volumes/KmiDi-external/musicgen-local/docs/ml/APPROVAL_CHECKLIST.md)

### train-jepa.sh output
BLOCKED: `/Volumes/KmiDi-external/DatasetsEXTERNAL/midi`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/_sortedEXTERNAL`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
BLOCKED: `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi`
  owner= approval_date= allowed_uses=eval-only / training / internal-only commercial=YES/NO training=YES/NO
FAIL: training gate closed (check /Volumes/KmiDi-external/musicgen-local/docs/ml/APPROVAL_CHECKLIST.md)
