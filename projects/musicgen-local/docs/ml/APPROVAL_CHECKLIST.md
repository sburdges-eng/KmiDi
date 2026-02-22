# Approval Checklist

Last Updated: 2026-02-13
Purpose: explicit signoff record before enabling training on imported assets.

## Global Gate
- Current state: `ALLOW_TRAINING=YES (approved)`
- Gate owner: seanburdges
- Gate decision date: 2026-02-13
- Notes: User requested training enablement.

## Source Signoff Matrix

| Source Root | Asset Type | Evidence Doc | Owner | Approval Date | Allowed Uses | Commercial Use | Training Allowed | Notes |
|---|---|---|---|---|---|---|---|---|
| `/Volumes/KmiDi-external/DatasetsEXTERNAL/midi` | MIDI dataset | `/Volumes/KmiDi-external/musicgen-local/docs/ml/LICENSE_EVIDENCE_INDEX.md` | seanburdges | 2026-02-13 | training, eval, internal-only | YES | YES | Enabled per user request. |
| `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI` | MIDI examples | `/Volumes/KmiDi-external/musicgen-local/docs/ml/LICENSE_EVIDENCE_INDEX.md` | seanburdges | 2026-02-13 | training, eval, internal-only | YES | YES | Enabled per user request. |
| `/Volumes/KmiDi-external/_sortedEXTERNAL` | Model artifacts (`.pt`, `.onnx`) | `/Volumes/KmiDi-external/musicgen-local/docs/ml/LICENSE_EVIDENCE_INDEX.md` | seanburdges | 2026-02-13 | training, eval, internal-only | YES | YES | Enabled per user request. |
| `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi` | Internal model artifacts | `/Volumes/KmiDi-external/musicgen-local/docs/ml/LICENSE_EVIDENCE_INDEX.md` | seanburdges | 2026-02-13 | training, eval, internal-only | YES | YES | Enabled per user request. |

## Decision Rules
1. `Training Allowed` must be `YES` per source before any training job consumes that source.
2. `Commercial Use` must be `YES` for product-facing model training.
3. If any source is `NO` or blank, training manifests must exclude it.
4. Update `DATA_PROVENANCE.md` gate state after each approved change.

## Change Log
- 2026-02-13: Checklist initialized.
- 2026-02-13: All sources approved and gate authorized by user request.
