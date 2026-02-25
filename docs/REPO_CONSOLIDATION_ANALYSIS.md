# KmiDi vs KmiDi-1 Consolidation Analysis

## Repository Roots Examined

- Primary working root: `KmiDi_recovery_20260218-0329`
- Legacy/consolidated artifacts:
  - `KmiDi_FINAL/`
  - `KmiDi_PROJECT/source/`
  - `KmiDi/` (minimal shell present)

## Feature Verification (Priority Set)

| Feature | Status | Evidence |
|---|---|---|
| Spectocloud | Present | `music_brain/visualization/spectocloud.py` |
| Tier1 generators | Restored in primary tree | `music_brain/tier1/midi_generator.py`, `audio_generator.py`, `voice_generator.py` |
| Tier2 LoRA | Present | `music_brain/tier2/lora_finetuner.py` |
| FFI bridge | Present | `src/bridge/kelly_ffi.cpp` |
| Tauri state | Present | `src-tauri/src/state.rs` |

## Consolidation Actions Performed

1. Added missing `tier1` generator/pipeline modules to primary tree.
2. Added root `BUILD.md` with Kelly target naming.
3. Added harmony compatibility path at `music_brain/harmony/deps`.
4. Updated API pipeline to favor full intent flow and audio render integration.
5. Added voice model deployment/runtime components (export + registry + classifier).

## Remaining Consolidation Risks

- `KmiDi/` directory is currently minimal; primary executable code path is the root tree.
- Some historical modules still rely on compatibility wrappers and should be normalized in a future pass.
- Full CI matrix build for C++/Tauri/Python remains recommended.
