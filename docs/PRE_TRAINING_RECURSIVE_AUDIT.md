# PRE-TRAINING RECURSIVE HARDENING AUDIT

Date: 2026-02-19  
Scope: `/home/runner/work/KmiDi/KmiDi` (staging sandbox)  
Canonical comparison reference: `sburdges-eng/KMIDI`

## Executive summary

This repository is **not ready for training freeze**.

The highest-risk blockers are:
1. **Non-deterministic / network-dependent build paths** in CMake (`FetchContent` from GitHub at configure time).
2. **No lockfile-based dependency pinning** for Python/npm/Cargo in the checked tree.
3. **Silent schema fallback behavior** in intent serialization/deserialization (`from_dict` defaulting without provenance).
4. **Insufficient deterministic training controls** (no global seed and deterministic backend settings in training orchestrator).

Canonical cross-reference result: key control files are currently in parity with `KMIDI` (same blob hashes for sampled files), so these are **shared structural risks**, not local regressions.

## Canonical cross-reference (KMIDI)

Compared local vs canonical (`sburdges-eng/KMIDI`) for core enforcement files:

- `pyproject.toml` local hash `400f898...` == canonical SHA `400f898...`
- `music_brain/session/intent_schema.py` local hash `0bbee34...` == canonical SHA `0bbee34...`
- `engine/intent_ir/build.rs` local hash `9976bd9...` == canonical SHA `9976bd9...`
- `python/penta_core/ml/model_registry.py` local hash `38a1ef7...` == canonical SHA `38a1ef7...`

Conclusion: no divergence in those audited controls; hardening should be applied in both staging and canonical if policy is shared.

## Structural risk analysis

### Critical

1. **Build depends on network during configure/build**
   - `CMakeLists.txt:92-99` (`readerwriterqueue` via `FetchContent_Declare`)
   - `CMakeLists.txt:145-152` (`RTNeural` fallback via `FetchContent_Declare`)
   - `CMakeLists.txt:445-452` (`googletest` via `FetchContent_Declare`)
   - Impact: violates offline compile/replay requirement and introduces mutable upstream risk.

2. **Build can proceed with missing FFI artifact**
   - `engine/intent_ir/build.rs:95-99` logs warnings instead of failing hard.
   - Impact: deferred runtime failure and nondeterministic packaging outcomes.

### High

3. **Unsafe C string exposure in Intent IR FFI error API**
   - `engine/intent_ir/src/intent_ir/ffi_exports.rs:234,238,247` returns `str.as_ptr()` as `*const c_char` without guaranteed NUL-termination contract.
   - Impact: UB risk at FFI boundary; undefined reads by C consumers.

4. **Model path resolution escapes base directory**
   - `python/penta_core/ml/model_registry.py:349-355` uses `.resolve()` with relative paths and no root containment check.
   - Impact: implicit trust of manifest path values; weak boundary governance.

### Medium

5. **Serialization defaults hide missing/invalid signal**
   - `music_brain/session/intent_schema.py:436-500` maps missing nested keys to empty/default values.
   - Impact: schema drift is masked; upstream cannot distinguish omitted vs explicit values.

6. **Remote runtime calls hardcoded in UI hook**
   - `src/hooks/useMusicBrain.ts:3,90-100` hardcoded `http://127.0.0.1:8000`.
   - Impact: implicit runtime intelligence boundary outside compile-time guards.

## Determinism analysis

### Violations

1. **No explicit training seed control**
   - `python/penta_core/ml/training_orchestrator.py` has no `seed/manual_seed/deterministic` controls.
   - DataLoader shuffles training data (`training_orchestrator.py:606-610`) without seeded generator.

2. **Dependency drift risk**
   - No checked lockfiles observed for:
     - Python (`poetry.lock` / equivalent): none
     - npm (`package-lock.json`): none
     - Cargo (`Cargo.lock`): none in repo root tree
   - Floating constraints:
     - `pyproject.toml:15-21` (`torch>=2.0`, `librosa>=0.10`, etc.)
     - `engine/intent_ir/Cargo.toml:13-14` (`reqwest = "0.11"`, `tokio = "1"`)

3. **Build-time mutable network sources**
   - CMake `FetchContent` tags/branches are external moving dependencies (especially `RTNeural` on `main` at `CMakeLists.txt:148`).

### Deterministic replay status

Current status: **FAIL**  
Reason: network fetches + floating dependency resolution + unseeded training pipeline.

## Schema drift analysis

1. **Manifest schema exists outside active registry loader location**
   - Schema + rich metadata lives in `KmiDi_FINAL/ml/models/registry.schema.json` and `registry.json`.
   - Active Python registry loader (`python/penta_core/ml/model_registry.py`) loads generic dict data and only validates schema when optional `jsonschema` is present (`model_registry.py:305-315`).
   - Drift risk: governance metadata can exist in manifests but be ignored by runtime model registration.

2. **Intent schema coercion without provenance**
   - `music_brain/session/intent_schema.py` clamps/coerces values but does not emit normalization provenance.
   - Drift risk: training/debug tooling cannot audit whether values are original vs corrected.

## Dependency graph concerns

1. **Build graph includes optional remote acquisitions** (CMake `FetchContent` blocks).
2. **Runtime graph includes hardcoded HTTP boundary** (`useMusicBrain.ts`).
3. **ML backend graph accepts multiple formats/backends** without strict artifact trust policy (`model_registry.py` backend mapping and path resolution).
4. **Tooling dependency gaps in environment validation**
   - Baseline runs failed due absent local tooling (`pytest`, `flake8`, Qt6), indicating missing preflight enforcement.

## Build independence compliance (required checks)

Required condition | Status | Evidence
---|---|---
Compile without network dependency | **FAIL** | `CMakeLists.txt` `FetchContent_*` blocks
Compile without external runtime model calls | **PARTIAL** | local model support exists, but UI path assumes live API endpoint (`useMusicBrain.ts`)
No remote dataset pulls required | **PARTIAL** | no explicit pull in audited runtime path, but training orchestration lacks strict local-artifact gate
Fail fast on missing local artifacts | **FAIL** | `engine/intent_ir/build.rs:95-99` warning-only behavior
Deterministic build replay | **FAIL** | floating dependencies, no lockfiles, mutable network fetches

## Freeze readiness assessment

- Circular dependency risk: **present via implicit runtime fallback boundaries** (Tauri commands may route C++ or Python HTTP dynamically).
- Schema ambiguity: **present** (silent defaulting/coercion without provenance).
- Implicit serialization ordering: **present risk** (dict-based JSON without explicit schema version gating in intent structures).
- Hidden runtime intelligence outside enforcement boundaries: **present** (hardcoded local HTTP API boundary not feature-gated for freeze mode).

## Training-readiness score

**43 / 100**

Scoring basis:
- Architecture boundaries: 62
- Build independence/determinism: 25
- Schema governance: 45
- Dependency governance: 30
- Test enforceability: 55

## Structured fix list

## Required fixes (freeze blockers)

1. **Add offline build gate + vendor mode for CMake fetches**
   - Files: `CMakeLists.txt`
   - Action: introduce `KMIDI_OFFLINE_BUILD` option; when ON, disallow `FetchContent` and require vendored/existing deps.

2. **Fail-fast when KellyFFI artifact is missing**
   - Files: `engine/intent_ir/build.rs`
   - Action: turn warning path into hard failure under freeze/CI profile (feature flag allowed for dev).

3. **Pin dependency resolution**
   - Files: `pyproject.toml`, `engine/intent_ir/Cargo.toml`, root npm project metadata
   - Action: add lockfile strategy and CI enforcement (`--frozen`/`--locked` modes).

4. **Deterministic training controls**
   - Files: `python/penta_core/ml/training_orchestrator.py`
   - Action: add global seed path (Python/Torch/CUDA), deterministic backend switch, seeded DataLoader generator.

5. **FFI error string safety**
   - Files: `engine/intent_ir/src/intent_ir/ffi_exports.rs`
   - Action: return static NUL-terminated C strings; avoid raw Rust `str.as_ptr()` conversion.

## Strong recommendations

1. **Schema provenance annotations for coercion/defaulting**
   - Files: `music_brain/session/intent_schema.py`
   - Action: track per-field normalization/fallback markers in output metadata.

2. **Root-bound model path enforcement**
   - Files: `python/penta_core/ml/model_registry.py`
   - Action: require resolved model paths remain within approved model roots.

3. **Freeze-mode runtime boundary gate**
   - Files: `src/hooks/useMusicBrain.ts`, Tauri config pathing
   - Action: feature-flag external API fallback (`KMIDI_USE_API`) with explicit deny-by-default in freeze builds.

## Optional optimizations

1. Add reproducibility manifest (`build provenance`: compiler/toolchain/dependency hashes).
2. Add schema compatibility tests between `KmiDi_FINAL/ml/models/registry.schema.json` and runtime registry loader.
3. Add adversarial serialization tests for NaN/Inf/path traversal payloads.

## Concrete patch suggestions (minimal diffs)

### Patch A (required): offline CMake gate

File: `CMakeLists.txt`

```diff
+option(KMIDI_OFFLINE_BUILD "Disallow network dependency resolution at configure/build time" ON)
 ...
 include(FetchContent)
+if(KMIDI_OFFLINE_BUILD)
+    message(FATAL_ERROR "KMIDI_OFFLINE_BUILD=ON forbids FetchContent network pulls. Vendor dependencies locally.")
+endif()
 FetchContent_Declare(
   readerwriterqueue
   GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
 ```

### Patch B (required): fail fast for missing FFI artifact

File: `engine/intent_ir/build.rs`

```diff
 if !ffi_lib_found {
-    println!("cargo:warning=Kelly FFI library not found. Build may fail.");
-    println!("cargo:warning=Run CMake build first: cd ../build && cmake .. && make KellyFFI");
-    println!("cargo:warning=Expected locations: {:?}", possible_lib_paths);
+    panic!(
+        "Kelly FFI library not found. Expected one of: {:?}. Run CMake build for KellyFFI first.",
+        possible_lib_paths
+    );
 }
 ```

### Patch C (required): deterministic training seed

File: `python/penta_core/ml/training_orchestrator.py`

```diff
+import random
 ...
+def _set_determinism(seed: int) -> None:
+    random.seed(seed)
+    torch.manual_seed(seed)
+    if torch.cuda.is_available():
+        torch.cuda.manual_seed_all(seed)
+    torch.backends.cudnn.deterministic = True
+    torch.backends.cudnn.benchmark = False
 ...
 def train(self, job: TrainingJob) -> Dict[str, float]:
+    _set_determinism(getattr(self.config, "seed", 42))
     train_loader = self._create_dummy_dataloader("train")
 ```

### Patch D (required): safe FFI error strings

File: `engine/intent_ir/src/intent_ir/ffi_exports.rs`

```diff
+const ERR_SUCCESS: &[u8] = b"Success\0";
+const ERR_UNKNOWN: &[u8] = b"Unknown error\0";
 ...
 pub extern "C" fn intent_ir_get_error_message(error_code: IntentIRErrorCode) -> *const c_char {
-    for (msg, code) in ERROR_MESSAGES {
-        if *code == error_code {
-            return msg.as_ptr() as *const c_char;
-        }
-    }
-    "Unknown error".as_ptr() as *const c_char
+    match error_code {
+        IntentIRErrorCode::Success => ERR_SUCCESS.as_ptr() as *const c_char,
+        _ => ERR_UNKNOWN.as_ptr() as *const c_char,
+    }
 }
 ```

## Recursive second-order impact review

After applying required fixes:

1. Offline gate may break convenience builds -> mitigate with explicit `KMIDI_OFFLINE_BUILD=OFF` for developer mode only.
2. Fail-fast FFI build behavior will increase early CI failures -> desired for freeze quality; add clear remediation text.
3. Deterministic training settings can reduce throughput -> acceptable in freeze/training reproducibility profile.
4. Strict model path/root checks may reject legacy manifests -> add compatibility warning path behind migration flag.

Re-evaluation status after proposed fixes:  
- Circular dependency risk: reducible to low with explicit runtime boundary flags.  
- Schema ambiguity: reducible to medium-low with provenance metadata.  
- Implicit serialization ordering: reducible with schema-version assertions.  
- Hidden runtime intelligence: reducible with freeze-mode API gating.

## Final freeze-readiness verdict

**NOT READY**

Blocking reasons:
- Offline-independent build requirement is currently unmet.
- Deterministic replay requirement is currently unmet.
- Fail-fast local artifact enforcement is currently unmet.
