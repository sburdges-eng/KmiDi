# Next Patch Batch Report — Reconciled Audit

Applied only verified open findings from **docs/CODE_REVIEW_REPORT.md**. No separate "A. VERIFIED OPEN FINDINGS" or "D. NEXT PATCH BATCH" sections exist in the repo; the patch batch was derived from the report’s **D. FINDINGS**, **J. FIX PLAN**, and **K. PATCH ORDER**.

---

## A. FILES CHANGED

**None.**

- **F3 (Spectocloud path traversal):** Marked in the report as *"Already mitigated in current code"*. Confirmed in repo: `music_brain/api.py` uses `_resolve_audio_path_sandbox(payload.midi_file_path)` before `_parse_midi_file` (lines 1330–1332) and `_resolve_audio_path_sandbox(payload.output_path)` for static and animation output (lines 1370–1371, 1388–1389). **No change; already fixed.**

- **F4 (Tauri Music Brain fallback env var):** Listed under "Safe automated fixes." Confirmed in repo: `src-tauri/src/bridge/musicbrain.rs` already uses `env::var("MUSIC_BRAIN_API_URL").or_else(|_| env::var("KMIDI_API_URL"))` (lines 10–11). `.env.example` documents both vars (lines 61–64). `docs/ENVIRONMENT.md` documents both in the Service URLs table (lines 121–122). **No change; already fixed.**

---

## B. TESTS ADDED OR UPDATED

**None.** No code or config was modified; existing tests already cover the current behavior (spectocloud path sandbox tests in `test_api_audit_fixes.py`).

---

## C. VALIDATION RUN

| Command | Result |
|--------|--------|
| `cd src-tauri && cargo check` | **Passed** (exit 0). Warnings are pre-existing (unused imports, dead code, cfg); not caused by this batch. |
| `pytest tests/unit/test_api_schema.py tests/unit/test_api_audit_fixes.py -q` | **Passed** (19 tests). |
| Frontend typecheck/build | **Not run** — no frontend or CI files changed. |
| Shell/script validation | **Not run** — no script files changed. |

---

## D. REMAINING OPEN ITEMS

Only items that remain open and were **not** patched (per report and rules):

- **F1 (FFI journey intensity):** Requires code change in `src/bridge/kelly_ffi.cpp` (`parse_wound_json` / `Wound.intensity`). Report lists under "Needs architecture/product decision"; not in the low-risk patch batch. **Blocked by architecture decision.**

- **F2 (FFI JSON escaping):** Requires JSON library or escape helper in `kelly_ffi.cpp`. Report lists under "Needs architecture/product decision." **Blocked by architecture decision.**

- **Native / C++:** Audio-thread and plugin paths were not directly reviewed; report Honesty Notes state they still need inspection. **Not patched; out of scope for this batch.**

---

## E. NOTES ON SCOPE CONTROL

- **No stale findings were patched.** F3 and F4 were re-verified in the repo and confirmed already fixed; no edits were applied.

- **Items intentionally left untouched because they were already fixed:**
  - **F3 (Spectocloud path):** Sandbox already applied in `api.py`; tests in `test_api_audit_fixes.py` (e.g. `test_spectocloud_rejects_midi_file_path_outside_sandbox`, `test_spectocloud_rejects_output_path_outside_sandbox`) cover it.
  - **F4 (Tauri env var):** `musicbrain.rs` already checks both `MUSIC_BRAIN_API_URL` and `KMIDI_API_URL`; `.env.example` and `ENVIRONMENT.md` already document both.

- No generic cleanup, opportunistic refactors, or other findings from earlier reports were used. Only **docs/CODE_REVIEW_REPORT.md** was used as the reconciled audit source.
