# KmiDi release-polish report

## Status

**READY**

## Minimal fixes made

- **Phase 1 — dev-setup direct invoke:** Set executable bit on `scripts/dev-setup.sh` (`chmod +x`) and staged the mode in git (`git update-index --chmod=+x`). Script was the only one in `scripts/` tracked as 100644; it is now 100755 like other shell scripts. Direct `./scripts/dev-setup.sh` runs without permission denied.
- **Phase 2 — dev flow / port collision:** Changed `dev:all` to run **React + Music Brain API only** (removed Tauri from the concurrently group). Tauri’s `beforeDevCommand` runs `npm run dev`, so running Tauri and React together in one command started two Vite instances and caused a port collision. `dev:all` now runs only `npm run dev` and `npm run dev:python`. Updated `AGENTS.md` (Running services table and Gotchas) and `scripts/dev-setup.sh` final echo to describe the split: `dev:all` = React + API; desktop = `npm run dev:tauri` separately (optionally with `npm run dev:python` in another terminal).

## Validations

- **`./scripts/dev-setup.sh` (direct invoke)**  
  Command: `./scripts/dev-setup.sh 2>&1 | head -5`  
  **RESULT: PASS** — Script runs; first line `=> KmiDi v1 dev setup (bootstrap + npm + pip)...` and bootstrap proceeds (no permission denied).

- **`npm run dev:all`**  
  Command: `npm run dev:all` (run in background ~8s then stopped).  
  **RESULT: PASS** — Starts only `react` and `api` (concurrently -n react,api). Vite on http://localhost:1420 and uvicorn on :8000; no Tauri, no second Vite, no port collision.

- **`npm run dev:tauri`**  
  Command: `npm run dev:tauri` (run in background ~10s then stopped).  
  **RESULT: PASS** — Runs BeforeDevCommand (`npm run dev`), Vite on :1420, then Tauri dev (cargo run). Desktop flow works when run on its own.

## Remaining non-blocking notes

- None. Both polish items are addressed: dev-setup is executable and documented; dev:all no longer causes a Tauri/Vite port collision and behavior is documented in AGENTS.md.

## Final decision

**READY.** The repo is release-ready. The two polish items are resolved: (1) `scripts/dev-setup.sh` is executable in-repo and direct invocation works; (2) `dev:all` runs only React and the Music Brain API, avoiding the Tauri/Vite port collision, and AGENTS.md clearly describes `dev:all` vs `dev:tauri`. Canonical pipelines and tests were already passing; this pass only applied minimal, freeze-safe changes and validations.
