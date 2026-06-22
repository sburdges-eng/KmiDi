# KmiDi

Audio and DAW tooling app that now includes the iDAW scope. **Not a finished project.**

## Role

KmiDi is the consolidated audio project for composition, MIDI, and DAW-adjacent workflows (KmiDi + iDAW).

## CAD Corpus Reference

CAD corpus context can be referenced when needed at:

- `docs/CAD_CORPUS_SUMMARY.md`

## Build / resolve

From the **workspace root** (parent of `apps/`):

- Resolve deps and build: `uv sync`
- Run tests: `uv run pytest apps/kmidi/tests -v`

No runnable server yet; this app is a stub for KmiDi + iDAW.

## Auto-apply on open

Resolving the workspace (e.g. `uv sync`) can run automatically when you open the repo. See the **top-level README** at the workspace root for how to enable it.
