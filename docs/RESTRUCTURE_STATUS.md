# Restructure status

Restructure aligns with the categorized layout (apps/, docs/, libs/, plugins/, tests/) per the build-branch plan.

## Done

- **Top-level dirs:** `apps/`, `libs/`, `plugins/` created.
- **plugin:** Top-level `plugin/` moved to `plugins/plugin/`.
- **docs:** Root narrative `.md` files moved into `docs/` (README, BUILD, AGENTS, QUICK_START kept at root).
- **Root config:** `pyproject.toml` has `[tool.uv.workspace]` with `members = ["apps/*", "libs/*"]`, `[tool.pytest.ini_options]` testpaths, and `migration_manifest.yaml` added.
- **Stub packages:** `apps/kmidi/` and `libs/ai-core/` have minimal `pyproject.toml` for workspace membership.

## Deferred (Phase B)

- **Python packages in libs/:** `music_brain` and `python`/`penta_core` remain at repo root so the current editable install and tests keep working. Moving them into `libs/ai-core/` will require a uv-based workspace install and path/import updates.
- **Frontend in apps/kmidi:** `src/`, `src-tauri/`, `frontend/` remain at root; moving them into `apps/kmidi/` would require CMake, Vite, and Tauri path updates.

## Reference

See the plan **Reference: sburdges-eng-workspace monorepo criteria** for the target uv workspace and test layout.
