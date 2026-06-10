# Architecture Review Checklist

Status: review checklist derived from approved workbook passes A, B, C, D, E, F, and G
Last updated: 2026-06-08

Purpose
- Give reviewers a compact merge checklist for changes that may affect architecture authority.
- Translate the handoff docs into repeatable review questions.

Use this checklist together with:
- `docs/ARCHITECTURE.md`
- `docs/REPO_MODULE_MAP.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`
- `docs/AGENT_ALLOWED_SURFACES.md`
- `docs/HUMAN_OWNED_SURFACES.md`
- `AGENTS.md`

## Fast triage

Before detailed review, answer:
- Does this change stay inside one module from `docs/REPO_MODULE_MAP.md`?
- Does it touch a human-owned surface?
- Does it change semantics, ownership, persistence, ABI, or dependency direction?
- Does it make any historical implementation or old doc become de facto authority?

If any answer is yes or unclear, escalate review depth.

## General architecture checklist

- [ ] Change scope matches one approved module, or the cross-module nature is explicit and justified.
- [ ] No authority moved implicitly between plugin/runtime, native engine, backend, bridge, or schema layers.
- [ ] No historical doc, generated artifact, or support surface is being treated as source of truth.
- [ ] New code follows the approved dependency direction from `docs/REPO_MODULE_MAP.md`.
- [ ] Any drift from the architecture docs is called out explicitly rather than hidden in implementation churn.

## Pass A — global principles

- [ ] Core usability still survives offline and AI/backend failure where required.
- [ ] User control and customization were not weakened for speed or convenience.
- [ ] Plugin/runtime stability and RT safety still outrank experimentation.

## Pass B — repo/module mapping

- [ ] No module boundary was redefined without human approval.
- [ ] `src/ui/` remains presentation under plugin/runtime authority, not a truth-owning layer.
- [ ] `src/project/` remains project/session/persistence authority under plugin/runtime.
- [ ] `src/ml/` only grows on the native side when runtime-needed; broader experimentation belongs in backend/orchestration.
- [ ] Support surfaces (`src_penta-core/`, `include/penta/`, `libs/daiw/`) did not become accidental architecture authority.
- [ ] Engine-separable extraction seams were preserved.

## Pass C — Intent IR / data contracts

- [ ] Engine-facing or persisted intent still converges into validated Intent IR.
- [ ] Generated artifacts were not hand-edited as authority.
- [ ] Any semantic/schema meaning change has explicit human review.
- [ ] Drift checks or sync steps were run when contract sources changed.

## Pass D — native runtime ownership

- [ ] Live runtime truth remains owned by the native engine.
- [ ] UI/plugin/backend code does not directly own runtime-critical native objects.
- [ ] RT-path changes remain allocation-free, lock-free, and exception-free.
- [ ] Shutdown/lifetime ordering was preserved.

## Pass E — JUCE / plugin / RT rules

- [ ] PluginProcessor/host shell still owns lifecycle, persistence boundary, and host integration.
- [ ] PluginEditor/UI code still owns presentation and transient UI state only.
- [ ] Non-RT to RT handoff still uses explicit bounded safe mechanisms.
- [ ] No unsafe callback, timer, worker, or listener lifetime regression was introduced.

## Pass F — FFI / ABI / ownership

- [ ] No new exported symbol was introduced without human approval.
- [ ] Allocator/free pairing remains explicit and matched.
- [ ] No ambiguous borrow/ownership rule was introduced.
- [ ] No exception or panic can cross the FFI boundary.

## Pass G — persistence / migration

- [ ] Project file remains canonical persisted project/session truth.
- [ ] Persisted intent remains canonical validated Intent IR.
- [ ] Native engine runtime state is still reconstructed, not persisted as authority.
- [ ] Save/load orchestration remains owned by plugin/runtime.
- [ ] Migration, restore-order, field lifecycle, and compatibility promises were not changed implicitly.
- [ ] Autosave remains recovery-oriented, not runtime authority.

## Required evidence before merge

At least one of the following should accompany review where relevant:
- file-level diff reasoning tied to an authority doc
- targeted tests or build verification
- schema sync / validator verification
- explicit note that the change is docs-only or governance-only
- explicit human approval for any architecture-surface change

## Escalate immediately if you see

- a broad refactor crossing plugin, engine, persistence, and FFI in one batch
- a convenience shortcut that bypasses validated boundaries
- support or legacy surfaces starting to define product truth
- persistence or ABI changes hidden inside formatting or modernization churn
- old Tauri-centered narratives being reintroduced as current architecture
