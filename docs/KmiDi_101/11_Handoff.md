# KmiDi 101 — Section 11: Handoff and How to Continue

This section explains how to extend the 101 module, how to find “what depends on X,” and how to pass the baton to the next writer or session.

---

## How to Extend the 101

- **Deepen a section:** Pick a section (0–10) and add more detail. Keep the same tone: short sentences, concrete file names, one idea per paragraph. When you add a new function or file to the narrative, consider adding a row to **10_Dependency_Map.md**.
- **Add a new section:** If a new major area appears (e.g. a new service or a new “room” in the Brain), add a new subsection in **02_Through_09_By_Area.md** or a new file under **docs/KmiDi_101/**, and link it from the overview or the handoff note.
- **Update the dependency map:** Section 10 is a living reference. When you trace a new path (e.g. “who calls the emotion thesaurus?”), add that to **10_Dependency_Map.md** with: name, what it does, what calls it, what it calls.

---

## How to Find “What Depends on X”

- **Follow the discovery workflow:** See **docs/KmiDi_101/DISCOVERY_WORKFLOW.md** for step-by-step search commands, scopes, and how to update the dependency map and handoff.
- **Search the codebase:** Use your editor or command-line search (e.g. `grep`) for the function name, module name, or file path. Look for imports and function calls. In this repo, **music_brain** is Python, **src/** is TypeScript/React, **src-tauri/** is Rust, **src_penta-core/** and **src/bridge/** are C++.
- **Explore in chunks:** If the repo is large, explore one folder or one layer at a time (e.g. “everything under music_brain/session,” or “all callers of process_intent”). Summarize what you find in a bullet list or table, then add that into the right 101 section or the dependency map.
- **Use the discovery log:** The "What has already been gathered" list at the top of **DISCOVERY_WORKFLOW.md** is the single place to check so you don't re-discover the same entry points and paths. If you discover something new, add a one-line note to the handoff block below and an entry to that list so the next person sees it.

---

## How to Pass the Baton

When you stop (e.g. at the end of a session or when handing off to another person):

1. **Update the handoff block** below: set “Last completed” to the section or file you just finished; set “Next” to the section or task to do next; list any key files to open; add a one-sentence discovery if something important came up.
2. **Save all 101 docs** so the next person sees the latest state.
3. **Point the next person to:**  
   - This file (**11_Handoff.md**),  
   - The **docs/KmiDi_101/** folder (00_Overview, 02_Through_09_By_Area, 10_Dependency_Map, DISCOVERY_WORKFLOW).  
   - If a KmiDi 101 plan exists in .cursor/plans or project docs, read it next.

The next writer/session should read the handoff note, then continue from “Next” without redoing completed work unless they need to fix or deepen it.

---

## Handoff note (template — update as you go)

*Copy the block below; fill in the four fields before you stop so the next writer/session can continue from "Next" without redoing work.*

---

**Last completed:** *[Section or file you just finished, e.g. "Section 5 deepened for music_brain/session" or "Discovery for process_intent added to 10_Dependency_Map" or "Initial 101 complete (sections 0–11)."]*

**Next:** *[Exact section or task to do next, e.g. "Trace callers of useMusicBrain in src/ and add to 10_Dependency_Map" or "Deepen Section 6 (Tauri commands)" or "Run find_deps.sh for intent_bridge.py."]*

**Key files to open:**  
*[List 3–6 files the next person should have open; include 101 docs and any code files central to "Next".]*  
- `docs/KmiDi_101/00_Overview.md`  
- `docs/KmiDi_101/02_Through_09_By_Area.md`  
- `docs/KmiDi_101/10_Dependency_Map.md`  
- `docs/KmiDi_101/DISCOVERY_WORKFLOW.md` (when the next task is dependency discovery)  
- If present: a KmiDi 101 plan in .cursor/plans or project docs

**One-sentence discovery:** *[Leave blank or one sentence when something important came up, e.g. “intent_bridge.py is used by MCP/tools, not by the main web UI.”]*

---


## Conclude project context

When wrapping up or handing off, the next writer/session has enough context if they know:

- **What KmiDi is:** AI-powered music creation from natural-language intent; React + Music Brain API + optional Tauri/C++ engine. See **00_Overview.md** (Sections 0–1).
- **Where the 101 lives:** **docs/KmiDi_101/** — 00_Overview, 02_Through_09_By_Area, 10_Dependency_Map, DISCOVERY_WORKFLOW, this file (11_Handoff).
- **How to resume:** Read the handoff block above; continue from **Next**; open **Key files to open**; check **DISCOVERY_WORKFLOW.md** "What has already been gathered" before re-tracing a symbol.
- **Single source for "what was done / what's next":** The handoff block (Last completed, Next, One-sentence discovery). Keep it updated so no one redoes work.
