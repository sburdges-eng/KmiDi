# KmiDi 101 — "What Depends on X" Discovery Workflow

Step-by-step workflow to find callers and callees for a symbol or file, then update the dependency map and handoff so the next person (or session) doesn't re-do the same work.

---

## What has already been gathered

Before starting a new discovery, check this list. If the symbol or path is here, the details are already in **10_Dependency_Map.md** (and possibly in the handoff block in **11_Handoff.md**).

- **process_intent** — Callers and callees in 10_Dependency_Map (Generate path, Other callers).
- **useMusicBrain** / **generateFromIntent** — In 10_Dependency_Map (Generate path).
- **intent_bridge.py** — In 10_Dependency_Map (Other callers); used by MCP/tools, not main web UI.

When you finish a new discovery, add one line here: symbol or path + short summary (e.g. "in 10_Dependency_Map, Generate path").

---

## Step 1 — Choose X

Pick the **symbol** (e.g. `process_intent`, `useMusicBrain`) or **file path** (e.g. `music_brain/session/intent_processor.py`) you want to trace. Decide whether you need:

- **Who calls X** only, or  
- **Who calls X** and **what X calls**.

---

## Step 2 — Pick scope

Search one or more of these; for a large repo, do one scope at a time.

| Scope | Directory(ies) | Language |
|-------|----------------|----------|
| **music_brain** | `music_brain/` | Python |
| **src** | `src/` | TypeScript/React |
| **src-tauri** | `src-tauri/` | Rust |
| **cpp** | `src_penta-core/`, `include/`, `src/bridge/` | C++ |
| **all** | All of the above | — |

---

## Step 3 — Search

**Option A — Use the script (if present):**

```bash
./scripts/docs/find_deps.sh <symbol> [scope]
```

Use `scope`: `music_brain` | `src` | `src-tauri` | `cpp` | `all` (default: `all`). The script prints file:line matches and an optional markdown table row template for **10_Dependency_Map.md**.

**Option B — Manual commands (from repo root):**

- **Python:**  
  `rg 'symbol_or_module' music_brain/`  
  Look for: `from … import`, `import …`, and function calls (symbol as identifier).

- **TypeScript/React:**  
  `rg 'symbol' src/`  
  Look for: imports, `invoke(`, hook/function names.

- **Rust:**  
  `rg 'symbol' src-tauri/`  
  Look for: `use`, function calls, `invoke`.

- **C++:**  
  `rg 'symbol' src_penta-core/ include/ src/bridge/`  
  Look for: `#include`, function calls.

Use word boundaries where it helps (e.g. `rg '\bprocess_intent\b' music_brain/`) to cut down noise.

---

## Step 4 — Summarize

Turn the hits into:

- A **bullet list** (file + what you learned), or  
- The **four-column table** format used in 10_Dependency_Map:

  | Name | What it does | What calls it | What it calls |
  |------|--------------|---------------|---------------|

  One row per important function, module, or file. Keep "What it does" to one short sentence.

---

## Step 5 — Update docs

- **10_Dependency_Map.md:** Add or extend a table in the right section (Generate path, Intent contract, Tauri/C++ bridge, or a new "Other" table). Paste or type the new rows in the same four-column format.
- **11_Handoff.md:** If something important came up (e.g. "X is only used by MCP, not the web UI"), add a one-line note to the handoff block under "One-sentence discovery."
- **This file:** Add an entry to **What has already been gathered** at the top so the next person doesn't re-trace X.

---

## Step 6 — Check before you start

- **Before** you start: Check "What has already been gathered" above. If X is there, use 10_Dependency_Map and skip re-running the same search. After you finish a new discovery, the entry you add in Step 5 is what the next person will see here.
