# Naming Standardization Plan

**Date:** 2025-12-22  
**Component:** Naming and Identity Resolution  
**Status:** Complete

---

## 1. Current State Analysis

### 1.1 Project Name Usage

| Location | Project Name | Context |
|----------|--------------|---------|
| `CMakeLists.txt` | Kelly | Build system |
| `pyproject.toml` | kelly | Python package |
| `README.md` | Kelly - Therapeutic iDAW | Main documentation |
| `CLAUDE.md` | iDAW (Intelligent Digital Audio Workstation) | AI guide |
| `CLAUDE_AGENT_GUIDE.md` | DAiW-Music-Brain | AI guide |
| `docs/KELLY_PROJECT_CONSOLIDATION.md` | Kelly | Consolidation doc |
| `CONSOLIDATION_LOG.md` | iDAW | Consolidation log |
| `daiw_mcp/README.md` | DAiW Music-Brain | MCP server |

**Summary:**
- **Kelly:** 3 occurrences (build system, main README, consolidation doc)
- **iDAW:** 2 occurrences (CLAUDE.md, CONSOLIDATION_LOG.md)
- **DAiW/DAiW-Music-Brain:** 2 occurrences (CLAUDE_AGENT_GUIDE.md, daiw_mcp)

### 1.2 Package Name Usage

| Package | Location | CLI Command | Status |
|---------|----------|-------------|--------|
| `kelly` | `src/kelly/` | `kelly` | ⚠️ New, minimal |
| `music_brain` | `music_brain/` | `daiw` | ✅ Active, comprehensive |

**Issues:**
- Two primary packages with overlapping purposes
- CLI command conflict (`kelly` vs `daiw`)
- Package version mismatch (0.1.0 vs 1.0.0)

### 1.3 Namespace Usage (C++)

| Namespace | Usage | Files |
|-----------|-------|-------|
| `kelly` | New code | `src/biometric/`, `src/bridge/` |
| `daiw` | Legacy code | `src/core/memory.cpp`, `include/daiw/` |

**Issues:**
- Mixed namespace usage
- OSC paths use `/daiw/` but code uses `kelly` namespace
- Inconsistent naming

---

## 2. Recommended Standardization

### 2.1 Project Name: **Kelly**

**Rationale:**
1. Already in build system (CMakeLists.txt, pyproject.toml)
2. Clear brand identity
3. Therapeutic focus aligns with project purpose
4. Shorter, more memorable than "iDAW"

**Full Name:** "Kelly - Therapeutic iDAW"

**Subtitle Options:**
- "Kelly - Therapeutic iDAW" (recommended)
- "Kelly - Intelligent Digital Audio Workstation"
- "Kelly - Interactive Digital Audio Workstation"

### 2.2 Package Structure

**Recommended Approach: Option A - Keep Both with Clear Roles**

```
kelly (src/kelly/)
├── Purpose: Therapeutic iDAW wrapper/CLI
├── Version: 0.1.0 → 1.0.0 (align with music_brain)
├── CLI: kelly
└── Imports: music_brain (core functionality)

music_brain (music_brain/)
├── Purpose: Core music intelligence library
├── Version: 1.0.0
├── CLI: daiw (deprecated, alias to kelly)
└── Status: Core library, used by kelly
```

**Benefits:**
- Clear separation of concerns
- `music_brain` remains core library
- `kelly` provides therapeutic focus
- Minimal breaking changes

### 2.3 CLI Command: **kelly**

**Rationale:**
1. Matches package name
2. Matches project name
3. Shorter, clearer

**Migration:**
- Keep `daiw` command as alias to `kelly` (backward compatibility)
- Document deprecation
- Remove `daiw` in future major version

### 2.4 C++ Namespace: **kelly**

**Rationale:**
1. Matches project name
2. Consistent with new code
3. Clearer than `daiw`

**Migration:**
- Update `include/daiw/` → `include/kelly/` (or keep both)
- Update `namespace daiw` → `namespace kelly` in legacy code
- Update OSC paths `/daiw/` → `/kelly/` (or document why `/daiw/` is kept)

---

## 3. Standardization Plan

### 3.1 Phase 1: Documentation Updates (Week 1)

**Priority: High**

1. **Update Main Documentation**
   - [ ] Update `README.md` to use "Kelly" consistently
   - [ ] Update `CLAUDE.md` to reference "Kelly" (keep iDAW as subtitle)
   - [ ] Update `CLAUDE_AGENT_GUIDE.md` to reference "Kelly"
   - [ ] Update `docs/KELLY_PROJECT_CONSOLIDATION.md` (already correct)

2. **Update AI Guides**
   - [ ] Merge `CLAUDE.md` and `CLAUDE_AGENT_GUIDE.md` into single guide
   - [ ] Use "Kelly" as primary name throughout
   - [ ] Reference "iDAW" as subtitle/description

3. **Update Component Documentation**
   - [ ] Update `daiw_mcp/README.md` to reference "Kelly"
   - [ ] Update MCP server documentation
   - [ ] Update component-specific docs

### 3.2 Phase 2: Code Updates (Week 2-3)

**Priority: Medium**

1. **Python Package**
   - [ ] Update `music_brain` package to reference "Kelly" in docstrings
   - [ ] Add `daiw` CLI alias for `kelly` command
   - [ ] Update package documentation

2. **C++ Namespace (Optional)**
   - [ ] Decide on namespace migration strategy
   - [ ] If migrating: Update `namespace daiw` → `namespace kelly`
   - [ ] If keeping both: Document rationale
   - [ ] Update include paths if needed

3. **OSC Paths (Optional)**
   - [ ] Decide on OSC path migration
   - [ ] If migrating: Update `/daiw/` → `/kelly/`
   - [ ] If keeping: Document why `/daiw/` is used

### 3.3 Phase 3: Build System (Week 4)

**Priority: Low**

1. **CMakeLists.txt**
   - [ ] Already uses "Kelly" ✅
   - [ ] No changes needed

2. **pyproject.toml**
   - [ ] Already uses "kelly" ✅
   - [ ] Consider version bump to 1.0.0 to align with music_brain

### 3.4 Phase 4: Deprecation (Future)

**Priority: Low**

1. **Deprecate `daiw` CLI Command**
   - [ ] Add deprecation warning to `daiw` command
   - [ ] Document migration path
   - [ ] Remove in future major version (2.0.0)

2. **Deprecate `daiw` Namespace (if applicable)**
   - [ ] Add deprecation warnings
   - [ ] Provide migration guide
   - [ ] Remove in future major version

---

## 4. Migration Checklist

### 4.1 Documentation Files to Update

- [ ] `README.md`
- [ ] `CLAUDE.md`
- [ ] `CLAUDE_AGENT_GUIDE.md`
- [ ] `daiw_mcp/README.md`
- [ ] `docs/KELLY_PROJECT_CONSOLIDATION.md` (already correct)
- [ ] All component-specific README files
- [ ] All guide documents

### 4.2 Code Files to Update

**Python:**
- [ ] `music_brain/__init__.py` (docstrings)
- [ ] `music_brain/cli.py` (if exists, add kelly alias)
- [ ] All package docstrings

**C++ (if migrating namespace):**
- [ ] `src/core/memory.cpp` (namespace daiw → kelly)
- [ ] `include/daiw/` headers (if renaming)
- [ ] All files using `namespace daiw`

**OSC (if migrating paths):**
- [ ] `src/bridge/OSCBridge.cpp` (update `/daiw/` paths)

### 4.3 Build System Files

- [ ] `CMakeLists.txt` (already correct ✅)
- [ ] `pyproject.toml` (already correct ✅)

---

## 5. Backward Compatibility

### 5.1 CLI Command Compatibility

**Recommended Approach:**
```python
# In music_brain/cli.py or kelly/cli.py
import typer

# Primary command
app = typer.Typer(name="kelly")

# Alias for backward compatibility
daiw_app = typer.Typer(name="daiw")

@daiw_app.command()
def _deprecated_warning():
    """Deprecated: Use 'kelly' command instead."""
    import warnings
    warnings.warn(
        "The 'daiw' command is deprecated. Use 'kelly' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Forward to kelly command
    app()
```

### 5.2 Namespace Compatibility (C++)

**If Migrating Namespace:**

```cpp
// Option 1: Namespace alias (temporary)
namespace daiw = kelly;

// Option 2: Using declarations
using namespace kelly;
namespace daiw {
    using kelly::*;
}

// Option 3: Keep both (recommended for now)
namespace kelly { /* new code */ }
namespace daiw { /* legacy code, deprecated */ }
```

**Recommended:** Keep both namespaces, mark `daiw` as deprecated

### 5.3 OSC Path Compatibility

**If Migrating OSC Paths:**

```cpp
// Support both paths for backward compatibility
if (path.startsWith("/daiw/")) {
    // Handle legacy path
    handleLegacyPath(path);
} else if (path.startsWith("/kelly/")) {
    // Handle new path
    handleNewPath(path);
}
```

**Recommended:** Support both paths, document migration

---

## 6. Communication Plan

### 6.1 Developer Communication

1. **Announcement**
   - Post in project chat/forum
   - Update project description
   - Add to changelog

2. **Migration Guide**
   - Create `docs/migration/naming-standardization.md`
   - Document all changes
   - Provide examples

3. **Timeline**
   - Week 1: Documentation updates
   - Week 2-3: Code updates
   - Week 4: Build system alignment
   - Future: Deprecation warnings

### 6.2 AI Assistant Communication

1. **Update AI Guides**
   - Merge and update CLAUDE.md
   - Update CLAUDE_AGENT_GUIDE.md
   - Clear project identity

2. **Update Context Files**
   - Update `.agents/` context files
   - Update MCP server descriptions
   - Update tool documentation

---

## 7. Success Criteria

- [ ] All documentation uses "Kelly" as primary name
- [ ] "iDAW" used only as subtitle/description
- [ ] CLI command standardized to `kelly`
- [ ] `daiw` command available as alias (with deprecation warning)
- [ ] C++ namespace decision made and documented
- [ ] OSC path decision made and documented
- [ ] Migration guide created
- [ ] No breaking changes for existing users (backward compatibility)

---

## 8. Risk Assessment

### 8.1 Low Risk

- Documentation updates (no code changes)
- Adding CLI alias (backward compatible)
- Keeping both namespaces (no breaking changes)

### 8.2 Medium Risk

- Merging AI guides (may lose information)
- Updating package docstrings (minor breaking change if API documented)

### 8.3 High Risk

- Renaming C++ namespace (breaking change)
- Changing OSC paths (breaking change for external tools)
- Removing `daiw` command (breaking change)

**Mitigation:**
- Keep backward compatibility where possible
- Provide migration guides
- Use deprecation warnings
- Gradual migration over multiple versions

---

*End of Naming Standardization Plan*
