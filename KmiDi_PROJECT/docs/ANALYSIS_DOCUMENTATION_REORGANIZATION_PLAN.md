# Documentation Reorganization Plan

**Date:** 2025-12-22  
**Component:** Documentation Organization  
**Status:** Complete

---

## 1. Current State

### 1.1 Documentation Distribution

| Location | Files | Status |
|----------|-------|--------|
| Root | 8 | ⚠️ Needs organization |
| `docs/` | 190+ | ⚠️ Needs master index |
| `Production_Workflows/` | 49 | ✅ Well-organized |
| `Songwriting_Guides/` | 26 | ✅ Well-organized |
| `Theory_Reference/` | 9 | ✅ Well-organized |
| `vault/` | 36 | ⚠️ Overlaps with other dirs |
| `docs_music-brain/` | 15 | ⚠️ May be legacy |
| `docs_penta-core/` | 25 | ✅ Component-specific |
| `Obsidian_Documentation/` | 15 | ⚠️ May duplicate vault/ |
| **Total** | **422** | |

### 1.2 Issues Identified

1. **No Master Index**
   - No `docs/README.md` as entry point
   - Difficult to discover documentation
   - No clear navigation

2. **Duplicate Content**
   - Production guides in `Production_Workflows/` and `vault/Production_Guides/`
   - Songwriting guides in `Songwriting_Guides/` and `vault/Songwriting_Guides/`
   - Multiple AI assistant guides

3. **Root-Level Clutter**
   - Historical analysis documents
   - Design documents
   - Old roadmaps

4. **Unclear Organization**
   - Mix of current and historical docs
   - No clear categorization
   - Difficult to find authoritative sources

---

## 2. Proposed Structure

### 2.1 New Documentation Hierarchy

```
docs/
├── README.md                          # Master documentation index ⭐
│
├── getting-started/
│   ├── README.md                      # Getting started index
│   ├── installation.md                # Installation guide
│   ├── quickstart.md                  # Quick start guide
│   ├── architecture-overview.md       # System architecture
│   └── project-identity.md            # Project naming and identity
│
├── guides/
│   ├── README.md                      # Guides index
│   ├── ai-assistant-guide.md          # Consolidated AI guide ⭐
│   ├── production-guides/             # Move from Production_Workflows/
│   │   ├── README.md                  # Production guides index
│   │   ├── country-production.md
│   │   ├── rock-production.md
│   │   ├── metal-production.md
│   │   ├── electronic-edm-production.md
│   │   ├── ambient-atmospheric-production.md
│   │   ├── indie-alternative-production.md
│   │   ├── rnb-soul-production.md
│   │   └── ... (all 49 guides)
│   ├── songwriting-guides/            # Move from Songwriting_Guides/
│   │   ├── README.md                  # Songwriting guides index
│   │   ├── rule-breaking.md
│   │   ├── lyric-writing.md
│   │   ├── chord-progressions.md
│   │   ├── songwriting-fundamentals.md
│   │   └── ... (all 26 guides)
│   └── theory-reference/              # Move from Theory_Reference/
│       ├── README.md                  # Theory reference index
│       ├── music-theory-vocabulary.md
│       ├── audio-recording-vocabulary.md
│       ├── logic-pro-settings.md
│       └── ... (all 9 files)
│
├── api-reference/
│   ├── README.md                      # API reference index
│   ├── python-api.md                  # Python API documentation
│   ├── cpp-api.md                     # C++ API documentation
│   ├── mcp-tools.md                   # MCP tools reference
│   └── cli-reference.md               # CLI command reference
│
├── development/
│   ├── README.md                      # Development docs index
│   ├── build-system.md                # Build system documentation
│   ├── build-exclusions.md            # Document why files are excluded ⭐
│   ├── testing.md                     # Testing guide
│   ├── contributing.md                # Contributing guide
│   ├── code-review-checklist.md       # Code review guidelines
│   ├── dependency-management.md       # Dependency documentation
│   └── troubleshooting.md             # Troubleshooting guide
│
├── components/
│   ├── README.md                      # Components index
│   ├── music-brain.md                 # Music Brain component docs
│   ├── penta-core.md                  # Penta-Core component docs
│   ├── idaw-core.md                   # iDAW_Core (JUCE plugins) docs
│   ├── mcp-servers.md                 # MCP servers documentation
│   └── python-packages.md             # Python package structure
│
└── project-history/
    ├── README.md                      # Project history index
    ├── consolidation-log.md           # Consolidation history
    ├── changelog.md                   # Change log
    ├── roadmap.md                     # Project roadmap
    └── archived/                      # Historical documents
        ├── analysis-summaries/        # Move ANALYSIS_*.md files
        ├── design-documents/          # Move DESIGN_*.md files
        ├── old-roadmaps/              # Move old ROADMAP_*.md files
        └── summaries/                 # Move DELIVERY_SUMMARY.md, etc.
```

### 2.2 Key Features

1. **Master Index** (`docs/README.md`)
   - Clear navigation
   - Quick links to common tasks
   - Discovery mechanism

2. **Consolidated AI Guide**
   - Merge `CLAUDE.md` + `CLAUDE_AGENT_GUIDE.md`
   - Single authoritative source
   - Clear project identity

3. **Organized Guides**
   - Production guides in one place
   - Songwriting guides in one place
   - Theory reference in one place

4. **Component Documentation**
   - Each major component documented
   - Clear component relationships
   - Architecture overview

5. **Project History**
   - Historical documents archived
   - Consolidation log preserved
   - Change log maintained

---

## 3. Migration Plan

### 3.1 Phase 1: Create Structure (Week 1)

**Priority: High**

1. **Create New Directories**
   ```bash
   mkdir -p docs/{getting-started,guides/{production-guides,songwriting-guides,theory-reference},api-reference,development,components,project-history/archived/{analysis-summaries,design-documents,old-roadmaps,summaries}}
   ```

2. **Create Master Index**
   - [ ] Create `docs/README.md` with navigation
   - [ ] Link to all major sections
   - [ ] Add quick start links

3. **Create Section Indexes**
   - [ ] `docs/getting-started/README.md`
   - [ ] `docs/guides/README.md`
   - [ ] `docs/api-reference/README.md`
   - [ ] `docs/development/README.md`
   - [ ] `docs/components/README.md`
   - [ ] `docs/project-history/README.md`

### 3.2 Phase 2: Consolidate AI Guides (Week 1)

**Priority: High**

1. **Merge Guides**
   - [ ] Read both `CLAUDE.md` and `CLAUDE_AGENT_GUIDE.md`
   - [ ] Identify unique content in each
   - [ ] Merge into `docs/guides/ai-assistant-guide.md`
   - [ ] Use "Kelly" as primary name throughout
   - [ ] Update all project name references

2. **Update References**
   - [ ] Update all files referencing CLAUDE.md
   - [ ] Update all files referencing CLAUDE_AGENT_GUIDE.md
   - [ ] Update MCP server documentation

3. **Deprecate Old Files**
   - [ ] Move `CLAUDE.md` to `docs/project-history/archived/`
   - [ ] Move `CLAUDE_AGENT_GUIDE.md` to `docs/project-history/archived/`
   - [ ] Add deprecation notice with link to new guide

### 3.3 Phase 3: Move Guides (Week 2)

**Priority: Medium**

1. **Production Guides**
   - [ ] Move `Production_Workflows/*.md` → `docs/guides/production-guides/`
   - [ ] Create `docs/guides/production-guides/README.md` index
   - [ ] Resolve duplicates with `vault/Production_Guides/`
   - [ ] Keep vault/ for Obsidian, docs/ for general docs

2. **Songwriting Guides**
   - [ ] Move `Songwriting_Guides/*.md` → `docs/guides/songwriting-guides/`
   - [ ] Create `docs/guides/songwriting-guides/README.md` index
   - [ ] Resolve duplicates with `vault/Songwriting_Guides/`

3. **Theory Reference**
   - [ ] Move `Theory_Reference/*.md` → `docs/guides/theory-reference/`
   - [ ] Create `docs/guides/theory-reference/README.md` index

### 3.4 Phase 4: Archive Historical Docs (Week 2)

**Priority: Low**

1. **Analysis Documents**
   - [ ] Move `ANALYSIS_*.md` → `docs/project-history/archived/analysis-summaries/`
   - [ ] Move `docs/ANALYSIS_*.md` → `docs/project-history/archived/analysis-summaries/`

2. **Design Documents**
   - [ ] Move `DESIGN_*.md` → `docs/project-history/archived/design-documents/`
   - [ ] Move `docs/DESIGN_*.md` → `docs/project-history/archived/design-documents/`

3. **Summaries**
   - [ ] Move `*_SUMMARY.md` → `docs/project-history/archived/summaries/`
   - [ ] Move `DELIVERY_SUMMARY.md` → `docs/project-history/archived/summaries/`

4. **Old Roadmaps**
   - [ ] Move `ROADMAP_*.md` → `docs/project-history/archived/old-roadmaps/`
   - [ ] Keep current roadmap in `docs/project-history/roadmap.md`

### 3.5 Phase 5: Create New Documentation (Week 3)

**Priority: Medium**

1. **Getting Started**
   - [ ] Create `docs/getting-started/installation.md`
   - [ ] Create `docs/getting-started/quickstart.md`
   - [ ] Create `docs/getting-started/architecture-overview.md`
   - [ ] Create `docs/getting-started/project-identity.md`

2. **Development**
   - [ ] Create `docs/development/build-system.md`
   - [ ] Create `docs/development/build-exclusions.md` ⭐
   - [ ] Create `docs/development/testing.md`
   - [ ] Create `docs/development/contributing.md`

3. **Components**
   - [ ] Create `docs/components/music-brain.md`
   - [ ] Create `docs/components/penta-core.md`
   - [ ] Create `docs/components/idaw-core.md`
   - [ ] Create `docs/components/mcp-servers.md`
   - [ ] Create `docs/components/python-packages.md`

### 3.6 Phase 6: Update Cross-References (Week 4)

**Priority: High**

1. **Update Internal Links**
   - [ ] Search for all markdown links
   - [ ] Update to new paths
   - [ ] Test all links

2. **Update Code References**
   - [ ] Update docstrings referencing old paths
   - [ ] Update comments referencing old paths
   - [ ] Update MCP tool descriptions

3. **Update External References**
   - [ ] Update README.md links
   - [ ] Update component README files
   - [ ] Update CI/CD documentation

---

## 4. Master Index Template

```markdown
# Kelly Documentation

**Kelly** is a therapeutic interactive Digital Audio Workstation (iDAW) that translates emotions into music.

## Quick Links

- [Installation Guide](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Architecture Overview](getting-started/architecture-overview.md)
- [AI Assistant Guide](guides/ai-assistant-guide.md)

## Documentation Structure

### Getting Started
- [Installation](getting-started/installation.md) - How to install Kelly
- [Quick Start](getting-started/quickstart.md) - Get up and running quickly
- [Architecture Overview](getting-started/architecture-overview.md) - System architecture
- [Project Identity](getting-started/project-identity.md) - Project naming and identity

### Guides
- [AI Assistant Guide](guides/ai-assistant-guide.md) - Guide for AI assistants working with Kelly
- [Production Guides](guides/production-guides/) - Genre-specific production guides
- [Songwriting Guides](guides/songwriting-guides/) - Songwriting methodology and guides
- [Theory Reference](guides/theory-reference/) - Music theory and audio reference

### API Reference
- [Python API](api-reference/python-api.md) - Python API documentation
- [C++ API](api-reference/cpp-api.md) - C++ API documentation
- [MCP Tools](api-reference/mcp-tools.md) - MCP tools reference
- [CLI Reference](api-reference/cli-reference.md) - CLI command reference

### Development
- [Build System](development/build-system.md) - Build system documentation
- [Build Exclusions](development/build-exclusions.md) - Why files are excluded from build
- [Testing](development/testing.md) - Testing guide
- [Contributing](development/contributing.md) - Contributing guidelines
- [Troubleshooting](development/troubleshooting.md) - Troubleshooting guide

### Components
- [Music Brain](components/music-brain.md) - Music intelligence component
- [Penta-Core](components/penta-core.md) - Penta-Core audio engines
- [iDAW Core](components/idaw-core.md) - JUCE plugin suite
- [MCP Servers](components/mcp-servers.md) - MCP server components
- [Python Packages](components/python-packages.md) - Python package structure

### Project History
- [Consolidation Log](project-history/consolidation-log.md) - Repository consolidation history
- [Changelog](project-history/changelog.md) - Change log
- [Roadmap](project-history/roadmap.md) - Project roadmap
- [Archived Documents](project-history/archived/) - Historical documents

## Search

Looking for something specific? Try:
- [All Guides](guides/)
- [All API Documentation](api-reference/)
- [All Development Docs](development/)

## Contributing

See [Contributing Guide](development/contributing.md) for information on contributing to Kelly's documentation.
```

---

## 5. Backward Compatibility

### 5.1 Redirect Strategy

**Option 1: Symbolic Links (Unix/Mac)**
```bash
# Create symlinks for old paths
ln -s docs/guides/ai-assistant-guide.md CLAUDE.md
ln -s docs/guides/ai-assistant-guide.md CLAUDE_AGENT_GUIDE.md
```

**Option 2: Redirect Files**
```markdown
<!-- In old location (CLAUDE.md) -->
# This document has moved

Please see [AI Assistant Guide](../docs/guides/ai-assistant-guide.md)
```

**Option 3: Keep Both (Recommended)**
- Keep old files with deprecation notice
- Link to new location
- Remove in future major version

### 5.2 Update Process

1. **Create new structure** (preserve old)
2. **Move files** (keep copies initially)
3. **Update links** (gradually)
4. **Add deprecation notices** (to old files)
5. **Monitor usage** (check links, references)
6. **Remove old files** (after sufficient time)

---

## 6. Success Criteria

- [ ] Master documentation index created
- [ ] All guides organized into logical structure
- [ ] AI guides consolidated into single document
- [ ] Historical documents archived
- [ ] All cross-references updated
- [ ] No broken links
- [ ] Clear navigation path for new developers
- [ ] Backward compatibility maintained

---

## 7. Timeline Estimate

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 1: Create Structure | 1 day | High |
| Phase 2: Consolidate AI Guides | 2 days | High |
| Phase 3: Move Guides | 2 days | Medium |
| Phase 4: Archive Historical | 1 day | Low |
| Phase 5: Create New Docs | 3 days | Medium |
| Phase 6: Update References | 2 days | High |
| **Total** | **~2 weeks** | |

---

*End of Documentation Reorganization Plan*
