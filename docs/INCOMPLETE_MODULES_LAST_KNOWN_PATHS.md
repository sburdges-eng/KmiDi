# Incomplete Modules — Last Known Paths

**Purpose:** Restore or reimplement incomplete/deleted modules. Use `git show <commit>:<path>` to extract from history.

**References:** [GIT_RESTORE_PATHWAYS.md](GIT_RESTORE_PATHWAYS.md), [ISSUES_LIST.md](ISSUES_LIST.md), [SPECTOCLOUD.md](SPECTOCLOUD.md)

---

## 1. Visualization

| Module | Last known path | Commit | Notes |
|--------|-----------------|--------|-------|
| **spectocloud.py** | `KmiDi_CANON/brain/music_brain/visualization/spectocloud.py` | `6d4d67c5` (restored) | **COMPLETE (2026-01-31)** — Full particle/spectral render restored from 6d4d67c5, adapted for current layout. Optional deps: matplotlib, mido. |
| **spectocloud_cli.py** | `KmiDi_BACKUP/project/source/python/music_brain/visualization/spectocloud_cli.py` | `d8149ca2` (deleted) | CLI for render/animate; used mido for MIDI load. Optional; can restore if CLI needed. |
| **SPECTOCLOUD_README.md** | (forensic/archive per SPECTOCLOUD.md) | — | Optional; check forensic repo. |

**Restore spectocloud_cli:**

```bash
git show d8149ca2:KmiDi_BACKUP/project/source/python/music_brain/visualization/spectocloud_cli.py > /tmp/spectocloud_cli.py
# Then adapt for KmiDi_CANON/brain/music_brain/visualization/
```

---

## 2. Audio

| Module | Last known path | Commit | Notes |
|--------|-----------------|--------|-------|
| **refinery.py** | `music_brain/audio/refinery.py` | `0886e9d5` | Librosa-based. Restore: `git show 0886e9d5:music_brain/audio/refinery.py` → `KmiDi_CANON/brain/music_brain/audio/`. |
| **audio_cataloger.py** | `KmiDi_BACKUP/project/legacy/legacy/Python_Tools/audio/audio_cataloger.py` | `d8149ca2` (deleted) | BPM/key detection with librosa. |
| **audio_cataloger.py** (scripts) | `KmiDi_BACKUP/project/scripts/audio_cataloger.py` | `d8149ca2` (deleted) | Alternative location. |

**Restore refinery:**

```bash
git show 0886e9d5:music_brain/audio/refinery.py > /tmp/refinery.py
# Adapt imports for KmiDi_CANON; target: KmiDi_CANON/brain/music_brain/audio/refinery.py
```

---

## 3. KmiDi_BACKUP (deleted in d8149ca2)

| Module | Last known path | Notes |
|--------|-----------------|-------|
| **spectocloud_cli** | `KmiDi_BACKUP/project/source/python/music_brain/visualization/spectocloud_cli.py` | See §1. |
| **spectocloud examples** | `KmiDi_BACKUP/project/examples/spectocloud_animation.py`, `spectocloud_example.py` | Animation and basic example. |
| **audio_cataloger** | `KmiDi_BACKUP/project/legacy/legacy/Python_Tools/audio/audio_cataloger.py` | |
| **audio_cataloger** | `KmiDi_BACKUP/project/scripts/audio_cataloger.py` | |
| **groove/harmony/data** | `KmiDi_BACKUP/project/data/groove_extractor.py`, `groove_applicator.py`, `harmony_generator.py` | Likely superseded by `music_brain/groove/`, `music_brain/harmony/`. |
| **emotion_thesaurus** | `KmiDi_BACKUP/project/data/emotion_thesaurus/emotion_thesaurus.py` | |
| **brain_server** | `KmiDi_BACKUP/project/scripts/brain_server.py` | |
| **idaw_* scripts** | `KmiDi_BACKUP/project/scripts/idaw_*.py` | Various iDAW integration scripts. |

---

## 4. ISSUES_REPORT (historical)

| Item | Last known path | Commit |
|------|-----------------|--------|
| **ISSUES_REPORT.md** | `KmiDi_PROJECT/ISSUES_REPORT.md` | `21991118` |

**View:**

```bash
git show 21991118:KmiDi_PROJECT/ISSUES_REPORT.md
```

Content summarized in [ISSUES_LIST.md](ISSUES_LIST.md).

---

## 5. Forensic DAiW-Music-Brain

**Repo:** `~/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain`

| Module | Forensic path | Key commits |
|--------|---------------|-------------|
| intent_schema | `music_brain/session/intent_schema.py` | dc58aac, 0391216 |
| intent_processor | `music_brain/session/intent_processor.py` | 0391216, 0b23796 |
| generator, interrogator, teaching | `music_brain/session/*` | b3cd0eb, c5b3450 |

**Restore (read-only):**

```bash
FORENSIC="/Users/seanburdges/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain"
git -C "$FORENSIC" show 0391216:music_brain/session/intent_processor.py > /tmp/intent_processor_forensic.py
```

---

## 6. Current Canon Layout (reference)

| Layer | Path |
|-------|------|
| Brain root | `KmiDi_CANON/brain/` |
| music_brain | `KmiDi_CANON/brain/music_brain/` |
| mcp_workstation | `KmiDi_CANON/brain/mcp_workstation/` |
| penta_core | `KmiDi_CANON/brain/penta_core/` |
| kmidi_gui | `KmiDi_CANON/brain/kmidi_gui/` |
| api_server | `KmiDi_CANON/brain/api_server.py` |
| visualization | `KmiDi_CANON/brain/music_brain/visualization/spectocloud.py` |

---

## 7. Quick restore commands

```bash
cd "/Users/seanburdges/Dev/KmiDi MIDI Companion"

# spectocloud_cli (deleted)
git show d8149ca2:KmiDi_BACKUP/project/source/python/music_brain/visualization/spectocloud_cli.py

# refinery (old layout — may need path adjustment)
git log -p -1 0886e9d5 -- "**/refinery.py"

# ISSUES_REPORT
git show 21991118:KmiDi_PROJECT/ISSUES_REPORT.md
```

---

*Last updated: 2026-01-31*
