# Instructions for GitHub Copilot

## Repository Setup

**Source Repository (always up-to-date):**
```
/Users/seanburdges/Desktop/KmiDi-remote
```
- Connected to: https://github.com/sburdges-eng/KmiDi.git
- This is the SOURCE OF TRUTH

**Target Repository (to be overwritten):**
```
/Users/seanburdges/Desktop/KmiDi
```
- This should ALWAYS be overwritten with data from KmiDi-remote
- DO NOT work directly in this folder

---

## Your Task: Keep KmiDi Synchronized

### Step 1: Update KmiDi-remote from GitHub
```bash
cd /Users/seanburdges/Desktop/KmiDi-remote
git pull origin main
```

### Step 2: Overwrite KmiDi with KmiDi-remote
```bash
# Use rsync to completely sync (overwrite) KmiDi with KmiDi-remote
rsync -av --delete \
    --exclude='.git/' \
    --exclude='.DS_Store' \
    /Users/seanburdges/Desktop/KmiDi-remote/ \
    /Users/seanburdges/Desktop/KmiDi/
```

**What this does:**
- `-av`: Archive mode, verbose output
- `--delete`: Remove files in KmiDi that don't exist in KmiDi-remote
- `--exclude='.git/'`: Keep separate git histories
- Overwrites ALL files in KmiDi with latest from KmiDi-remote

---

## Complete Sync Command (One-liner)
```bash
cd /Users/seanburdges/Desktop/KmiDi-remote && \
git pull origin main && \
rsync -av --delete --exclude='.git/' --exclude='.DS_Store' \
/Users/seanburdges/Desktop/KmiDi-remote/ /Users/seanburdges/Desktop/KmiDi/
```

---

## Important Rules

1. ✅ **Always work in:** `/Users/seanburdges/Desktop/KmiDi-remote`
2. ✅ **Always push to:** `https://github.com/sburdges-eng/KmiDi.git`
3. ❌ **Never work in:** `/Users/seanburdges/Desktop/KmiDi` (this gets overwritten)
4. ✅ **Before starting work:** Pull from GitHub
5. ✅ **After finishing work:** Push to GitHub
6. ✅ **Sync KmiDi:** Run the rsync command to overwrite

---

## Workflow for Copilot

```bash
# 1. Pull latest from GitHub
cd /Users/seanburdges/Desktop/KmiDi-remote
git pull origin main

# 2. Do your work (edit files, write code, etc.)
# ... make changes ...

# 3. Commit and push your changes
git add .
git commit -m "feat: your changes here"
git push origin main

# 4. Overwrite local KmiDi folder
rsync -av --delete --exclude='.git/' --exclude='.DS_Store' \
/Users/seanburdges/Desktop/KmiDi-remote/ /Users/seanburdges/Desktop/KmiDi/
```

---

## Why Two Folders?

- **KmiDi-remote**: Working repository for Claude + Copilot collaboration
- **KmiDi**: Claude's working directory (kept in sync via overwrite)

This ensures both AIs always have the latest code without manual coordination.
