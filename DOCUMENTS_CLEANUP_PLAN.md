# Documents/Disk Space Cleanup Plan

**Date:** January 19, 2025

## 🔍 Problem Identified

Your system has **587GB** total in home directory, with **359GB** in Library alone!

## 📊 Space Breakdown

### Total Home Directory: 587GB

**Top Space Consumers:**

1. **Library: 359GB** (1.6M files) ⚠️ **MAJOR ISSUE**
   - Application Support: **146GB** 🔴
   - Group Containers: **69GB** 🔴
   - CloudStorage: **69GB** 🟡
   - Containers: **48GB** 🟡
   - Caches: **11GB** 🟢 (can safely clean)

2. **RECOVERY_OPS: 105GB** 🟡
   - Likely recovery/backup files

3. **audio: 28GB** 🟡
   - Audio files directory

4. **.cursor: 27GB** 🟡
   - Cursor IDE data

5. **Desktop: 24GB** 🟡
   - Desktop files

## 🎯 Cleanup Strategy

### Phase 1: Safe Cache Cleanup (Start Here)

**Caches: 11GB** - Safe to clean
```bash
# macOS built-in cleanup
rm -rf ~/Library/Caches/*

# Specific app caches (safe)
rm -rf ~/Library/Caches/com.apple.*
rm -rf ~/Library/Caches/Homebrew
rm -rf ~/Library/Caches/pip
```

### Phase 2: Analyze Application Support (146GB)

This is the biggest issue. Need to identify what apps are using space:

**Common culprits:**
- Docker Desktop (VMs, images)
- Xcode (derived data, archives)
- Browser profiles (Chrome, Firefox)
- Electron apps (Discord, Slack, etc.)
- Code editors (VS Code extensions, etc.)

**Investigation needed:**
```bash
# Find largest Application Support subdirectories
du -h -d 1 ~/Library/Application\ Support | sort -hr | head -20
```

### Phase 3: Group Containers (69GB)

**Common large apps:**
- Cloud storage sync (iCloud, Dropbox, OneDrive)
- Messaging apps (Messages, WhatsApp)
- Productivity apps

**Investigation:**
```bash
du -h -d 1 ~/Library/Group\ Containers | sort -hr | head -20
```

### Phase 4: CloudStorage (69GB)

This might be synced cloud files. Check:
- Is iCloud Desktop/Documents enabled?
- Dropbox/OneDrive sync folders
- Consider moving to external storage

### Phase 5: Other Directories

1. **RECOVERY_OPS (105GB)**
   - Review if recovery files are still needed
   - Move to external storage if needed

2. **Desktop (24GB)**
   - Clean up old files
   - Move to organized folders
   - Enable iCloud Desktop if appropriate

3. **.cursor (27GB)**
   - Check if this is cache/index data
   - May be safe to clean periodically

## 🛠️ Tools Created

1. `audit_large_files.sh` - System-wide audit
2. `analyze_library.sh` - Deep Library analysis
3. `audit_documents.sh` - Documents-specific audit

## 📋 Next Steps

### Immediate (Safe):

1. **Clean caches:**
   ```bash
   rm -rf ~/Library/Caches/*
   # Saves ~11GB
   ```

2. **Find what's in Application Support:**
   ```bash
   ./analyze_library.sh
   # Or manually:
   du -h -d 1 ~/Library/Application\ Support | sort -hr
   ```

### Short-term (Review First):

3. **Analyze Application Support** - Find which apps are using 146GB
4. **Review Group Containers** - Check cloud sync and messaging apps
5. **Review RECOVERY_OPS** - Determine if 105GB of recovery files needed

### Long-term:

6. **Set up regular cleanup** - Cache cleaning scripts
7. **External storage** - Move large archives/recovery files
8. **Cloud storage review** - Optimize iCloud/OneDrive sync

## ⚠️ Safety Notes

- **Library/Caches** - Safe to clean (regenerates)
- **Application Support** - Review before deleting (app data)
- **Group Containers** - Be careful (shared app data)
- **CloudStorage** - May be synced files (check first)

## 📈 Expected Impact

- **Immediate:** 11GB from caches
- **Potential:** 100-200GB+ after full cleanup
- **Total reduction:** Could free 150-300GB+

## 🔗 Related Cleanup

Remember we already cleaned your KmiDi project:
- **1.1GB freed** from project cleanup
- **35,360 files** quarantined

---

**Priority:** Start with cache cleanup, then investigate Application Support (the 146GB mystery).
