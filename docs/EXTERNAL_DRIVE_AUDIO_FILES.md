# Audio files on external drive

**Full-volume search** — root = `$KMIDI_DATA_ROOT` (set in `.env` when drive is mounted; no hardcoded volume name). No depth limit. Run: 2026-03-02. Expanded: same date (additional extensions + uppercase).

## Total: 128,456 audio files

### By extension (combined)

| Extension | Count   |
|-----------|--------|
| .wav      | 111,489 |
| .mp3      | 8,450   |
| .flac     | 7,323   |
| .ogg      | 93      |
| .m4a      | (remainder) |
| .aif      | 1,067   |
| .caf      | 34      |

Original search: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a` (127,355). Additional search: `.aif`, `.caf` plus uppercase variants for all — **1,101** extra (`.aif`: 1,067; `.caf`: 34).

## By top-level directory

| Dir (under volume)     | Audio file count |
|------------------------|------------------|
| Datasets               | 116,712          |
| My MacEXTERNAL         | 7,439            |
| COLD_STORAGEEXTERNAL   | 2,451            |
| DevEXTERNAL            | 442              |
| KmiDi_MASTER_VAULTEXTERNAL | 285  |
| KmiDi                  | 26               |

*(Directory breakdown is from the original 5-extension search; additional .aif/.caf are under the same volume.)*

## Copies / duplicates

- **Same filename, different paths:** 7,985 basenames appear more than once → **45,048** files share their name with at least one other (e.g. many `0000.wav` in different m4singer folders).
- **Same content (MD5), full scan:** **352** duplicate groups (**2,109** files that are byte-identical copies). List: **`docs/EXTERNAL_DRIVE_AUDIO_DUPLICATES.txt`** (groups separated by `---`). Keeping one file per group would remove **1,757** redundant copies.

**Re-run content-duplicate scan** (hashes all 128k files; run in tmux; ~30–60 min):

```bash
./scripts/audio_duplicate_scan.sh
```

## Deduplicate and consolidate under Datasets

Two steps: (1) remove content-identical duplicates, (2) move all remaining audio from other top-level dirs into `Datasets/consolidated/from_<TopDir>/...`. Use `KMIDI_DATA_ROOT` from `.env` (or default `/Volumes/Sean's SSD`).

**1. Delete duplicates** (keeps one per group; prefers path under `Datasets/`):

```bash
./scripts/delete_audio_duplicates.sh          # dry-run (default)
./scripts/delete_audio_duplicates.sh --execute   # actually delete
```

**2. Move audio into Datasets** (run after step 1):

```bash
./scripts/move_audio_to_datasets.sh           # dry-run (default)
./scripts/move_audio_to_datasets.sh --execute    # actually move
```

Result: all audio under `$KMIDI_DATA_ROOT/Datasets`; files that were under `DevEXTERNAL`, `My MacEXTERNAL`, etc. live under `Datasets/consolidated/from_DevEXTERNAL/`, `from_My MacEXTERNAL/`, etc.

**Run (2026-03-02):** Delete ran; some paths under `DevEXTERNAL` and under `Datasets/.../CLEANUP_RECOVERY...` returned "Permission denied" (kept one per group where deletion succeeded). Move ran successfully: **10,057** files moved into `Datasets/consolidated/from_*`. Fix permissions on those dirs if you want to remove the remaining duplicates or re-run move.

## Full list

- **Complete (all extensions):** **`docs/EXTERNAL_DRIVE_AUDIO_COMPLETE_LIST.txt`** — 128,456 paths, one per line.
- Original only: `docs/EXTERNAL_DRIVE_AUDIO_FULL_LIST.txt` (127,355).
- Additional only: `docs/EXTERNAL_DRIVE_AUDIO_ADDITIONAL_LIST.txt` (1,101).

## Re-run full-volume search

Use `KMIDI_DATA_ROOT` from `.env` (no hardcoded drive name). From repo root:

```bash
# Load env (e.g. set -a; source .env; set +a) then:
ROOT="${KMIDI_DATA_ROOT:?Set KMIDI_DATA_ROOT in .env}"

# Original extensions
find "$ROOT" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) 2>/dev/null > docs/EXTERNAL_DRIVE_AUDIO_FULL_LIST.txt

# Additional extensions (aif, caf + uppercase variants)
find "$ROOT" -type f \( -name "*.WAV" -o -name "*.MP3" -o -name "*.FLAC" -o -name "*.OGG" -o -name "*.M4A" -o -name "*.aiff" -o -name "*.AIFF" -o -name "*.aif" -o -name "*.AIF" -o -name "*.m4b" -o -name "*.M4B" -o -name "*.aac" -o -name "*.AAC" -o -name "*.opus" -o -name "*.OPUS" -o -name "*.alac" -o -name "*.ALAC" -o -name "*.weba" -o -name "*.WEBA" -o -name "*.caf" -o -name "*.CAF" \) 2>/dev/null > docs/EXTERNAL_DRIVE_AUDIO_ADDITIONAL_LIST.txt

# Combine
cat docs/EXTERNAL_DRIVE_AUDIO_FULL_LIST.txt docs/EXTERNAL_DRIVE_AUDIO_ADDITIONAL_LIST.txt > docs/EXTERNAL_DRIVE_AUDIO_COMPLETE_LIST.txt
wc -l docs/EXTERNAL_DRIVE_AUDIO_*.txt
```

Or run the script: `./scripts/catalog-external-audio.sh` (uses `KMIDI_DATA_ROOT`).
