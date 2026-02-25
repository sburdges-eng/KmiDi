# Migrate data into KmiDi-compile before deleting sibling folders

Use this when you want to **keep only KmiDi-compile** and move any needed data into it so scripts still work after you delete AUDIO_MIDI_DATA, sbdrive mounts, etc.

---

## 1. Create data dirs inside KmiDi-compile

```bash
cd "/path/to/KmiDi MIDI Companion/KmiDi-compile"

mkdir -p datasets/audio
mkdir -p datasets/audio/kelly-audio-data   # optional, for training layout
mkdir -p models/checkpoints                # for emotion classifier fallback
```

---

## 2. Point env at KmiDi-compile

Add to `KmiDi-compile/.env.example` (and your real `.env` if you use one):

```bash
# Data roots — use paths inside KmiDi-compile after migration
KELLY_AUDIO_DATA_ROOT=<absolute-path-to-KmiDi-compile>/datasets/audio
AUDIO_DATA_ROOT=<absolute-path-to-KmiDi-compile>/datasets/audio
```

Example on your machine:

```bash
KELLY_AUDIO_DATA_ROOT=/Users/seanburdges/KmiDi MIDI Companion/KmiDi-compile/datasets/audio
AUDIO_DATA_ROOT=/Users/seanburdges/KmiDi MIDI Companion/KmiDi-compile/datasets/audio
```

---

## 3. Copy (or symlink) from AUDIO_MIDI_DATA

If you want to keep a copy of `AUDIO_MIDI_DATA` inside KmiDi-compile:

**Option A – copy (uses more disk, safest)**

```bash
WORKSPACE="/Users/seanburdges/KmiDi MIDI Companion"
KMIDI="$WORKSPACE/KmiDi-compile"

cp -R "$WORKSPACE/AUDIO_MIDI_DATA/kelly-audio-data/"* "$KMIDI/datasets/audio/" 2>/dev/null || true
# If you use SSD_Transfer content:
cp -R "$WORKSPACE/AUDIO_MIDI_DATA/SSD_Transfer/kelly-audio-data/"* "$KMIDI/datasets/audio/" 2>/dev/null || true
```

**Option B – symlink (no extra space, link breaks if you delete AUDIO_MIDI_DATA)**

```bash
WORKSPACE="/Users/seanburdges/KmiDi MIDI Companion"
KMIDI="$WORKSPACE/KmiDi-compile"

ln -s "$WORKSPACE/AUDIO_MIDI_DATA/kelly-audio-data" "$KMIDI/datasets/audio/kelly-audio-data"
```

Only use Option B if you plan to keep AUDIO_MIDI_DATA elsewhere or on another volume.

---

## 4. Run the migration script (optional)

From the repo root (KmiDi-compile):

```bash
python scripts/migrate_data_into_kmidi_compile.py --dry-run
python scripts/migrate_data_into_kmidi_compile.py --copy    # copy from sibling AUDIO_MIDI_DATA
# or
python scripts/migrate_data_into_kmidi_compile.py --link   # symlink instead of copy
```

(See that script’s `--help` for exact behavior.)

---

## 5. Checkpoints (already handled)

`music_brain/emotion/audio_emotion_classifier.py` already looks in `KmiDi-compile/models/checkpoints` first. Create that dir if you use the classifier:

```bash
mkdir -p models/checkpoints
```

Optional: set `KMIDI_CHECKPOINTS_DIR` in `.env` to override the search path.

---

## 6. Checklist before deleting sibling project data

- [ ] `python scripts/check_external_dependencies.py` run and reviewed  
- [ ] `KELLY_AUDIO_DATA_ROOT` and `AUDIO_DATA_ROOT` set to `KmiDi-compile/datasets/audio` (or your chosen path)  
- [ ] Any needed data from AUDIO_MIDI_DATA copied or linked into `KmiDi-compile/datasets/audio`  
- [ ] `audio_emotion_classifier.py` updated or configured to use `models/checkpoints` (or env)  
- [ ] Training/scripts that use `KELLY_AUDIO_DATA_ROOT` tested with the new path  

Then it is **safe to delete** all project data outside `KmiDi-compile/`.
