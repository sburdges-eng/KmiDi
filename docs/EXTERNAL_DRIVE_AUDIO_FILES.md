# Audio files on external drive (Sean's SSD)

Search scope: `/Volumes/Sean's SSD`, max depth 8. Run date: 2026-03-02.

## Quick counts (partial)

| Extension | Count (maxdepth 8) |
|-----------|--------------------|
| .wav      | 7,470+             |
| .mp3      | 379+               |

(Flac/ogg/m4a not yet fully counted in this run.)

## Locations (sample paths)

Audio files were found under at least:

- **DevEXTERNAL/KmiDi MIDI Companion/audio/MP3/** — many .mp3 (loops, vocals, SFX).
- **DevEXTERNAL/_FORENSIC_READONLY_KMIDI/daiw_complete/** — .wav (e.g. GUITAR L#01.wav, GUITAR R#09.wav).
- **DevEXTERNAL/_FORENSIC_READONLY_KMIDI/Attachments/audio_moved/Emotion_Scale_Library/** — .mp3.

A full scan with `maxdepth 6` returned **402** audio files (first 200 paths shown in terminal output).

## Re-run search

From repo root:

```bash
# Count only (faster)
find "/Volumes/Sean's SSD" -maxdepth 8 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) 2>/dev/null | wc -l

# List all (can be slow)
find "/Volumes/Sean's SSD" -maxdepth 8 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) 2>/dev/null > /tmp/external_audio_list.txt
```

For deeper or full-volume scan, increase `-maxdepth` or remove it (can be very slow).
