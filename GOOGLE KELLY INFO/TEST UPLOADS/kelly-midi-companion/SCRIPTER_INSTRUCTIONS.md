# Logic Pro Scripter - Kelly MIDI Companion

## Quick Setup

1. **Open Logic Pro**
2. **Create a Software Instrument track**
3. **Add Scripter MIDI FX plugin** (MIDI Effects → Scripter)
4. **Open Scripter editor** (click the script icon)
5. **Copy entire contents of `Logic_Pro_Scripter_Kelly.js`**
6. **Paste into Scripter**
7. **Click "Run Script"**

## Features

The script integrates all Kelly JSON data:

### Emotion Thesaurus
- **30+ emotions** from anger.json, joy.json, sad.json, fear.json, etc.
- **Valence/Arousal/Intensity** mapping
- **Auto mode**: Select emotion, script picks progression
- **Manual mode**: Override with valence/arousal sliders

### Chord Progressions
- **20+ progressions** from chord_progressions.json
- Emotion-to-progression mapping
- Common progressions (I-V-vi-IV, etc.)
- Rule-breaking progressions (modal interchange, tritone subs)

### Genre Pocket Maps
- **6 groove styles**: Funk, Jazz Swing, Neo-Soul, Rock, Lo-Fi, Straight
- Timing offsets (laid back vs. on the beat)
- Velocity ranges by genre
- Swing feel parameters

## Parameters

1. **Emotion** - Select from 30+ emotions
2. **Valence** - -1.0 (negative) to +1.0 (positive)
3. **Arousal** - 0.0 (calm) to 1.0 (excited)
4. **Intensity** - 0.0 (subtle) to 1.0 (overwhelming)
5. **Root Note** - C to B
6. **Progression** - Select specific progression
7. **Chord Length** - 1-8 beats per chord
8. **Humanize** - 0.0 (tight) to 1.0 (loose)
9. **Generate Mode** - Auto (from emotion) or Manual
10. **Genre/Groove** - Funk, Jazz, Neo-Soul, Rock, Lo-Fi, Straight
11. **Add 7th** - Probability of adding 7th to chords

## Usage Tips

### Auto Mode (Recommended)
1. Select an emotion (e.g., "hope", "grief", "rage")
2. Script automatically:
   - Picks matching chord progression
   - Sets mode (major/minor)
   - Adjusts tempo feel
   - Applies rule-breaking if needed

### Manual Mode
1. Set "Generate Mode" to "Manual progression"
2. Select specific progression
3. Adjust Valence/Arousal/Intensity manually
4. Fine-tune with other parameters

### Genre/Groove
- **Funk**: Deep pocket, behind the beat
- **Jazz Swing**: Triplet swing feel
- **Neo-Soul**: Very laid back
- **Rock**: On the beat, driving
- **Lo-Fi**: Imperfect, behind the beat
- **Straight**: Perfect grid

## Example Workflows

### "When I Found You Sleeping" (Grief → Hope)
1. Start with emotion: "grief"
2. Progression: "grief_i_iv_i_v"
3. Genre: "lo_fi"
4. Intensity: 0.8
5. Gradually change emotion to "hope" over time

### Angry Rock Track
1. Emotion: "rage"
2. Progression: "rage_i_bii_i_v"
3. Genre: "rock"
4. Intensity: 1.0
5. Humanize: 0.2 (tight)

### Bittersweet Ballad
1. Emotion: "bittersweetness"
2. Progression: "bittersweet_i_vi_iv_v"
3. Genre: "neo_soul"
4. Intensity: 0.5
5. Add 7th: 0.8

## Troubleshooting

**No chords playing?**
- Check track is playing
- Verify Scripter is enabled
- Check Root Note is set correctly
- Try different progression

**Chords sound wrong?**
- Adjust Root Note
- Try different progression
- Check emotion matches desired feel

**Too mechanical?**
- Increase Humanize
- Try different Genre/Groove
- Increase Add 7th for more color

## Data Sources

All data comes from Kelly JSON files:
- `anger.json`, `joy.json`, `sad.json`, `fear.json`, `disgust.json`, `surprise.json`
- `chord_progressions.json`
- `common_progressions.json`
- `chord_progressions_db.json`
- `genre_pocket_maps.json`
- `song_intent_examples.json`

The script embeds key data structures from these files for real-time use in Logic Pro.

