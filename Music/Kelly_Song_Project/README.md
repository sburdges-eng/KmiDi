# Kelly Song Project
## "When I Found You" - A Song for Kelly

### ABOUT THIS PROJECT
This song is written to remember Kelly - blonde hair, green eyes, 29 years old. A helicopter pilot for the Army. She sat by the dartboards at Molly McPherson's, always alone but knew everyone. Late nights after 10pm. Her place was dark, maroon, cozy - nick-knacks or plants. She had a loud laugh that changed her whole face. She wore flannel or tank tops, and a necklace. She was always smiling.

She knew Sean better than anyone at that time. They would sit on her couch and just talk. No TV, no music. It felt easy. She smelled like shea butter shampoo.

This song uses intentional misdirection - every line sounds like falling in love for the first time, until the final verse reveals the truth: this was the moment of finding her gone.

---

## SONG STRUCTURE

**Key:** A minor (Am)  
**Tempo:** 72 BPM  
**Time Signature:** 4/4  
**Feel:** Sparse, intimate, devastatingly beautiful  
**Reference tracks:** Snuff (Corey Taylor), Wicked Game (Chris Isaak), Something to Remind You (Staind/Aaron Lewis)

---

## CHORD PROGRESSIONS

### INTRO (8 bars - fingerpicked, let notes ring)
```
| Am      | Am/G    | Fmaj7   | E       |
| Am      | Am/G    | Dm      | E       |
```

**Voicings (standard tuning):**
- Am: x02210
- Am/G: 302210
- Fmaj7: x33210 (or 1x3210 for lower voicing)
- E: 022100
- Dm: xx0231

**Picking pattern (6/8 feel within 4/4):**
```
Beat:  1  +  2  +  3  +  4  +
       T  i  m  a  m  i  m  a
       (root on 1, let strings ring)
```

---

### VERSE 1 (8 bars - gentle strumming, down-focused)
```
| Am      | C       | G       | Em      |
| Am      | C       | Fmaj7   | E       |
```

**Strumming pattern:**
```
Beat:  1   2   3   4
       D   -   D   du
       (sparse, let chords breathe)
```

**Lyrics - Verse 1:**
```
I had nowhere else to run, so I ran to you
You were lying there, your eyes already knew
Just a lamp between us, no world outside
For a moment there I swore that I'd arrived
```

---

### VERSE 2 (8 bars - same progression, slight crescendo)
```
| Am      | C       | G       | Em      |
| Am      | C       | Fmaj7   | E       |
```

**Lyrics - Verse 2:**
```
I reached for you, desperate to feel your breath
First time I saw all of you, nothing left
Shea butter sweet, still lingering in your hair
I finally fell apart with you right there
```

---

### CHORUS/HOOK (8 bars - fuller strumming, more open)
This is the emotional misdirection - sounds like the overwhelming feeling of first love.

```
| Fmaj7   | C       | Am      | G       |
| Dm      | Am      | Esus4   | E       |
```

**Strumming pattern:**
```
Beat:  1   2   3   4
       D   DU  -   DU
       (more movement, still restrained)
```

**Lyrics - Chorus:**
```
And I swear I couldn't breathe
Couldn't stand, fell to my knees
Everything I'd never known
Hit me there, cut to the bone
```

**Alternative/Additional Hook Lines (choose what fits):**
```
This is what they write about
This is what I'd lived without
Now I know what poets mean
The most [beautiful/devastating] thing I've seen
```

---

### INSTRUMENTAL BREAK (4 bars - arpeggiated with tension)
```
| Am      | Bdim    | C       | C#dim   |
```

**Voicings for diminished:**
- Bdim: x2343x
- C#dim: x4565x

**Playing style:** Slow arpeggios, let the dissonance of diminished chords create unease. This is where something starts to feel wrong.

---

### VERSE 3 (8 bars - stripped back, almost whispered)
```
| Am      | C       | G       | Em      |
| Am      | C       | Fmaj7   | E       |
```

**Lyrics - Verse 3:**
```
Thirty minutes passed and not a single sound
The kind of quiet where you lose all solid ground
I could've stayed forever in that space
Just me and you, your eyes still on my face
```

---

### BRIDGE (4 bars - sustained, building to reveal)
```
| Dm      | Am      | Dm      | E (let ring) |
```

**Lyrics - Bridge:**
```
Your silence left a hole
Your absence still takes its toll
```

---

### FINAL VERSE / REVEAL (4 bars - devastation)
The turn. Everything recontextualizes.

```
| Am      | Fmaj7   | Dm      | E       | Am (hold) |
```

**Playing style:** Stripped to almost nothing. Single notes, maybe just bass notes of chords. Voice carries it.

**Lyrics - Final:**
```
But you weren't waiting — you were already gone
Now the woman of my dreams won't let me move on
```

**Alternative final couplet options:**
```
You weren't sleeping — you were already gone
Now the woman of my dreams is the one who haunts them

But you weren't resting — you were already gone  
The woman of my dreams won't leave me alone
```

---

### OUTRO (4-8 bars - fingerpicked, fading)
Return to intro progression, slower, fading out.
```
| Am      | Am/G    | Fmaj7   | E       |
| Am      | (hold and fade)
```

---

## FULL SONG MAP

```
[0:00]  INTRO - 8 bars (~26 sec) - fingerpicked
[0:26]  VERSE 1 - 8 bars (~26 sec) - light strum
[0:52]  VERSE 2 - 8 bars (~26 sec) - building slightly  
[1:18]  CHORUS - 8 bars (~26 sec) - fuller
[1:44]  INSTRUMENTAL - 4 bars (~13 sec) - arpeggiated, diminished tension
[1:57]  VERSE 3 - 8 bars (~26 sec) - pulled back
[2:23]  BRIDGE - 4 bars (~13 sec) - sustained
[2:36]  FINAL/REVEAL - 5 bars (~17 sec) - stripped, devastating
[2:53]  OUTRO - 4-8 bars (~13-26 sec) - fingerpicked fade

TOTAL: approximately 3:00 - 3:30
```

---

## RECORDING NOTES

**Guitar Setup:**
- Acoustic guitar, recorded in stereo (two tracks: GUITAR L, GUITAR R)
- Light compression, room reverb
- Keep it intimate - close mic positioning

**Vocal Approach:**
- Conversational, not performative
- Let the words carry the weight
- Final verse almost spoken/whispered
- No vocal runs or embellishments - this is about the story

**Mix Philosophy:**
- Guitar slightly left/right panned for width
- Vocal center, present but not harsh
- Room, not studio - this should feel like 2am on a couch

---

## LOGIC PRO SETUP

See `setup_logic_project.py` for automated project creation.
See `generate_midi.py` for MIDI reference tracks.

**Manual Setup:**
1. Create new project: 72 BPM, 4/4, Key of A minor
2. Create 3 tracks:
   - Track 1: "Guitar L" - Audio track, input 1, pan -30
   - Track 2: "Guitar R" - Audio track, input 2, pan +30
   - Track 3: "Vocal" - Audio track, input 1 (or preferred), pan center
3. Add markers for each section (see song map above)
4. Import MIDI reference track to guide recording

---

## FOR KELLY

Blonde hair. Green eyes. Flannel and tank tops. A necklace.
A loud laugh that changed her whole face.
A couch in a dark, maroon, cozy room.
Shea butter in her hair.
Thirty minutes of silence.

This song is so her face isn't just behind that door anymore.
This song is so she's sitting by the dartboards again, smiling.

