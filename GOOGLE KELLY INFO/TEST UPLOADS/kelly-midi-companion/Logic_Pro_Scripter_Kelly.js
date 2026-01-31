// ============================================================================
// Kelly MIDI Companion - Logic Pro Scripter Script
// ============================================================================
// 
// This script integrates Kelly's emotion thesaurus and chord progressions
// into Logic Pro's Scripter MIDI plugin.
//
// Usage:
// 1. Open Logic Pro
// 2. Create a Software Instrument track
// 3. Add Scripter MIDI FX plugin
// 4. Paste this entire script
// 5. Adjust emotion parameters in the UI
//
// ============================================================================

var NeedsTimingInfo = true;
var PluginParameters = [];

// ============================================================================
// EMOTION THESAURUS DATA (from JSON files)
// ============================================================================

var EmotionThesaurus = {
    // Emotion categories with valence/arousal/intensity mapping
    emotions: {
        // SADNESS CLUSTER
        "grief": { valence: -0.9, arousal: 0.3, intensity: 1.0, mode: "minor", tempo: 0.6 },
        "melancholy": { valence: -0.6, arousal: 0.2, intensity: 0.5, mode: "aeolian", tempo: 0.7 },
        "despair": { valence: -1.0, arousal: 0.4, intensity: 0.9, mode: "locrian", tempo: 0.5 },
        "sorrow": { valence: -0.8, arousal: 0.3, intensity: 0.7, mode: "minor", tempo: 0.65 },
        "wistfulness": { valence: -0.3, arousal: 0.2, intensity: 0.3, mode: "dorian", tempo: 0.75 },
        "longing": { valence: -0.4, arousal: 0.5, intensity: 0.6, mode: "minor", tempo: 0.8 },
        
        // ANGER CLUSTER
        "rage": { valence: -0.8, arousal: 1.0, intensity: 1.0, mode: "phrygian", tempo: 1.4 },
        "fury": { valence: -0.9, arousal: 0.95, intensity: 0.95, mode: "locrian", tempo: 1.5 },
        "frustration": { valence: -0.5, arousal: 0.7, intensity: 0.6, mode: "minor", tempo: 1.2 },
        "irritation": { valence: -0.3, arousal: 0.5, intensity: 0.4, mode: "minor", tempo: 1.1 },
        "resentment": { valence: -0.6, arousal: 0.5, intensity: 0.5, mode: "phrygian", tempo: 1.0 },
        
        // FEAR CLUSTER
        "terror": { valence: -0.9, arousal: 1.0, intensity: 1.0, mode: "locrian", tempo: 1.6 },
        "panic": { valence: -0.8, arousal: 0.95, intensity: 0.95, mode: "phrygian", tempo: 1.5 },
        "anxiety": { valence: -0.5, arousal: 0.7, intensity: 0.6, mode: "minor", tempo: 1.3 },
        "worry": { valence: -0.3, arousal: 0.5, intensity: 0.4, mode: "aeolian", tempo: 1.1 },
        "dread": { valence: -0.7, arousal: 0.6, intensity: 0.6, mode: "locrian", tempo: 1.2 },
        
        // JOY CLUSTER
        "ecstasy": { valence: 1.0, arousal: 1.0, intensity: 1.0, mode: "lydian", tempo: 1.8 },
        "elation": { valence: 0.9, arousal: 0.9, intensity: 0.9, mode: "ionian", tempo: 1.6 },
        "happiness": { valence: 0.7, arousal: 0.7, intensity: 0.7, mode: "major", tempo: 1.4 },
        "contentment": { valence: 0.5, arousal: 0.6, intensity: 0.3, mode: "major", tempo: 1.0 },
        "serenity": { valence: 0.4, arousal: 0.5, intensity: 0.2, mode: "major", tempo: 0.9 },
        "peace": { valence: 0.3, arousal: 0.4, intensity: 0.15, mode: "major", tempo: 0.85 },
        "calm": { valence: 0.2, arousal: 0.3, intensity: 0.1, mode: "major", tempo: 0.8 },
        "relief": { valence: 0.6, arousal: 0.5, intensity: 0.5, mode: "major", tempo: 1.1 },
        "gratitude": { valence: 0.6, arousal: 0.6, intensity: 0.4, mode: "major", tempo: 1.2 },
        "hope": { valence: 0.7, arousal: 0.6, intensity: 0.6, mode: "major", tempo: 1.3 },
        
        // TRUST/LOVE CLUSTER
        "love": { valence: 0.9, arousal: 0.8, intensity: 0.7, mode: "major", tempo: 1.2 },
        "adoration": { valence: 0.85, arousal: 0.85, intensity: 0.75, mode: "lydian", tempo: 1.3 },
        "affection": { valence: 0.6, arousal: 0.6, intensity: 0.5, mode: "major", tempo: 1.1 },
        "tenderness": { valence: 0.5, arousal: 0.5, intensity: 0.4, mode: "major", tempo: 1.0 },
        "warmth": { valence: 0.4, arousal: 0.5, intensity: 0.35, mode: "major", tempo: 0.95 },
        
        // ANTICIPATION CLUSTER
        "excitement": { valence: 0.9, arousal: 0.7, intensity: 0.9, mode: "major", tempo: 1.5 },
        "eagerness": { valence: 0.7, arousal: 0.6, intensity: 0.75, mode: "major", tempo: 1.3 },
        "curiosity": { valence: 0.5, arousal: 0.4, intensity: 0.6, mode: "mixolydian", tempo: 1.1 },
        
        // SURPRISE CLUSTER
        "amazement": { valence: 0.9, arousal: 0.5, intensity: 0.95, mode: "lydian", tempo: 1.4 },
        "astonishment": { valence: 0.85, arousal: 0.4, intensity: 0.9, mode: "major", tempo: 1.3 },
        "shock": { valence: 0.8, arousal: -0.1, intensity: 0.85, mode: "minor", tempo: 1.5 },
        "wonder": { valence: 0.6, arousal: 0.6, intensity: 0.6, mode: "lydian", tempo: 1.2 },
        
        // DISGUST CLUSTER
        "revulsion": { valence: -0.8, arousal: 0.7, intensity: 0.9, mode: "locrian", tempo: 1.1 },
        "loathing": { valence: -0.85, arousal: 0.6, intensity: 0.85, mode: "phrygian", tempo: 1.0 },
        "contempt": { valence: -0.6, arousal: 0.5, intensity: 0.7, mode: "minor", tempo: 0.9 },
        
        // COMPLEX EMOTIONS
        "bittersweetness": { valence: 0.1, arousal: 0.4, intensity: 0.6, mode: "dorian", tempo: 1.0 },
        "nostalgia": { valence: 0.2, arousal: 0.3, intensity: 0.5, mode: "aeolian", tempo: 0.9 },
        "catharsis": { valence: 0.8, arousal: 0.3, intensity: 0.7, mode: "major", tempo: 1.1 },
        "yearning": { valence: -0.2, arousal: 0.6, intensity: 0.7, mode: "minor", tempo: 1.2 },
        "ambivalence": { valence: 0.0, arousal: 0.5, intensity: 0.5, mode: "mixolydian", tempo: 1.0 }
    },
    
    // Get emotion data by name
    getEmotion: function(name) {
        var key = name.toLowerCase();
        return this.emotions[key] || this.emotions["neutral"] || { 
            valence: 0.0, arousal: 0.5, intensity: 0.5, mode: "major", tempo: 1.0 
        };
    },
    
    // Find nearest emotion by valence/arousal
    findNearest: function(valence, arousal) {
        var nearest = null;
        var minDist = Infinity;
        
        for (var key in this.emotions) {
            var e = this.emotions[key];
            var dist = Math.sqrt(
                Math.pow(e.valence - valence, 2) + 
                Math.pow(e.arousal - arousal, 2)
            );
            if (dist < minDist) {
                minDist = dist;
                nearest = { name: key, data: e };
            }
        }
        return nearest;
    }
};

// ============================================================================
// CHORD PROGRESSIONS (from chord_progressions.json and common_progressions.json)
// ============================================================================

var ChordProgressions = {
    // Common progressions by emotional context
    progressions: {
        // SAD/MELANCHOLIC (from sad.json emotional context)
        "sad_i_vi_iii_vii": { 
            chords: [0, 9, 4, 11], // C, Am, Em, Bm (in C)
            name: "Sad i-VI-III-VII",
            emotion: "melancholy",
            numerals: ["i", "VI", "III", "VII"],
            intervals: [0, 9, 4, 11]
        },
        "grief_i_iv_i_v": {
            chords: [0, 5, 0, 7], // C, Fm, C, G (in C minor)
            name: "Grief i-iv-i-V",
            emotion: "grief",
            numerals: ["i", "iv", "i", "V"],
            intervals: [0, 5, 0, 7]
        },
        "melancholy_i_vii_vi_vii": {
            chords: [0, 11, 9, 11], // C, Bm, Am, Bm
            name: "Melancholy i-VII-VI-VII",
            emotion: "melancholy",
            numerals: ["i", "VII", "VI", "VII"],
            intervals: [0, 11, 9, 11]
        },
        "sorrow_vi_iv_i_v": {
            chords: [9, 5, 0, 7], // Am, F, C, G (Axis variation)
            name: "Sorrow vi-IV-I-V",
            emotion: "sorrow",
            numerals: ["vi", "IV", "I", "V"],
            intervals: [9, 5, 0, 7]
        },
        
        // ANGRY/INTENSE (from anger.json)
        "rage_i_bii_i_v": {
            chords: [0, 1, 0, 7], // C, Db, C, G (tritone substitution)
            name: "Rage i-bII-i-V",
            emotion: "rage",
            numerals: ["i", "bII", "i", "V"],
            intervals: [0, 1, 0, 7]
        },
        "tension_i_iv_bvi_v": {
            chords: [0, 5, 8, 7], // C, Fm, Ab, G
            name: "Tension i-iv-bVI-V",
            emotion: "frustration",
            numerals: ["i", "iv", "bVI", "V"],
            intervals: [0, 5, 8, 7]
        },
        "fury_i_bvii_iv_i": {
            chords: [0, 10, 5, 0], // C, Bb, F, C (Rock Mixolydian)
            name: "Fury I-bVII-IV-I",
            emotion: "fury",
            numerals: ["I", "bVII", "IV", "I"],
            intervals: [0, 10, 5, 0]
        },
        
        // FEAR/ANXIOUS (from fear.json)
        "anxiety_ii_v_i_vi": {
            chords: [2, 7, 0, 9], // Dm, G, C, Am
            name: "Anxiety ii-V-i-vi",
            emotion: "anxiety",
            numerals: ["ii", "V", "i", "vi"],
            intervals: [2, 7, 0, 9]
        },
        "dread_i_bvi_bvii_i": {
            chords: [0, 8, 10, 0], // C, Ab, Bb, C
            name: "Dread i-bVI-bVII-i",
            emotion: "dread",
            numerals: ["i", "bVI", "bVII", "i"],
            intervals: [0, 8, 10, 0]
        },
        "terror_i_dim_vii_dim": {
            chords: [0, 3, 11], // C, Eb, B (diminished)
            name: "Terror i-dim-vii-dim",
            emotion: "terror",
            numerals: ["i", "dim", "vii", "dim"],
            intervals: [0, 3, 11]
        },
        
        // HOPEFUL (from joy.json - hope)
        "hope_i_v_vi_iv": {
            chords: [0, 7, 9, 5], // C, G, Am, F (Axis of Awesome)
            name: "Hope I-V-vi-IV",
            emotion: "hope",
            numerals: ["I", "V", "vi", "IV"],
            intervals: [0, 7, 9, 5]
        },
        "rising_i_iv_vi_v": {
            chords: [0, 5, 9, 7], // C, F, Am, G
            name: "Rising I-IV-vi-V",
            emotion: "excitement",
            numerals: ["I", "IV", "vi", "V"],
            intervals: [0, 5, 9, 7]
        },
        "gratitude_i_vi_iv_v": {
            chords: [0, 9, 5, 7], // C, Am, F, G (50s progression)
            name: "Gratitude I-vi-IV-V",
            emotion: "gratitude",
            numerals: ["I", "vi", "IV", "V"],
            intervals: [0, 9, 5, 7]
        },
        
        // JOYFUL (from joy.json)
        "joy_i_v_vi_iii": {
            chords: [0, 7, 9, 4], // C, G, Am, Em
            name: "Joy I-V-vi-iii",
            emotion: "happiness",
            numerals: ["I", "V", "vi", "iii"],
            intervals: [0, 7, 9, 4]
        },
        "elation_i_iv_v_iv": {
            chords: [0, 5, 7, 5], // C, F, G, F
            name: "Elation I-IV-V-IV",
            emotion: "elation",
            numerals: ["I", "IV", "V", "IV"],
            intervals: [0, 5, 7, 5]
        },
        "ecstasy_i_v_iii_vi": {
            chords: [0, 7, 4, 9], // C, G, Em, Am
            name: "Ecstasy I-V-iii-vi",
            emotion: "ecstasy",
            numerals: ["I", "V", "iii", "vi"],
            intervals: [0, 7, 4, 9]
        },
        
        // LOVE/TRUST (from joy.json - love)
        "love_i_vi_ii_v": {
            chords: [0, 9, 2, 7], // C, Am, Dm, G
            name: "Love I-vi-ii-V",
            emotion: "love",
            numerals: ["I", "vi", "ii", "V"],
            intervals: [0, 9, 2, 7]
        },
        "tenderness_i_iii_vi_iv": {
            chords: [0, 4, 9, 5], // C, Em, Am, F
            name: "Tenderness I-iii-vi-IV",
            emotion: "tenderness",
            numerals: ["I", "iii", "vi", "IV"],
            intervals: [0, 4, 9, 5]
        },
        
        // SURPRISE (from surprise.json)
        "amazement_i_vii_iv_v": {
            chords: [0, 11, 5, 7], // C, B, F, G
            name: "Amazement I-VII-IV-V",
            emotion: "amazement",
            numerals: ["I", "VII", "IV", "V"],
            intervals: [0, 11, 5, 7]
        },
        "wonder_i_iv_v_iii": {
            chords: [0, 5, 7, 4], // C, F, G, Em
            name: "Wonder I-IV-V-iii",
            emotion: "wonder",
            numerals: ["I", "IV", "V", "iii"],
            intervals: [0, 5, 7, 4]
        },
        
        // DISGUST (from disgust.json)
        "revulsion_i_bii_biii_i": {
            chords: [0, 1, 3, 0], // C, Db, Eb, C
            name: "Revulsion i-bII-bIII-i",
            emotion: "revulsion",
            numerals: ["i", "bII", "bIII", "i"],
            intervals: [0, 1, 3, 0]
        },
        
        // COMPLEX EMOTIONS
        "bittersweet_i_vi_iv_v": {
            chords: [0, 9, 5, 7], // C, Am, F, G (major with minor vi)
            name: "Bittersweet I-vi-IV-V",
            emotion: "bittersweetness",
            numerals: ["I", "vi", "IV", "V"],
            intervals: [0, 9, 5, 7]
        },
        "nostalgia_i_vi_ii_v": {
            chords: [0, 9, 2, 7], // C, Am, Dm, G
            name: "Nostalgia I-vi-ii-V",
            emotion: "nostalgia",
            numerals: ["I", "vi", "ii", "V"],
            intervals: [0, 9, 2, 7]
        },
        "yearning_vi_i_iv_v": {
            chords: [9, 0, 5, 7], // Am, C, F, G
            name: "Yearning vi-I-IV-V",
            emotion: "yearning",
            numerals: ["vi", "I", "IV", "V"],
            intervals: [9, 0, 5, 7]
        }
    },
    
    // Get progression by emotion
    getByEmotion: function(emotionName) {
        var emotion = emotionName.toLowerCase();
        for (var key in this.progressions) {
            if (this.progressions[key].emotion === emotion) {
                return this.progressions[key];
            }
        }
        // Try partial match
        for (var key in this.progressions) {
            if (this.progressions[key].emotion.indexOf(emotion) !== -1 || 
                emotion.indexOf(this.progressions[key].emotion) !== -1) {
                return this.progressions[key];
            }
        }
        // Default to hope progression
        return this.progressions["hope_i_v_vi_iv"];
    },
    
    // Get progression by name
    getByName: function(name) {
        return this.progressions[name] || this.progressions["hope_i_v_vi_iv"];
    },
    
    // Get all progressions for an emotion category
    getByCategory: function(category) {
        var results = [];
        for (var key in this.progressions) {
            if (this.progressions[key].emotion.indexOf(category) !== -1) {
                results.push(this.progressions[key]);
            }
        }
        return results;
    }
};

// ============================================================================
// PLUGIN PARAMETERS
// ============================================================================

PluginParameters.push({
    name: "Emotion",
    type: "menu",
    valueStrings: Object.keys(EmotionThesaurus.emotions).sort(),
    defaultValue: "hope"
});

PluginParameters.push({
    name: "Valence",
    type: "lin",
    minValue: -1.0,
    maxValue: 1.0,
    numberOfSteps: 200,
    defaultValue: 0.0
});

PluginParameters.push({
    name: "Arousal",
    type: "lin",
    minValue: 0.0,
    maxValue: 1.0,
    numberOfSteps: 100,
    defaultValue: 0.5
});

PluginParameters.push({
    name: "Intensity",
    type: "lin",
    minValue: 0.0,
    maxValue: 1.0,
    numberOfSteps: 100,
    defaultValue: 0.7
});

PluginParameters.push({
    name: "Root Note",
    type: "menu",
    valueStrings: ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    defaultValue: "C"
});

PluginParameters.push({
    name: "Progression",
    type: "menu",
    valueStrings: Object.keys(ChordProgressions.progressions).sort(),
    defaultValue: "hope_i_v_vi_iv"
});

PluginParameters.push({
    name: "Chord Length (beats)",
    type: "lin",
    minValue: 1.0,
    maxValue: 8.0,
    numberOfSteps: 7,
    defaultValue: 2.0
});

PluginParameters.push({
    name: "Humanize",
    type: "lin",
    minValue: 0.0,
    maxValue: 1.0,
    numberOfSteps: 100,
    defaultValue: 0.4
});

PluginParameters.push({
    name: "Generate Mode",
    type: "menu",
    valueStrings: ["Auto (from emotion)", "Manual progression"],
    defaultValue: "Auto (from emotion)"
});

PluginParameters.push({
    name: "Genre/Groove",
    type: "menu",
    valueStrings: Object.keys(GenrePocketMaps.genres).sort(),
    defaultValue: "straight"
});

PluginParameters.push({
    name: "Add 7th",
    type: "lin",
    minValue: 0.0,
    maxValue: 1.0,
    numberOfSteps: 100,
    defaultValue: 0.5
});

// ============================================================================
// STATE
// ============================================================================

var currentEmotion = "hope";
var currentProgression = null;
var currentRoot = 60; // C4 MIDI note
var chordLength = 2.0;
var progressionIndex = 0;
var lastBeat = 0;
var humanizeAmount = 0.4;

// ============================================================================
// MAIN PROCESSING
// ============================================================================

function ProcessMIDI() {
    var timingInfo = GetTimingInfo();
    
    if (!timingInfo.playing) {
        progressionIndex = 0;
        return;
    }
    
    // Get current parameters
    var emotionParam = GetParameter("Emotion");
    var valenceParam = GetParameter("Valence");
    var arousalParam = GetParameter("Arousal");
    var intensityParam = GetParameter("Intensity");
    var rootParam = GetParameter("Root Note");
    var progressionParam = GetParameter("Progression");
    var chordLengthParam = GetParameter("Chord Length (beats)");
    var humanizeParam = GetParameter("Humanize");
    var generateMode = GetParameter("Generate Mode");
    var genreParam = GetParameter("Genre/Groove");
    var add7thParam = GetParameter("Add 7th");
    
    // Update state
    currentRoot = 60 + (rootParam * 12); // Convert menu index to MIDI note
    chordLength = chordLengthParam;
    humanizeAmount = humanizeParam;
    
    // Get emotion data
    var emotionData;
    if (generateMode === 0) {
        // Auto mode - use selected emotion
        var emotionNames = Object.keys(EmotionThesaurus.emotions).sort();
        currentEmotion = emotionNames[emotionParam];
        emotionData = EmotionThesaurus.getEmotion(currentEmotion);
        
        // Override with manual valence/arousal if significantly different
        if (Math.abs(emotionData.valence - valenceParam) > 0.2 || 
            Math.abs(emotionData.arousal - arousalParam) > 0.2) {
            emotionData = EmotionThesaurus.findNearest(valenceParam, arousalParam).data;
        }
        
        // Get progression from emotion
        currentProgression = ChordProgressions.getByEmotion(currentEmotion);
    } else {
        // Manual mode - use selected progression
        var progressionNames = Object.keys(ChordProgressions.progressions).sort();
        currentProgression = ChordProgressions.getByName(progressionNames[progressionParam]);
        emotionData = EmotionThesaurus.getEmotion(currentProgression.emotion);
    }
    
    // Calculate when to trigger next chord
    var currentBeat = timingInfo.blockStartBeat;
    var beatsSinceLastChord = currentBeat - lastBeat;
    
    if (beatsSinceLastChord >= chordLength || lastBeat === 0) {
        // Get genre data for humanization
        var genreNames = Object.keys(GenrePocketMaps.genres).sort();
        var genre = GenrePocketMaps.getGenre(genreNames[genreParam]);
        
        // Get chord interval from progression
        var chordInterval = currentProgression.chords[progressionIndex % currentProgression.chords.length];
        var chordRoot = currentRoot + chordInterval;
        
        // Determine chord quality from emotion and progression
        var isMinor = emotionData.valence < 0;
        var chordQuality = isMinor ? "min" : "maj";
        
        // Build chord triad
        var chordNotes = buildChord(chordRoot, chordInterval, chordQuality);
        
        // Add seventh based on parameter and intensity
        if (add7thParam > 0.3 || intensityParam > 0.6) {
            if (isMinor) {
                chordNotes.push(chordRoot + 10); // Minor seventh
            } else {
                chordNotes.push(chordRoot + 11); // Major seventh
            }
        }
        
        // Send chord with humanization
        for (var i = 0; i < chordNotes.length; i++) {
            var humanized = applyHumanization(chordNotes[i], humanizeAmount, genreNames[genreParam]);
            
            var velocity = Math.round(
                genre.velocityRange[0] + 
                (intensityParam * (genre.velocityRange[1] - genre.velocityRange[0]))
            );
            
            var event = new NoteOn;
            event.pitch = humanized.note;
            event.velocity = velocity;
            event.send();
        }
        
        // Update state
        progressionIndex++;
        lastBeat = currentBeat;
    }
    
    // Handle note offs (simplified - hold for chord length)
    // In a full implementation, you'd track individual notes
}

// ============================================================================
// GENRE POCKET MAPS (from genre_pocket_maps.json)
// ============================================================================

var GenrePocketMaps = {
    genres: {
        "funk": {
            swing: 0.15,
            timingOffset: 12, // ms behind beat
            velocityRange: [100, 120],
            description: "Deep pocket, emphasis on 2 and 4"
        },
        "jazz_swing": {
            swing: 0.67, // Triplet swing
            timingOffset: 5,
            velocityRange: [80, 110],
            description: "Classic triplet swing feel"
        },
        "neo_soul": {
            swing: 0.25,
            timingOffset: 15, // Laid back
            velocityRange: [70, 100],
            description: "Laid back, behind the beat"
        },
        "rock": {
            swing: 0.0,
            timingOffset: -5, // Slightly ahead
            velocityRange: [90, 127],
            description: "Straight, driving, on the beat"
        },
        "lo_fi": {
            swing: 0.1,
            timingOffset: 20, // Very laid back
            velocityRange: [60, 90],
            description: "Very laid back, imperfect"
        },
        "straight": {
            swing: 0.0,
            timingOffset: 0,
            velocityRange: [80, 100],
            description: "Perfect grid, no swing"
        }
    },
    
    getGenre: function(name) {
        return this.genres[name] || this.genres["straight"];
    }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getNoteName(midiNote) {
    var names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    return names[midiNote % 12] + Math.floor(midiNote / 12);
}

function buildChord(rootNote, interval, quality) {
    // Build chord from root note (interval is already applied to get rootNote)
    var notes = [rootNote];
    
    if (quality === "maj" || quality === "major") {
        notes.push(rootNote + 4);  // Major third
        notes.push(rootNote + 7);  // Perfect fifth
    } else if (quality === "min" || quality === "minor") {
        notes.push(rootNote + 3);  // Minor third
        notes.push(rootNote + 7);  // Perfect fifth
    } else if (quality === "dim") {
        notes.push(rootNote + 3);  // Minor third
        notes.push(rootNote + 6);  // Diminished fifth
    } else if (quality === "aug") {
        notes.push(rootNote + 4);  // Major third
        notes.push(rootNote + 8);  // Augmented fifth
    } else {
        // Default to major
        notes.push(rootNote + 4);
        notes.push(rootNote + 7);
    }
    
    return notes;
}

function applyHumanization(note, amount, genre) {
    var genreData = GenrePocketMaps.getGenre(genre);
    // Pitch jitter (small microtonal variation)
    var pitchJitter = (Math.random() - 0.5) * amount * 0.1; // Small variation
    var humanizedNote = Math.max(0, Math.min(127, Math.round(note + pitchJitter)));
    return {
        note: humanizedNote,
        timing: (genreData.timingOffset / 1000.0) * amount // Timing offset in beats
    };
}

// ============================================================================
// INITIALIZATION
// ============================================================================

function Reset() {
    progressionIndex = 0;
    lastBeat = 0;
    currentProgression = ChordProgressions.getByName("hope_i_v_vi_iv");
}

// Initialize
Reset();

