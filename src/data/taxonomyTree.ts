/**
 * Complete Musical Command Taxonomy
 *
 * Every possible musical command/technique organized as a browseable tree.
 * Each leaf node contributes probability distributions (not fixed values)
 * and declares which context clusters it participates in.
 */

import type { ParamDistribution } from '../types/Interpretation';

export interface TaxonomyNode {
  id: string;
  label: string;
  labelKey?: string;
  children?: TaxonomyNode[];
  paramContributions?: Record<string, ParamDistribution>;
  clusterMemberships?: string[];
  keywords: string[];
  description?: string;
}

// Shorthand helpers
const dw = (weights: Record<string, number>): ParamDistribution => ({ type: 'discrete', weights });
const rn = (min: number, max: number, bias: number): ParamDistribution => ({ type: 'range', min, max, bias });

export const TAXONOMY_TREE: TaxonomyNode[] = [
  // ═══ 1. HARMONY ═══
  {
    id: 'harmony', label: 'Harmony', keywords: ['harmony', 'chords', 'keys', 'tonality'],
    children: [
      {
        id: 'harmony.keys', label: 'Keys', keywords: ['key', 'root note', 'tonic'],
        children: ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B'].map(k => ({
          id: `harmony.keys.${k.replace('#', 'sharp').replace('b', 'flat').toLowerCase()}`,
          label: k, keywords: [k.toLowerCase(), `key of ${k.toLowerCase()}`],
          paramContributions: { key: dw({ [k]: 1.0 }) },
        })),
      },
      {
        id: 'harmony.scales', label: 'Scales & Modes', keywords: ['scales', 'modes'],
        children: [
          { id: 'harmony.scales.major', label: 'Major (Ionian)', keywords: ['major', 'ionian', 'happy', 'bright'], paramContributions: { mode_weights: dw({ major: 0.85, lydian: 0.1, mixolydian: 0.05 }) }, clusterMemberships: ['pop-upbeat', 'pop-ballad', 'gospel', 'country-traditional', 'bossa-nova'] },
          { id: 'harmony.scales.minor', label: 'Minor (Aeolian)', keywords: ['minor', 'aeolian', 'sad', 'dark', 'natural minor'], paramContributions: { mode_weights: dw({ minor: 0.85, dorian: 0.1, phrygian: 0.05 }) }, clusterMemberships: ['metal-shred', 'punk-rock', 'grunge', 'edm-drop', 'hip-hop-trap', 'post-rock', 'cinematic-epic'] },
          { id: 'harmony.scales.dorian', label: 'Dorian', keywords: ['dorian', 'jazzy', 'soulful', 'minor but bright'], paramContributions: { mode_weights: dw({ dorian: 0.85, minor: 0.1, mixolydian: 0.05 }) }, clusterMemberships: ['metal-shred', 'blues-slow', 'jazz-modal', 'funk-groove', 'lo-fi-chill', 'neo-soul', 'synthwave-retro', 'reggae', 'world-afrobeat'] },
          { id: 'harmony.scales.phrygian', label: 'Phrygian', keywords: ['phrygian', 'spanish', 'metal', 'exotic', 'dark'], paramContributions: { mode_weights: dw({ phrygian: 0.85, minor: 0.1, locrian: 0.05 }) }, clusterMemberships: ['metal-shred', 'punk-rock', 'edm-drop', 'hip-hop-trap'] },
          { id: 'harmony.scales.lydian', label: 'Lydian', keywords: ['lydian', 'dreamy', 'bright', 'floating', 'ethereal'], paramContributions: { mode_weights: dw({ lydian: 0.85, major: 0.1, mixolydian: 0.05 }) }, clusterMemberships: ['ambient-pad', 'jazz-modal', 'bossa-nova', 'prog-rock'] },
          { id: 'harmony.scales.mixolydian', label: 'Mixolydian', keywords: ['mixolydian', 'bluesy', 'dominant', 'rock'], paramContributions: { mode_weights: dw({ mixolydian: 0.85, major: 0.1, dorian: 0.05 }) }, clusterMemberships: ['blues-chicago', 'funk-groove', 'country-traditional', 'gospel', 'neo-soul', 'world-afrobeat'] },
          { id: 'harmony.scales.locrian', label: 'Locrian', keywords: ['locrian', 'diminished', 'unstable', 'dissonant'], paramContributions: { mode_weights: dw({ locrian: 0.85, phrygian: 0.15 }) }, clusterMemberships: ['metal-shred'] },
          { id: 'harmony.scales.harmonic-minor', label: 'Harmonic Minor', keywords: ['harmonic minor', 'classical', 'middle eastern'], paramContributions: { mode_weights: dw({ harmonic_minor: 0.85, minor: 0.15 }) }, clusterMemberships: ['metal-shred', 'classical-romantic'] },
          { id: 'harmony.scales.melodic-minor', label: 'Melodic Minor', keywords: ['melodic minor', 'jazz minor'], paramContributions: { mode_weights: dw({ melodic_minor: 0.85, minor: 0.15 }) }, clusterMemberships: ['jazz-modal', 'jazz-bebop'] },
          { id: 'harmony.scales.pentatonic-major', label: 'Pentatonic Major', keywords: ['pentatonic major', 'country', 'folk', 'simple'], paramContributions: { mode_weights: dw({ pentatonic_major: 0.85, major: 0.15 }) }, clusterMemberships: ['country-traditional', 'pop-upbeat', 'indie-folk'] },
          { id: 'harmony.scales.pentatonic-minor', label: 'Pentatonic Minor', keywords: ['pentatonic', 'pentatonic minor', 'blues', 'rock'], paramContributions: { mode_weights: dw({ pentatonic_minor: 0.85, minor: 0.15 }) }, clusterMemberships: ['blues-slow', 'blues-chicago', 'hip-hop-boom-bap'] },
          { id: 'harmony.scales.blues', label: 'Blues Scale', keywords: ['blues scale', 'bluesy', 'blue note'], paramContributions: { mode_weights: dw({ blues: 0.85, pentatonic_minor: 0.15 }) }, clusterMemberships: ['blues-slow', 'blues-chicago'] },
          { id: 'harmony.scales.whole-tone', label: 'Whole Tone', keywords: ['whole tone', 'augmented', 'dreamy', 'impressionist'], paramContributions: { mode_weights: dw({ whole_tone: 0.9, lydian: 0.1 }) }, clusterMemberships: ['ambient-pad'] },
          { id: 'harmony.scales.chromatic', label: 'Chromatic', keywords: ['chromatic', 'all notes', 'atonal'], paramContributions: { mode_weights: dw({ chromatic: 1.0 }) }, clusterMemberships: ['jazz-bebop'] },
          { id: 'harmony.scales.diminished', label: 'Diminished', keywords: ['diminished', 'octatonic', 'symmetric'], paramContributions: { mode_weights: dw({ diminished: 0.9, locrian: 0.1 }) }, clusterMemberships: ['jazz-bebop', 'metal-shred'] },
        ],
      },
      {
        id: 'harmony.chords', label: 'Chords', keywords: ['chords', 'voicings'],
        children: [
          { id: 'harmony.chords.major-triad', label: 'Major Triad', keywords: ['major chord', 'major triad'], paramContributions: { harmonic_tension: rn(0, 0.2, 0.1) }, clusterMemberships: ['pop-upbeat', 'country-traditional'] },
          { id: 'harmony.chords.minor-triad', label: 'Minor Triad', keywords: ['minor chord', 'minor triad'], paramContributions: { harmonic_tension: rn(0.1, 0.3, 0.2) }, clusterMemberships: ['pop-ballad', 'indie-folk'] },
          { id: 'harmony.chords.dim-triad', label: 'Diminished Triad', keywords: ['diminished', 'dim'], paramContributions: { harmonic_tension: rn(0.5, 0.8, 0.65) }, clusterMemberships: ['classical-romantic'] },
          { id: 'harmony.chords.aug-triad', label: 'Augmented Triad', keywords: ['augmented', 'aug'], paramContributions: { harmonic_tension: rn(0.4, 0.7, 0.55) }, clusterMemberships: ['jazz-modal'] },
          { id: 'harmony.chords.maj7', label: 'Major 7th', keywords: ['maj7', 'major seventh', 'dreamy'], paramContributions: { harmonic_tension: rn(0.1, 0.3, 0.2) }, clusterMemberships: ['jazz-modal', 'bossa-nova', 'neo-soul'] },
          { id: 'harmony.chords.min7', label: 'Minor 7th', keywords: ['min7', 'minor seventh'], paramContributions: { harmonic_tension: rn(0.15, 0.35, 0.25) }, clusterMemberships: ['jazz-modal', 'neo-soul', 'lo-fi-chill'] },
          { id: 'harmony.chords.dom7', label: 'Dominant 7th', keywords: ['dom7', 'dominant seventh', '7th'], paramContributions: { harmonic_tension: rn(0.2, 0.5, 0.35) }, clusterMemberships: ['blues-slow', 'blues-chicago', 'funk-groove'] },
          { id: 'harmony.chords.sus2', label: 'Sus2', keywords: ['sus2', 'suspended second', 'open'], paramContributions: { harmonic_tension: rn(0.1, 0.25, 0.15) }, clusterMemberships: ['ambient-pad', 'post-rock'] },
          { id: 'harmony.chords.sus4', label: 'Sus4', keywords: ['sus4', 'suspended fourth'], paramContributions: { harmonic_tension: rn(0.15, 0.3, 0.2) }, clusterMemberships: ['pop-upbeat', 'post-rock'] },
          { id: 'harmony.chords.power-chord', label: 'Power Chord', keywords: ['power chord', 'fifth', 'rock'], paramContributions: { harmonic_tension: rn(0, 0.15, 0.05) }, clusterMemberships: ['punk-rock', 'metal-shred', 'grunge'] },
          { id: 'harmony.chords.dom9', label: 'Dominant 9th', keywords: ['dom9', '9th', 'funk chord'], paramContributions: { harmonic_tension: rn(0.25, 0.5, 0.35) }, clusterMemberships: ['funk-groove', 'neo-soul'] },
          { id: 'harmony.chords.dom7alt', label: 'Altered Dominant', keywords: ['alt', 'altered', 'jazz altered'], paramContributions: { harmonic_tension: rn(0.5, 0.9, 0.7) }, clusterMemberships: ['jazz-bebop', 'jazz-modal'] },
        ],
      },
      {
        id: 'harmony.progressions', label: 'Progressions', keywords: ['progression', 'chord progression'],
        children: [
          { id: 'harmony.progressions.I-V-vi-IV', label: 'I-V-vi-IV', keywords: ['pop progression', 'four chord', 'axis'], paramContributions: { mode_weights: dw({ major: 0.9, mixolydian: 0.1 }) }, clusterMemberships: ['pop-upbeat'] },
          { id: 'harmony.progressions.ii-V-I', label: 'ii-V-I', keywords: ['two five one', 'jazz standard'], paramContributions: { mode_weights: dw({ major: 0.5, dorian: 0.3, mixolydian: 0.2 }) }, clusterMemberships: ['jazz-modal', 'jazz-bebop', 'bossa-nova'] },
          { id: 'harmony.progressions.twelve-bar-blues', label: '12-Bar Blues', keywords: ['twelve bar', 'blues form', 'I-IV-V'], paramContributions: { mode_weights: dw({ mixolydian: 0.5, blues: 0.3, dorian: 0.2 }) }, clusterMemberships: ['blues-slow', 'blues-chicago'] },
          { id: 'harmony.progressions.i-VII-VI-VII', label: 'i-VII-VI-VII', keywords: ['dorian drift', 'modal'], paramContributions: { mode_weights: dw({ dorian: 0.8, minor: 0.2 }) }, clusterMemberships: ['jazz-modal', 'lo-fi-chill'] },
        ],
      },
      {
        id: 'harmony.techniques', label: 'Techniques', keywords: ['harmony technique'],
        children: [
          { id: 'harmony.techniques.modal-interchange', label: 'Modal Interchange', keywords: ['modal interchange', 'borrowed chord', 'parallel key'], paramContributions: { harmonic_tension: rn(0.3, 0.6, 0.45) }, clusterMemberships: ['prog-rock', 'neo-soul'] },
          { id: 'harmony.techniques.tritone-sub', label: 'Tritone Substitution', keywords: ['tritone sub', 'tritone substitution'], paramContributions: { harmonic_tension: rn(0.4, 0.7, 0.55) }, clusterMemberships: ['jazz-bebop', 'jazz-modal'] },
          { id: 'harmony.techniques.pedal-point', label: 'Pedal Point', keywords: ['pedal point', 'drone', 'sustained bass'], paramContributions: { harmonic_tension: rn(0.1, 0.4, 0.2) }, clusterMemberships: ['ambient-drone', 'post-rock', 'cinematic-epic'] },
          { id: 'harmony.techniques.voice-leading', label: 'Voice Leading', keywords: ['voice leading', 'smooth motion', 'stepwise'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'jazz-modal'] },
        ],
      },
    ],
  },

  // ═══ 2. MELODY ═══
  {
    id: 'melody', label: 'Melody', keywords: ['melody', 'tune', 'melodic'],
    children: [
      {
        id: 'melody.contour', label: 'Contour', keywords: ['contour', 'shape', 'direction'],
        children: [
          { id: 'melody.contour.ascending', label: 'Ascending', keywords: ['ascending', 'rising', 'upward'], paramContributions: { melodic_activity: rn(0.4, 0.7, 0.55) }, clusterMemberships: ['cinematic-epic', 'edm-drop'] },
          { id: 'melody.contour.descending', label: 'Descending', keywords: ['descending', 'falling', 'downward'], paramContributions: { melodic_activity: rn(0.3, 0.6, 0.45) }, clusterMemberships: ['blues-slow', 'pop-ballad'] },
          { id: 'melody.contour.arch', label: 'Arch', keywords: ['arch', 'hill', 'up and down'], paramContributions: { melodic_activity: rn(0.4, 0.7, 0.55) }, clusterMemberships: ['pop-upbeat', 'classical-romantic'] },
          { id: 'melody.contour.wave', label: 'Wave', keywords: ['wave', 'undulating', 'arpeggio'], paramContributions: { melodic_activity: rn(0.5, 0.8, 0.65) }, clusterMemberships: ['synthwave-retro', 'ambient-pad'] },
          { id: 'melody.contour.static', label: 'Static', keywords: ['static', 'monotone', 'repeated note'], paramContributions: { melodic_activity: rn(0, 0.2, 0.1) }, clusterMemberships: ['ambient-drone', 'hip-hop-trap'] },
        ],
      },
      {
        id: 'melody.activity', label: 'Activity', keywords: ['activity', 'note density'],
        children: [
          { id: 'melody.activity.sparse', label: 'Sparse', keywords: ['sparse melody', 'few notes', 'space'], paramContributions: { melodic_activity: rn(0, 0.25, 0.1) }, clusterMemberships: ['ambient-pad', 'ambient-drone'] },
          { id: 'melody.activity.moderate', label: 'Moderate', keywords: ['moderate melody', 'balanced'], paramContributions: { melodic_activity: rn(0.3, 0.6, 0.45) }, clusterMemberships: ['pop-upbeat', 'indie-folk'] },
          { id: 'melody.activity.active', label: 'Active', keywords: ['active melody', 'busy', 'moving'], paramContributions: { melodic_activity: rn(0.6, 0.85, 0.7) }, clusterMemberships: ['jazz-bebop', 'funk-groove'] },
          { id: 'melody.activity.virtuosic', label: 'Virtuosic', keywords: ['virtuosic', 'shredding', 'rapid', 'technical'], paramContributions: { melodic_activity: rn(0.8, 1.0, 0.9) }, clusterMemberships: ['metal-shred', 'jazz-bebop'] },
        ],
      },
      {
        id: 'melody.ornamentation', label: 'Ornamentation', keywords: ['ornament', 'embellishment'],
        children: [
          { id: 'melody.ornamentation.trill', label: 'Trill', keywords: ['trill', 'rapid alternation'], paramContributions: {}, clusterMemberships: ['classical-romantic'] },
          { id: 'melody.ornamentation.grace-note', label: 'Grace Note', keywords: ['grace note', 'appoggiatura', 'acciaccatura'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'jazz-bebop'] },
          { id: 'melody.ornamentation.bend', label: 'Bend', keywords: ['bend', 'string bend', 'pitch bend'], paramContributions: {}, clusterMemberships: ['blues-slow', 'blues-chicago', 'metal-shred'] },
          { id: 'melody.ornamentation.slide', label: 'Slide', keywords: ['slide', 'glide', 'portamento'], paramContributions: {}, clusterMemberships: ['blues-slow', 'country-traditional'] },
        ],
      },
    ],
  },

  // ═══ 3. RHYTHM ═══
  {
    id: 'rhythm', label: 'Rhythm', keywords: ['rhythm', 'beat', 'groove', 'time'],
    children: [
      {
        id: 'rhythm.time-signatures', label: 'Time Signatures', keywords: ['time signature', 'meter'],
        children: [
          { id: 'rhythm.time-signatures.4-4', label: '4/4', keywords: ['four four', 'common time'], paramContributions: {}, clusterMemberships: ['pop-upbeat', 'rock', 'edm-house'] },
          { id: 'rhythm.time-signatures.3-4', label: '3/4 (Waltz)', keywords: ['three four', 'waltz', 'triple'], paramContributions: {}, clusterMemberships: ['classical-romantic'] },
          { id: 'rhythm.time-signatures.6-8', label: '6/8', keywords: ['six eight', 'compound'], paramContributions: {}, clusterMemberships: ['blues-slow', 'gospel'] },
          { id: 'rhythm.time-signatures.5-4', label: '5/4', keywords: ['five four', 'odd meter', 'take five'], paramContributions: {}, clusterMemberships: ['prog-rock', 'jazz-modal'] },
          { id: 'rhythm.time-signatures.7-8', label: '7/8', keywords: ['seven eight', 'odd meter', 'prog'], paramContributions: {}, clusterMemberships: ['prog-rock'] },
        ],
      },
      {
        id: 'rhythm.patterns', label: 'Patterns', keywords: ['pattern', 'beat pattern'],
        children: [
          { id: 'rhythm.patterns.four-on-floor', label: 'Four on the Floor', keywords: ['four on floor', 'kick every beat', 'house beat'], paramContributions: { groove_strength: rn(0.7, 0.9, 0.8) }, clusterMemberships: ['edm-house', 'edm-drop'] },
          { id: 'rhythm.patterns.backbeat', label: 'Backbeat', keywords: ['backbeat', 'snare on 2 and 4', 'rock beat'], paramContributions: { groove_strength: rn(0.5, 0.8, 0.65) }, clusterMemberships: ['pop-upbeat', 'punk-rock', 'grunge'] },
          { id: 'rhythm.patterns.shuffle', label: 'Shuffle', keywords: ['shuffle', 'triplet feel', 'bouncy'], paramContributions: { swing: rn(0.4, 0.7, 0.55) }, clusterMemberships: ['blues-chicago', 'country-traditional'] },
          { id: 'rhythm.patterns.clave', label: 'Clave', keywords: ['clave', 'son clave', 'latin rhythm'], paramContributions: { groove_strength: rn(0.8, 1.0, 0.9) }, clusterMemberships: ['latin-salsa', 'bossa-nova'] },
          { id: 'rhythm.patterns.breakbeat', label: 'Breakbeat', keywords: ['breakbeat', 'break', 'amen'], paramContributions: { groove_strength: rn(0.7, 0.95, 0.85) }, clusterMemberships: ['edm-dnb', 'hip-hop-boom-bap'] },
          { id: 'rhythm.patterns.boom-bap', label: 'Boom Bap', keywords: ['boom bap', 'hip-hop beat', 'old school'], paramContributions: { groove_strength: rn(0.7, 0.9, 0.8) }, clusterMemberships: ['hip-hop-boom-bap'] },
        ],
      },
      {
        id: 'rhythm.syncopation', label: 'Syncopation', keywords: ['syncopation', 'offbeat'],
        children: [
          { id: 'rhythm.syncopation.none', label: 'None', keywords: ['no syncopation', 'on-beat', 'straight'], paramContributions: { groove_strength: rn(0, 0.3, 0.15) }, clusterMemberships: ['pop-upbeat'] },
          { id: 'rhythm.syncopation.light', label: 'Light', keywords: ['light syncopation', 'subtle'], paramContributions: { groove_strength: rn(0.2, 0.5, 0.35) }, clusterMemberships: ['bossa-nova', 'indie-folk'] },
          { id: 'rhythm.syncopation.heavy', label: 'Heavy', keywords: ['heavy syncopation', 'funky'], paramContributions: { groove_strength: rn(0.6, 0.9, 0.75) }, clusterMemberships: ['funk-groove', 'latin-salsa'] },
          { id: 'rhythm.syncopation.offbeat', label: 'Offbeat', keywords: ['offbeat', 'reggae', 'ska'], paramContributions: { groove_strength: rn(0.6, 0.85, 0.7) }, clusterMemberships: ['reggae'] },
        ],
      },
      {
        id: 'rhythm.density', label: 'Density', keywords: ['density', 'busy', 'sparse'],
        children: [
          { id: 'rhythm.density.sparse', label: 'Sparse', keywords: ['sparse', 'minimal', 'space', 'slow'], paramContributions: { density: dw({ sparse: 0.85, medium: 0.15 }) }, clusterMemberships: ['ambient-pad', 'ambient-drone', 'blues-slow', 'jazz-ballad'] },
          { id: 'rhythm.density.medium', label: 'Medium', keywords: ['medium density', 'balanced', 'moderate'], paramContributions: { density: dw({ medium: 0.8, sparse: 0.1, dense: 0.1 }) }, clusterMemberships: ['pop-upbeat', 'indie-folk', 'neo-soul'] },
          { id: 'rhythm.density.dense', label: 'Dense', keywords: ['dense', 'busy', 'fast', 'complex', 'rapid'], paramContributions: { density: dw({ dense: 0.85, medium: 0.15 }) }, clusterMemberships: ['metal-shred', 'jazz-bebop', 'edm-dnb', 'punk-rock'] },
        ],
      },
      {
        id: 'rhythm.swing', label: 'Swing', keywords: ['swing', 'shuffle'],
        children: [
          { id: 'rhythm.swing.straight', label: 'Straight', keywords: ['straight', 'no swing', 'even'], paramContributions: { swing: rn(0, 0.1, 0.05) }, clusterMemberships: ['pop-upbeat', 'edm-house', 'punk-rock', 'metal-shred'] },
          { id: 'rhythm.swing.light-swing', label: 'Light Swing', keywords: ['light swing', 'gentle swing', 'lilt'], paramContributions: { swing: rn(0.2, 0.45, 0.3) }, clusterMemberships: ['bossa-nova', 'lo-fi-chill', 'neo-soul'] },
          { id: 'rhythm.swing.heavy-swing', label: 'Heavy Swing', keywords: ['heavy swing', 'hard swing', 'jazz swing'], paramContributions: { swing: rn(0.5, 0.85, 0.7) }, clusterMemberships: ['jazz-modal', 'jazz-bebop', 'blues-chicago'] },
        ],
      },
      {
        id: 'rhythm.groove-feel', label: 'Groove Feel', keywords: ['groove', 'feel'],
        children: [
          { id: 'rhythm.groove-feel.straight-driving', label: 'Straight/Driving', keywords: ['straight', 'driving', 'tight', 'rigid'], paramContributions: { groove_feel: dw({ 'Straight/Driving': 1.0 }), timing_feel: dw({ on: 0.7, ahead: 0.3 }) }, clusterMemberships: ['metal-shred', 'punk-rock', 'edm-house'] },
          { id: 'rhythm.groove-feel.laid-back', label: 'Laid Back', keywords: ['laid-back', 'relaxed', 'chill', 'behind the beat'], paramContributions: { groove_feel: dw({ 'Laid Back': 1.0 }), timing_feel: dw({ behind: 0.8, on: 0.2 }) }, clusterMemberships: ['lo-fi-chill', 'reggae', 'neo-soul', 'r-and-b-slow'] },
          { id: 'rhythm.groove-feel.swung', label: 'Swung', keywords: ['swing', 'swung', 'shuffle', 'bouncy'], paramContributions: { groove_feel: dw({ Swung: 1.0 }), swing: rn(0.4, 0.8, 0.6) }, clusterMemberships: ['jazz-modal', 'blues-chicago'] },
          { id: 'rhythm.groove-feel.syncopated', label: 'Syncopated', keywords: ['syncopated', 'funky', 'offbeat'], paramContributions: { groove_feel: dw({ Syncopated: 1.0 }), groove_strength: rn(0.7, 1.0, 0.85) }, clusterMemberships: ['funk-groove', 'latin-salsa', 'world-afrobeat'] },
          { id: 'rhythm.groove-feel.rubato', label: 'Rubato/Free', keywords: ['rubato', 'free tempo', 'expressive'], paramContributions: { groove_feel: dw({ 'Rubato/Free': 1.0 }) }, clusterMemberships: ['classical-romantic', 'jazz-ballad'] },
          { id: 'rhythm.groove-feel.mechanical', label: 'Mechanical', keywords: ['mechanical', 'robotic', 'quantized', 'precise'], paramContributions: { groove_feel: dw({ Mechanical: 1.0 }), timing_feel: dw({ on: 0.9, ahead: 0.1 }) }, clusterMemberships: ['edm-house', 'video-game-chiptune'] },
          { id: 'rhythm.groove-feel.organic', label: 'Organic/Breathing', keywords: ['organic', 'human', 'natural', 'breathing'], paramContributions: { groove_feel: dw({ 'Organic/Breathing': 1.0 }) }, clusterMemberships: ['indie-folk', 'neo-soul', 'gospel'] },
          { id: 'rhythm.groove-feel.push-pull', label: 'Push-Pull', keywords: ['push-pull', 'tension', 'pushing'], paramContributions: { groove_feel: dw({ 'Push-Pull': 1.0 }) }, clusterMemberships: ['jazz-modal', 'prog-rock'] },
        ],
      },
      {
        id: 'rhythm.polyrhythm', label: 'Polyrhythm', keywords: ['polyrhythm', 'cross rhythm'],
        children: [
          { id: 'rhythm.polyrhythm.2-against-3', label: '2:3', keywords: ['two against three', '2:3', 'hemiola'], paramContributions: { groove_strength: rn(0.6, 0.9, 0.75) }, clusterMemberships: ['world-afrobeat', 'latin-salsa'] },
          { id: 'rhythm.polyrhythm.3-against-4', label: '3:4', keywords: ['three against four', '3:4'], paramContributions: { groove_strength: rn(0.65, 0.9, 0.8) }, clusterMemberships: ['world-afrobeat', 'prog-rock'] },
        ],
      },
    ],
  },

  // ═══ 4. DYNAMICS ═══
  {
    id: 'dynamics', label: 'Dynamics', keywords: ['dynamics', 'volume', 'loud', 'quiet'],
    children: [
      { id: 'dynamics.levels.pp', label: 'Pianissimo (pp)', keywords: ['pianissimo', 'very quiet', 'whisper', 'pp'], paramContributions: { dynamics: dw({ pp: 0.9, p: 0.1 }) }, clusterMemberships: ['ambient-pad', 'ambient-drone'] },
      { id: 'dynamics.levels.p', label: 'Piano (p)', keywords: ['piano dynamics', 'quiet', 'soft', 'gentle', 'p'], paramContributions: { dynamics: dw({ p: 0.8, pp: 0.1, mp: 0.1 }) }, clusterMemberships: ['jazz-ballad', 'pop-ballad'] },
      { id: 'dynamics.levels.mp', label: 'Mezzo Piano (mp)', keywords: ['mezzo piano', 'moderately quiet', 'mp'], paramContributions: { dynamics: dw({ mp: 0.8, p: 0.1, mf: 0.1 }) }, clusterMemberships: ['indie-folk', 'bossa-nova'] },
      { id: 'dynamics.levels.mf', label: 'Mezzo Forte (mf)', keywords: ['mezzo forte', 'moderate', 'medium volume', 'mf'], paramContributions: { dynamics: dw({ mf: 0.8, mp: 0.1, f: 0.1 }) }, clusterMemberships: ['pop-upbeat', 'country-traditional'] },
      { id: 'dynamics.levels.f', label: 'Forte (f)', keywords: ['forte', 'loud', 'strong', 'f'], paramContributions: { dynamics: dw({ f: 0.8, mf: 0.1, ff: 0.1 }) }, clusterMemberships: ['metal-shred', 'punk-rock'] },
      { id: 'dynamics.levels.ff', label: 'Fortissimo (ff)', keywords: ['fortissimo', 'very loud', 'powerful', 'ff', 'aggressive'], paramContributions: { dynamics: dw({ ff: 0.9, f: 0.1 }) }, clusterMemberships: ['metal-shred', 'punk-rock', 'edm-drop'] },
      {
        id: 'dynamics.changes', label: 'Dynamic Changes', keywords: ['dynamic change'],
        children: [
          { id: 'dynamics.changes.crescendo', label: 'Crescendo', keywords: ['crescendo', 'building', 'getting louder'], paramContributions: { narrative_arc: dw({ 'Climb-to-Climax': 0.7, 'Rise and Fall': 0.3 }) }, clusterMemberships: ['cinematic-epic', 'post-rock'] },
          { id: 'dynamics.changes.decrescendo', label: 'Decrescendo', keywords: ['decrescendo', 'diminuendo', 'fading', 'dying away'], paramContributions: { narrative_arc: dw({ Descent: 0.6, 'Slow Reveal': 0.4 }) }, clusterMemberships: ['ambient-pad'] },
          { id: 'dynamics.changes.sforzando', label: 'Sforzando', keywords: ['sforzando', 'sudden accent', 'sfz'], paramContributions: {}, clusterMemberships: ['classical-romantic'] },
        ],
      },
    ],
  },

  // ═══ 5. ARTICULATION ═══
  {
    id: 'articulation', label: 'Articulation', keywords: ['articulation', 'technique', 'playing style'],
    children: [
      { id: 'articulation.staccato', label: 'Staccato', keywords: ['staccato', 'short', 'detached', 'choppy'], paramContributions: {}, clusterMemberships: ['funk-groove', 'edm-house'] },
      { id: 'articulation.legato', label: 'Legato', keywords: ['legato', 'smooth', 'connected', 'flowing'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'metal-shred', 'ambient-pad'] },
      { id: 'articulation.vibrato', label: 'Vibrato', keywords: ['vibrato', 'wobble', 'expression'], paramContributions: {}, clusterMemberships: ['blues-slow', 'classical-romantic'] },
      { id: 'articulation.portamento', label: 'Portamento', keywords: ['portamento', 'slide', 'glide between notes'], paramContributions: {}, clusterMemberships: ['synthwave-retro', 'ambient-pad'] },
      { id: 'articulation.tremolo', label: 'Tremolo', keywords: ['tremolo', 'rapid repeat', 'shake'], paramContributions: {}, clusterMemberships: ['classical-romantic'] },
      { id: 'articulation.pizzicato', label: 'Pizzicato', keywords: ['pizzicato', 'plucked strings'], paramContributions: {}, clusterMemberships: ['classical-romantic'] },
      { id: 'articulation.muted', label: 'Muted', keywords: ['muted', 'damped', 'ghost note'], paramContributions: {}, clusterMemberships: ['funk-groove', 'reggae'] },
      { id: 'articulation.palm-mute', label: 'Palm Mute', keywords: ['palm mute', 'palm-muted', 'chug', 'chugging'], paramContributions: {}, clusterMemberships: ['metal-shred', 'punk-rock'] },
      { id: 'articulation.fingerpicking', label: 'Fingerpicking', keywords: ['fingerpicking', 'fingerstyle', 'travis picking'], paramContributions: {}, clusterMemberships: ['indie-folk', 'country-traditional', 'bossa-nova'] },
      { id: 'articulation.strumming', label: 'Strumming', keywords: ['strumming', 'strum', 'rhythm guitar'], paramContributions: {}, clusterMemberships: ['pop-upbeat', 'indie-folk', 'reggae'] },
      { id: 'articulation.slap', label: 'Slap', keywords: ['slap', 'slap bass', 'pop and slap'], paramContributions: {}, clusterMemberships: ['funk-groove'] },
      { id: 'articulation.tapping', label: 'Tapping', keywords: ['tapping', 'two-hand tapping'], paramContributions: {}, clusterMemberships: ['metal-shred', 'prog-rock'] },
      { id: 'articulation.harmonics', label: 'Harmonics', keywords: ['harmonics', 'natural harmonics', 'artificial harmonics'], paramContributions: {}, clusterMemberships: ['post-rock', 'metal-shred'] },
    ],
  },

  // ═══ 6. TEXTURE ═══
  {
    id: 'texture', label: 'Texture', keywords: ['texture', 'layering', 'thickness'],
    children: [
      { id: 'texture.density.monophonic', label: 'Monophonic', keywords: ['monophonic', 'single voice', 'unison'], paramContributions: { texture_density: rn(0, 0.15, 0.05) }, clusterMemberships: ['ambient-drone'] },
      { id: 'texture.density.thin', label: 'Thin', keywords: ['thin', 'sparse texture', 'minimal'], paramContributions: { texture_density: rn(0.1, 0.3, 0.2) }, clusterMemberships: ['lo-fi-chill', 'indie-folk'] },
      { id: 'texture.density.moderate', label: 'Moderate', keywords: ['moderate texture', 'balanced'], paramContributions: { texture_density: rn(0.3, 0.6, 0.45) }, clusterMemberships: ['pop-upbeat', 'neo-soul'] },
      { id: 'texture.density.thick', label: 'Thick', keywords: ['thick', 'lush', 'full', 'rich', 'layered'], paramContributions: { texture_density: rn(0.6, 0.85, 0.7) }, clusterMemberships: ['cinematic-epic', 'synthwave-retro', 'gospel'] },
      { id: 'texture.density.massive', label: 'Massive', keywords: ['massive', 'wall of sound', 'huge', 'epic'], paramContributions: { texture_density: rn(0.8, 1.0, 0.9) }, clusterMemberships: ['cinematic-epic', 'post-rock', 'edm-drop'] },
      {
        id: 'texture.space', label: 'Space', keywords: ['space', 'room', 'environment'],
        children: [
          { id: 'texture.space.intimate', label: 'Intimate', keywords: ['intimate', 'close', 'dry'], paramContributions: { space_probability: rn(0, 0.1, 0.05) }, clusterMemberships: ['lo-fi-chill', 'hip-hop-boom-bap'] },
          { id: 'texture.space.room', label: 'Room', keywords: ['room', 'natural', 'studio'], paramContributions: { space_probability: rn(0.1, 0.25, 0.15) }, clusterMemberships: ['indie-folk', 'blues-slow'] },
          { id: 'texture.space.hall', label: 'Hall', keywords: ['hall', 'spacious', 'concert hall'], paramContributions: { space_probability: rn(0.2, 0.4, 0.3) }, clusterMemberships: ['classical-romantic', 'gospel'] },
          { id: 'texture.space.cathedral', label: 'Cathedral', keywords: ['cathedral', 'vast', 'reverberant'], paramContributions: { space_probability: rn(0.35, 0.55, 0.45) }, clusterMemberships: ['ambient-pad', 'post-rock'] },
          { id: 'texture.space.infinite', label: 'Infinite', keywords: ['infinite', 'endless', 'void'], paramContributions: { space_probability: rn(0.5, 0.8, 0.65) }, clusterMemberships: ['ambient-drone'] },
        ],
      },
    ],
  },

  // ═══ 7. STRUCTURE ═══
  {
    id: 'structure', label: 'Structure', keywords: ['structure', 'form', 'arrangement', 'song form'],
    children: [
      {
        id: 'structure.sections', label: 'Sections', keywords: ['section', 'part'],
        children: [
          { id: 'structure.sections.intro', label: 'Intro', keywords: ['intro', 'introduction', 'opening'], paramContributions: {}, clusterMemberships: [] },
          { id: 'structure.sections.verse', label: 'Verse', keywords: ['verse', 'stanza'], paramContributions: {}, clusterMemberships: [] },
          { id: 'structure.sections.pre-chorus', label: 'Pre-Chorus', keywords: ['pre-chorus', 'build', 'lift'], paramContributions: {}, clusterMemberships: [] },
          { id: 'structure.sections.chorus', label: 'Chorus', keywords: ['chorus', 'hook', 'refrain'], paramContributions: {}, clusterMemberships: [] },
          { id: 'structure.sections.bridge', label: 'Bridge', keywords: ['bridge', 'middle eight', 'contrast'], paramContributions: {}, clusterMemberships: [] },
          { id: 'structure.sections.outro', label: 'Outro', keywords: ['outro', 'ending', 'coda'], paramContributions: {}, clusterMemberships: [] },
          { id: 'structure.sections.build', label: 'Build', keywords: ['build', 'buildup', 'riser'], paramContributions: {}, clusterMemberships: ['edm-drop'] },
          { id: 'structure.sections.drop', label: 'Drop', keywords: ['drop', 'bass drop'], paramContributions: {}, clusterMemberships: ['edm-drop', 'edm-dnb'] },
          { id: 'structure.sections.breakdown', label: 'Breakdown', keywords: ['breakdown', 'strip back'], paramContributions: {}, clusterMemberships: ['metal-shred', 'edm-drop'] },
          { id: 'structure.sections.solo', label: 'Solo', keywords: ['solo', 'improvisation'], paramContributions: {}, clusterMemberships: ['metal-shred', 'blues-chicago', 'jazz-bebop'] },
        ],
      },
      {
        id: 'structure.narrative-arc', label: 'Narrative Arc', keywords: ['arc', 'trajectory', 'energy'],
        children: [
          { id: 'structure.narrative-arc.climb-to-climax', label: 'Climb to Climax', keywords: ['build', 'climax', 'crescendo', 'peak'], paramContributions: { narrative_arc: dw({ 'Climb-to-Climax': 1.0 }) }, clusterMemberships: ['cinematic-epic', 'post-rock', 'edm-drop'] },
          { id: 'structure.narrative-arc.slow-reveal', label: 'Slow Reveal', keywords: ['reveal', 'gradual', 'unfolding'], paramContributions: { narrative_arc: dw({ 'Slow Reveal': 1.0 }) }, clusterMemberships: ['ambient-pad', 'post-rock'] },
          { id: 'structure.narrative-arc.rise-and-fall', label: 'Rise and Fall', keywords: ['rise', 'fall', 'wave', 'arc'], paramContributions: { narrative_arc: dw({ 'Rise and Fall': 1.0 }) }, clusterMemberships: ['classical-romantic', 'pop-ballad'] },
          { id: 'structure.narrative-arc.sudden-shift', label: 'Sudden Shift', keywords: ['sudden', 'shift', 'surprise', 'contrast'], paramContributions: { narrative_arc: dw({ 'Sudden Shift': 1.0 }) }, clusterMemberships: ['grunge', 'prog-rock'] },
          { id: 'structure.narrative-arc.descent', label: 'Descent', keywords: ['descent', 'decline', 'winding down'], paramContributions: { narrative_arc: dw({ Descent: 1.0 }) }, clusterMemberships: ['ambient-drone'] },
          { id: 'structure.narrative-arc.spiral', label: 'Spiral', keywords: ['spiral', 'cyclical', 'repetitive'], paramContributions: { narrative_arc: dw({ Spiral: 1.0 }) }, clusterMemberships: ['edm-house', 'ambient-drone'] },
        ],
      },
    ],
  },

  // ═══ 8. TIMBRE ═══
  {
    id: 'timbre', label: 'Timbre', keywords: ['timbre', 'instrument', 'sound', 'tone'],
    children: [
      {
        id: 'timbre.instruments', label: 'Instruments', keywords: ['instruments'],
        children: [
          { id: 'timbre.instruments.piano', label: 'Piano', keywords: ['piano', 'keys', 'keyboard', 'grand piano'], paramContributions: {}, clusterMemberships: ['jazz-modal', 'jazz-ballad', 'pop-ballad', 'classical-romantic', 'neo-soul', 'lo-fi-chill'] },
          { id: 'timbre.instruments.electric-piano', label: 'Electric Piano', keywords: ['electric piano', 'rhodes', 'wurlitzer', 'ep'], paramContributions: {}, clusterMemberships: ['neo-soul', 'lo-fi-chill', 'funk-groove'] },
          { id: 'timbre.instruments.organ', label: 'Organ', keywords: ['organ', 'hammond', 'b3'], paramContributions: {}, clusterMemberships: ['gospel', 'blues-chicago', 'reggae'] },
          { id: 'timbre.instruments.acoustic-guitar', label: 'Acoustic Guitar', keywords: ['acoustic guitar', 'acoustic', 'nylon'], paramContributions: {}, clusterMemberships: ['indie-folk', 'country-traditional', 'bossa-nova', 'pop-ballad'] },
          { id: 'timbre.instruments.electric-guitar', label: 'Electric Guitar', keywords: ['electric guitar', 'guitar', 'six string'], paramContributions: {}, clusterMemberships: ['metal-shred', 'punk-rock', 'grunge', 'blues-chicago', 'post-rock', 'indie-folk'] },
          { id: 'timbre.instruments.bass-guitar', label: 'Bass Guitar', keywords: ['bass', 'bass guitar', 'electric bass'], paramContributions: {}, clusterMemberships: ['funk-groove', 'blues-slow', 'reggae', 'neo-soul', 'punk-rock'] },
          { id: 'timbre.instruments.upright-bass', label: 'Upright Bass', keywords: ['upright bass', 'double bass', 'acoustic bass', 'walking bass'], paramContributions: {}, clusterMemberships: ['jazz-modal', 'jazz-bebop', 'jazz-ballad'] },
          { id: 'timbre.instruments.synth-lead', label: 'Synth Lead', keywords: ['synth', 'synthesizer', 'lead synth'], paramContributions: {}, clusterMemberships: ['synthwave-retro', 'edm-drop', 'edm-house', 'pop-upbeat'] },
          { id: 'timbre.instruments.synth-pad', label: 'Synth Pad', keywords: ['pad', 'synth pad', 'atmospheric'], paramContributions: {}, clusterMemberships: ['ambient-pad', 'ambient-drone', 'synthwave-retro', 'r-and-b-slow'] },
          { id: 'timbre.instruments.synth-bass', label: 'Synth Bass', keywords: ['synth bass', 'sub bass', 'bass synth'], paramContributions: {}, clusterMemberships: ['edm-drop', 'edm-house', 'edm-dnb', 'hip-hop-trap'] },
          { id: 'timbre.instruments.strings', label: 'Strings', keywords: ['strings', 'orchestral strings', 'string section'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'cinematic-epic', 'pop-ballad'] },
          { id: 'timbre.instruments.brass', label: 'Brass', keywords: ['brass', 'horns', 'brass section'], paramContributions: {}, clusterMemberships: ['latin-salsa', 'gospel', 'cinematic-epic', 'world-afrobeat'] },
          { id: 'timbre.instruments.sax', label: 'Saxophone', keywords: ['sax', 'saxophone', 'alto sax', 'tenor sax'], paramContributions: {}, clusterMemberships: ['jazz-modal', 'jazz-bebop', 'neo-soul'] },
          { id: 'timbre.instruments.drums', label: 'Drums', keywords: ['drums', 'drum kit', 'percussion'], paramContributions: {}, clusterMemberships: [] },
          { id: 'timbre.instruments.choir', label: 'Choir', keywords: ['choir', 'vocals', 'voice', 'singing'], paramContributions: {}, clusterMemberships: ['gospel', 'cinematic-epic'] },
          { id: 'timbre.instruments.808', label: '808', keywords: ['808', 'tr-808', 'trap bass', 'sub'], paramContributions: {}, clusterMemberships: ['hip-hop-trap', 'hip-hop-boom-bap'] },
          { id: 'timbre.instruments.violin', label: 'Violin', keywords: ['violin', 'fiddle'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'country-traditional'] },
          { id: 'timbre.instruments.cello', label: 'Cello', keywords: ['cello'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'cinematic-epic'] },
          { id: 'timbre.instruments.flute', label: 'Flute', keywords: ['flute'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'bossa-nova'] },
          { id: 'timbre.instruments.harp', label: 'Harp', keywords: ['harp', 'arpeggiated'], paramContributions: {}, clusterMemberships: ['classical-romantic', 'ambient-pad'] },
        ],
      },
      {
        id: 'timbre.color', label: 'Tonal Color', keywords: ['tone', 'color', 'brightness'],
        children: [
          { id: 'timbre.color.bright', label: 'Bright', keywords: ['bright', 'shimmering', 'sparkling'], paramContributions: {}, clusterMemberships: ['pop-upbeat', 'video-game-chiptune'] },
          { id: 'timbre.color.warm', label: 'Warm', keywords: ['warm', 'round', 'full'], paramContributions: {}, clusterMemberships: ['neo-soul', 'lo-fi-chill', 'jazz-ballad'] },
          { id: 'timbre.color.dark', label: 'Dark', keywords: ['dark', 'murky', 'deep'], paramContributions: {}, clusterMemberships: ['metal-shred', 'ambient-drone', 'hip-hop-trap'] },
          { id: 'timbre.color.gritty', label: 'Gritty', keywords: ['gritty', 'rough', 'raw', 'dirty'], paramContributions: {}, clusterMemberships: ['grunge', 'punk-rock', 'blues-chicago'] },
        ],
      },
      {
        id: 'timbre.effects', label: 'Effects', keywords: ['effects', 'fx', 'processing'],
        children: [
          { id: 'timbre.effects.distortion', label: 'Distortion', keywords: ['distortion', 'distorted', 'heavy', 'gain'], paramContributions: {}, clusterMemberships: ['metal-shred', 'punk-rock', 'grunge'] },
          { id: 'timbre.effects.overdrive', label: 'Overdrive', keywords: ['overdrive', 'drive', 'crunch', 'tube'], paramContributions: {}, clusterMemberships: ['metal-shred', 'blues-chicago'] },
          { id: 'timbre.effects.clean', label: 'Clean', keywords: ['clean', 'pristine', 'pure', 'unprocessed'], paramContributions: {}, clusterMemberships: ['blues-slow', 'jazz-modal', 'bossa-nova', 'indie-folk'] },
          { id: 'timbre.effects.reverb', label: 'Reverb', keywords: ['reverb', 'spacious', 'reverberant'], paramContributions: { space_probability: rn(0.15, 0.45, 0.3) }, clusterMemberships: ['ambient-pad', 'post-rock', 'ambient-drone'] },
          { id: 'timbre.effects.delay', label: 'Delay', keywords: ['delay', 'echo', 'repeat'], paramContributions: {}, clusterMemberships: ['post-rock', 'ambient-pad', 'reggae'] },
          { id: 'timbre.effects.chorus-effect', label: 'Chorus', keywords: ['chorus effect', 'thicken', 'shimmer'], paramContributions: {}, clusterMemberships: ['synthwave-retro', 'indie-folk'] },
          { id: 'timbre.effects.wah', label: 'Wah', keywords: ['wah', 'wah-wah', 'funk wah'], paramContributions: {}, clusterMemberships: ['funk-groove'] },
          { id: 'timbre.effects.compression', label: 'Compression', keywords: ['compression', 'compressed', 'squash'], paramContributions: {}, clusterMemberships: [] },
          { id: 'timbre.effects.bitcrusher', label: 'Bitcrusher', keywords: ['bitcrusher', 'bitcrushed', '8-bit', 'lo-res'], paramContributions: {}, clusterMemberships: ['video-game-chiptune'] },
          { id: 'timbre.effects.lo-fi', label: 'Lo-Fi', keywords: ['lo-fi', 'lofi', 'tape', 'vinyl', 'vintage'], paramContributions: {}, clusterMemberships: ['lo-fi-chill'] },
        ],
      },
    ],
  },

  // ═══ 9. EXPRESSION ═══
  {
    id: 'expression', label: 'Expression', keywords: ['expression', 'mood', 'emotion', 'feeling'],
    children: [
      { id: 'expression.mood.nostalgic', label: 'Nostalgic', keywords: ['nostalgic', 'nostalgia', 'memories', 'wistful'], paramContributions: { mode_weights: dw({ mixolydian: 0.4, major: 0.3, dorian: 0.3 }) }, clusterMemberships: ['lo-fi-chill', 'synthwave-retro'] },
      { id: 'expression.mood.melancholic', label: 'Melancholic', keywords: ['melancholic', 'melancholy', 'somber', 'blue'], paramContributions: { mode_weights: dw({ minor: 0.5, dorian: 0.3, phrygian: 0.2 }) }, clusterMemberships: ['post-rock', 'pop-ballad'] },
      { id: 'expression.mood.grief', label: 'Grief', keywords: ['grief', 'loss', 'mourning', 'devastation'], paramContributions: { mode_weights: dw({ minor: 0.6, phrygian: 0.2, dorian: 0.2 }) }, clusterMemberships: ['ambient-pad', 'classical-romantic'] },
      { id: 'expression.mood.tender', label: 'Tender', keywords: ['tender', 'gentle', 'delicate', 'fragile'], paramContributions: { dynamics: dw({ p: 0.4, mp: 0.4, pp: 0.2 }) }, clusterMemberships: ['jazz-ballad', 'pop-ballad'] },
      { id: 'expression.mood.romantic', label: 'Romantic', keywords: ['romantic', 'love', 'passionate'], paramContributions: { mode_weights: dw({ major: 0.4, lydian: 0.3, dorian: 0.3 }) }, clusterMemberships: ['classical-romantic', 'r-and-b-slow'] },
      { id: 'expression.mood.hopeful', label: 'Hopeful', keywords: ['hopeful', 'hope', 'uplifting', 'optimistic'], paramContributions: { mode_weights: dw({ major: 0.5, lydian: 0.3, mixolydian: 0.2 }) }, clusterMemberships: ['pop-upbeat', 'gospel'] },
      { id: 'expression.mood.joyful', label: 'Joyful', keywords: ['joyful', 'happy', 'cheerful', 'bright', 'celebration'], paramContributions: { mode_weights: dw({ major: 0.5, lydian: 0.3, mixolydian: 0.2 }) }, clusterMemberships: ['pop-upbeat', 'gospel', 'latin-salsa'] },
      { id: 'expression.mood.energetic', label: 'Energetic', keywords: ['energetic', 'high energy', 'pumping', 'adrenaline'], paramContributions: { density: dw({ dense: 0.6, medium: 0.4 }), dynamics: dw({ f: 0.5, ff: 0.3, mf: 0.2 }) }, clusterMemberships: ['edm-drop', 'punk-rock', 'metal-shred'] },
      { id: 'expression.mood.fierce', label: 'Fierce', keywords: ['fierce', 'intense', 'aggressive', 'brutal'], paramContributions: { density: dw({ dense: 0.7, medium: 0.3 }), dynamics: dw({ ff: 0.6, f: 0.4 }) }, clusterMemberships: ['metal-shred', 'punk-rock', 'edm-dnb'] },
      { id: 'expression.mood.mysterious', label: 'Mysterious', keywords: ['mysterious', 'enigmatic', 'dark', 'haunting'], paramContributions: { mode_weights: dw({ phrygian: 0.3, minor: 0.3, dorian: 0.2, locrian: 0.2 }) }, clusterMemberships: ['ambient-pad', 'cinematic-epic'] },
      { id: 'expression.mood.ethereal', label: 'Ethereal', keywords: ['ethereal', 'otherworldly', 'heavenly', 'floating'], paramContributions: { mode_weights: dw({ lydian: 0.4, major: 0.3, whole_tone: 0.3 }), space_probability: rn(0.2, 0.5, 0.35) }, clusterMemberships: ['ambient-pad', 'post-rock'] },
      { id: 'expression.mood.calm', label: 'Calm', keywords: ['calm', 'peaceful', 'serene', 'tranquil', 'zen'], paramContributions: { mode_weights: dw({ major: 0.4, lydian: 0.3, mixolydian: 0.3 }), density: dw({ sparse: 0.6, medium: 0.4 }) }, clusterMemberships: ['ambient-pad', 'jazz-ballad', 'bossa-nova'] },
    ],
  },

  // ═══ 10. PRODUCTION ═══
  {
    id: 'production', label: 'Production', keywords: ['production', 'mix', 'spatial', 'engineering'],
    children: [
      {
        id: 'production.register', label: 'Register', keywords: ['register', 'frequency range'],
        children: [
          { id: 'production.register.sub-bass', label: 'Sub Bass', keywords: ['sub bass', 'sub', 'rumble', 'below 80hz'], paramContributions: {}, clusterMemberships: ['hip-hop-trap', 'edm-drop'] },
          { id: 'production.register.bass', label: 'Bass', keywords: ['bass register', 'low', 'bottom'], paramContributions: {}, clusterMemberships: ['reggae', 'edm-dnb'] },
          { id: 'production.register.mid', label: 'Mid', keywords: ['mid range', 'middle', 'presence'], paramContributions: {}, clusterMemberships: [] },
          { id: 'production.register.high', label: 'High', keywords: ['high register', 'treble', 'bright', 'air'], paramContributions: {}, clusterMemberships: [] },
        ],
      },
      {
        id: 'production.spatial', label: 'Spatial', keywords: ['spatial', 'stereo', 'width'],
        children: [
          { id: 'production.spatial.mono', label: 'Mono', keywords: ['mono', 'center', 'focused'], paramContributions: {}, clusterMemberships: ['lo-fi-chill'] },
          { id: 'production.spatial.narrow-stereo', label: 'Narrow Stereo', keywords: ['narrow', 'tight stereo'], paramContributions: {}, clusterMemberships: ['hip-hop-boom-bap'] },
          { id: 'production.spatial.wide-stereo', label: 'Wide Stereo', keywords: ['wide', 'stereo', 'spacious', 'panoramic'], paramContributions: {}, clusterMemberships: ['post-rock', 'cinematic-epic', 'ambient-pad'] },
        ],
      },
      {
        id: 'production.mix-style', label: 'Mix Style', keywords: ['mix', 'style', 'aesthetic'],
        children: [
          { id: 'production.mix-style.raw', label: 'Raw', keywords: ['raw', 'unpolished', 'live feel'], paramContributions: {}, clusterMemberships: ['punk-rock', 'grunge'] },
          { id: 'production.mix-style.polished', label: 'Polished', keywords: ['polished', 'clean', 'professional', 'hi-fi'], paramContributions: {}, clusterMemberships: ['pop-upbeat', 'edm-house'] },
          { id: 'production.mix-style.lo-fi', label: 'Lo-Fi', keywords: ['lo-fi mix', 'lofi', 'tape saturation', 'vinyl crackle'], paramContributions: {}, clusterMemberships: ['lo-fi-chill'] },
          { id: 'production.mix-style.saturated', label: 'Saturated', keywords: ['saturated', 'warm', 'analog', 'tape'], paramContributions: {}, clusterMemberships: ['neo-soul', 'synthwave-retro'] },
        ],
      },
    ],
  },
];

/** Flatten the taxonomy tree to a Map for O(1) lookup by ID */
export function flattenTaxonomy(nodes: TaxonomyNode[]): Map<string, TaxonomyNode> {
  const map = new Map<string, TaxonomyNode>();
  function walk(nodeList: TaxonomyNode[]) {
    for (const node of nodeList) {
      map.set(node.id, node);
      if (node.children) walk(node.children);
    }
  }
  walk(nodes);
  return map;
}
