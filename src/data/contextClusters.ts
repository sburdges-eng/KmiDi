/**
 * Context Cluster Registry
 *
 * Defines genre/style clusters that activate when certain combinations of
 * taxonomy nodes and/or keywords are present together. The same musical
 * parameter (e.g., "dorian mode") means completely different things in
 * different contexts — clusters capture these emergent meanings.
 */

import type { ParamDistribution } from '../types/Interpretation';

export interface TriggerRule {
  requiredNodes?: string[];
  requiredKeywords?: string[];
  minMatchCount?: number;
}

export interface ContextCluster {
  id: string;
  label: string;
  genreHints: string[];
  triggerConditions: TriggerRule[];
  parameterOverrides: Record<string, ParamDistribution>;
}

function g(center: number, spread: number): ParamDistribution {
  return { type: 'gaussian', center, spread };
}
function d(weights: Record<string, number>): ParamDistribution {
  return { type: 'discrete', weights };
}
function r(min: number, max: number, bias: number): ParamDistribution {
  return { type: 'range', min, max, bias };
}

export const CONTEXT_CLUSTERS: ContextCluster[] = [
  {
    id: 'metal-shred',
    label: 'Metal/Prog Shred',
    genreHints: ['metal', 'prog', 'djent', 'shred'],
    triggerConditions: [{ requiredKeywords: ['fast', 'guitar', 'overdrive', 'distortion', 'metal', 'shred', 'heavy', 'aggressive'], requiredNodes: ['harmony.scales.dorian', 'harmony.scales.phrygian', 'timbre.effects.distortion', 'timbre.effects.overdrive'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(160, 15), density: r(0.8, 1.0, 0.9), dynamics: d({ f: 0.4, ff: 0.6 }), timing_feel: d({ on: 0.6, ahead: 0.4 }), groove_strength: r(0.5, 0.8, 0.6) },
  },
  {
    id: 'blues-slow',
    label: 'Slow Blues',
    genreHints: ['blues', 'delta blues', 'slow blues'],
    triggerConditions: [{ requiredKeywords: ['slow', 'blues', 'bluesy', 'clean', 'pentatonic', 'laid-back', 'soulful'], requiredNodes: ['harmony.scales.pentatonic-minor', 'harmony.scales.blues', 'harmony.scales.minor', 'timbre.effects.clean'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(72, 8), density: d({ sparse: 0.6, medium: 0.4 }), dynamics: d({ mp: 0.5, mf: 0.3, p: 0.2 }), timing_feel: d({ behind: 0.8, on: 0.2 }), space_probability: r(0.2, 0.4, 0.3) },
  },
  {
    id: 'blues-chicago',
    label: 'Chicago Blues',
    genreHints: ['blues', 'chicago blues', 'electric blues'],
    triggerConditions: [{ requiredKeywords: ['blues', 'shuffle', 'overdrive', 'guitar', 'chicago'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.mixolydian', 'rhythm.patterns.shuffle'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(110, 10), density: d({ medium: 0.7, dense: 0.3 }), swing: r(0.4, 0.7, 0.6), dynamics: d({ mf: 0.5, f: 0.3, mp: 0.2 }) },
  },
  {
    id: 'jazz-modal',
    label: 'Modal Jazz',
    genreHints: ['jazz', 'modal jazz', 'cool jazz'],
    triggerConditions: [{ requiredKeywords: ['jazz', 'jazzy', 'modal', 'swing', 'walking bass', 'sax', 'piano'], requiredNodes: ['harmony.scales.dorian', 'harmony.scales.lydian', 'timbre.instruments.sax', 'timbre.instruments.piano'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(130, 20), swing: r(0.5, 0.8, 0.7), density: d({ medium: 0.5, dense: 0.3, sparse: 0.2 }), harmonic_rhythm: d({ fast: 0.5, medium: 0.4, slow: 0.1 }) },
  },
  {
    id: 'jazz-bebop',
    label: 'Bebop',
    genreHints: ['jazz', 'bebop', 'hard bop'],
    triggerConditions: [{ requiredKeywords: ['bebop', 'fast', 'sax', 'trumpet', 'walking bass', 'jazz'], requiredNodes: ['harmony.scales.mixolydian', 'harmony.scales.dorian', 'timbre.instruments.sax', 'timbre.instruments.trumpet'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(200, 25), density: d({ dense: 0.7, medium: 0.3 }), swing: r(0.6, 0.9, 0.75), harmonic_rhythm: d({ fast: 0.7, medium: 0.3 }) },
  },
  {
    id: 'jazz-ballad',
    label: 'Jazz Ballad',
    genreHints: ['jazz', 'ballad', 'standards'],
    triggerConditions: [{ requiredKeywords: ['ballad', 'slow', 'jazz', 'piano', 'brushes', 'gentle'], requiredNodes: ['harmony.scales.major', 'harmony.scales.lydian', 'timbre.instruments.piano'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(65, 8), density: d({ sparse: 0.6, medium: 0.4 }), dynamics: d({ pp: 0.3, p: 0.4, mp: 0.3 }), space_probability: r(0.25, 0.45, 0.35) },
  },
  {
    id: 'ambient-pad',
    label: 'Ambient Pad',
    genreHints: ['ambient', 'atmospheric', 'new age'],
    triggerConditions: [{ requiredKeywords: ['ambient', 'pad', 'atmospheric', 'ethereal', 'dreamy', 'floating', 'spacious', 'drone'], requiredNodes: ['harmony.scales.lydian', 'timbre.instruments.synth-pad', 'timbre.effects.reverb'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(65, 10), density: d({ sparse: 0.7, medium: 0.3 }), dynamics: d({ pp: 0.4, p: 0.4, mp: 0.2 }), space_probability: r(0.3, 0.6, 0.4) },
  },
  {
    id: 'ambient-drone',
    label: 'Ambient Drone',
    genreHints: ['ambient', 'drone', 'experimental'],
    triggerConditions: [{ requiredKeywords: ['drone', 'minimal', 'ambient', 'texture', 'reverb', 'spacious'], requiredNodes: ['timbre.instruments.synth-pad', 'timbre.effects.reverb'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(50, 5), density: d({ sparse: 0.9, medium: 0.1 }), dynamics: d({ pp: 0.5, p: 0.5 }), space_probability: r(0.4, 0.7, 0.5) },
  },
  {
    id: 'funk-groove',
    label: 'Funk Groove',
    genreHints: ['funk', 'groove', 'disco'],
    triggerConditions: [{ requiredKeywords: ['funk', 'funky', 'groovy', 'groove', 'syncopated', 'slap', 'tight'], requiredNodes: ['harmony.scales.dorian', 'harmony.scales.mixolydian', 'rhythm.groove-feel.syncopated', 'articulation.slap'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(105, 10), groove_strength: r(0.8, 1.0, 0.9), timing_feel: d({ ahead: 0.5, on: 0.5 }), density: d({ medium: 0.6, dense: 0.4 }) },
  },
  {
    id: 'pop-upbeat',
    label: 'Upbeat Pop',
    genreHints: ['pop', 'dance pop', 'synth pop'],
    triggerConditions: [{ requiredKeywords: ['pop', 'upbeat', 'catchy', 'bright', 'happy', 'energetic', 'cheerful', 'bouncy'], requiredNodes: ['harmony.scales.major', 'harmony.scales.lydian'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(118, 10), density: d({ medium: 0.7, dense: 0.3 }), dynamics: d({ mf: 0.5, f: 0.3, mp: 0.2 }) },
  },
  {
    id: 'pop-ballad',
    label: 'Pop Ballad',
    genreHints: ['pop', 'ballad', 'power ballad'],
    triggerConditions: [{ requiredKeywords: ['ballad', 'slow', 'piano', 'gentle', 'soft', 'emotional'], requiredNodes: ['harmony.scales.major', 'harmony.scales.minor', 'timbre.instruments.piano'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(72, 8), density: d({ sparse: 0.5, medium: 0.5 }), dynamics: d({ mp: 0.4, p: 0.3, mf: 0.3 }) },
  },
  {
    id: 'edm-drop',
    label: 'EDM Drop',
    genreHints: ['edm', 'dubstep', 'bass music'],
    triggerConditions: [{ requiredKeywords: ['edm', 'drop', 'build', 'buildup', 'electronic', 'massive', 'bass drop'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.phrygian', 'timbre.instruments.synth-bass', 'structure.sections.drop'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(128, 5), density: r(0.7, 1.0, 0.9), dynamics: d({ ff: 0.7, f: 0.3 }) },
  },
  {
    id: 'edm-house',
    label: 'House',
    genreHints: ['house', 'deep house', 'tech house'],
    triggerConditions: [{ requiredKeywords: ['house', 'four-on-floor', 'electronic', 'club', 'dance'], requiredNodes: ['rhythm.patterns.four-on-floor', 'timbre.instruments.synth-lead'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(124, 3), groove_strength: r(0.7, 0.9, 0.8), density: d({ medium: 0.6, dense: 0.4 }) },
  },
  {
    id: 'edm-dnb',
    label: 'Drum & Bass',
    genreHints: ['dnb', 'drum and bass', 'jungle'],
    triggerConditions: [{ requiredKeywords: ['dnb', 'drum and bass', 'jungle', 'breakbeat', 'fast'], requiredNodes: ['harmony.scales.minor', 'rhythm.patterns.breakbeat', 'rhythm.density.dense'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(174, 4), density: d({ dense: 0.8, medium: 0.2 }), groove_strength: r(0.7, 0.95, 0.85) },
  },
  {
    id: 'hip-hop-boom-bap',
    label: 'Boom Bap',
    genreHints: ['hip-hop', 'boom bap', 'old school'],
    triggerConditions: [{ requiredKeywords: ['hip-hop', 'hip hop', 'boom bap', 'rap', 'sample', 'vinyl'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.dorian', 'rhythm.patterns.boom-bap'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(90, 5), groove_strength: r(0.7, 0.9, 0.8), timing_feel: d({ behind: 0.6, on: 0.4 }) },
  },
  {
    id: 'hip-hop-trap',
    label: 'Trap',
    genreHints: ['trap', 'hip-hop', 'drill'],
    triggerConditions: [{ requiredKeywords: ['trap', '808', 'hi-hat', 'drill', 'dark'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.phrygian', 'timbre.instruments.808'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(140, 10), density: d({ medium: 0.4, dense: 0.4, sparse: 0.2 }), dynamics: d({ f: 0.5, ff: 0.3, mf: 0.2 }) },
  },
  {
    id: 'lo-fi-chill',
    label: 'Lo-fi Chill',
    genreHints: ['lo-fi', 'chill', 'study beats'],
    triggerConditions: [{ requiredKeywords: ['lo-fi', 'lofi', 'chill', 'vinyl', 'tape', 'warm', 'cozy', 'study', 'relax'], requiredNodes: ['harmony.scales.dorian', 'harmony.scales.mixolydian', 'timbre.effects.lo-fi'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(78, 6), timing_feel: d({ behind: 0.8, on: 0.2 }), density: d({ sparse: 0.6, medium: 0.4 }), space_probability: r(0.15, 0.3, 0.25) },
  },
  {
    id: 'classical-romantic',
    label: 'Romantic Classical',
    genreHints: ['classical', 'romantic', 'orchestral'],
    triggerConditions: [{ requiredKeywords: ['classical', 'romantic', 'orchestral', 'strings', 'legato', 'sweeping'], requiredNodes: ['timbre.instruments.strings', 'timbre.instruments.piano', 'articulation.legato'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(85, 15), dynamics: d({ pp: 0.15, p: 0.2, mp: 0.15, mf: 0.15, f: 0.2, ff: 0.15 }), density: d({ medium: 0.5, dense: 0.3, sparse: 0.2 }) },
  },
  {
    id: 'punk-rock',
    label: 'Punk Rock',
    genreHints: ['punk', 'punk rock', 'hardcore'],
    triggerConditions: [{ requiredKeywords: ['punk', 'fast', 'distortion', 'aggressive', 'raw', 'loud', 'power chords', 'angry'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.phrygian', 'timbre.effects.distortion', 'harmony.chords.power-chord'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(175, 15), density: r(0.7, 1.0, 0.85), dynamics: d({ ff: 0.7, f: 0.3 }) },
  },
  {
    id: 'grunge',
    label: 'Grunge',
    genreHints: ['grunge', 'alternative', '90s rock'],
    triggerConditions: [{ requiredKeywords: ['grunge', 'distortion', 'quiet-loud', 'raw', 'angst', 'guitar'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.phrygian', 'timbre.effects.distortion'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(120, 12), dynamics: d({ pp: 0.2, p: 0.1, ff: 0.4, f: 0.3 }), density: d({ medium: 0.4, dense: 0.4, sparse: 0.2 }) },
  },
  {
    id: 'indie-folk',
    label: 'Indie Folk',
    genreHints: ['indie', 'folk', 'acoustic'],
    triggerConditions: [{ requiredKeywords: ['indie', 'folk', 'acoustic', 'fingerpicking', 'gentle', 'organic'], requiredNodes: ['harmony.scales.major', 'harmony.scales.mixolydian', 'timbre.instruments.acoustic-guitar', 'articulation.fingerpicking'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(100, 10), density: d({ sparse: 0.4, medium: 0.5, dense: 0.1 }), dynamics: d({ mp: 0.4, p: 0.3, mf: 0.3 }) },
  },
  {
    id: 'country-traditional',
    label: 'Country',
    genreHints: ['country', 'bluegrass', 'americana'],
    triggerConditions: [{ requiredKeywords: ['country', 'twang', 'fiddle', 'pedal steel', 'honky-tonk'], requiredNodes: ['harmony.scales.major', 'harmony.scales.mixolydian', 'timbre.instruments.acoustic-guitar'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(110, 8), timing_feel: d({ on: 0.7, behind: 0.3 }), density: d({ medium: 0.6, sparse: 0.3, dense: 0.1 }) },
  },
  {
    id: 'reggae',
    label: 'Reggae',
    genreHints: ['reggae', 'dub', 'ska'],
    triggerConditions: [{ requiredKeywords: ['reggae', 'offbeat', 'dub', 'ska', 'island', 'laid-back'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.dorian', 'rhythm.syncopation.offbeat'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(80, 5), timing_feel: d({ behind: 0.7, on: 0.3 }), groove_strength: r(0.7, 0.9, 0.8) },
  },
  {
    id: 'bossa-nova',
    label: 'Bossa Nova',
    genreHints: ['bossa nova', 'brazilian', 'mpb'],
    triggerConditions: [{ requiredKeywords: ['bossa', 'bossa nova', 'brazilian', 'nylon', 'gentle'], requiredNodes: ['harmony.scales.major', 'harmony.scales.lydian', 'timbre.instruments.acoustic-guitar'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(130, 8), swing: r(0.3, 0.6, 0.5), density: d({ sparse: 0.4, medium: 0.6 }), dynamics: d({ mp: 0.4, p: 0.3, mf: 0.3 }) },
  },
  {
    id: 'r-and-b-slow',
    label: 'Slow R&B',
    genreHints: ['r&b', 'rnb', 'slow jam'],
    triggerConditions: [{ requiredKeywords: ['r&b', 'rnb', 'soul', 'smooth', 'slow jam', 'sensual'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.dorian', 'timbre.instruments.synth-pad'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(68, 6), timing_feel: d({ behind: 0.7, on: 0.3 }), groove_strength: r(0.6, 0.8, 0.7), dynamics: d({ mp: 0.4, p: 0.3, mf: 0.3 }) },
  },
  {
    id: 'prog-rock',
    label: 'Progressive Rock',
    genreHints: ['prog', 'progressive', 'art rock'],
    triggerConditions: [{ requiredKeywords: ['prog', 'progressive', 'complex', 'odd meter', 'time signature'], requiredNodes: ['rhythm.time-signatures.5-4', 'rhythm.time-signatures.7-8', 'harmony.scales.dorian', 'harmony.scales.lydian'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(120, 20), density: d({ medium: 0.3, dense: 0.5, sparse: 0.2 }), harmonic_rhythm: d({ fast: 0.4, medium: 0.4, slow: 0.2 }) },
  },
  {
    id: 'post-rock',
    label: 'Post-Rock',
    genreHints: ['post-rock', 'atmospheric rock', 'shoegaze'],
    triggerConditions: [{ requiredKeywords: ['post-rock', 'atmospheric', 'reverb', 'crescendo', 'building', 'wall of sound', 'ethereal'], requiredNodes: ['harmony.scales.minor', 'timbre.effects.reverb', 'timbre.effects.delay', 'dynamics.changes.crescendo'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(90, 10), dynamics: d({ pp: 0.3, p: 0.2, ff: 0.3, f: 0.2 }), space_probability: r(0.1, 0.35, 0.2) },
  },
  {
    id: 'synthwave-retro',
    label: 'Synthwave/Retrowave',
    genreHints: ['synthwave', 'retrowave', '80s', 'outrun'],
    triggerConditions: [{ requiredKeywords: ['synthwave', 'retrowave', '80s', 'neon', 'retro', 'arpeggiated', 'driving'], requiredNodes: ['harmony.scales.minor', 'harmony.scales.dorian', 'timbre.instruments.synth-lead'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(118, 8), density: d({ medium: 0.6, dense: 0.4 }), dynamics: d({ mf: 0.4, f: 0.4, mp: 0.2 }) },
  },
  {
    id: 'latin-salsa',
    label: 'Latin Salsa',
    genreHints: ['salsa', 'latin', 'cumbia'],
    triggerConditions: [{ requiredKeywords: ['salsa', 'latin', 'clave', 'brass', 'percussion', 'cumbia'], requiredNodes: ['harmony.scales.major', 'harmony.scales.mixolydian', 'rhythm.patterns.clave', 'timbre.instruments.brass'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(180, 10), groove_strength: r(0.85, 1.0, 0.95), density: d({ dense: 0.6, medium: 0.4 }) },
  },
  {
    id: 'gospel',
    label: 'Gospel',
    genreHints: ['gospel', 'worship', 'spiritual'],
    triggerConditions: [{ requiredKeywords: ['gospel', 'worship', 'choir', 'organ', 'spiritual', 'praise'], requiredNodes: ['harmony.scales.major', 'harmony.scales.mixolydian', 'timbre.instruments.organ', 'timbre.instruments.choir'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(100, 12), dynamics: d({ mp: 0.2, mf: 0.2, f: 0.3, ff: 0.3 }), harmonic_rhythm: d({ medium: 0.5, fast: 0.3, slow: 0.2 }) },
  },
  {
    id: 'cinematic-epic',
    label: 'Cinematic Epic',
    genreHints: ['cinematic', 'film', 'epic', 'trailer'],
    triggerConditions: [{ requiredKeywords: ['cinematic', 'epic', 'orchestral', 'film', 'dramatic', 'sweeping', 'massive'], requiredNodes: ['harmony.scales.minor', 'timbre.instruments.strings', 'timbre.instruments.brass', 'timbre.instruments.choir'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(90, 15), density: r(0.5, 1.0, 0.8), dynamics: d({ ff: 0.3, f: 0.3, pp: 0.2, p: 0.2 }) },
  },
  {
    id: 'neo-soul',
    label: 'Neo-Soul',
    genreHints: ['neo-soul', 'soul', 'erykah badu'],
    triggerConditions: [{ requiredKeywords: ['neo-soul', 'neo soul', 'soul', 'warm', 'smooth', 'organic', 'groove'], requiredNodes: ['harmony.scales.dorian', 'harmony.scales.mixolydian', 'timbre.instruments.piano', 'timbre.instruments.bass-guitar'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(85, 8), groove_strength: r(0.6, 0.85, 0.75), timing_feel: d({ behind: 0.7, on: 0.3 }), harmonic_rhythm: d({ medium: 0.5, slow: 0.3, fast: 0.2 }) },
  },
  {
    id: 'world-afrobeat',
    label: 'Afrobeat',
    genreHints: ['afrobeat', 'afro', 'world'],
    triggerConditions: [{ requiredKeywords: ['afrobeat', 'afro', 'polyrhythm', 'percussion', 'brass', 'groove'], requiredNodes: ['harmony.scales.dorian', 'harmony.scales.mixolydian', 'rhythm.polyrhythm.2-against-3', 'timbre.instruments.brass'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(115, 8), groove_strength: r(0.85, 1.0, 0.9), density: d({ dense: 0.5, medium: 0.5 }) },
  },
  {
    id: 'video-game-chiptune',
    label: 'Chiptune/8-bit',
    genreHints: ['chiptune', '8-bit', 'video game', 'retro'],
    triggerConditions: [{ requiredKeywords: ['chiptune', '8-bit', 'retro', 'video game', 'pixel', 'bitcrushed'], requiredNodes: ['timbre.effects.bitcrusher', 'harmony.scales.major', 'harmony.scales.minor'], minMatchCount: 2 }],
    parameterOverrides: { tempo: g(140, 15), density: d({ medium: 0.4, dense: 0.5, sparse: 0.1 }), dynamics: d({ mf: 0.5, f: 0.3, mp: 0.2 }) },
  },
];
