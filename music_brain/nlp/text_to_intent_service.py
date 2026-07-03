"""
Text-to-Intent Service

Wraps TextEmotionParser, KeywordExtractor, and emotional_mapping to convert
free-text descriptions into probabilistic musical parameter distributions.

Returns distributions (not fixed values) so the frontend can sample different
results from the same input, controlled by the interpretive slider.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from music_brain.nlp.keyword_extractor import KeywordExtractor, ExtractedKeywords

# Lazy imports for optional dependencies
_emotion_parser = None
_emotional_mapping = None


def _get_emotion_parser():
    global _emotion_parser
    if _emotion_parser is None:
        try:
            from music_brain.emotion.text_emotion_parser import TextEmotionParser

            _emotion_parser = TextEmotionParser()
        except ImportError:
            _emotion_parser = None
    return _emotion_parser


def _get_emotional_mapping():
    global _emotional_mapping
    if _emotional_mapping is None:
        try:
            from music_brain.data_utils import emotional_mapping

            _emotional_mapping = emotional_mapping
        except ImportError:
            _emotional_mapping = None
    return _emotional_mapping


@dataclass
class ParamDistribution:
    """A probability distribution over possible values."""

    type: str  # 'gaussian', 'discrete', 'range'
    center: Optional[float] = None
    spread: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    bias: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.center is not None:
            d["center"] = self.center
        if self.spread is not None:
            d["spread"] = self.spread
        if self.weights is not None:
            d["weights"] = self.weights
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.bias is not None:
            d["bias"] = self.bias
        return d


@dataclass
class ClusterActivation:
    """A detected context cluster with confidence."""

    id: str
    confidence: float
    label: str


@dataclass
class InterpretationResult:
    """Full result of text-to-intent interpretation."""

    param_distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    activated_clusters: List[Dict[str, Any]] = field(default_factory=list)
    activated_taxonomy_ids: List[str] = field(default_factory=list)
    detected_keywords: List[str] = field(default_factory=list)
    confidence: float = 0.5


# Context cluster definitions (mirroring frontend contextClusters.ts)
# These define what combinations of keywords activate which genre/style
CLUSTER_TRIGGERS: Dict[str, Dict[str, Any]] = {
    "metal-shred": {
        "label": "Metal/Prog Shred",
        "keywords": {
            "fast",
            "guitar",
            "overdrive",
            "distortion",
            "distorted",
            "metal",
            "shred",
            "heavy",
            "aggressive",
            "blistering",
        },
        "modes": {"dorian", "phrygian", "minor", "harmonic minor", "locrian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=160, spread=15),
            "density": ParamDistribution("range", min=0.8, max=1.0, bias=0.9),
            "dynamics": ParamDistribution("discrete", weights={"f": 0.4, "ff": 0.6}),
            "timing_feel": ParamDistribution("discrete", weights={"on": 0.6, "ahead": 0.4}),
            "groove_strength": ParamDistribution("range", min=0.5, max=0.8, bias=0.6),
        },
    },
    "blues-slow": {
        "label": "Slow Blues",
        "keywords": {
            "slow",
            "blues",
            "bluesy",
            "clean",
            "pentatonic",
            "laid-back",
            "laid back",
            "soulful",
            "mellow",
        },
        "modes": {
            "pentatonic",
            "minor pentatonic",
            "blues scale",
            "minor",
            "mixolydian",
            "dorian",
        },
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=72, spread=8),
            "density": ParamDistribution("discrete", weights={"sparse": 0.6, "medium": 0.4}),
            "dynamics": ParamDistribution("discrete", weights={"mp": 0.5, "mf": 0.3, "p": 0.2}),
            "timing_feel": ParamDistribution("discrete", weights={"behind": 0.8, "on": 0.2}),
            "space_probability": ParamDistribution("range", min=0.2, max=0.4, bias=0.3),
        },
    },
    "jazz-modal": {
        "label": "Modal Jazz",
        "keywords": {
            "jazz",
            "jazzy",
            "walking bass",
            "swing",
            "swinging",
            "cool",
            "modal",
            "sax",
            "saxophone",
            "piano",
        },
        "modes": {"dorian", "lydian", "mixolydian", "minor"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=130, spread=20),
            "density": ParamDistribution(
                "discrete", weights={"medium": 0.5, "dense": 0.3, "sparse": 0.2}
            ),
            "swing": ParamDistribution("range", min=0.5, max=0.8, bias=0.7),
            "harmonic_rhythm": ParamDistribution(
                "discrete", weights={"fast": 0.5, "medium": 0.4, "slow": 0.1}
            ),
        },
    },
    "ambient-pad": {
        "label": "Ambient Pad",
        "keywords": {
            "ambient",
            "pad",
            "synth pad",
            "reverb",
            "atmospheric",
            "ethereal",
            "dreamy",
            "floating",
            "spacious",
            "drone",
        },
        "modes": {"lydian", "major", "mixolydian", "whole tone"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=65, spread=10),
            "density": ParamDistribution("discrete", weights={"sparse": 0.7, "medium": 0.3}),
            "dynamics": ParamDistribution("discrete", weights={"pp": 0.4, "p": 0.4, "mp": 0.2}),
            "space_probability": ParamDistribution("range", min=0.3, max=0.6, bias=0.4),
        },
    },
    "funk-groove": {
        "label": "Funk Groove",
        "keywords": {
            "funk",
            "funky",
            "groovy",
            "groove",
            "syncopated",
            "slap",
            "slap bass",
            "tight",
            "bass",
        },
        "modes": {"dorian", "mixolydian", "minor", "pentatonic"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=105, spread=10),
            "groove_strength": ParamDistribution("range", min=0.8, max=1.0, bias=0.9),
            "timing_feel": ParamDistribution("discrete", weights={"ahead": 0.5, "on": 0.5}),
            "density": ParamDistribution("discrete", weights={"medium": 0.6, "dense": 0.4}),
        },
    },
    "pop-upbeat": {
        "label": "Upbeat Pop",
        "keywords": {
            "pop",
            "upbeat",
            "catchy",
            "bright",
            "happy",
            "energetic",
            "cheerful",
            "bouncy",
        },
        "modes": {"major", "lydian", "mixolydian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=118, spread=10),
            "density": ParamDistribution("discrete", weights={"medium": 0.7, "dense": 0.3}),
            "dynamics": ParamDistribution("discrete", weights={"mf": 0.5, "f": 0.3, "mp": 0.2}),
        },
    },
    "lo-fi-chill": {
        "label": "Lo-fi Chill",
        "keywords": {
            "lo-fi",
            "lofi",
            "chill",
            "vinyl",
            "tape",
            "warm",
            "cozy",
            "study",
            "relax",
            "mellow",
        },
        "modes": {"dorian", "mixolydian", "minor", "pentatonic"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=78, spread=6),
            "timing_feel": ParamDistribution("discrete", weights={"behind": 0.8, "on": 0.2}),
            "density": ParamDistribution("discrete", weights={"sparse": 0.6, "medium": 0.4}),
            "space_probability": ParamDistribution("range", min=0.15, max=0.3, bias=0.25),
        },
    },
    "edm-drop": {
        "label": "EDM Drop",
        "keywords": {
            "edm",
            "drop",
            "build",
            "buildup",
            "electronic",
            "synth",
            "bass drop",
            "heavy",
            "massive",
        },
        "modes": {"minor", "phrygian", "aeolian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=128, spread=5),
            "density": ParamDistribution("range", min=0.7, max=1.0, bias=0.9),
            "dynamics": ParamDistribution("discrete", weights={"ff": 0.7, "f": 0.3}),
        },
    },
    "hip-hop-boom-bap": {
        "label": "Boom Bap",
        "keywords": {
            "hip-hop",
            "hip hop",
            "boom bap",
            "rap",
            "oldschool",
            "drums",
            "sample",
            "vinyl",
        },
        "modes": {"minor", "dorian", "pentatonic"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=90, spread=5),
            "groove_strength": ParamDistribution("range", min=0.7, max=0.9, bias=0.8),
            "timing_feel": ParamDistribution("discrete", weights={"behind": 0.6, "on": 0.4}),
        },
    },
    "cinematic-epic": {
        "label": "Cinematic Epic",
        "keywords": {
            "cinematic",
            "epic",
            "orchestral",
            "film",
            "movie",
            "dramatic",
            "sweeping",
            "massive",
            "strings",
            "brass",
            "choir",
        },
        "modes": {"minor", "aeolian", "dorian", "phrygian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=90, spread=15),
            "density": ParamDistribution("range", min=0.5, max=1.0, bias=0.8),
            "dynamics": ParamDistribution(
                "discrete", weights={"ff": 0.3, "f": 0.3, "pp": 0.2, "p": 0.2}
            ),
        },
    },
    "punk-rock": {
        "label": "Punk Rock",
        "keywords": {
            "punk",
            "fast",
            "distortion",
            "distorted",
            "aggressive",
            "raw",
            "loud",
            "power chords",
            "angry",
        },
        "modes": {"minor", "phrygian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=175, spread=15),
            "density": ParamDistribution("range", min=0.7, max=1.0, bias=0.85),
            "dynamics": ParamDistribution("discrete", weights={"ff": 0.7, "f": 0.3}),
        },
    },
    "reggae": {
        "label": "Reggae",
        "keywords": {"reggae", "offbeat", "dub", "ska", "island", "bass", "laid-back"},
        "modes": {"minor", "dorian", "mixolydian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=80, spread=5),
            "timing_feel": ParamDistribution("discrete", weights={"behind": 0.7, "on": 0.3}),
            "groove_strength": ParamDistribution("range", min=0.7, max=0.9, bias=0.8),
        },
    },
    "post-rock": {
        "label": "Post-Rock",
        "keywords": {
            "post-rock",
            "atmospheric",
            "reverb",
            "crescendo",
            "building",
            "guitar",
            "delay",
            "ethereal",
            "wall of sound",
        },
        "modes": {"minor", "aeolian", "dorian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=90, spread=10),
            "dynamics": ParamDistribution(
                "discrete", weights={"pp": 0.3, "p": 0.2, "ff": 0.3, "f": 0.2}
            ),
            "space_probability": ParamDistribution("range", min=0.1, max=0.35, bias=0.2),
        },
    },
    "synthwave-retro": {
        "label": "Synthwave/Retrowave",
        "keywords": {
            "synthwave",
            "retrowave",
            "80s",
            "neon",
            "retro",
            "synth",
            "arpeggiated",
            "arpeggio",
            "driving",
        },
        "modes": {"minor", "dorian", "phrygian"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=118, spread=8),
            "density": ParamDistribution("discrete", weights={"medium": 0.6, "dense": 0.4}),
            "dynamics": ParamDistribution("discrete", weights={"mf": 0.4, "f": 0.4, "mp": 0.2}),
        },
    },
    "neo-soul": {
        "label": "Neo-Soul",
        "keywords": {
            "neo-soul",
            "neo soul",
            "soul",
            "warm",
            "smooth",
            "keys",
            "piano",
            "bass",
            "groove",
            "organic",
        },
        "modes": {"dorian", "mixolydian", "minor"},
        "min_match": 2,
        "overrides": {
            "tempo": ParamDistribution("gaussian", center=85, spread=8),
            "groove_strength": ParamDistribution("range", min=0.6, max=0.85, bias=0.75),
            "timing_feel": ParamDistribution("discrete", weights={"behind": 0.7, "on": 0.3}),
        },
    },
}


class TextToIntentService:
    """
    Convert free-text music descriptions into probabilistic parameter
    distributions.

    Uses:
    - TextEmotionParser for emotional valence/arousal
    - KeywordExtractor for concrete musical terms
    - CLUSTER_TRIGGERS for context-dependent interpretation
    - MusicalParameters (from emotional_mapping) for emotion→music mapping
    """

    def __init__(self):
        self.keyword_extractor = KeywordExtractor()

    def parse(
        self,
        text: str,
        locale: str = "en",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse text into probabilistic musical parameters.

        Returns a dict matching the ParseTextResponse schema expected by
        the frontend.
        """
        # 1. Extract musical keywords
        keywords = self.keyword_extractor.extract(text)

        # 2. Parse emotion (optional — works without it)
        emotion_distributions = self._parse_emotion(text)

        # 3. Match context clusters
        clusters = self._match_clusters(keywords)

        # 4. Merge distributions: emotion base + cluster overrides
        merged = self._merge_distributions(emotion_distributions, clusters)

        # 5. Inject keyword-derived distributions
        if keywords.mode_words:
            merged = self._apply_mode_keywords(merged, keywords.mode_words)

        # 6. Build taxonomy node activations
        taxonomy_ids = list(set(keywords.taxonomy_hints))

        # 7. Calculate overall confidence
        confidence = self._calculate_confidence(keywords, clusters)

        return {
            "param_distributions": {k: v.to_dict() for k, v in merged.items()},
            "activated_clusters": [
                {"id": c_id, "confidence": c_data["score"], "label": c_data["label"]}
                for c_id, c_data in clusters.items()
            ],
            "activated_taxonomy_ids": taxonomy_ids,
            "detected_keywords": keywords.all_matched,
            "confidence": confidence,
        }

    def _parse_emotion(self, text: str) -> Dict[str, ParamDistribution]:
        """Use TextEmotionParser + emotional_mapping for emotion-based params."""
        distributions: Dict[str, ParamDistribution] = {}

        parser = _get_emotion_parser()
        mapping = _get_emotional_mapping()

        if parser is None or mapping is None:
            return distributions

        try:
            parsed = parser.parse(text)
            emotional_state_dict = parsed.to_emotional_state_dict()

            # Create EmotionalState
            state = mapping.EmotionalState(
                valence=emotional_state_dict["valence"],
                arousal=emotional_state_dict["arousal"],
                primary_emotion=emotional_state_dict["primary_emotion"],
                secondary_emotions=emotional_state_dict.get("secondary_emotions", []),
                has_intrusions=emotional_state_dict.get("has_intrusions", False),
            )

            # Get musical parameters (already has ranges + distributions!)
            params = mapping.get_parameters_for_state(state)

            # Convert MusicalParameters to ParamDistribution objects
            distributions["tempo"] = ParamDistribution(
                "gaussian",
                center=params.tempo_suggested,
                spread=(params.tempo_max - params.tempo_min) / 4,
            )
            distributions["mode_weights"] = ParamDistribution(
                "discrete",
                weights=dict(params.mode_weights),
            )
            distributions["density"] = ParamDistribution(
                "discrete",
                weights=(
                    {params.density.value: 0.7, "medium": 0.3}
                    if params.density.value != "medium"
                    else {"medium": 0.7, "sparse": 0.15, "dense": 0.15}
                ),
            )
            distributions["timing_feel"] = ParamDistribution(
                "discrete",
                weights=(
                    {params.timing_feel.value: 0.7, "on": 0.3}
                    if params.timing_feel.value != "on"
                    else {"on": 0.6, "behind": 0.2, "ahead": 0.2}
                ),
            )
            distributions["dynamics"] = ParamDistribution(
                "discrete",
                weights=(
                    {params.dynamics: 0.6, "mf": 0.2, "mp": 0.2}
                    if params.dynamics not in ("mf", "mp")
                    else {params.dynamics: 0.5, "f": 0.25, "p": 0.25}
                ),
            )
            distributions["space_probability"] = ParamDistribution(
                "range",
                min=max(0, params.space_probability - 0.1),
                max=min(1, params.space_probability + 0.1),
                bias=params.space_probability,
            )
            distributions["dissonance"] = ParamDistribution(
                "range",
                min=max(0, params.dissonance - 0.1),
                max=min(1, params.dissonance + 0.1),
                bias=params.dissonance,
            )
        except Exception:
            pass

        return distributions

    def _match_clusters(self, keywords: ExtractedKeywords) -> Dict[str, Dict[str, Any]]:
        """Match extracted keywords against cluster triggers."""
        matched: Dict[str, Dict[str, Any]] = {}
        all_words = set(w.lower() for w in keywords.all_matched)
        all_modes = set(w.lower() for w in keywords.mode_words)

        for cluster_id, cluster_def in CLUSTER_TRIGGERS.items():
            trigger_keywords = cluster_def["keywords"]
            trigger_modes = cluster_def["modes"]
            min_match = cluster_def.get("min_match", 2)

            # Count keyword matches
            keyword_matches = all_words & trigger_keywords
            mode_matches = all_modes & trigger_modes

            total_matches = len(keyword_matches) + len(mode_matches)

            if total_matches >= min_match:
                score = min(1.0, total_matches / max(1, min_match + 1))
                matched[cluster_id] = {
                    "score": round(score, 3),
                    "label": cluster_def["label"],
                    "matched_keywords": list(keyword_matches),
                    "matched_modes": list(mode_matches),
                    "overrides": cluster_def["overrides"],
                }

        # Sort by score, keep top 5
        sorted_clusters = dict(
            sorted(matched.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
        )
        return sorted_clusters

    def _merge_distributions(
        self,
        emotion_base: Dict[str, ParamDistribution],
        clusters: Dict[str, Dict[str, Any]],
    ) -> Dict[str, ParamDistribution]:
        """Merge emotion-derived distributions with cluster overrides."""
        merged = dict(emotion_base)

        # Apply cluster overrides weighted by score, weakest first so the
        # highest-scored cluster has the final (strongest) say.
        for cluster_id, cluster_data in reversed(list(clusters.items())):
            score = cluster_data["score"]
            overrides: Dict[str, ParamDistribution] = cluster_data["overrides"]

            for param, dist in overrides.items():
                if param not in merged:
                    merged[param] = dist
                else:
                    # Blend: cluster with high score overrides more
                    existing = merged[param]
                    merged[param] = self._blend_distributions(existing, dist, score)

        # Default distributions for params that weren't set
        if "tempo" not in merged:
            merged["tempo"] = ParamDistribution("gaussian", center=120, spread=20)
        if "mode_weights" not in merged:
            merged["mode_weights"] = ParamDistribution(
                "discrete",
                weights={
                    "major": 0.25,
                    "minor": 0.25,
                    "dorian": 0.15,
                    "mixolydian": 0.15,
                    "lydian": 0.1,
                    "phrygian": 0.1,
                },
            )
        if "density" not in merged:
            merged["density"] = ParamDistribution(
                "discrete",
                weights={"sparse": 0.3, "medium": 0.4, "dense": 0.3},
            )

        return merged

    def _blend_distributions(
        self,
        a: ParamDistribution,
        b: ParamDistribution,
        b_weight: float,
    ) -> ParamDistribution:
        """Blend two distributions, with b weighted by b_weight."""
        a_weight = 1.0 - b_weight

        if a.type == "gaussian" and b.type == "gaussian":
            return ParamDistribution(
                "gaussian",
                center=(a.center or 0) * a_weight + (b.center or 0) * b_weight,
                spread=max(a.spread or 0, b.spread or 0),
            )
        elif a.type == "discrete" and b.type == "discrete":
            merged_weights: Dict[str, float] = {}
            all_keys = set(list(a.weights or {}) + list(b.weights or {}))
            for k in all_keys:
                av = (a.weights or {}).get(k, 0)
                bv = (b.weights or {}).get(k, 0)
                merged_weights[k] = av * a_weight + bv * b_weight
            # Normalize
            total = sum(merged_weights.values())
            if total > 0:
                merged_weights = {k: v / total for k, v in merged_weights.items()}
            return ParamDistribution("discrete", weights=merged_weights)
        elif a.type == "range" and b.type == "range":
            return ParamDistribution(
                "range",
                min=(a.min or 0) * a_weight + (b.min or 0) * b_weight,
                max=(a.max or 1) * a_weight + (b.max or 1) * b_weight,
                bias=(a.bias or 0.5) * a_weight + (b.bias or 0.5) * b_weight,
            )
        else:
            # Type mismatch — higher weight wins
            return b if b_weight > 0.5 else a

    def _apply_mode_keywords(
        self,
        distributions: Dict[str, ParamDistribution],
        mode_words: List[str],
    ) -> Dict[str, ParamDistribution]:
        """Boost mode weights for explicitly mentioned modes."""
        if "mode_weights" not in distributions:
            distributions["mode_weights"] = ParamDistribution("discrete", weights={})

        weights = dict(distributions["mode_weights"].weights or {})

        # Boost mentioned modes significantly
        for mode in mode_words:
            canonical = mode.lower().replace(" ", "_")
            # Map some aliases
            alias_map = {
                "pentatonic": "pentatonic_minor",
                "minor_pentatonic": "pentatonic_minor",
                "major_pentatonic": "pentatonic_major",
                "blues_scale": "blues",
                "harmonic_minor": "harmonic_minor",
                "melodic_minor": "melodic_minor",
                "aeolian": "minor",
            }
            canonical = alias_map.get(canonical, canonical)
            weights[canonical] = weights.get(canonical, 0) + 0.5

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 3) for k, v in weights.items()}

        distributions["mode_weights"] = ParamDistribution("discrete", weights=weights)
        return distributions

    def _calculate_confidence(
        self,
        keywords: ExtractedKeywords,
        clusters: Dict[str, Dict[str, Any]],
    ) -> float:
        """Calculate overall interpretation confidence."""
        keyword_score = min(1.0, len(keywords.all_matched) / 5)
        cluster_score = max(c["score"] for c in clusters.values()) if clusters else 0.3
        return round((keyword_score * 0.4 + cluster_score * 0.6), 3)
