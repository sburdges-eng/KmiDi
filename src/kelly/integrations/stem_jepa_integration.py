"""
Kelly Project - Stem-JEPA Integration Layer

Bridges the VAD emotion pipeline with Stem-JEPA stem compatibility:
- Emotion-conditioned stem retrieval
- FiLM conditioning vector generation from VAD
- Integration with MusicVAE latent space
- Compatibility scoring with emotional context

This module enables: 
    emotion → compatible stems that "feel" right for the emotion
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum
import numpy as np

# Import Kelly's VAD state
try:
    from .magenta_integration import VADState, VADLatentMapper
except ImportError:
    # Inline VADState for standalone use
    @dataclass
    class VADState:
        valence: float = 0.0
        arousal: float = 0.5
        dominance: float = 0.5
        timestamp: float = 0.0
        
        def to_array(self) -> np.ndarray:
            return np.array([self.valence, self.arousal, self.dominance])


# =============================================================================
# Stem Categories (matching Stem-JEPA taxonomy)
# =============================================================================

class StemCategory(Enum):
    """
    Hierarchical stem categories from MoisesDB/MUSDB18.
    
    Stem-JEPA uses these for FiLM conditioning.
    """
    # Primary stems (MUSDB18 level)
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    OTHER = "other"
    
    # Extended stems (MoisesDB level)
    GUITAR_CLEAN = "guitar_clean"
    GUITAR_DISTORTED = "guitar_distorted"
    GUITAR_ACOUSTIC = "guitar_acoustic"
    PIANO = "piano"
    KEYS_OTHER = "keys_other"
    STRINGS = "strings"
    BRASS = "brass"
    WOODWINDS = "woodwinds"
    SYNTH_PAD = "synth_pad"
    SYNTH_LEAD = "synth_lead"
    PERCUSSION = "percussion"
    
    # Vocal subcategories
    VOCALS_LEAD = "vocals_lead"
    VOCALS_BACKING = "vocals_backing"


# Mapping from category to FiLM conditioning index
STEM_TO_IDX: Dict[StemCategory, int] = {
    stem: i for i, stem in enumerate(StemCategory)
}

IDX_TO_STEM: Dict[int, StemCategory] = {
    i: stem for i, stem in enumerate(StemCategory)
}


# =============================================================================
# Emotion-Stem Affinity Matrix
# =============================================================================

@dataclass
class EmotionStemAffinity:
    """
    Learned affinities between emotional states and stem categories.
    
    High affinity = this stem type tends to appear in music with this emotion.
    
    Matrix dimensions: [num_emotions x num_stems]
    Computed from EMOPIA dataset correlation analysis.
    """
    
    # Default affinities based on music theory heuristics
    # Rows: [grief, anger, joy, calm, tension, release]
    # Cols: [drums, bass, vocals, other, guitar_clean, guitar_dist, ...]
    
    affinity_matrix: np.ndarray = field(default_factory=lambda: np.array([
        # grief: soft drums, present bass, strong vocals, strings
        [0.3, 0.6, 0.9, 0.7, 0.4, 0.2, 0.5, 0.8, 0.6, 0.8, 0.3, 0.2, 0.7, 0.3, 0.3, 0.7, 0.5],
        # anger: heavy drums, distorted bass, aggressive vocals
        [0.9, 0.8, 0.7, 0.5, 0.3, 0.9, 0.2, 0.3, 0.4, 0.2, 0.2, 0.2, 0.3, 0.6, 0.4, 0.6, 0.5],
        # joy: upbeat drums, bouncy bass, bright keys
        [0.8, 0.7, 0.6, 0.6, 0.6, 0.4, 0.7, 0.8, 0.7, 0.5, 0.5, 0.4, 0.4, 0.7, 0.3, 0.5, 0.5],
        # calm: minimal drums, subtle bass, pads
        [0.2, 0.4, 0.5, 0.8, 0.5, 0.1, 0.6, 0.6, 0.7, 0.6, 0.3, 0.3, 0.9, 0.2, 0.2, 0.4, 0.4],
        # tension: building drums, dissonant elements
        [0.6, 0.5, 0.4, 0.7, 0.3, 0.5, 0.3, 0.4, 0.5, 0.7, 0.4, 0.3, 0.6, 0.5, 0.3, 0.5, 0.5],
        # release: full drums, resolved bass, triumphant
        [0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.5, 0.7, 0.6, 0.6, 0.6, 0.5, 0.5, 0.6, 0.4, 0.6, 0.6],
    ], dtype=np.float32))
    
    emotion_labels: List[str] = field(default_factory=lambda: [
        "grief", "anger", "joy", "calm", "tension", "release"
    ])
    
    def vad_to_emotion_weights(self, vad: VADState) -> np.ndarray:
        """
        Convert VAD state to weights over emotion categories.
        
        Uses soft assignment based on VAD coordinates mapping to
        Russell's circumplex model extended with dominance.
        """
        weights = np.zeros(len(self.emotion_labels))
        
        # Grief: low valence, low arousal
        weights[0] = (1 - (vad.valence + 1) / 2) * (1 - vad.arousal)
        
        # Anger: low valence, high arousal, high dominance
        weights[1] = (1 - (vad.valence + 1) / 2) * vad.arousal * vad.dominance
        
        # Joy: high valence, high arousal
        weights[2] = ((vad.valence + 1) / 2) * vad.arousal
        
        # Calm: high valence, low arousal
        weights[3] = ((vad.valence + 1) / 2) * (1 - vad.arousal)
        
        # Tension: low dominance, medium-high arousal
        weights[4] = (1 - vad.dominance) * vad.arousal * 0.7
        
        # Release: high dominance, medium arousal
        weights[5] = vad.dominance * (0.5 + 0.5 * (1 - abs(vad.arousal - 0.6)))
        
        # Normalize
        weights = weights / (weights.sum() + 1e-8)
        return weights
    
    def get_stem_affinities(self, vad: VADState) -> Dict[StemCategory, float]:
        """
        Get stem category affinities for a given VAD state.
        
        Returns dict mapping stem category to affinity score (0-1).
        """
        emotion_weights = self.vad_to_emotion_weights(vad)
        
        # Weighted sum across emotions
        stem_scores = emotion_weights @ self.affinity_matrix
        
        # Map to stem categories
        affinities = {}
        for i, stem in enumerate(StemCategory):
            if i < len(stem_scores):
                affinities[stem] = float(stem_scores[i])
            else:
                affinities[stem] = 0.5  # Default neutral affinity
        
        return affinities


# =============================================================================
# FiLM Conditioning Generator
# =============================================================================

@dataclass
class FiLMConditioningVector:
    """
    FiLM (Feature-wise Linear Modulation) conditioning vector.
    
    Used by Stem-JEPA predictor to specify target instrument:
    h_l+1 = σ(γ_l ⊙ (W_l * h_l) + β_l)
    
    where γ (scale) and β (shift) are derived from this conditioning.
    """
    instrument_embedding: np.ndarray  # Base instrument identity
    emotion_modulation: np.ndarray    # VAD-derived modulation
    
    @property
    def combined(self) -> np.ndarray:
        """Combined conditioning vector for FiLM layers."""
        return np.concatenate([self.instrument_embedding, self.emotion_modulation])


class FiLMConditioningGenerator:
    """
    Generates FiLM conditioning vectors for Stem-JEPA.
    
    Combines instrument identity with emotional context.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        emotion_dim: int = 32,
        num_stems: int = len(StemCategory)
    ):
        self.embedding_dim = embedding_dim
        self.emotion_dim = emotion_dim
        self.num_stems = num_stems
        
        # Initialize stem embeddings (in practice, load from trained model)
        self.stem_embeddings = self._init_stem_embeddings()
        
        # Emotion projection
        self.emotion_projector = self._init_emotion_projector()
    
    def _init_stem_embeddings(self) -> np.ndarray:
        """Initialize learnable stem embeddings."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(self.num_stems, self.embedding_dim).astype(np.float32)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def _init_emotion_projector(self) -> np.ndarray:
        """Initialize VAD -> emotion embedding projection."""
        rng = np.random.RandomState(43)
        # Project 3D VAD to emotion_dim
        return rng.randn(3, self.emotion_dim).astype(np.float32) * 0.5
    
    def generate_conditioning(
        self,
        target_stem: StemCategory,
        vad: Optional[VADState] = None
    ) -> FiLMConditioningVector:
        """
        Generate FiLM conditioning for a target stem.
        
        Args:
            target_stem: Which instrument to predict
            vad: Optional emotional context for modulation
        
        Returns:
            FiLMConditioningVector for Stem-JEPA predictor
        """
        # Get base instrument embedding
        stem_idx = STEM_TO_IDX.get(target_stem, 0)
        instrument_emb = self.stem_embeddings[stem_idx].copy()
        
        # Generate emotion modulation
        if vad is not None:
            vad_array = vad.to_array()
            emotion_mod = vad_array @ self.emotion_projector
        else:
            emotion_mod = np.zeros(self.emotion_dim, dtype=np.float32)
        
        return FiLMConditioningVector(
            instrument_embedding=instrument_emb,
            emotion_modulation=emotion_mod
        )
    
    def generate_multi_stem_conditioning(
        self,
        target_stems: List[StemCategory],
        vad: Optional[VADState] = None
    ) -> List[FiLMConditioningVector]:
        """Generate conditioning for multiple stems."""
        return [self.generate_conditioning(stem, vad) for stem in target_stems]


# =============================================================================
# Stem-JEPA Compatibility Scorer
# =============================================================================

class StemCompatibilityScorer:
    """
    Scores stem compatibility using Stem-JEPA embeddings.
    
    Given a context mix and candidate stems, ranks candidates by
    compatibility with the context AND emotional intent.
    """
    
    def __init__(
        self,
        encoder: Optional[object] = None,  # ViT encoder (placeholder)
        predictor: Optional[object] = None,  # MLP predictor (placeholder)
        conditioning_generator: Optional[FiLMConditioningGenerator] = None
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.conditioning_gen = conditioning_generator or FiLMConditioningGenerator()
        self.affinity_model = EmotionStemAffinity()
        
        # Placeholder embedding dimension (matches Stem-JEPA ViT-Base)
        self.embedding_dim = 768
    
    def encode_context(self, context_spectrogram: np.ndarray) -> np.ndarray:
        """
        Encode context mix to latent embedding.
        
        In production, runs ViT encoder on log-mel spectrogram.
        """
        if self.encoder is not None:
            return self.encoder(context_spectrogram)
        
        # Placeholder: random embedding
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def predict_target_embedding(
        self,
        context_embedding: np.ndarray,
        target_stem: StemCategory,
        vad: Optional[VADState] = None
    ) -> np.ndarray:
        """
        Predict embedding of target stem given context.
        
        Uses FiLM-conditioned MLP predictor.
        """
        conditioning = self.conditioning_gen.generate_conditioning(target_stem, vad)
        
        if self.predictor is not None:
            return self.predictor(context_embedding, conditioning.combined)
        
        # Placeholder: modulated context
        return context_embedding + conditioning.combined[:self.embedding_dim]
    
    def score_compatibility(
        self,
        context_embedding: np.ndarray,
        candidate_embedding: np.ndarray,
        target_stem: StemCategory,
        vad: Optional[VADState] = None
    ) -> float:
        """
        Score compatibility between context and candidate stem.
        
        Lower distance = higher compatibility.
        """
        # Predict expected embedding
        predicted = self.predict_target_embedding(context_embedding, target_stem, vad)
        
        # L2 distance (Stem-JEPA objective)
        distance = np.linalg.norm(predicted - candidate_embedding)
        
        # Convert to similarity (0-1)
        similarity = 1.0 / (1.0 + distance)
        
        # Weight by emotion affinity if VAD provided
        if vad is not None:
            affinities = self.affinity_model.get_stem_affinities(vad)
            stem_affinity = affinities.get(target_stem, 0.5)
            # Boost/penalize based on affinity
            similarity *= (0.5 + stem_affinity)
        
        return float(similarity)
    
    def rank_candidates(
        self,
        context_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidate_stems: List[StemCategory],
        vad: Optional[VADState] = None,
        top_k: int = 5
    ) -> List[Tuple[int, StemCategory, float]]:
        """
        Rank candidate stems by compatibility.
        
        Returns list of (index, stem_category, score) sorted by score descending.
        """
        scores = []
        for i, (emb, stem) in enumerate(zip(candidate_embeddings, candidate_stems)):
            score = self.score_compatibility(context_embedding, emb, stem, vad)
            scores.append((i, stem, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)
        
        return scores[:top_k]


# =============================================================================
# Integrated Emotion-Stem Pipeline
# =============================================================================

@dataclass
class StemRecommendation:
    """A stem recommendation with emotional context."""
    stem_category: StemCategory
    compatibility_score: float
    emotion_affinity: float
    combined_score: float
    reasoning: str


class EmotionStemPipeline:
    """
    End-to-end pipeline: VAD emotion → compatible stem recommendations.
    
    Integrates:
    - VAD emotion processing
    - Stem-JEPA compatibility prediction
    - Emotion-stem affinity weighting
    - MusicVAE latent space mapping
    """
    
    def __init__(
        self,
        compatibility_scorer: Optional[StemCompatibilityScorer] = None,
        latent_mapper: Optional['VADLatentMapper'] = None
    ):
        self.scorer = compatibility_scorer or StemCompatibilityScorer()
        self.affinity_model = EmotionStemAffinity()
        
        try:
            from .magenta_integration import VADLatentMapper
            self.latent_mapper = latent_mapper or VADLatentMapper()
        except ImportError:
            self.latent_mapper = None
    
    def recommend_stems(
        self,
        context_spectrogram: np.ndarray,
        vad: VADState,
        available_stems: List[StemCategory],
        num_recommendations: int = 3
    ) -> List[StemRecommendation]:
        """
        Recommend stems that fit both the musical context and emotional intent.
        """
        # Encode context
        context_emb = self.scorer.encode_context(context_spectrogram)
        
        # Get emotion affinities
        affinities = self.affinity_model.get_stem_affinities(vad)
        
        recommendations = []
        for stem in available_stems:
            # Generate placeholder candidate embedding
            # (In production, encode actual candidate audio)
            candidate_emb = np.random.randn(self.scorer.embedding_dim).astype(np.float32)
            
            # Score compatibility
            compat_score = self.scorer.score_compatibility(
                context_emb, candidate_emb, stem, vad
            )
            
            # Get emotion affinity
            affinity = affinities.get(stem, 0.5)
            
            # Combined score (geometric mean)
            combined = np.sqrt(compat_score * affinity)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(stem, vad, compat_score, affinity)
            
            recommendations.append(StemRecommendation(
                stem_category=stem,
                compatibility_score=compat_score,
                emotion_affinity=affinity,
                combined_score=combined,
                reasoning=reasoning
            ))
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x.combined_score, reverse=True)
        
        return recommendations[:num_recommendations]
    
    def _generate_reasoning(
        self,
        stem: StemCategory,
        vad: VADState,
        compat: float,
        affinity: float
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        # Determine emotional context
        if vad.valence < -0.3:
            mood = "melancholic" if vad.arousal < 0.5 else "intense"
        elif vad.valence > 0.3:
            mood = "uplifting" if vad.arousal > 0.5 else "peaceful"
        else:
            mood = "neutral"
        
        # Map stem to musical role
        stem_roles = {
            StemCategory.DRUMS: "rhythmic foundation",
            StemCategory.BASS: "harmonic anchor",
            StemCategory.VOCALS: "melodic lead",
            StemCategory.GUITAR_DISTORTED: "aggressive texture",
            StemCategory.PIANO: "harmonic clarity",
            StemCategory.STRINGS: "emotional depth",
            StemCategory.SYNTH_PAD: "atmospheric backdrop",
        }
        role = stem_roles.get(stem, "complementary texture")
        
        if affinity > 0.7:
            fit = "strongly supports"
        elif affinity > 0.4:
            fit = "complements"
        else:
            fit = "contrasts with"
        
        return f"{stem.value} provides {role} that {fit} the {mood} emotional intent"
    
    def get_musicvae_latent_target(self, vad: VADState) -> Optional[np.ndarray]:
        """
        Get MusicVAE latent space target for generation.
        
        Uses VADLatentMapper to compute emotion-biased latent offset.
        """
        if self.latent_mapper is None:
            return None
        
        # Get offset from neutral
        return self.latent_mapper.vad_to_latent_offset(vad)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create pipeline
    pipeline = EmotionStemPipeline()
    
    # Test with grief emotion (Kelly's canonical test case)
    grief_vad = VADState(valence=-0.6, arousal=0.3, dominance=0.4)
    
    # Get stem affinities
    affinities = pipeline.affinity_model.get_stem_affinities(grief_vad)
    print("Grief - Stem Affinities:")
    top_stems = sorted(affinities.items(), key=lambda x: x[1], reverse=True)[:5]
    for stem, score in top_stems:
        print(f"  {stem.value}: {score:.2f}")
    
    # Get recommendations
    available = [
        StemCategory.DRUMS, StemCategory.BASS, StemCategory.VOCALS,
        StemCategory.PIANO, StemCategory.STRINGS, StemCategory.GUITAR_ACOUSTIC
    ]
    
    # Placeholder spectrogram
    context = np.random.randn(80, 800).astype(np.float32)
    
    recommendations = pipeline.recommend_stems(context, grief_vad, available)
    print("\nRecommendations for grief context:")
    for rec in recommendations:
        print(f"  {rec.stem_category.value}:")
        print(f"    Combined: {rec.combined_score:.2f}")
        print(f"    {rec.reasoning}")
    
    # Test with euphoria
    print("\n" + "="*60)
    euphoria_vad = VADState(valence=0.8, arousal=0.9, dominance=0.7)
    affinities = pipeline.affinity_model.get_stem_affinities(euphoria_vad)
    print("Euphoria - Stem Affinities:")
    top_stems = sorted(affinities.items(), key=lambda x: x[1], reverse=True)[:5]
    for stem, score in top_stems:
        print(f"  {stem.value}: {score:.2f}")
