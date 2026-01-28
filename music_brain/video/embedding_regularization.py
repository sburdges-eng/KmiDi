"""
Embedding Regularization for Video Generation

Implements regularization techniques to improve embedding prediction accuracy
for emotion-to-visual mappings while maintaining fast inference.

Key features:
- L1/L2 regularization for embedding weights
- Dropout for robust embeddings
- Fast inference optimized for real-time video generation
- Integration with C++ RT-safe architecture
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from pathlib import Path


class RegularizationType(Enum):
    """Types of regularization."""
    L1 = "l1"           # Lasso regularization (sparse weights)
    L2 = "l2"           # Ridge regularization (small weights)
    ELASTIC = "elastic"  # Combination of L1 and L2
    DROPOUT = "dropout"  # Random dropout during training
    NONE = "none"


@dataclass
class RegularizationConfig:
    """Configuration for embedding regularization."""
    
    # Regularization type
    regularization_type: RegularizationType = RegularizationType.L2
    
    # Regularization strength
    l1_lambda: float = 0.0001  # L1 regularization strength
    l2_lambda: float = 0.001   # L2 regularization strength
    elastic_ratio: float = 0.5  # Mix of L1/L2 for elastic net (0=L2, 1=L1)
    
    # Dropout (for training robustness)
    dropout_rate: float = 0.1   # Probability of dropping units
    use_dropout_inference: bool = False  # Usually False for inference
    
    # Gradient clipping (prevents exploding gradients)
    clip_gradients: bool = True
    max_gradient_norm: float = 1.0
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10  # Epochs without improvement before stopping
    min_delta: float = 0.0001  # Minimum improvement to count
    
    # Weight decay
    weight_decay: float = 0.0001  # Optimizer weight decay
    
    # Batch normalization
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    
    # Performance optimization
    use_fast_inference: bool = True  # Disable regularization during inference
    cache_embeddings: bool = True    # Cache computed embeddings
    use_quantization: bool = False   # 8-bit quantization for speed
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for emotion-visual embeddings."""
    
    # Embedding dimensions
    emotion_embedding_dim: int = 128    # Emotion embedding size
    visual_embedding_dim: int = 256     # Visual parameter embedding size
    hidden_dims: List[int] = field(default_factory=lambda: [512, 384, 256])
    
    # Embedding initialization
    init_method: str = "xavier_uniform"  # xavier_uniform, kaiming_normal, etc.
    init_scale: float = 0.02
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd, rmsprop
    beta1: float = 0.9       # Adam beta1
    beta2: float = 0.999     # Adam beta2
    epsilon: float = 1e-8    # Adam epsilon
    
    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 1000
    warmup_steps: int = 500
    
    # Regularization
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    
    # Fast inference optimization
    use_fp16: bool = False  # Half precision for faster inference
    compile_to_cpp: bool = True  # Compile to C++ for RT performance
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingRegularizer:
    """
    Handles regularization for emotion-visual embeddings.
    
    Provides:
    - Fast inference with optional regularization
    - Training-time regularization for better generalization
    - Integration with C++ RT-safe audio engine
    - Embedding caching for performance
    
    Example:
        >>> config = RegularizationConfig(
        ...     regularization_type=RegularizationType.L2,
        ...     l2_lambda=0.001,
        ...     use_fast_inference=True
        ... )
        >>> regularizer = EmbeddingRegularizer(config)
        >>> loss = regularizer.compute_regularization_loss(weights)
    """
    
    def __init__(self, config: Optional[RegularizationConfig] = None):
        """
        Initialize the embedding regularizer.
        
        Args:
            config: Regularization configuration
        """
        self.config = config or RegularizationConfig()
        self._training_mode = True
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def compute_regularization_loss(
        self,
        weights: np.ndarray,
        training: bool = True
    ) -> float:
        """
        Compute regularization loss for given weights.
        
        Args:
            weights: Weight matrix to regularize
            training: Whether in training mode (False skips for fast inference)
        
        Returns:
            Regularization loss value
        
        Note:
            Returns 0.0 during inference if use_fast_inference=True
        """
        # Skip during fast inference
        if not training and self.config.use_fast_inference:
            return 0.0
        
        loss = 0.0
        
        if self.config.regularization_type == RegularizationType.L1:
            # L1 regularization: sum of absolute values
            loss = self.config.l1_lambda * np.sum(np.abs(weights))
        
        elif self.config.regularization_type == RegularizationType.L2:
            # L2 regularization: sum of squares
            loss = self.config.l2_lambda * np.sum(weights ** 2)
        
        elif self.config.regularization_type == RegularizationType.ELASTIC:
            # Elastic Net: combination of L1 and L2
            l1_component = np.sum(np.abs(weights))
            l2_component = np.sum(weights ** 2)
            
            ratio = self.config.elastic_ratio
            loss = (
                ratio * self.config.l1_lambda * l1_component +
                (1 - ratio) * self.config.l2_lambda * l2_component
            )
        
        return loss
    
    def apply_dropout(
        self,
        embeddings: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply dropout to embeddings.
        
        Args:
            embeddings: Input embeddings
            training: Whether in training mode
        
        Returns:
            Embeddings with dropout applied (if training)
        
        Note:
            Dropout is typically disabled during inference for consistency.
        """
        if not training or self.config.dropout_rate == 0.0:
            return embeddings
        
        if not training and not self.config.use_dropout_inference:
            return embeddings
        
        # Create dropout mask
        mask = np.random.binomial(
            1,
            1 - self.config.dropout_rate,
            size=embeddings.shape
        )
        
        # Scale to maintain expected value
        scale = 1.0 / (1.0 - self.config.dropout_rate)
        
        return embeddings * mask * scale
    
    def clip_gradients(
        self,
        gradients: np.ndarray
    ) -> np.ndarray:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            gradients: Gradient array
        
        Returns:
            Clipped gradients
        """
        if not self.config.clip_gradients:
            return gradients
        
        # Compute gradient norm
        grad_norm = np.linalg.norm(gradients)
        
        # Clip if necessary
        if grad_norm > self.config.max_gradient_norm:
            gradients = gradients * (self.config.max_gradient_norm / grad_norm)
        
        return gradients
    
    def set_training_mode(self, training: bool) -> None:
        """
        Set training vs inference mode.
        
        Args:
            training: True for training, False for inference
        """
        self._training_mode = training
        
        # Clear cache when switching to training
        if training:
            self._embedding_cache.clear()
    
    def cache_embedding(
        self,
        key: str,
        embedding: np.ndarray
    ) -> None:
        """
        Cache an embedding for fast retrieval.
        
        Args:
            key: Cache key (e.g., emotion name)
            embedding: Embedding to cache
        """
        if self.config.cache_embeddings:
            self._embedding_cache[key] = embedding.copy()
    
    def get_cached_embedding(
        self,
        key: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve a cached embedding.
        
        Args:
            key: Cache key
        
        Returns:
            Cached embedding or None if not found
        """
        return self._embedding_cache.get(key)
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()


class FastEmbeddingPredictor:
    """
    Fast embedding predictor optimized for real-time video generation.
    
    Uses regularized embeddings with optimizations for speed:
    - Cached embeddings for common emotions
    - Optional quantization
    - C++ compatible inference path
    - Minimal memory allocations
    
    Example:
        >>> predictor = FastEmbeddingPredictor()
        >>> visual_embedding = predictor.predict(
        ...     emotion="grief",
        ...     intensity=0.8
        ... )
    """
    
    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the fast embedding predictor.
        
        Args:
            embedding_config: Embedding configuration
        """
        self.config = embedding_config or EmbeddingConfig()
        self.regularizer = EmbeddingRegularizer(self.config.regularization)
        
        # Initialize embedding matrices (stub - would be learned)
        self._emotion_embeddings: Dict[str, np.ndarray] = {}
        self._projection_matrix: Optional[np.ndarray] = None
        
        # Performance tracking
        self._inference_count = 0
        self._cache_hits = 0
    
    def predict(
        self,
        emotion: str,
        intensity: float = 1.0,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Predict visual embedding from emotion.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity 0.0-1.0
            use_cache: Whether to use cached embeddings
        
        Returns:
            Visual parameter embedding
        
        Note:
            This is a stub. Real implementation would use learned weights.
        """
        self._inference_count += 1
        
        # Check cache first (fast path)
        if use_cache and self.config.regularization.cache_embeddings:
            cache_key = f"{emotion}_{intensity:.2f}"
            cached = self.regularizer.get_cached_embedding(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached
        
        # TODO: Implement actual embedding prediction
        # For now, return a stub embedding
        embedding = np.random.randn(self.config.visual_embedding_dim) * 0.1
        
        # Apply intensity scaling
        embedding = embedding * intensity
        
        # Cache the result
        if use_cache and self.config.regularization.cache_embeddings:
            cache_key = f"{emotion}_{intensity:.2f}"
            self.regularizer.cache_embedding(cache_key, embedding)
        
        return embedding
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        cache_hit_rate = 0.0
        if self._inference_count > 0:
            cache_hit_rate = self._cache_hits / self._inference_count
        
        return {
            "inference_count": self._inference_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.regularizer._embedding_cache),
        }
    
    def save_embeddings(self, path: Path) -> bool:
        """
        Save learned embeddings to disk.
        
        Args:
            path: File path to save to
        
        Returns:
            True if successful
        
        Note:
            This is a stub. Real implementation would serialize
            learned weights in a C++ compatible format.
        """
        # TODO: Implement saving
        # - Save to numpy format for Python
        # - Export to binary format for C++
        # - Include metadata and config
        
        return False
    
    def load_embeddings(self, path: Path) -> bool:
        """
        Load learned embeddings from disk.
        
        Args:
            path: File path to load from
        
        Returns:
            True if successful
        """
        # TODO: Implement loading
        # - Support numpy format
        # - Support C++ binary format
        # - Validate dimensions
        
        return False


def create_regularized_mapper(
    regularization_strength: float = 0.001,
    use_dropout: bool = True,
    use_fast_inference: bool = True
) -> Tuple[EmbeddingConfig, EmbeddingRegularizer]:
    """
    Create a regularized emotion-visual mapper configuration.
    
    Convenience function for common configurations.
    
    Args:
        regularization_strength: L2 regularization strength
        use_dropout: Whether to use dropout during training
        use_fast_inference: Whether to optimize for fast inference
    
    Returns:
        Tuple of (EmbeddingConfig, EmbeddingRegularizer)
    
    Example:
        >>> config, regularizer = create_regularized_mapper(
        ...     regularization_strength=0.001,
        ...     use_fast_inference=True
        ... )
    """
    # Create regularization config
    reg_config = RegularizationConfig(
        regularization_type=RegularizationType.L2,
        l2_lambda=regularization_strength,
        dropout_rate=0.1 if use_dropout else 0.0,
        use_fast_inference=use_fast_inference,
        cache_embeddings=use_fast_inference,
    )
    
    # Create embedding config
    embedding_config = EmbeddingConfig(
        emotion_embedding_dim=128,
        visual_embedding_dim=256,
        regularization=reg_config,
        compile_to_cpp=use_fast_inference,
    )
    
    # Create regularizer
    regularizer = EmbeddingRegularizer(reg_config)
    
    return embedding_config, regularizer
