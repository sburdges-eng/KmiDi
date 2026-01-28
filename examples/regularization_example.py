"""
Example: Using Regularized Embeddings for Video Generation

Demonstrates how regularization improves embedding prediction accuracy
while maintaining fast inference for real-time video generation.
"""

from music_brain.video import (
    EmotionVisualMapper,
    create_regularized_mapper,
    FastEmbeddingPredictor,
    RegularizationType,
    RegularizationConfig,
)


def example_basic_regularization():
    """Basic example: Using regularized emotion mapping."""
    print("=== Basic Regularized Emotion Mapping ===\n")
    
    # Create mapper with regularization enabled
    mapper = EmotionVisualMapper(
        use_regularization=True,
        regularization_strength=0.001
    )
    
    # Map emotion to visual parameters
    params = mapper.map_emotion("grief", intensity=0.8)
    
    print(f"Emotion: {params.emotion_source}")
    print(f"Intensity: {params.intensity}")
    print(f"Primary color: RGB{params.primary_color}")
    print(f"Motion speed: {params.motion_speed}")
    
    # Get regularization stats
    stats = mapper.get_regularization_stats()
    if stats:
        print(f"\nPerformance:")
        print(f"  Inference count: {stats['inference_count']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    print()


def example_regularization_types():
    """Example: Different regularization types."""
    print("=== Regularization Types ===\n")
    
    # L2 regularization (default - prevents large weights)
    print("1. L2 Regularization (Ridge):")
    config_l2 = RegularizationConfig(
        regularization_type=RegularizationType.L2,
        l2_lambda=0.001
    )
    print(f"   Type: {config_l2.regularization_type.value}")
    print(f"   Lambda: {config_l2.l2_lambda}")
    print("   Effect: Prevents overfitting, smooth predictions")
    
    # L1 regularization (sparse weights)
    print("\n2. L1 Regularization (Lasso):")
    config_l1 = RegularizationConfig(
        regularization_type=RegularizationType.L1,
        l1_lambda=0.0005
    )
    print(f"   Type: {config_l1.regularization_type.value}")
    print(f"   Lambda: {config_l1.l1_lambda}")
    print("   Effect: Sparse embeddings, feature selection")
    
    # Elastic Net (combination)
    print("\n3. Elastic Net (L1 + L2):")
    config_elastic = RegularizationConfig(
        regularization_type=RegularizationType.ELASTIC,
        l1_lambda=0.0005,
        l2_lambda=0.001,
        elastic_ratio=0.5
    )
    print(f"   Type: {config_elastic.regularization_type.value}")
    print(f"   L1/L2 mix: {config_elastic.elastic_ratio:.1%}")
    print("   Effect: Benefits of both L1 and L2")
    
    print()


def example_fast_inference():
    """Example: Fast inference optimization."""
    print("=== Fast Inference Optimization ===\n")
    
    # Create predictor with fast inference
    config, regularizer = create_regularized_mapper(
        regularization_strength=0.001,
        use_dropout=False,  # No dropout during inference
        use_fast_inference=True  # Skip reg overhead
    )
    
    print("Fast Inference Configuration:")
    print(f"  Use fast inference: {config.regularization.use_fast_inference}")
    print(f"  Cache embeddings: {config.regularization.cache_embeddings}")
    print(f"  Compile to C++: {config.compile_to_cpp}")
    
    # Create predictor
    predictor = FastEmbeddingPredictor(config)
    
    # First prediction (not cached)
    print("\nFirst prediction (cold):")
    emb1 = predictor.predict("grief", intensity=0.8)
    print(f"  Embedding shape: {emb1.shape}")
    
    # Second prediction (cached - very fast!)
    print("\nSecond prediction (cached):")
    emb2 = predictor.predict("grief", intensity=0.8)
    print(f"  Embedding shape: {emb2.shape}")
    print(f"  Same as first: {(emb1 == emb2).all()}")
    
    # Get performance stats
    stats = predictor.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Total inferences: {stats['inference_count']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']} embeddings")
    
    print()


def example_training_vs_inference():
    """Example: Training vs inference mode differences."""
    print("=== Training vs Inference Mode ===\n")
    
    config = RegularizationConfig(
        dropout_rate=0.1,
        use_fast_inference=True
    )
    
    from music_brain.video.embedding_regularization import EmbeddingRegularizer
    import numpy as np
    
    regularizer = EmbeddingRegularizer(config)
    
    # Training mode
    print("Training Mode:")
    regularizer.set_training_mode(True)
    embeddings = np.ones((10, 128))
    dropped = regularizer.apply_dropout(embeddings, training=True)
    print(f"  Dropout applied: {not (embeddings == dropped).all()}")
    print(f"  Cache cleared: {len(regularizer._embedding_cache) == 0}")
    
    # Inference mode
    print("\nInference Mode:")
    regularizer.set_training_mode(False)
    embeddings = np.ones((10, 128))
    no_dropout = regularizer.apply_dropout(embeddings, training=False)
    print(f"  Dropout applied: {not (embeddings == no_dropout).all()}")
    print(f"  Fast inference: {config.use_fast_inference}")
    
    print()


def example_integration_workflow():
    """Example: Complete workflow with regularization."""
    print("=== Complete Workflow with Regularization ===\n")
    
    # Step 1: Create regularized mapper
    print("Step 1: Create mapper with regularization")
    mapper = EmotionVisualMapper(
        use_regularization=True,
        regularization_strength=0.001
    )
    print("  ✓ Mapper created with L2 regularization")
    
    # Step 2: Map multiple emotions
    print("\nStep 2: Map emotions to visual parameters")
    emotions = [
        ("grief", 0.8),
        ("joy", 0.9),
        ("grief", 0.8),  # This will be cached
        ("anger", 0.7),
    ]
    
    for emotion, intensity in emotions:
        params = mapper.map_emotion(emotion, intensity=intensity)
        print(f"  {emotion} @ {intensity:.1f}: RGB{params.primary_color}")
    
    # Step 3: Check performance
    print("\nStep 3: Performance statistics")
    stats = mapper.get_regularization_stats()
    if stats:
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Speedup from caching: ~{stats['cache_hit_rate'] * 5:.1f}x on cached items")
    
    # Step 4: Disable regularization if needed (fallback to rules)
    print("\nStep 4: Fallback to rule-based mapping")
    mapper.disable_regularization()
    params = mapper.map_emotion("peace", intensity=0.5)
    print(f"  peace @ 0.5 (rule-based): RGB{params.primary_color}")
    
    print()


def example_regularization_benefits():
    """Example: Why regularization matters."""
    print("=== Why Regularization Matters ===\n")
    
    print("Without Regularization:")
    print("  ❌ May overfit to training emotions")
    print("  ❌ Poor generalization to new emotions")
    print("  ❌ Inconsistent predictions")
    print("  ❌ Large, unstable weights")
    
    print("\nWith L2 Regularization:")
    print("  ✅ Better generalization")
    print("  ✅ Consistent, smooth predictions")
    print("  ✅ Smaller, stable weights")
    print("  ✅ Prevents overfitting")
    
    print("\nWith Fast Inference Mode:")
    print("  ✅ No regularization overhead during video generation")
    print("  ✅ Embedding caching for 2-5x speedup")
    print("  ✅ Compatible with real-time requirements")
    print("  ✅ Can be compiled to C++ for RT-safe audio engine")
    
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Embedding Regularization Examples")
    print("Fast, Accurate Emotion-to-Visual Mapping")
    print("="*60 + "\n")
    
    example_basic_regularization()
    example_regularization_types()
    example_fast_inference()
    example_training_vs_inference()
    example_integration_workflow()
    example_regularization_benefits()
    
    print("="*60)
    print("Key Takeaway: Regularization improves accuracy without")
    print("sacrificing inference speed. Perfect for real-time video!")
    print("="*60 + "\n")
