"""Unit tests for embedding regularization."""

import pytest
import numpy as np
from music_brain.video import (
    EmbeddingRegularizer,
    FastEmbeddingPredictor,
    EmbeddingConfig,
    RegularizationConfig,
    RegularizationType,
    create_regularized_mapper,
    EmotionVisualMapper,
)


class TestRegularizationConfig:
    """Test RegularizationConfig dataclass."""
    
    def test_regularization_config_defaults(self):
        """Test default regularization config values."""
        config = RegularizationConfig()
        assert config.regularization_type == RegularizationType.L2
        assert config.l2_lambda == 0.001
        assert config.dropout_rate == 0.1
        assert config.use_fast_inference is True
        assert config.cache_embeddings is True
    
    def test_regularization_config_custom(self):
        """Test custom regularization config."""
        config = RegularizationConfig(
            regularization_type=RegularizationType.L1,
            l1_lambda=0.0005,
            dropout_rate=0.2,
            use_fast_inference=False
        )
        assert config.regularization_type == RegularizationType.L1
        assert config.l1_lambda == 0.0005
        assert config.dropout_rate == 0.2
        assert config.use_fast_inference is False


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass."""
    
    def test_embedding_config_defaults(self):
        """Test default embedding config values."""
        config = EmbeddingConfig()
        assert config.emotion_embedding_dim == 128
        assert config.visual_embedding_dim == 256
        assert len(config.hidden_dims) == 3
        assert config.learning_rate == 0.001
        assert config.compile_to_cpp is True
    
    def test_embedding_config_with_regularization(self):
        """Test embedding config with custom regularization."""
        reg_config = RegularizationConfig(l2_lambda=0.002)
        config = EmbeddingConfig(regularization=reg_config)
        assert config.regularization.l2_lambda == 0.002


class TestEmbeddingRegularizer:
    """Test EmbeddingRegularizer class."""
    
    def test_regularizer_creation(self):
        """Test creating an EmbeddingRegularizer."""
        regularizer = EmbeddingRegularizer()
        assert regularizer is not None
        assert regularizer.config is not None
    
    def test_l2_regularization_loss(self):
        """Test L2 regularization loss computation."""
        config = RegularizationConfig(
            regularization_type=RegularizationType.L2,
            l2_lambda=0.001
        )
        regularizer = EmbeddingRegularizer(config)
        
        # Create test weights
        weights = np.array([1.0, 2.0, 3.0])
        
        # Compute L2 loss: lambda * sum(w^2)
        loss = regularizer.compute_regularization_loss(weights, training=True)
        expected_loss = 0.001 * (1.0**2 + 2.0**2 + 3.0**2)
        
        assert np.isclose(loss, expected_loss)
    
    def test_l1_regularization_loss(self):
        """Test L1 regularization loss computation."""
        config = RegularizationConfig(
            regularization_type=RegularizationType.L1,
            l1_lambda=0.001
        )
        regularizer = EmbeddingRegularizer(config)
        
        weights = np.array([1.0, -2.0, 3.0])
        
        # Compute L1 loss: lambda * sum(|w|)
        loss = regularizer.compute_regularization_loss(weights, training=True)
        expected_loss = 0.001 * (1.0 + 2.0 + 3.0)
        
        assert np.isclose(loss, expected_loss)
    
    def test_elastic_regularization_loss(self):
        """Test Elastic Net regularization."""
        config = RegularizationConfig(
            regularization_type=RegularizationType.ELASTIC,
            l1_lambda=0.001,
            l2_lambda=0.001,
            elastic_ratio=0.5
        )
        regularizer = EmbeddingRegularizer(config)
        
        weights = np.array([1.0, 2.0, 3.0])
        loss = regularizer.compute_regularization_loss(weights, training=True)
        
        # Should be combination of L1 and L2
        assert loss > 0.0
    
    def test_fast_inference_skip_regularization(self):
        """Test that regularization is skipped during fast inference."""
        config = RegularizationConfig(use_fast_inference=True)
        regularizer = EmbeddingRegularizer(config)
        
        weights = np.array([1.0, 2.0, 3.0])
        
        # Should return 0 during inference
        loss = regularizer.compute_regularization_loss(weights, training=False)
        assert loss == 0.0
    
    def test_dropout_application(self):
        """Test dropout application."""
        config = RegularizationConfig(dropout_rate=0.5)
        regularizer = EmbeddingRegularizer(config)
        
        embeddings = np.ones((10, 128))
        
        # Apply dropout during training
        dropped = regularizer.apply_dropout(embeddings, training=True)
        
        # Some values should be zero, others scaled
        assert not np.array_equal(embeddings, dropped)
        # Mean should be approximately preserved (with scaling)
        assert np.abs(np.mean(dropped) - np.mean(embeddings)) < 0.3
    
    def test_dropout_disabled_inference(self):
        """Test that dropout is disabled during inference."""
        config = RegularizationConfig(dropout_rate=0.5)
        regularizer = EmbeddingRegularizer(config)
        
        embeddings = np.ones((10, 128))
        
        # No dropout during inference
        result = regularizer.apply_dropout(embeddings, training=False)
        assert np.array_equal(embeddings, result)
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        config = RegularizationConfig(
            clip_gradients=True,
            max_gradient_norm=1.0
        )
        regularizer = EmbeddingRegularizer(config)
        
        # Create gradients with large norm
        gradients = np.array([10.0, 10.0, 10.0])
        
        clipped = regularizer.clip_gradients(gradients)
        
        # Norm should be clipped to max_gradient_norm
        clipped_norm = np.linalg.norm(clipped)
        assert np.isclose(clipped_norm, 1.0)
    
    def test_embedding_cache(self):
        """Test embedding caching."""
        config = RegularizationConfig(cache_embeddings=True)
        regularizer = EmbeddingRegularizer(config)
        
        key = "test_emotion"
        embedding = np.random.randn(128)
        
        # Cache embedding
        regularizer.cache_embedding(key, embedding)
        
        # Retrieve cached embedding
        cached = regularizer.get_cached_embedding(key)
        assert cached is not None
        assert np.array_equal(cached, embedding)
    
    def test_training_mode_clears_cache(self):
        """Test that switching to training mode clears cache."""
        regularizer = EmbeddingRegularizer()
        
        # Add to cache
        regularizer.cache_embedding("test", np.ones(128))
        assert len(regularizer._embedding_cache) > 0
        
        # Switch to training mode
        regularizer.set_training_mode(True)
        assert len(regularizer._embedding_cache) == 0


class TestFastEmbeddingPredictor:
    """Test FastEmbeddingPredictor class."""
    
    def test_predictor_creation(self):
        """Test creating a FastEmbeddingPredictor."""
        predictor = FastEmbeddingPredictor()
        assert predictor is not None
        assert predictor.config is not None
        assert predictor.regularizer is not None
    
    def test_predict_emotion_embedding(self):
        """Test predicting visual embedding from emotion."""
        predictor = FastEmbeddingPredictor()
        
        embedding = predictor.predict("grief", intensity=0.8)
        
        assert embedding is not None
        assert embedding.shape == (predictor.config.visual_embedding_dim,)
    
    def test_predict_with_caching(self):
        """Test that predictions are cached."""
        predictor = FastEmbeddingPredictor()
        
        # First prediction
        emb1 = predictor.predict("joy", intensity=0.5, use_cache=True)
        
        # Second prediction should be from cache (same result)
        emb2 = predictor.predict("joy", intensity=0.5, use_cache=True)
        
        assert np.array_equal(emb1, emb2)
        
        # Check cache hit stats
        stats = predictor.get_performance_stats()
        assert stats["cache_hits"] > 0
    
    def test_predict_different_intensities(self):
        """Test that different intensities produce different embeddings."""
        predictor = FastEmbeddingPredictor()
        
        emb1 = predictor.predict("grief", intensity=0.3, use_cache=False)
        emb2 = predictor.predict("grief", intensity=0.8, use_cache=False)
        
        # Different intensities should produce different embeddings
        assert not np.array_equal(emb1, emb2)
    
    def test_performance_stats(self):
        """Test getting performance statistics."""
        predictor = FastEmbeddingPredictor()
        
        # Make some predictions
        predictor.predict("joy", 0.5, use_cache=True)
        predictor.predict("joy", 0.5, use_cache=True)  # Cache hit
        predictor.predict("grief", 0.8, use_cache=True)
        
        stats = predictor.get_performance_stats()
        
        assert "inference_count" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate" in stats
        assert stats["inference_count"] == 3
        assert stats["cache_hits"] >= 1


class TestCreateRegularizedMapper:
    """Test create_regularized_mapper convenience function."""
    
    def test_create_default(self):
        """Test creating with default parameters."""
        config, regularizer = create_regularized_mapper()
        
        assert config is not None
        assert regularizer is not None
        assert config.regularization.l2_lambda == 0.001
        assert config.compile_to_cpp is True
    
    def test_create_custom_strength(self):
        """Test creating with custom regularization strength."""
        config, regularizer = create_regularized_mapper(
            regularization_strength=0.005
        )
        
        assert config.regularization.l2_lambda == 0.005
    
    def test_create_with_dropout(self):
        """Test creating with dropout enabled."""
        config, regularizer = create_regularized_mapper(
            use_dropout=True
        )
        
        assert config.regularization.dropout_rate == 0.1
    
    def test_create_fast_inference(self):
        """Test creating with fast inference enabled."""
        config, regularizer = create_regularized_mapper(
            use_fast_inference=True
        )
        
        assert config.regularization.use_fast_inference is True
        assert config.regularization.cache_embeddings is True


class TestEmotionVisualMapperWithRegularization:
    """Test EmotionVisualMapper with regularization integration."""
    
    def test_mapper_without_regularization(self):
        """Test mapper in default mode (no regularization)."""
        mapper = EmotionVisualMapper(use_regularization=False)
        
        params = mapper.map_emotion("grief", intensity=0.8)
        assert params.emotion_source == "grief"
        assert params.intensity == 0.8
    
    def test_mapper_with_regularization_enabled(self):
        """Test mapper with regularization enabled."""
        mapper = EmotionVisualMapper(use_regularization=True)
        
        params = mapper.map_emotion("joy", intensity=0.7)
        assert params.emotion_source == "joy"
        # Should work regardless of regularization availability
    
    def test_enable_regularization_at_runtime(self):
        """Test enabling regularization after creation."""
        mapper = EmotionVisualMapper(use_regularization=False)
        
        # Enable regularization
        enabled = mapper.enable_regularization(regularization_strength=0.002)
        
        # Should return True if regularization module available
        # (May be False if module not imported)
        if enabled:
            assert mapper._use_regularization is True
    
    def test_disable_regularization(self):
        """Test disabling regularization."""
        mapper = EmotionVisualMapper(use_regularization=True)
        
        mapper.disable_regularization()
        assert mapper._use_regularization is False
        assert mapper._embedding_predictor is None
    
    def test_get_regularization_stats(self):
        """Test getting regularization statistics."""
        mapper = EmotionVisualMapper(use_regularization=True)
        
        # Make some predictions
        mapper.map_emotion("grief", 0.8)
        mapper.map_emotion("joy", 0.5)
        
        stats = mapper.get_regularization_stats()
        # May be None if regularization not available
        if stats:
            assert "inference_count" in stats


class TestRegularizationTypes:
    """Test RegularizationType enum."""
    
    def test_regularization_types_exist(self):
        """Test that all regularization types are defined."""
        assert RegularizationType.L1
        assert RegularizationType.L2
        assert RegularizationType.ELASTIC
        assert RegularizationType.DROPOUT
        assert RegularizationType.NONE
    
    def test_regularization_type_values(self):
        """Test regularization type string values."""
        assert RegularizationType.L1.value == "l1"
        assert RegularizationType.L2.value == "l2"
        assert RegularizationType.ELASTIC.value == "elastic"


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_fast_inference_mode(self):
        """Test fast inference mode skips unnecessary computation."""
        config = RegularizationConfig(use_fast_inference=True)
        regularizer = EmbeddingRegularizer(config)
        
        weights = np.random.randn(1000, 1000)  # Large weights
        
        # Should be very fast (returns 0 immediately)
        loss = regularizer.compute_regularization_loss(weights, training=False)
        assert loss == 0.0
    
    def test_embedding_cache_performance(self):
        """Test that caching improves performance."""
        predictor = FastEmbeddingPredictor()
        
        # First prediction (not cached)
        _ = predictor.predict("grief", 0.8, use_cache=True)
        
        # Second prediction (cached)
        _ = predictor.predict("grief", 0.8, use_cache=True)
        
        stats = predictor.get_performance_stats()
        
        # Should have at least one cache hit
        assert stats["cache_hit_rate"] > 0.0
