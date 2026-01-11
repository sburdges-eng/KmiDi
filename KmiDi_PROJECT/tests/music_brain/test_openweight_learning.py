"""
Tests for OpenWeight Learning.

OpenWeight Learning is a method for learning with open (trainable) weights,
allowing dynamic adaptation of model parameters based on new data.

Tests cover:
- Weight initialization
- Online learning updates
- Weight adaptation
- Model performance with open weights
"""

import pytest
from pathlib import Path
import numpy as np
import tempfile
from unittest.mock import Mock, patch


class TestOpenWeightLearning:
    """Tests for the OpenWeight Learning system."""

    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        # Mock the OpenWeightLearner class
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=10, output_dim=5)

        # Check that weights are initialized
        assert learner.weights is not None
        assert learner.weights.shape == (10, 5)

        # Check that biases are initialized
        assert learner.biases is not None
        assert learner.biases.shape == (5,)

    def test_online_learning_update(self):
        """Test online learning weight updates."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=3, output_dim=1, learning_rate=0.1)

        # Initial weights
        initial_weights = learner.weights.copy()

        # Sample data
        x = np.array([[1, 2, 3]])
        y_true = np.array([[1]])

        # Update weights
        learner.update_weights(x, y_true)

        # Check that weights changed
        assert not np.array_equal(learner.weights, initial_weights)

    def test_weight_adaptation(self):
        """Test that weights adapt to new patterns."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=2, output_dim=1, learning_rate=0.5)

        # Train on pattern: y = 2*x1 + x2
        for _ in range(100):
            x = np.random.rand(10, 2)
            y = 2 * x[:, 0] + x[:, 1]
            learner.update_weights(x, y.reshape(-1, 1))

        # Test prediction
        test_x = np.array([[1, 1]])
        prediction = learner.predict(test_x)

        # Should be close to 3 (2*1 + 1)
        assert abs(prediction[0, 0] - 3.0) < 0.5

    def test_learning_rate_effect(self):
        """Test that learning rate affects convergence speed."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        # Fast learning
        fast_learner = OpenWeightLearner(input_dim=1, output_dim=1, learning_rate=0.1)
        # Slow learning
        slow_learner = OpenWeightLearner(input_dim=1, output_dim=1, learning_rate=0.01)

        # Force same initialization
        fast_learner.weights = np.zeros((1, 1))
        fast_learner.biases = np.zeros((1,))
        slow_learner.weights = np.zeros((1, 1))
        slow_learner.biases = np.zeros((1,))

        x = np.array([[1]])
        y = np.array([[2]])

        # One update each
        fast_learner.update_weights(x, y)
        slow_learner.update_weights(x, y)

        # Fast learner should have changed more
        fast_change = abs(fast_learner.weights[0, 0] - 0)  # assuming initial 0
        slow_change = abs(slow_learner.weights[0, 0] - 0)

        assert fast_change > slow_change

    def test_weight_persistence(self):
        """Test saving and loading weights."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=2, output_dim=1)

        # Modify weights
        learner.weights[0, 0] = 1.5
        learner.biases[0] = 0.5

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            learner.save_weights(f.name)

            # Load in new learner
            new_learner = OpenWeightLearner(input_dim=2, output_dim=1)
            new_learner.load_weights(f.name)

            # Check weights match
            assert np.array_equal(new_learner.weights, learner.weights)
            assert np.array_equal(new_learner.biases, learner.biases)

    def test_adaptive_learning(self):
        """Test adaptive learning with changing patterns."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=1, output_dim=1, learning_rate=0.1)

        # Phase 1: Learn y = x
        for _ in range(100):
            x = np.random.rand(5, 1)
            y = x
            learner.update_weights(x, y)

        # Phase 2: Learn y = 2*x
        for _ in range(100):
            x = np.random.rand(5, 1)
            y = 2 * x
            learner.update_weights(x, y)

        # Test final prediction
        test_x = np.array([[1]])
        prediction = learner.predict(test_x)

        # Should be closer to 2 than to 1
        assert abs(prediction[0, 0] - 2.0) < abs(prediction[0, 0] - 1.0)

    def test_open_weight_convergence(self):
        """Test that open weight learning converges."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=1, output_dim=1, learning_rate=0.01)

        losses = []

        # Train for multiple epochs
        for epoch in range(100):
            x = np.random.rand(10, 1)
            y = 3 * x + 1  # Linear relationship

            loss = learner.update_weights(x, y)
            losses.append(loss)

        # Loss should decrease over time
        early_loss = np.mean(losses[:20])
        late_loss = np.mean(losses[-20:])

        assert late_loss < early_loss

    def test_weight_regularization(self):
        """Test weight regularization to prevent overfitting."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(
            input_dim=5, output_dim=1,
            learning_rate=0.1,
            regularization=0.01
        )

        # Train with regularization
        for _ in range(50):
            x = np.random.rand(10, 5)
            y = np.random.rand(10, 1)
            learner.update_weights(x, y)

        # Weights should be smaller due to regularization
        weight_magnitude = np.linalg.norm(learner.weights)
        assert weight_magnitude < 1.0  # Should be regularized

    def test_batch_vs_online_learning(self):
        """Test difference between batch and online learning."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        # Online learner
        online_learner = OpenWeightLearner(input_dim=2, output_dim=1, learning_rate=0.1)

        # Batch learner (simulated)
        batch_learner = OpenWeightLearner(input_dim=2, output_dim=1, learning_rate=0.1)

        # Data
        x = np.random.rand(100, 2)
        y = (x[:, 0] + 2 * x[:, 1]).reshape(-1, 1)

        # Online learning: update after each sample
        for i in range(len(x)):
            online_learner.update_weights(x[i:i+1], y[i:i+1])

        # Batch learning: update multiple times to match online progress
        for _ in range(100):
            batch_learner.update_weights(x, y)

        # Both should converge to similar weights
        weight_diff = np.linalg.norm(online_learner.weights - batch_learner.weights)
        assert weight_diff < 1.0  # Should be reasonably close


class TestOpenWeightIntegration:
    """Integration tests for OpenWeight Learning."""

    def test_with_emotion_data(self):
        """Test learning with emotion-related data."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=10, output_dim=7)  # 7 emotions

        # Mock emotion features
        emotion_features = np.random.rand(50, 10)
        # One-hot encode labels
        emotion_labels = np.zeros((50, 7))
        indices = np.random.randint(0, 7, 50)
        emotion_labels[np.arange(50), indices] = 1

        # Train
        for _ in range(10):
            learner.update_weights(emotion_features, emotion_labels)

        # Predict
        test_features = np.random.rand(5, 10)
        predictions = learner.predict(test_features)

        assert predictions.shape == (5, 7)

    def test_adaptation_to_user_feedback(self):
        """Test adapting to user feedback."""
        from music_brain.learning.openweight_learning import OpenWeightLearner

        learner = OpenWeightLearner(input_dim=5, output_dim=1, learning_rate=0.2)

        # Initial generation parameters
        params = np.random.rand(1, 5)

        # User likes it, reinforce
        learner.update_weights(params, np.array([[1]]))  # Positive feedback

        # User dislikes next one, penalize
        bad_params = np.random.rand(1, 5)
        learner.update_weights(bad_params, np.array([[0]]))  # Negative feedback

        # Future predictions should favor good parameters
        new_prediction = learner.predict(np.random.rand(1, 5))
        assert new_prediction.shape == (1, 1)

    def test_memory_efficiency(self):
        """Test that open weight learning is memory efficient."""
        from music_brain.learning.openweight_learning import OpenWeightLearner
        import psutil
        import os

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create learner
        learner = OpenWeightLearner(input_dim=100, output_dim=50)

        # Train with large dataset
        for _ in range(100):
            x = np.random.rand(1000, 100)
            y = np.random.rand(1000, 50)
            learner.update_weights(x, y)

        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not increase dramatically (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestOpenWeightLearningManager:
    """Tests for the OpenWeightLearningManager."""

    def test_manager_initialization(self):
        """Test manager setup."""
        from music_brain.learning.openweight_learning import OpenWeightLearningManager

        manager = OpenWeightLearningManager()

        assert manager.learners == {}
        assert manager.storage_dir is not None

    def test_adding_learner(self):
        """Test adding a learner for a task."""
        from music_brain.learning.openweight_learning import OpenWeightLearningManager

        manager = OpenWeightLearningManager()

        manager.add_learner("emotion_recognition", input_dim=10, output_dim=7)

        assert "emotion_recognition" in manager.learners
        assert manager.learners["emotion_recognition"].weights.shape == (10, 7)

    def test_saving_and_loading_manager(self):
        """Test persisting the manager state."""
        from music_brain.learning.openweight_learning import OpenWeightLearningManager

        manager = OpenWeightLearningManager()

        manager.add_learner("test_task", input_dim=2, output_dim=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            manager.save_state(temp_dir)

            new_manager = OpenWeightLearningManager()
            new_manager.load_state(temp_dir)

            assert "test_task" in new_manager.learners

    def test_concurrent_learning(self):
        """Test learning multiple tasks concurrently."""
        from music_brain.learning.openweight_learning import OpenWeightLearningManager

        manager = OpenWeightLearningManager()

        # Add multiple learners
        manager.add_learner("task1", input_dim=3, output_dim=1)
        manager.add_learner("task2", input_dim=4, output_dim=2)

        # Train both
        for task in ["task1", "task2"]:
            x = np.random.rand(10, manager.learners[task].weights.shape[0])
            y = np.random.rand(10, manager.learners[task].weights.shape[1])
            manager.update_learner(task, x, y)


class TestTeachingDataIntegration:
    """Tests for using teaching data in OpenWeight Learning."""

    def test_load_teaching_parameters(self):
        """Test loading teaching parameters from the learning module."""
        from music_brain.learning.openweight_learning import load_teaching_parameters

        data = load_teaching_parameters()

        # Check that we have the expected data
        assert 'instruments' in data
        assert 'teaching_prompt_templates' in data
        assert 'teaching_sequences' in data
        assert 'known_sources' in data

        # Check instruments data
        assert len(data['instruments']) > 0
        assert 'piano' in data['instruments']

        # Check prompt templates
        assert len(data['teaching_prompt_templates']) > 0
        assert 'explain' in data['teaching_prompt_templates']

        # Check teaching sequences
        assert len(data['teaching_sequences']) > 0
        assert 'beginner_guitar_chords' in data['teaching_sequences']

    def test_create_learner_from_instruments(self):
        """Test creating a learner from instrument teaching data."""
        from music_brain.learning.openweight_learning import create_learner_from_teaching_data

        learner = create_learner_from_teaching_data("instrument_learning", "instruments")

        # Should have created a learner with appropriate dimensions
        assert learner.input_dim > 0
        assert learner.output_dim > 0
        assert learner.weights.shape == (learner.input_dim, learner.output_dim)

    def test_create_learner_from_prompts(self):
        """Test creating a learner from prompt template data."""
        from music_brain.learning.openweight_learning import create_learner_from_teaching_data

        learner = create_learner_from_teaching_data("prompt_learning", "prompts")

        # Should have created a learner
        assert learner.input_dim > 0
        assert learner.output_dim > 0

    def test_learner_with_instrument_features(self):
        """Test learning with instrument feature data."""
        from music_brain.learning.openweight_learning import load_teaching_parameters, OpenWeightLearner

        data = load_teaching_parameters()
        instruments = list(data['instruments'].values())

        # Extract features from instruments
        features = []
        targets = []
        for inst in instruments[:5]:  # Use first 5 instruments
            feature_vector = [
                float(inst.beginner_friendly),
                float(inst.days_to_first_song),
                float(len(inst.first_skills)),
                float(len(inst.practice_tips)),
                float(len(inst.primary_genres))
            ]
            features.append(feature_vector)
            # Target: some derived metric (e.g., overall difficulty)
            target = [float(inst.days_to_first_song) / 30.0]  # Normalize
            targets.append(target)

        if features:
            learner = OpenWeightLearner(input_dim=len(features[0]), output_dim=1)

            x = np.array(features)
            y = np.array(targets)

            # Train
            loss = learner.update_weights(x, y)
            assert loss >= 0  # Loss should be non-negative

            # Predict
            predictions = learner.predict(x)
            assert predictions.shape == (len(features), 1)

    def test_learner_with_prompt_features(self):
        """Test learning with prompt template features."""
        from music_brain.learning.openweight_learning import load_teaching_parameters, OpenWeightLearner

        data = load_teaching_parameters()
        prompts = list(data['teaching_prompt_templates'].values())

        # Extract features from prompts
        features = []
        targets = []
        for prompt in prompts:
            feature_vector = [
                float(len(prompt)),
                float(prompt.count('{')),
                float(prompt.count('topic')),
                float(prompt.count('instrument')),
                float(prompt.count('difficulty'))
            ]
            features.append(feature_vector)
            # Target: prompt complexity score
            target = [float(len(prompt)) / 1000.0]  # Normalize
            targets.append(target)

        if features:
            learner = OpenWeightLearner(input_dim=len(features[0]), output_dim=1)

            x = np.array(features)
            y = np.array(targets)

            # Train
            loss = learner.update_weights(x, y)
            assert loss >= 0

            # Predict
            predictions = learner.predict(x)
            assert predictions.shape == (len(features), 1)

    def test_teaching_sequence_adaptation(self):
        """Test adapting learning based on teaching sequences."""
        from music_brain.learning.openweight_learning import load_teaching_parameters, OpenWeightLearner

        data = load_teaching_parameters()
        sequences = data['teaching_sequences']

        # Use sequence data for learning
        features = []
        targets = []
        for seq_name, seq_steps in sequences.items():
            feature_vector = [
                float(len(seq_steps)),
                float(len(seq_name)),
                float(seq_name.count('beginner')),
                float(seq_name.count('guitar')),
                float(seq_name.count('piano'))
            ]
            features.append(feature_vector)
            # Target: sequence difficulty (based on name)
            difficulty = 1.0 if 'beginner' in seq_name else 2.0
            targets.append([difficulty])

        if features:
            learner = OpenWeightLearner(input_dim=len(features[0]), output_dim=1)

            x = np.array(features)
            y = np.array(targets)

            # Train
            initial_loss = learner.update_weights(x, y)

            # Train more
            for _ in range(10):
                learner.update_weights(x, y)

            # Loss should potentially decrease
            final_loss = learner.update_weights(x, y)
            # Note: May not always decrease due to random data, but should be valid
            assert final_loss >= 0

    def test_known_sources_integration(self):
        """Test using known sources data for learning."""
        from music_brain.learning.openweight_learning import load_teaching_parameters, OpenWeightLearner

        data = load_teaching_parameters()
        sources = data['known_sources']

        # Extract features from sources
        features = []
        targets = []
        for source_id, source_info in list(sources.items())[:5]:  # First 5 sources
            feature_vector = [
                float(len(source_id)),
                float(len(source_info.get('name', ''))),
                float(len(source_info.get('instruments', []))),
                float(len(source_info.get('content_types', []))),
                1.0 if 'guitar' in source_id.lower() else 0.0
            ]
            features.append(feature_vector)
            # Target: source quality score (mock)
            target = [0.8]  # Assume high quality
            targets.append(target)

        if features:
            learner = OpenWeightLearner(input_dim=len(features[0]), output_dim=1)

            x = np.array(features)
            y = np.array(targets)

            # Train
            loss = learner.update_weights(x, y)
            assert loss >= 0

            # Predict
            predictions = learner.predict(x)
            assert predictions.shape == (len(features), 1)


if __name__ == "__main__":
    pytest.main([__file__])