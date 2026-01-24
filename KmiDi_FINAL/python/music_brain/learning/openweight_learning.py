"""
OpenWeight Learning Module

Implements learning with open (trainable) weights for dynamic adaptation
in music generation and analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pickle
import os
from pathlib import Path


@dataclass
class OpenWeightLearner:
    """
    A learner with open (trainable) weights that can adapt online.

    Supports:
    - Online learning updates
    - Weight regularization
    - Memory-efficient updates
    - Persistence of learned weights
    - Standard ML training parameters
    """

    # Model identity
    model_id: str = "openweight_learner"
    model_type: str = "OpenWeight"
    task: str = "adaptive_learning"

    # Architecture
    input_size: int = 128
    output_size: int = 64
    input_dim: int = 128
    output_dim: int = 64
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"
    dropout: float = 0.0
    architecture_type: str = "linear"  # linear, mlp, cnn, lstm

    # Data parameters
    data_path: str = ""
    data_version: str = "v1"
    sample_rate: int = 16000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    max_duration: float = 5.0
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    optimizer: str = "adam"  # adam, sgd, rmsprop
    loss_fn: str = "mse"  # mse, cross_entropy, mae
    scheduler: Optional[str] = None  # step, cosine, exponential
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    min_delta: float = 1e-4

    # Hardware parameters
    device: str = "cpu"
    num_workers: int = 0
    pin_memory: bool = False

    # Output parameters
    output_dir: str = ""
    export_onnx: bool = True
    export_coreml: bool = True

    # Logging parameters
    log_interval: int = 10
    save_interval: int = 50

    # Metadata
    author: str = ""
    notes: str = ""
    labels: List[str] = field(default_factory=list)

    # Legacy parameters
    regularization: float = 0.0

    # C++ ML Interface parameters (from MLInterface.h)
    model_directory: str = ""
    use_gpu: bool = False
    use_coreml: bool = True
    timeout_ms: float = 100.0
    inference_thread_priority: int = 0

    # Additional training parameters from comprehensive analysis
    loss: str = "cross_entropy"  # Renamed from loss_fn for consistency
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

    # Augmentation parameters
    use_augmentation: bool = True
    aug_time_stretch: float = 0.3
    aug_pitch_shift: float = 0.3
    aug_noise: float = 0.2

    def __post_init__(self):
        """Initialize weights and biases."""
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (self.input_dim + self.output_dim))
        self.weights = np.random.uniform(-limit, limit, (self.input_dim, self.output_dim))
        self.biases = np.zeros(self.output_dim)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions with current weights.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Predictions (batch_size, output_dim)
        """
        return np.dot(x, self.weights) + self.biases

    def update_weights(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """
        Update weights using online learning.

        Args:
            x: Input features (batch_size, input_dim)
            y_true: True labels/targets (batch_size, output_dim)

        Returns:
            Loss value
        """
        predictions = self.predict(x)
        errors = predictions - y_true

        # Compute loss (MSE)
        loss = np.mean(errors ** 2)

        # Compute gradients
        d_weights = np.dot(x.T, errors) / len(x)
        d_biases = np.mean(errors, axis=0)

        # Add regularization
        if self.regularization > 0:
            d_weights += self.regularization * self.weights

        # Update weights
        self.weights -= self.learning_rate * d_weights
        self.biases -= self.learning_rate * d_biases

        return loss

    def save_weights(self, filepath: str):
        """Save weights to file."""
        data = {
            'weights': self.weights,
            'biases': self.biases,
            # Model identity
            'model_id': self.model_id,
            'model_type': self.model_type,
            'task': self.task,
            # Architecture
            'input_size': self.input_size,
            'output_size': self.output_size,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'architecture_type': self.architecture_type,
            # Data parameters
            'data_path': self.data_path,
            'data_version': self.data_version,
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'max_duration': self.max_duration,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            # Training parameters
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'loss_fn': self.loss_fn,
            'scheduler': self.scheduler,
            'warmup_epochs': self.warmup_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'min_delta': self.min_delta,
            # Hardware parameters
            'device': self.device,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            # Output parameters
            'output_dir': self.output_dir,
            'export_onnx': self.export_onnx,
            'export_coreml': self.export_coreml,
            # Logging parameters
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            # Metadata
            'author': self.author,
            'notes': self.notes,
            'labels': self.labels,
            # Legacy parameters
            'regularization': self.regularization,
            # C++ ML Interface parameters
            'model_directory': self.model_directory,
            'use_gpu': self.use_gpu,
            'use_coreml': self.use_coreml,
            'timeout_ms': self.timeout_ms,
            'inference_thread_priority': self.inference_thread_priority,
            # Additional training parameters
            'loss': self.loss,
            'focal_gamma': self.focal_gamma,
            'label_smoothing': self.label_smoothing,
            # Augmentation parameters
            'use_augmentation': self.use_augmentation,
            'aug_time_stretch': self.aug_time_stretch,
            'aug_pitch_shift': self.aug_pitch_shift,
            'aug_noise': self.aug_noise
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, filepath: str):
        """Load weights from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.weights = data['weights']
        self.biases = data['biases']
        # Model identity
        self.model_id = data.get('model_id', self.model_id)
        self.model_type = data.get('model_type', self.model_type)
        self.task = data.get('task', self.task)
        # Architecture
        self.input_size = data.get('input_size', self.input_size)
        self.output_size = data.get('output_size', self.output_size)
        self.input_dim = data.get('input_dim', self.input_dim)
        self.output_dim = data.get('output_dim', self.output_dim)
        self.hidden_layers = data.get('hidden_layers', self.hidden_layers)
        self.activation = data.get('activation', self.activation)
        self.dropout = data.get('dropout', self.dropout)
        self.architecture_type = data.get('architecture_type', self.architecture_type)
        # Data parameters
        self.data_path = data.get('data_path', self.data_path)
        self.data_version = data.get('data_version', self.data_version)
        self.sample_rate = data.get('sample_rate', self.sample_rate)
        self.n_mels = data.get('n_mels', self.n_mels)
        self.n_fft = data.get('n_fft', self.n_fft)
        self.hop_length = data.get('hop_length', self.hop_length)
        self.max_duration = data.get('max_duration', self.max_duration)
        self.train_split = data.get('train_split', self.train_split)
        self.val_split = data.get('val_split', self.val_split)
        self.test_split = data.get('test_split', self.test_split)
        # Training parameters
        self.epochs = data.get('epochs', self.epochs)
        self.batch_size = data.get('batch_size', self.batch_size)
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.weight_decay = data.get('weight_decay', self.weight_decay)
        self.optimizer = data.get('optimizer', self.optimizer)
        self.loss_fn = data.get('loss_fn', self.loss_fn)
        self.scheduler = data.get('scheduler', self.scheduler)
        self.warmup_epochs = data.get('warmup_epochs', self.warmup_epochs)
        self.early_stopping_patience = data.get('early_stopping_patience', self.early_stopping_patience)
        self.min_delta = data.get('min_delta', self.min_delta)
        # Hardware parameters
        self.device = data.get('device', self.device)
        self.num_workers = data.get('num_workers', self.num_workers)
        self.pin_memory = data.get('pin_memory', self.pin_memory)
        # Output parameters
        self.output_dir = data.get('output_dir', self.output_dir)
        self.export_onnx = data.get('export_onnx', self.export_onnx)
        self.export_coreml = data.get('export_coreml', self.export_coreml)
        # Logging parameters
        self.log_interval = data.get('log_interval', self.log_interval)
        self.save_interval = data.get('save_interval', self.save_interval)
        # Metadata
        self.author = data.get('author', self.author)
        self.notes = data.get('notes', self.notes)
        self.labels = data.get('labels', self.labels)
        # Legacy parameters
        self.regularization = data.get('regularization', self.regularization)
        # C++ ML Interface parameters
        self.model_directory = data.get('model_directory', self.model_directory)
        self.use_gpu = data.get('use_gpu', self.use_gpu)
        self.use_coreml = data.get('use_coreml', self.use_coreml)
        self.timeout_ms = data.get('timeout_ms', self.timeout_ms)
        self.inference_thread_priority = data.get('inference_thread_priority', self.inference_thread_priority)
        # Additional training parameters
        self.loss = data.get('loss', self.loss)
        self.focal_gamma = data.get('focal_gamma', self.focal_gamma)
        self.label_smoothing = data.get('label_smoothing', self.label_smoothing)
        # Augmentation parameters
        self.use_augmentation = data.get('use_augmentation', self.use_augmentation)
        self.aug_time_stretch = data.get('aug_time_stretch', self.aug_time_stretch)
        self.aug_pitch_shift = data.get('aug_pitch_shift', self.aug_pitch_shift)
        self.aug_noise = data.get('aug_noise', self.aug_noise)


@dataclass
class OpenWeightLearningManager:
    """
    Manager for multiple OpenWeight learners.

    Handles:
    - Multiple learning tasks
    - Persistence of learners
    - Concurrent learning
    """

    learners: Dict[str, OpenWeightLearner] = field(default_factory=dict)
    storage_dir: Optional[Path] = None

    def __post_init__(self):
        """Initialize storage directory."""
        if self.storage_dir is None:
            self.storage_dir = Path.home() / '.kmid_openweight'
        self.storage_dir.mkdir(exist_ok=True)

    def add_learner(self, task_name: str, input_dim: int, output_dim: int,
                   learning_rate: float = 0.01, regularization: float = 0.0):
        """
        Add a learner for a specific task.

        Args:
            task_name: Name of the learning task
            input_dim: Input dimension
            output_dim: Output dimension
            learning_rate: Learning rate
            regularization: Regularization strength
        """
        self.learners[task_name] = OpenWeightLearner(
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            regularization=regularization
        )

    def update_learner(self, task_name: str, x: np.ndarray, y: np.ndarray) -> float:
        """
        Update a specific learner.

        Args:
            task_name: Name of the task
            x: Input features
            y: Target values

        Returns:
            Loss value
        """
        if task_name not in self.learners:
            raise ValueError(f"No learner found for task: {task_name}")

        return self.learners[task_name].update_weights(x, y)

    def predict_with_learner(self, task_name: str, x: np.ndarray) -> np.ndarray:
        """
        Make predictions with a specific learner.

        Args:
            task_name: Name of the task
            x: Input features

        Returns:
            Predictions
        """
        if task_name not in self.learners:
            raise ValueError(f"No learner found for task: {task_name}")

        return self.learners[task_name].predict(x)

    def save_state(self, directory: Optional[str] = None):
        """
        Save all learners to disk.

        Args:
            directory: Directory to save to (default: storage_dir)
        """
        save_dir = Path(directory) if directory else self.storage_dir

        for task_name, learner in self.learners.items():
            filepath = save_dir / f"{task_name}_weights.pkl"
            learner.save_weights(str(filepath))

    def load_state(self, directory: Optional[str] = None):
        """
        Load all learners from disk.

        Args:
            directory: Directory to load from (default: storage_dir)
        """
        load_dir = Path(directory) if directory else self.storage_dir

        for filepath in load_dir.glob("*_weights.pkl"):
            task_name = filepath.stem.replace('_weights', '')
            if task_name not in self.learners:
                # Try to infer dimensions from file
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.add_learner(
                    task_name,
                    input_dim=data['input_dim'],
                    output_dim=data['output_dim'],
                    learning_rate=data.get('learning_rate', 0.01),
                    regularization=data.get('regularization', 0.0)
                )
            self.learners[task_name].load_weights(str(filepath))

    def get_learner_stats(self, task_name: str) -> Dict[str, Any]:
        """
        Get statistics for a learner.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with statistics
        """
        if task_name not in self.learners:
            raise ValueError(f"No learner found for task: {task_name}")

        learner = self.learners[task_name]
        return {
            # Model identity
            'model_id': learner.model_id,
            'model_type': learner.model_type,
            'task': learner.task,
            # Architecture
            'input_size': learner.input_size,
            'output_size': learner.output_size,
            'input_dim': learner.input_dim,
            'output_dim': learner.output_dim,
            'hidden_layers': learner.hidden_layers,
            'activation': learner.activation,
            'dropout': learner.dropout,
            'architecture_type': learner.architecture_type,
            # Data parameters
            'data_path': learner.data_path,
            'data_version': learner.data_version,
            'sample_rate': learner.sample_rate,
            'n_mels': learner.n_mels,
            'n_fft': learner.n_fft,
            'hop_length': learner.hop_length,
            'max_duration': learner.max_duration,
            'train_split': learner.train_split,
            'val_split': learner.val_split,
            'test_split': learner.test_split,
            # Training parameters
            'epochs': learner.epochs,
            'batch_size': learner.batch_size,
            'learning_rate': learner.learning_rate,
            'weight_decay': learner.weight_decay,
            'optimizer': learner.optimizer,
            'loss_fn': learner.loss_fn,
            'scheduler': learner.scheduler,
            'warmup_epochs': learner.warmup_epochs,
            'early_stopping_patience': learner.early_stopping_patience,
            'min_delta': learner.min_delta,
            # Hardware parameters
            'device': learner.device,
            'num_workers': learner.num_workers,
            'pin_memory': learner.pin_memory,
            # Output parameters
            'output_dir': learner.output_dir,
            'export_onnx': learner.export_onnx,
            'export_coreml': learner.export_coreml,
            # Logging parameters
            'log_interval': learner.log_interval,
            'save_interval': learner.save_interval,
            # Metadata
            'author': learner.author,
            'notes': learner.notes,
            'labels': learner.labels,
            # Legacy parameters
            'regularization': learner.regularization,
            # C++ ML Interface parameters
            'model_directory': learner.model_directory,
            'use_gpu': learner.use_gpu,
            'use_coreml': learner.use_coreml,
            'timeout_ms': learner.timeout_ms,
            'inference_thread_priority': learner.inference_thread_priority,
            # Additional training parameters
            'loss': learner.loss,
            'focal_gamma': learner.focal_gamma,
            'label_smoothing': learner.label_smoothing,
            # Augmentation parameters
            'use_augmentation': learner.use_augmentation,
            'aug_time_stretch': learner.aug_time_stretch,
            'aug_pitch_shift': learner.aug_pitch_shift,
            'aug_noise': learner.aug_noise,
            # Weight statistics
            'weight_mean': float(np.mean(learner.weights)),
            'weight_std': float(np.std(learner.weights)),
            'bias_mean': float(np.mean(learner.biases)),
            'bias_std': float(np.std(learner.biases))
        }


def load_teaching_parameters() -> Dict[str, Any]:
    """
    Load teaching parameter data from the learning module.

    Returns:
        Dictionary containing various teaching parameters
    """
    # Import here to avoid circular imports
    from music_brain.learning.instruments import INSTRUMENTS
    from music_brain.learning.pedagogy import TEACHING_PROMPT_TEMPLATES, TEACHING_SEQUENCES
    from music_brain.learning.resources import KNOWN_SOURCES

    return {
        'instruments': INSTRUMENTS,
        'teaching_prompt_templates': TEACHING_PROMPT_TEMPLATES,
        'teaching_sequences': TEACHING_SEQUENCES,
        'known_sources': KNOWN_SOURCES
    }


def create_learner_from_teaching_data(task_name: str, data_type: str = 'instruments') -> OpenWeightLearner:
    """
    Create an OpenWeightLearner initialized with teaching data.

    Args:
        task_name: Name of the learning task
        data_type: Type of teaching data to use ('instruments', 'prompts', etc.)

    Returns:
        Initialized OpenWeightLearner
    """
    teaching_data = load_teaching_parameters()

    if data_type == 'instruments':
        # Use instrument features for learning
        instruments = list(teaching_data['instruments'].values())
        # Extract numerical features from instruments
        features = []
        for inst in instruments[:10]:  # Use first 10 for demo
            feature_vector = [
                inst.beginner_friendly,
                inst.days_to_first_song,
                len(inst.first_skills),
                len(inst.practice_tips),
                len(inst.primary_genres)
            ]
            features.append(feature_vector)

        if features:
            input_dim = len(features[0])
            output_dim = 5  # Arbitrary output dimension
            learner = OpenWeightLearner(input_dim=input_dim, output_dim=output_dim)

            # Initialize with some data
            x = np.array(features)
            y = np.random.rand(len(features), output_dim)  # Random targets for demo
            learner.update_weights(x, y)

            return learner

    elif data_type == 'prompts':
        # Use prompt template features
        prompts = list(teaching_data['teaching_prompt_templates'].values())
        # Simple feature extraction from prompts
        features = []
        for prompt in prompts:
            feature_vector = [
                len(prompt),
                prompt.count('{'),
                prompt.count('topic'),
                prompt.count('instrument')
            ]
            features.append(feature_vector)

        if features:
            input_dim = len(features[0])
            output_dim = 3
            learner = OpenWeightLearner(input_dim=input_dim, output_dim=output_dim)

            x = np.array(features)
            y = np.random.rand(len(features), output_dim)
            learner.update_weights(x, y)

            return learner

    # Default learner
    return OpenWeightLearner(input_dim=10, output_dim=5)


# Convenience functions
def create_emotion_learner() -> OpenWeightLearner:
    """Create a learner for emotion recognition."""
    return OpenWeightLearner(input_dim=128, output_dim=7)  # 7 basic emotions


def create_generation_learner() -> OpenWeightLearner:
    """Create a learner for generation parameter adaptation."""
    return OpenWeightLearner(input_dim=20, output_dim=10)  # Various generation params


def create_adaptive_learner(input_dim: int, output_dim: int,
                          learning_rate: float = 0.05) -> OpenWeightLearner:
    """Create an adaptive learner with optimized defaults."""
    return OpenWeightLearner(
        input_dim=input_dim,
        output_dim=output_dim,
        learning_rate=learning_rate,
        regularization=0.001  # Light regularization
    )

