"""
Model Manager - Memory-constrained model loading for 16GB systems

Ensures models are loaded with Q4 quantization and proper memory management.
"""

import logging
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
import gc

from .utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model loading with 16GB constraint compliance"""

    # Memory constraints for 16GB systems
    MAX_MODEL_SIZE_GB = 2.0  # Maximum model size in memory (Q4 quantized 3B model ~1.5-2GB)
    WARN_MODEL_SIZE_GB = 1.5  # Warn if approaching limit

    def __init__(self):
        """Initialize model manager"""
        self.memory_monitor = MemoryMonitor()
        self.loaded_models: Dict[str, Any] = {}  # Track loaded models

    def check_memory_before_load(self, model_size_estimate_gb: float = 2.0) -> tuple[bool, str]:
        """
        Check if sufficient memory is available before loading model

        Args:
            model_size_estimate_gb: Estimated model size in GB

        Returns:
            (can_load, warning_message)
        """
        process_mem, system_available, system_total = self.memory_monitor.get_memory_usage()

        # Check if model would exceed worker limit
        if process_mem + model_size_estimate_gb > MemoryMonitor.WORKER_MAX_GB:
            return (
                False,
                f"Model would exceed worker memory limit: {process_mem:.2f}GB + {model_size_estimate_gb:.2f}GB > {MemoryMonitor.WORKER_MAX_GB}GB",
            )

        # Check if system has enough available memory
        if system_available < model_size_estimate_gb + 2.0:  # Model + 2GB reserve
            return (
                False,
                f"Insufficient system memory: {system_available:.2f}GB available, need {model_size_estimate_gb + 2.0:.2f}GB",
            )

        if process_mem + model_size_estimate_gb > MemoryMonitor.WORKER_WARN_GB:
            return (
                True,
                f"Warning: Memory usage will be high: {process_mem:.2f}GB + {model_size_estimate_gb:.2f}GB",
            )

        return True, ""

    def load_model(
        self, model_path: Path, model_type: str = "phoneme_aligner", quantization: str = "Q4"
    ) -> Optional[Any]:
        """
        Load ML model with memory constraints

        Args:
            model_path: Path to model file
            model_type: Type of model (phoneme_aligner, etc.)
            quantization: Quantization type (must be Q4 for 16GB systems)

        Returns:
            Loaded model or None if load fails
        """
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        # Enforce Q4 quantization for 16GB systems
        if quantization != "Q4":
            logger.warning(
                f"Non-Q4 quantization requested ({quantization}), but Q4 required for 16GB systems"
            )
            quantization = "Q4"

        # Estimate model size (Q4: ~1.5-2GB for 3B parameter model)
        model_size_gb = 2.0  # Conservative estimate

        # Check memory before loading
        can_load, warning = self.check_memory_before_load(model_size_gb)
        if not can_load:
            logger.error(f"Cannot load model: {warning}")
            return None

        if warning:
            logger.warning(f"Memory warning: {warning}")

        logger.info(f"Loading model: {model_path} (Q4 quantization, ~{model_size_gb}GB)")

        # Placeholder: In production, would load actual model
        # Example for llama.cpp:
        # from llama_cpp import Llama
        # model = Llama(
        #     model_path=str(model_path),
        #     n_ctx=4096,  # Context window
        #     n_threads=4,  # CPU threads
        #     n_gpu_layers=0,  # No GPU layers on 16GB Mac
        #     verbose=False
        # )

        # Track loaded model
        model_key = f"{model_type}_{model_path.name}"

        # For now, return placeholder
        model = None  # Would be actual model object

        if model is not None:
            self.loaded_models[model_key] = model

            # Verify memory after load
            process_mem, _, _ = self.memory_monitor.get_memory_usage()
            logger.info(f"Model loaded. Process memory: {process_mem:.2f}GB")

            if process_mem > MemoryMonitor.WORKER_WARN_GB:
                logger.warning(f"High memory usage after model load: {process_mem:.2f}GB")

        return model

    def unload_model(self, model_key: Optional[str] = None) -> None:
        """
        Unload model and free memory

        Args:
            model_key: Specific model to unload, or None to unload all
        """
        if model_key:
            if model_key in self.loaded_models:
                logger.info(f"Unloading model: {model_key}")
                del self.loaded_models[model_key]
        else:
            logger.info(f"Unloading all models ({len(self.loaded_models)} models)")
            self.loaded_models.clear()

        # Force garbage collection
        gc.collect()

        # Verify memory freed
        process_mem, _, _ = self.memory_monitor.get_memory_usage()
        logger.info(f"Model unloaded. Process memory: {process_mem:.2f}GB")

    def get_loaded_models(self) -> Dict[str, Any]:
        """Get all currently loaded models"""
        return self.loaded_models.copy()

    def ensure_single_model(self, new_model_key: str) -> None:
        """
        Ensure only one model is loaded at a time

        Unloads all other models before loading new one.
        """
        if len(self.loaded_models) > 0:
            logger.info("Unloading existing models to ensure single model constraint")
            self.unload_model()
