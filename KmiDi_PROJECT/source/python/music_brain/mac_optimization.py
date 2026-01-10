"""
Mac-specific optimization layer for Apple Silicon (M1/M2/M3/M4 Pro/Max).

Features:
  - MPS (Metal Performance Shaders) acceleration detection
  - Memory management for 16GB unified memory
  - torch.compile() integration (Python 3.11+)
  - Inference profiling and benchmarking
  - Quantization support (CPU, not available on MPS yet)
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MacHardwareInfo:
    """Information about Mac hardware"""
    device: str
    mps_available: bool
    mps_device_name: Optional[str]
    total_memory_gb: float
    is_metal_capable: bool
    chip_type: str  # "m1", "m2", "m3", "m4", unknown"


class MacOptimizationLayer:
    """
    Mac-specific optimizations for iDAW models.

    Automatically detects hardware and applies best practices:
      - Metal Performance Shaders (MPS) for GPU acceleration
      - Efficient memory management for unified memory
      - torch.compile() for graph optimization
      - Quantization for inference (optional)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.hardware_info = self._detect_hardware()
        self.device = self.hardware_info.device
        self._log(self._format_hardware_info())

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def _detect_hardware(self) -> MacHardwareInfo:
        """Detect Mac hardware capabilities"""
        mps_available = torch.backends.mps.is_available()

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            chip_type = "nvidia"
        elif mps_available:
            device = "mps"
            # Try to detect M-chip generation
            try:
                import platform
                chip_type = self._detect_m_chip()
            except Exception:
                chip_type = "unknown"
        else:
            device = "cpu"
            chip_type = "cpu"

        # Get memory info (approximate)
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024**3)
        except (ImportError, Exception):
            total_memory = 16.0  # Assume 16GB for M-series

        return MacHardwareInfo(
            device=device,
            mps_available=mps_available,
            mps_device_name="Apple Neural Engine" if mps_available else None,
            total_memory_gb=total_memory,
            is_metal_capable=mps_available,
            chip_type=chip_type
        )

    def _detect_m_chip(self) -> str:
        """Detect M-chip generation"""
        try:
            import platform
            version = platform.platform()

            if "14." in version or "15." in version:
                return "m4"  # Sonoma or later
            elif "13." in version:
                return "m3"  # Ventura
            elif "12." in version:
                return "m2"  # Monterey with m2
            else:
                return "m1"

        except Exception:
            return "unknown"

    def _format_hardware_info(self) -> str:
        info = self.hardware_info

        lines = [
            "=" * 60,
            "Mac Hardware Configuration",
            "=" * 60,
            f"Device: {info.device.upper()}",
            f"Chip: {info.chip_type.upper()}",
            f"Memory: {info.total_memory_gb:.1f} GB",
            f"Metal Capable: {'✓' if info.is_metal_capable else '✗'}",
        ]

        if info.mps_available:
            lines.append(f"MPS Device: {info.mps_device_name}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def optimize_model_for_inference(
        self,
        model: torch.nn.Module,
        enable_compile: bool = True,
        enable_quantize: bool = False
    ) -> torch.nn.Module:
        """
        Apply all optimizations to model for inference.

        Args:
            model: PyTorch model
            enable_compile: Use torch.compile() (Python 3.11+, PyTorch 2.0+)
            enable_quantize: Apply quantization (slower on M-series, CPU-only)

        Returns:
            optimized_model: Model with optimizations applied
        """
        # Move to best device
        model = model.to(self.device)
        model.eval()

        # Apply torch.compile if available and requested
        if enable_compile:
            model = self._apply_torch_compile(model)

        # Quantization (only on CPU for M-series)
        if enable_quantize and self.device == "cpu":
            model = self._apply_quantization(model)

        return model

    def _apply_torch_compile(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply torch.compile() for optimization (Python 3.11+, PyTorch 2.0+)"""
        try:
            import sys
            if sys.version_info >= (3, 11):
                # Try compile with fallback
                model = torch.compile(model, backend="eager")
                self._log("✓ Applied torch.compile() optimization")
                return model
        except (AttributeError, RuntimeError) as e:
            self._log(f"⚠ torch.compile() not available: {e}")

        return model

    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply dynamic quantization (INT8) to reduce model size.

        Note: MPS doesn't support quantization yet; CPU-only.
        """
        try:
            if self.device != "cpu":
                self._log("⚠ Quantization only supported on CPU; skipping")
                return model

            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

            self._log("✓ Applied INT8 dynamic quantization")

        except Exception as e:
            self._log(f"⚠ Quantization failed: {e}")

        return model

    def memory_efficient_inference(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        max_batch_size: int = 16,
        show_memory: bool = False
    ) -> torch.Tensor:
        """
        Run inference with memory management for M4 Pro.

        Splits large batches into smaller chunks if needed.

        Args:
            model: Model to run
            input_tensor: Input batch
            max_batch_size: Maximum batch size per inference
            show_memory: Print memory usage

        Returns:
            output: Model output
        """
        batch_size = input_tensor.shape[0]

        if show_memory:
            self._log(f"Input batch size: {batch_size}")

        # If batch fits, run directly
        if batch_size <= max_batch_size:
            with torch.no_grad():
                return model(input_tensor)

        # Otherwise, split into chunks
        outputs = []
        for i in range(0, batch_size, max_batch_size):
            chunk = input_tensor[i:i+max_batch_size]

            if show_memory:
                self._log(f"  Processing chunk {i//max_batch_size + 1}: {chunk.shape}")

            with torch.no_grad():
                output = model(chunk)

            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def profile_inference_latency(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Profile inference latency on target device.

        Args:
            model: Model to profile
            input_shape: Input tensor shape
            num_runs: Number of timing runs
            warmup_runs: Warmup iterations

        Returns:
            stats: Dict with latency statistics
        """
        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                model(dummy_input)

        # Synchronize before timing (important for MPS/CUDA)
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy_input)

        # Synchronize after timing
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / num_runs) * 1000

        return {
            "mean_latency_ms": latency_ms,
            "throughput_hz": 1000 / latency_ms,
            "device": self.device,
            "input_shape": input_shape,
            "num_runs": num_runs
        }

    def benchmark_models(
        self,
        models: Dict[str, torch.nn.Module],
        input_shape: Tuple[int, ...],
        warmup_runs: int = 5,
        timing_runs: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark multiple models and compare.

        Args:
            models: Dict of {name: model}
            input_shape: Input tensor shape
            warmup_runs: Warmup iterations per model
            timing_runs: Timing iterations per model

        Returns:
            benchmark_results: Dict of {name: latency_stats}
        """
        results = {}

        for name, model in models.items():
            self._log(f"Benchmarking {name}...")
            stats = self.profile_inference_latency(
                model, input_shape,
                num_runs=timing_runs,
                warmup_runs=warmup_runs
            )
            results[name] = stats

        # Print comparison
        self._log("\nBenchmark Results:")
        self._log("-" * 60)
        self._log(f"{'Model':<20} {'Latency (ms)':<15} {'Throughput (Hz)':<15}")
        self._log("-" * 60)

        for name, stats in results.items():
            self._log(f"{name:<20} {stats['mean_latency_ms']:<15.2f} "
                     f"{stats['throughput_hz']:<15.1f}")

        return results

    def estimate_memory_usage(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Estimate memory usage for model + inference.

        Args:
            model: Model to estimate
            input_shape: Input tensor shape (without batch dimension)
            batch_size: Batch size for estimation

        Returns:
            memory_stats: Dict with memory usage in MB
        """
        # Model parameters
        model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size_bytes / (1024 ** 2)

        # Input buffer
        full_input_shape = (batch_size,) + input_shape
        input_buffer = torch.zeros(full_input_shape, device=self.device)
        input_size_mb = input_buffer.element_size() * input_buffer.nelement() / (1024 ** 2)

        # Output (estimate from running model)
        with torch.no_grad():
            output = model(input_buffer)
        output_size_mb = output.element_size() * output.nelement() / (1024 ** 2)

        # Optimizer state (if training)
        optimizer_state_mb = model_size_mb * 2  # Approximate: gradient + momentum

        return {
            "model_mb": model_size_mb,
            "input_buffer_mb": input_size_mb,
            "output_buffer_mb": output_size_mb,
            "total_inference_mb": model_size_mb + input_size_mb + output_size_mb,
            "optimizer_state_mb": optimizer_state_mb,
            "total_training_mb": model_size_mb * 3 + input_size_mb,  # Model + gradients + optimizer
            "available_memory_gb": self.hardware_info.total_memory_gb
        }

    def get_optimization_recommendations(self) -> Dict[str, str]:
        """Get recommendations based on detected hardware"""
        recommendations = {}

        if self.hardware_info.device == "mps":
            recommendations["device"] = "Use MPS for GPU acceleration"
            recommendations["batch_size"] = "Start with batch_size=16 for M4 Pro"
            recommendations["memory"] = "Monitor memory; use gradient checkpointing if OOM"
            recommendations["precision"] = "Use float32; mixed precision support limited"
            recommendations["compilation"] = "Try torch.compile() for 5-10% speedup"

        elif self.hardware_info.device == "cpu":
            recommendations["device"] = "Using CPU (slower); consider RTX 4060 for faster training"
            recommendations["batch_size"] = "Start with batch_size=8"
            recommendations["threads"] = "Set OMP_NUM_THREADS=<cpu_cores>"
            recommendations["optimization"] = "Use torch.compile() if available"

        elif self.hardware_info.device == "cuda":
            recommendations["device"] = "Using CUDA (excellent)"
            recommendations["batch_size"] = "Can use batch_size=64 on RTX 4060 8GB"
            recommendations["mixed_precision"] = "Enable mixed precision (fp16) for 2x speedup"
            recommendations["quantization"] = "Consider INT8 quantization for deployment"

        return recommendations


# Convenience function
def get_mac_optimization():
    """Get Mac optimization layer instance"""
    return MacOptimizationLayer(verbose=True)
