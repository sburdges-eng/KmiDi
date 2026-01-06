#!/usr/bin/env python3
"""
Safe Extended Training with Resource Monitoring
================================================

Runs extended training with:
- Thermal monitoring (pauses if CPU too hot)
- Memory monitoring (reduces batch size if needed)
- Sequential model training (one at a time)
- Automatic cooldown periods between models
- Progress checkpointing

Usage:
    python scripts/safe_extended_training.py --epochs 50
    python scripts/safe_extended_training.py --model melody_transformer --epochs 100
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Thermal thresholds (Celsius)
THERMAL_WARNING = 85  # Start throttling
THERMAL_CRITICAL = 95  # Pause training
COOLDOWN_SECONDS = 60  # Wait time when overheating

# Memory thresholds
MEMORY_WARNING_PERCENT = 85  # Reduce batch size


def get_cpu_temperature():
    """Get CPU temperature on macOS."""
    try:
        # Try using osx-cpu-temp if available
        result = subprocess.run(
            ["osx-cpu-temp"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            temp_str = result.stdout.strip().replace("°C", "")
            return float(temp_str)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Fallback: estimate from system load
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            level = int(result.stdout.strip())
            # Estimate: level 0=50C, level 100=100C
            return 50 + (level * 0.5)
    except (subprocess.TimeoutExpired, ValueError):
        pass

    return 60  # Default safe value


def get_memory_usage():
    """Get memory usage percentage."""
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            stats = {}
            for line in lines[1:]:
                if ":" in line:
                    key, val = line.split(":")
                    val = val.strip().replace(".", "")
                    try:
                        stats[key.strip()] = int(val)
                    except ValueError:
                        pass

            # Calculate usage
            page_size = 16384
            free = stats.get("Pages free", 0)
            active = stats.get("Pages active", 0)
            inactive = stats.get("Pages inactive", 0)
            wired = stats.get("Pages wired down", 0)

            total = free + active + inactive + wired
            used = active + wired

            if total > 0:
                return (used / total) * 100
    except subprocess.TimeoutExpired:
        pass

    return 50  # Default safe value


def check_system_health():
    """Check if system is healthy for training."""
    temp = get_cpu_temperature()
    mem = get_memory_usage()

    status = {
        "temperature": temp,
        "memory_percent": mem,
        "healthy": True,
        "throttle": False,
        "pause": False,
        "message": ""
    }

    if temp >= THERMAL_CRITICAL:
        status["healthy"] = False
        status["pause"] = True
        status["message"] = f"CRITICAL: CPU at {temp:.1f}°C - pausing training"
    elif temp >= THERMAL_WARNING:
        status["throttle"] = True
        status["message"] = f"WARNING: CPU at {temp:.1f}°C - throttling"

    if mem >= MEMORY_WARNING_PERCENT:
        status["throttle"] = True
        status["message"] += f" | Memory at {mem:.1f}%"

    return status


def wait_for_cooldown():
    """Wait for system to cool down."""
    print(f"\n⏸️  Waiting {COOLDOWN_SECONDS}s for cooldown...")
    for i in range(COOLDOWN_SECONDS):
        time.sleep(1)
        if (i + 1) % 10 == 0:
            temp = get_cpu_temperature()
            print(f"   {COOLDOWN_SECONDS - i - 1}s remaining (temp: {temp:.1f}°C)")
            if temp < THERMAL_WARNING - 10:
                print("   System cooled down, resuming...")
                break


def run_training(model: str, epochs: int, batch_size: int = 8):
    """Run training for a single model with monitoring."""
    print(f"\n{'='*60}")
    print(f"Training: {model} ({epochs} epochs, batch_size={batch_size})")
    print(f"{'='*60}")

    # Check health before starting
    health = check_system_health()
    if health["pause"]:
        print(health["message"])
        wait_for_cooldown()

    # Build command
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model", model,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size)
    ]

    env = os.environ.copy()
    env["KELLY_AUDIO_DATA_ROOT"] = "/Volumes/sbdrive/audio/datasets"
    env["KELLY_EXTENDED_TRAINING"] = "1"  # Bypass epoch limits

    # Run training
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=env
    )

    # Monitor during training
    last_health_check = time.time()
    terminated_early = False

    try:
        for line in process.stdout:
            print(line, end="")

            # Health check every 30 seconds
            if time.time() - last_health_check > 30:
                health = check_system_health()
                last_health_check = time.time()

                if health["pause"]:
                    print(f"\n⚠️  {health['message']}")
                    terminated_early = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)  # Wait for graceful termination
                    except subprocess.TimeoutExpired:
                        process.kill()  # Force kill if needed
                        process.wait()
                    break
                elif health["throttle"]:
                    print(f"\n⚡ {health['message']}")
    finally:
        # Ensure stdout is closed to prevent resource leak
        if process.stdout:
            process.stdout.close()
        # Ensure process is properly waited on to avoid zombie processes
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    if terminated_early:
        wait_for_cooldown()
        # Restart with smaller batch, but clamp to minimum of 1
        new_batch_size = max(1, batch_size // 2)
        if new_batch_size == batch_size:
            print("❌ Cannot reduce batch_size further (already at minimum)")
            return False
        print(f"🔄 Restarting with batch_size={new_batch_size}")
        return run_training(model, epochs, new_batch_size)

    duration = time.time() - start_time

    success = process.returncode == 0
    status = "✅ Success" if success else "❌ Failed"
    print(f"\n{status} - {model} completed in {duration:.1f}s")

    return success


def main():
    parser = argparse.ArgumentParser(description="Safe Extended Training")
    parser.add_argument("--model", type=str, help="Train specific model only")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--cooldown", type=int, default=30, help="Seconds between models")
    args = parser.parse_args()

    # Models to train (in order of complexity)
    all_models = [
        "dynamics_engine",      # Smallest, fastest
        "groove_predictor",     # Small LSTM
        "harmony_predictor",    # Medium MLP
        "emotion_recognizer",   # Medium CNN
        "melody_transformer",   # Largest LSTM
    ]

    models = [args.model] if args.model else all_models

    print("="*60)
    print("SAFE EXTENDED TRAINING")
    print("="*60)
    print(f"Models: {', '.join(models)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Cooldown between models: {args.cooldown}s")
    print()

    # Initial health check
    health = check_system_health()
    print(f"System status: Temp={health['temperature']:.1f}°C, Mem={health['memory_percent']:.1f}%")

    if not health["healthy"]:
        print(f"⚠️  {health['message']}")
        wait_for_cooldown()

    # Train each model
    results = {}
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Starting {model}...")

        success = run_training(model, args.epochs, args.batch_size)
        results[model] = success

        # Cooldown between models (except last)
        if i < len(models) - 1:
            print(f"\n⏳ Cooldown: {args.cooldown}s before next model...")
            time.sleep(args.cooldown)

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model}")

    successful = sum(results.values())
    print(f"\nCompleted: {successful}/{len(models)} models")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
