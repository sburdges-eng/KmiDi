# PRROT/PARROT 16GB Mac Safety Guide

## Overview

This document outlines the safety mechanisms in place to ensure PRROT/PARROT operates safely on 16GB Mac systems without causing memory pressure or system instability.

## Memory Constraints

### System Memory Layout (16GB Mac)

```
Total: 16GB
├── macOS System: ~4-6GB (base system + background processes)
├── DAW/Applications: ~2-4GB (when DAW is running)
├── PRROT Worker: ~8GB (maximum, typically 6-7GB with Q4 model)
└── System Reserve: ~2GB (required for stability)
```

### Worker Memory Limits

- **Maximum Worker Memory**: 8GB
- **Warning Threshold**: 6GB
- **Model Size (Q4 3B)**: ~1.5-2GB in memory
- **Worker Process**: Must exit fully after job completion

## Safety Mechanisms

### 1. Single Worker Process Lock

**Location**: `python/prrot/utils/process_manager.py`

- Ensures only **one** ML worker runs at a time
- Uses lock file (`/tmp/prrot_worker.lock`) to prevent concurrent workers
- Automatically detects and kills stale processes (>1 hour old)
- Registers signal handlers (SIGINT, SIGTERM) for cleanup

**Usage**:
```python
from prrot.utils.process_manager import ProcessManager

manager = ProcessManager()
if not manager.acquire_lock():
    # Another worker is running - exit
    sys.exit(1)

try:
    # Process job
    ...
finally:
    manager.release_lock()
```

### 2. Memory Monitoring

**Location**: `python/prrot/utils/memory_monitor.py`

- Monitors process memory usage in real-time
- Checks system available memory
- Warns when approaching limits
- Blocks operation if limits exceeded

**Checks**:
- Process memory < 8GB
- System available memory > 2GB
- Memory warnings at 6GB threshold

### 3. Model Manager

**Location**: `python/prrot/model_manager.py`

- Enforces Q4 quantization (required for 16GB)
- Validates memory before model loading
- Ensures single model loaded at a time
- Tracks loaded models for cleanup

**Features**:
- Pre-load memory checks
- Post-load memory verification
- Automatic model unloading
- Garbage collection after unload

### 4. Worker Process Lifecycle

**Location**: `python/prrot/worker.py`

**Lifecycle**:
1. Acquire process lock (single worker)
2. Check system memory availability
3. Check process memory limits
4. Load model (if needed, with memory checks)
5. Process job
6. Unload model
7. Force garbage collection
8. Release process lock
9. Exit completely

**Guarantees**:
- ✅ Worker exits fully after job completion
- ✅ Memory is reclaimed via garbage collection
- ✅ Process lock is released
- ✅ No persistent processes

### 5. External SSD Usage

**Location**: `python/prrot/utils/external_ssd.py`

**Usage Rules**:
- External SSD used **only** for:
  - Reference audio files
  - Voice profiles (JSON)
  - Model files (quantized)
  - Job artifacts
  - Cache files
- **Never** used for:
  - ❌ Swap/paging
  - ❌ Temporary processing files (use RAM)
  - ❌ System dependencies

**Bandwidth Optimization**:
- Batch file operations (10MB chunks)
- Cache frequently accessed files
- Reuse cached data when possible

## Best Practices

### For Users

1. **Close other applications** before running PRROT worker
   - Free up system memory
   - Ensure 10GB+ available before starting

2. **Run one job at a time**
   - Worker automatically enforces this
   - Wait for job completion before starting next

3. **Monitor system memory**
   - Use Activity Monitor
   - Check "Memory Pressure" indicator
   - Should stay in green/yellow range

4. **Use Q4 quantized models only**
   - Required for 16GB systems
   - Q8 or full precision not supported

5. **Close DAW when running analysis**
   - Frees up memory for worker
   - Prevents memory pressure

### For Developers

1. **Always use ModelManager** for model loading
   ```python
   from prrot.model_manager import ModelManager

   manager = ModelManager()
   model = manager.load_model(model_path, quantization="Q4")
   ```

2. **Always use ProcessManager** for worker processes
   ```python
   from prrot.utils.process_manager import ProcessManager

   manager = ProcessManager()
   if not manager.acquire_lock():
       return  # Exit if another worker running
   ```

3. **Check memory before operations**
   ```python
   can_load, warning = model_manager.check_memory_before_load(2.0)
   if not can_load:
       raise RuntimeError("Insufficient memory")
   ```

4. **Always unload models after use**
   ```python
   model_manager.unload_model()  # Unload all
   gc.collect()  # Force garbage collection
   ```

5. **Verify memory after operations**
   ```python
   process_mem, _, _ = memory_monitor.get_memory_usage()
   if process_mem > WORKER_WARN_GB:
       logger.warning(f"High memory usage: {process_mem:.2f}GB")
   ```

## Memory Verification

### Before Starting Worker

```bash
# Check available memory
python3 -c "import psutil; m = psutil.virtual_memory(); print(f'Available: {m.available/(1024**3):.2f}GB')"

# Should show > 10GB available
```

### During Worker Execution

The worker logs memory usage:
```
INFO - Process memory: 6.2GB
WARNING - Memory warning: Process memory (6.2GB) approaching limit
```

### After Worker Completion

```bash
# Verify worker exited
ps aux | grep prrot.worker

# Should show no processes (except grep itself)

# Verify memory released
python3 -c "import psutil; m = psutil.virtual_memory(); print(f'Available: {m.available/(1024**3):.2f}GB')"
```

## Troubleshooting

### "Another PRROT worker is already running"

**Cause**: Lock file exists from previous worker that didn't exit properly.

**Solution**:
```bash
# Remove stale lock file
rm /tmp/prrot_worker.lock /tmp/prrot_worker.pid

# Or kill stale process if PID file exists
cat /tmp/prrot_worker.pid | xargs kill 2>/dev/null
rm /tmp/prrot_worker.lock /tmp/prrot_worker.pid
```

### "Memory limit exceeded"

**Cause**: System or process memory insufficient.

**Solutions**:
1. Close other applications
2. Close DAW
3. Check for memory leaks in other processes
4. Restart system if needed

### "Insufficient system memory"

**Cause**: Less than 10GB available.

**Solutions**:
1. Close applications
2. Check Activity Monitor for memory hogs
3. Restart to free memory
4. Run worker when DAW is not running

### High Memory Usage After Job

**Cause**: Worker didn't exit properly or model not unloaded.

**Solutions**:
1. Check if worker process still running: `ps aux | grep prrot.worker`
2. Kill process: `pkill -f prrot.worker`
3. Verify memory released
4. Check for memory leaks in worker code

## System Requirements

### Minimum for 16GB Mac

- **Available Memory**: 10GB+ before starting worker
- **External SSD**: 4TB (USB 2.0 acceptable)
- **macOS**: 10.12+ (for compatibility)
- **Python**: 3.8+
- **Dependencies**: numpy, psutil

### Recommended

- **Available Memory**: 12GB+ before starting worker
- **External SSD**: USB 3.0+ (faster I/O)
- **Close DAW** when running analysis
- **Close other apps** to maximize available memory

## Testing Memory Safety

### Test Single Worker Constraint

```bash
# Terminal 1
python -m prrot.worker job1.json

# Terminal 2 (should fail)
python -m prrot.worker job2.json
# Expected: "Another PRROT worker is already running"
```

### Test Memory Limits

```python
# Force high memory usage test
from prrot.utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor()
within_limit, warning = monitor.check_memory_limit()
print(f"Within limit: {within_limit}, Warning: {warning}")
```

### Test Process Cleanup

```bash
# Start worker
python -m prrot.worker job.json &

# Wait for completion
wait

# Verify process exited
ps aux | grep prrot.worker
# Should show no processes
```

## Summary

PRROT/PARROT is designed to be safe for 16GB Mac systems through:

1. ✅ **Single worker enforcement** (process lock)
2. ✅ **Memory monitoring** (pre/post checks)
3. ✅ **Model management** (Q4 only, single model)
4. ✅ **Process lifecycle** (exit after job, cleanup)
5. ✅ **External SSD** (storage only, never swap)
6. ✅ **Garbage collection** (force cleanup after jobs)

All mechanisms are automatic and require no user intervention. The system will refuse to operate if memory constraints cannot be met.
