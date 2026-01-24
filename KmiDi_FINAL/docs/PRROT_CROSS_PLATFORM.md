# PRROT/PARROT Cross-Platform Support

**Date**: 2025-01-18
**Status**: ✅ **Cross-Platform Compatible**

## Supported Platforms

PRROT/PARROT is designed to work on multiple platforms:

- ✅ **macOS** (Apple Silicon and Intel)
- ✅ **Linux** (x86_64 and ARM64)
- ✅ **Windows** (x86_64)

## Platform-Specific Considerations

### Memory Constraints

The 16GB constraint was designed for Mac systems, but the same safety mechanisms apply to other platforms:

- **Linux/Windows with 16GB RAM**: Same constraints apply (8GB worker limit, 10GB system reserve)
- **Systems with more RAM**: The worker limit scales automatically based on available system memory
- **Systems with less RAM**: The system will refuse to start if insufficient memory is available

### Lock File Location

Process lock files use platform-appropriate temporary directories:

- **macOS/Linux**: `/tmp/prrot_worker.lock` (or `$TMPDIR` if set)
- **Windows**: `%TEMP%\prrot_worker.lock`

The implementation uses Python's `tempfile.gettempdir()` for cross-platform compatibility.

### Signal Handling

Signal handlers are registered appropriately for each platform:

- **Unix-like (macOS, Linux)**: SIGINT and SIGTERM handlers
- **Windows**: SIGINT handler (Ctrl+C), with graceful fallback if unavailable

### External SSD Paths

Default external SSD paths are platform-specific:

- **macOS**: `/Volumes/ExternalSSD/prrot`
- **Linux**: `/mnt/external_ssd/prrot`
- **Windows**: `E:/prrot` (configurable drive letter)

Users can override via environment variable or command-line argument:

```bash
export PRROT_EXTERNAL_SSD_PATH=/custom/path
python -m prrot.worker job.json --external-ssd /custom/path
```

## Build Instructions

### macOS

```bash
cd KmiDi_FINAL
mkdir -p build && cd build
cmake .. -DBUILD_KMIDI_CORE=ON
cmake --build . -j$(sysctl -n hw.ncpu)
```

### Linux

```bash
cd KmiDi_FINAL
mkdir -p build && cd build
cmake .. -DBUILD_KMIDI_CORE=ON
cmake --build . -j$(nproc)
```

### Windows (Visual Studio)

```powershell
cd KmiDi_FINAL
mkdir build
cd build
cmake .. -DBUILD_KMIDI_CORE=ON -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release -j
```

### Cross-Platform Build Command

For maximum compatibility:

```bash
# Automatically detects platform and uses appropriate parallel build count
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
```

## Python Dependencies

All Python dependencies are cross-platform:

- `psutil` - Cross-platform process and system utilities
- `numpy` - Cross-platform numerical computing
- Standard library modules (json, pathlib, logging, signal, tempfile)

No platform-specific Python packages are required.

## Memory Monitoring

Memory monitoring uses `psutil`, which provides cross-platform memory statistics:

```python
# Works identically on all platforms
memory = psutil.virtual_memory()
available_gb = memory.available / (1024**3)
```

## File Path Handling

All file paths use Python's `pathlib.Path`, which handles platform-specific path separators automatically:

```python
# Works on all platforms
from pathlib import Path
profile_path = Path("/path/to/profile.json")  # Unix
profile_path = Path("C:/path/to/profile.json")  # Windows
profile_path = Path.home() / "prrot_data"  # Cross-platform home directory
```

## Process Management

Process management is cross-platform:

- **Lock files**: Platform-appropriate temp directory
- **Process detection**: `psutil.pid_exists()` works on all platforms
- **Process termination**: `psutil.Process.terminate()` works cross-platform
- **Signal handling**: Platform-appropriate signals registered

## Testing on Different Platforms

### Verify Cross-Platform Compatibility

1. **Lock File Location**:
   ```python
   from prrot.utils.process_manager import ProcessManager
   import tempfile
   manager = ProcessManager()
   print(f"Lock file: {manager.LOCK_FILE}")
   print(f"Temp dir: {tempfile.gettempdir()}")
   ```

2. **Memory Monitoring**:
   ```python
   from prrot.utils.memory_monitor import MemoryMonitor
   monitor = MemoryMonitor()
   stats = monitor.get_memory_stats()
   print(f"Platform: {platform.system()}")
   print(f"Available memory: {stats['system_available_gb']:.2f}GB")
   ```

3. **External SSD Path**:
   ```python
   from prrot.utils.external_ssd import ExternalSSDManager
   import platform
   ssd = ExternalSSDManager()
   print(f"Platform: {platform.system()}")
   print(f"Base path: {ssd.base_path}")
   ```

## Known Platform Differences

### Windows-Specific

- **Path separators**: Use forward slashes in Python paths (Path handles this)
- **Signal handling**: Limited signal support (SIGINT works, SIGTERM may not)
- **Executable extensions**: `.exe` for executables, `.dll` for libraries

### Linux-Specific

- **Path conventions**: `/mnt/` for external drives
- **Package managers**: Use system package manager for dependencies
- **Shared libraries**: `.so` extensions

### macOS-Specific

- **Path conventions**: `/Volumes/` for external drives
- **Framework support**: Can use `.framework` bundles
- **Universal binaries**: Can build for both Intel and Apple Silicon

## Recommendations

1. **Use pathlib.Path**: Always use `Path` objects instead of string paths
2. **Check platform when needed**: Use `platform.system()` for platform detection
3. **Use psutil**: For all system/process operations (cross-platform)
4. **Test on target platform**: Verify behavior on actual target platforms

## Compatibility Matrix

| Feature | macOS | Linux | Windows |
|---------|-------|-------|---------|
| Tier C (C++) | ✅ | ✅ | ✅ |
| Tier B (Python) | ✅ | ✅ | ✅ |
| Memory Monitoring | ✅ | ✅ | ✅ |
| Process Locking | ✅ | ✅ | ✅ |
| Signal Handling | ✅ | ✅ | ⚠️ Limited |
| External SSD Paths | ✅ | ✅ | ✅ |
| CMake Build | ✅ | ✅ | ✅ |

## Status

✅ **Cross-platform compatibility verified**

All core functionality works on macOS, Linux, and Windows. Platform-specific optimizations are applied automatically where appropriate.

---

**Last Updated**: 2025-01-18
**Platform Support**: macOS, Linux, Windows
