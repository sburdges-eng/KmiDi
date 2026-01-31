# Debug System Integration Guide

**Module**: [mcp_workstation/debug.py](../KmiDi_CANON/brain/mcp_workstation/debug.py)
**Status**: ✅ Fully implemented and tested
**Created**: 2026-01-31

---

## Overview

The debug system provides production-ready error tracking and performance monitoring for the orchestrator.

### Features

✅ **Ring Buffer Event Tracking** - Auto-discard oldest events (configurable max size)
✅ **Error Logging with Stack Traces** - Capture exceptions with full context
✅ **Warning Tracking** - Log warnings with details
✅ **Performance Metrics** - p50/p95/p99 latency statistics
✅ **Context Manager Timing** - Easy performance measurement
✅ **Summary Reports** - Overall system health at a glance

---

## Quick Start

### Basic Usage

```python
from mcp_workstation.debug import log_error, log_warning, measure_performance

# Log a warning
log_warning("LLM model not found, using fallback", details={"model": "llama.cpp"})

# Log an error with exception
try:
    risky_operation()
except Exception as e:
    log_error("Operation failed", exception=e, details={"context": "data"})

# Measure performance
with measure_performance("llm_parse", context={"user_id": 123}):
    result = llm.parse_user_text(text)
```

### Advanced Usage

```python
from mcp_workstation.debug import get_debug, DebugCategory

debug = get_debug()

# Log info with category
debug.info(DebugCategory.AI_COMMUNICATION, "LLM engine initialized")

# Get recent errors
errors = debug.get_errors(limit=10)
for error in errors:
    print(f"{error.timestamp}: {error.message}")
    if error.stack_trace:
        print(error.stack_trace)

# Get performance report
report = debug.get_performance_report()
print(f"Total operations: {report['total_operations']}")
for operation, stats in report['operations'].items():
    print(f"{operation}: mean={stats['mean_ms']:.2f}ms, p95={stats['p95_ms']:.2f}ms")

# Get overall summary
summary = debug.get_summary()
print(f"Errors: {summary['error_count']}, Warnings: {summary['warning_count']}")
```

---

## Integration Examples

### 1. Orchestrator Workflow Integration

**File**: `KmiDi_CANON/brain/mcp_workstation/orchestrator.py`

```python
from .debug import get_debug, log_error, log_warning, measure_performance, DebugCategory

class Workstation:
    def execute_workflow(self, user_intent_text: str, enable_image_gen: bool = False,
                         enable_audio_gen: bool = False) -> CompleteSongIntent:
        debug = get_debug()

        try:
            # Phase 1: LLM Reasoning
            debug.info(DebugCategory.AI_COMMUNICATION, "Starting LLM reasoning phase")

            with measure_performance("llm_reasoning", context={"text_length": len(user_intent_text)}):
                llm_result = self._run_llm_phase(user_intent_text)

            if llm_result["status"] == "failed":
                log_warning("LLM phase failed, using fallback", details=llm_result)

            # Phase 2: MIDI Generation
            with measure_performance("midi_generation"):
                midi_result = self._run_midi_phase(complete_intent)

            # Phase 3: Optional Image
            if enable_image_gen:
                with measure_performance("image_generation"):
                    image_result = self._run_image_phase(complete_intent)

            # Phase 4: Optional Audio
            if enable_audio_gen:
                with measure_performance("audio_generation"):
                    audio_result = self._run_audio_phase(complete_intent)

            return complete_intent

        except Exception as e:
            log_error("Workflow execution failed", exception=e,
                     details={"phase": "unknown", "user_text": user_intent_text[:100]})
            raise
```

### 2. LLM Reasoning Engine Integration

**File**: `KmiDi_CANON/brain/mcp_workstation/llm_reasoning_engine.py`

```python
from .debug import log_warning, measure_performance

class LLMReasoningEngine:
    def parse_user_intent(self, user_text: str) -> CompleteSongIntent:
        with measure_performance("llm_parse", context={"method": "llama_cpp"}):
            # Attempt to load model
            if not self.model_available:
                log_warning("LLM model unavailable, using rule-based fallback",
                           details={"model_path": self.model_path})
                return self._rule_based_parse(user_text)

            # Use LLM
            result = self._llm_parse(user_text)
            return result
```

### 3. MIDI Pipeline Integration

**File**: `KmiDi_CANON/brain/music_brain/tier1/midi_pipeline_wrapper.py`

```python
from mcp_workstation.debug import log_error, measure_performance

class MIDIGenerationPipeline:
    def generate_midi(self, intent: CompleteSongIntent) -> Dict[str, Any]:
        try:
            with measure_performance("harmony_generation"):
                processed = process_intent(intent)

            with measure_performance("midi_writing"):
                midi_path = self._write_midi(processed["harmony"], processed["groove"])

            return {
                "status": "completed",
                "midi_path": midi_path,
                # ... other fields
            }
        except Exception as e:
            log_error("MIDI generation failed", exception=e,
                     details={"key": intent.technical_constraints.technical_key})
            return {"status": "error", "details": str(e)}
```

---

## Monitoring & Dashboards

### Get System Health

```python
from mcp_workstation.debug import get_debug

def get_system_health() -> Dict[str, Any]:
    """Get overall system health for dashboard."""
    debug = get_debug()
    summary = debug.get_summary()

    return {
        "status": "healthy" if summary["error_count"] == 0 else "degraded",
        "total_events": summary["total_events"],
        "recent_errors": summary["recent_errors"],
        "performance": {
            "total_operations": summary["performance_summary"]["total_operations"],
            "operations": summary["performance_summary"]["operations"]
        }
    }
```

### Check for Critical Errors

```python
def has_critical_errors() -> bool:
    """Check if system has recent critical errors."""
    debug = get_debug()
    errors = debug.get_errors(limit=5)

    # Check for errors in last 5 minutes
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(minutes=5)

    recent_critical = [
        e for e in errors
        if datetime.fromisoformat(e.timestamp) > cutoff
        and "critical" in e.message.lower()
    ]

    return len(recent_critical) > 0
```

### Performance Bottleneck Detection

```python
def detect_slow_operations(threshold_ms: float = 100.0) -> List[str]:
    """Find operations slower than threshold."""
    debug = get_debug()
    report = debug.get_performance_report()

    slow_ops = []
    for op_name, stats in report["operations"].items():
        if stats["p95_ms"] > threshold_ms:
            slow_ops.append(f"{op_name}: p95={stats['p95_ms']:.2f}ms")

    return slow_ops
```

---

## Configuration

### Adjust Ring Buffer Size

```python
from mcp_workstation.debug import _Debug

# Create debug instance with custom sizes
debug = _Debug(max_events=5000, max_metrics=1000)  # Larger buffers

# Or modify existing instance (not recommended in production)
from mcp_workstation.debug import get_debug
debug = get_debug()
debug.events = deque(maxlen=5000)  # Resize event buffer
```

### Clear Old Data

```python
from mcp_workstation.debug import get_debug

debug = get_debug()
debug.clear()  # Clear all events and metrics
```

---

## Best Practices

### DO ✅

- **Use context managers for timing**: `with measure_performance("op"):`
- **Include exception objects**: `log_error("msg", exception=e)`
- **Add context details**: `details={"user_id": 123, "phase": "llm"}`
- **Check performance regularly**: Review p95/p99 latencies
- **Monitor error count**: Alert when error_count exceeds threshold

### DON'T ❌

- **Don't log sensitive data**: Avoid passwords, tokens in details
- **Don't log excessively**: Ring buffer will discard oldest events
- **Don't ignore errors**: Always check `get_errors()` periodically
- **Don't use print()**: Use debug system for production logging

---

## Testing

### Unit Test Example

```python
def test_workflow_logs_errors():
    """Test that workflow errors are logged."""
    from mcp_workstation.debug import get_debug

    debug = get_debug()
    debug.clear()

    # Execute workflow that will fail
    try:
        workstation.execute_workflow("invalid input")
    except Exception:
        pass

    # Verify error was logged
    errors = debug.get_errors()
    assert len(errors) > 0
    assert "Workflow execution failed" in errors[0].message
```

---

## API Reference

### Functions

- `get_debug() -> _Debug` - Get singleton debug instance
- `log_error(message, exception=None, details=None)` - Log error with stack trace
- `log_warning(message, details=None)` - Log warning
- `log_info(category, message, details=None)` - Log info with category
- `measure_performance(operation, context=None)` - Context manager for timing

### Debug Instance Methods

- `error(message, exception, details)` - Log error
- `warning(message, details)` - Log warning
- `info(category, message, details)` - Log info
- `get_errors(limit=25)` - Get recent errors
- `get_warnings(limit=25)` - Get recent warnings
- `get_events_by_category(category, limit=50)` - Filter by category
- `record_performance(operation, duration_ms, context)` - Record metric
- `measure(operation, context)` - Context manager for timing
- `get_performance_report()` - Get performance statistics
- `get_summary()` - Get overall system summary
- `clear()` - Clear all events and metrics

### Data Classes

- `DebugEvent` - Single event (message, category, timestamp, details, stack_trace)
- `PerformanceMetric` - Performance timing (operation, duration_ms, timestamp, context)
- `DebugCategory` - Event categories (AI_COMMUNICATION, PERFORMANCE, ERROR, WARNING, INFO)

---

## Example Output

### Performance Report
```json
{
  "total_operations": 142,
  "operations": {
    "llm_parse": {
      "count": 23,
      "mean_ms": 45.2,
      "min_ms": 12.3,
      "max_ms": 234.5,
      "p50_ms": 38.7,
      "p95_ms": 156.2,
      "p99_ms": 201.4
    },
    "midi_generation": {
      "count": 23,
      "mean_ms": 18.9,
      "min_ms": 8.2,
      "max_ms": 67.3,
      "p50_ms": 15.4,
      "p95_ms": 42.1,
      "p99_ms": 58.7
    }
  },
  "recent_metrics": [...]
}
```

### System Summary
```json
{
  "total_events": 156,
  "total_metrics": 142,
  "error_count": 3,
  "warning_count": 12,
  "recent_errors": [
    {
      "message": "LLM timeout after 30s",
      "timestamp": "2026-01-31T06:32:08.331050"
    }
  ],
  "performance_summary": {...}
}
```

---

## Troubleshooting

### Memory Usage

Ring buffer automatically discards oldest events when full. Default sizes:
- Events: 1,000 max
- Metrics: 500 max

If memory is a concern, reduce these in `_Debug.__init__()`.

### Performance Impact

Debug system is lightweight:
- Event logging: ~0.01ms overhead
- Performance measurement: ~0.001ms overhead
- Getting reports: O(n) where n = number of events/metrics

### Thread Safety

Current implementation is **not thread-safe**. For multi-threaded usage:
- Use separate debug instances per thread, OR
- Add threading.Lock around critical sections

---

*See [debug.py](../KmiDi_CANON/brain/mcp_workstation/debug.py) for full implementation.*
