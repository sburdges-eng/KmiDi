# DAiW Bridge Plugin Tests

## Overview

This directory contains comprehensive C++ unit tests for the DAiW Bridge JUCE plugin.

## Test Files

- **`PluginProcessorTest.cpp`** - Tests for audio processor functionality
- **`PluginEditorTest.cpp`** - Tests for UI editor components
- **`OSCCommunicationTest.cpp`** - Tests for OSC message handling and JSON parsing

## Building Tests

### Option 1: Using CMake

```bash
cd cpp/DAiWBridge
mkdir build
cd build
cmake ..
make
./DAiWBridgeTests
```

### Option 2: Using JUCE Projucer

1. Open `DAiWBridge.jucer` in Projucer
2. Add test files to the project
3. Generate Xcode/Visual Studio project
4. Build and run tests

### Option 3: Using Catch2 (if available)

```bash
# Install Catch2
# Then build with Catch2 support
cmake -DUSE_CATCH2=ON ..
make
./DAiWBridgeTests
```

## Test Categories

### Basic Functionality

- Plugin initialization
- Connection status
- Prepare to play
- Process block

### OSC Communication

- Message format validation
- Send generate request
- Send ping
- Receive responses

### JSON Parsing

- Valid JSON parsing
- Invalid JSON handling
- Missing properties
- Boundary value validation

### MIDI Events

- Event creation
- Timing calculations
- Value clamping
- Multiple events

### Error Handling

- Error message handling
- Unknown OSC messages
- Invalid data

### Performance

- Process block performance
- Memory management

## Running Tests

### All Tests

```bash
./DAiWBridgeTests
```

### Specific Test Category

```bash
./DAiWBridgeTests "[processor]"  # Processor tests only
./DAiWBridgeTests "[osc]"        # OSC tests only
./DAiWBridgeTests "[json]"       # JSON parsing tests
```

### With Verbose Output

```bash
./DAiWBridgeTests -s  # Show success messages
./DAiWBridgeTests -v  # Verbose output
```

## Test Coverage

### PluginProcessor

- ✅ Initialization
- ✅ Connection management
- ✅ Audio processing
- ✅ OSC message sending
- ✅ JSON parsing
- ✅ MIDI event scheduling
- ✅ Error handling
- ✅ State management

### PluginEditor

- ✅ UI initialization
- ✅ Resize handling
- ✅ Paint operations
- ✅ Component lifecycle

### OSC Communication

- ✅ Message format validation
- ✅ Response parsing
- ✅ Error handling
- ✅ Timing calculations

## Integration with Python Brain Server

These tests focus on the C++ plugin side. For end-to-end testing with the Python brain server, see:

- `examples/test_osc_client.py` - Python test client
- `tests/test_osc_server.py` - Python server tests

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Build and test
  run: |
    cd cpp/DAiWBridge
    mkdir build && cd build
    cmake ..
    make
    ./DAiWBridgeTests
```

## Debugging Failed Tests

1. **Check JUCE modules**: Ensure all required JUCE modules are linked
2. **Verify OSC setup**: Tests don't require actual OSC connection, but verify OSC modules are available
3. **Check JSON parsing**: Verify `juce::JSON` is working correctly
4. **Memory issues**: Use AddressSanitizer or Valgrind for memory debugging

## Adding New Tests

1. Add test case to appropriate test file
2. Use `CATCH_TEST_CASE` macro (or JUCE UnitTest if Catch2 unavailable)
3. Follow existing test patterns
4. Test both success and failure cases
5. Include boundary value testing

## See Also

- `docs/JUCE_PLUGIN_GUIDE.md` - Plugin building guide
- `docs/OSC_SERVER_GUIDE.md` - OSC protocol documentation
- `docs/PHASE_3_REFINEMENTS.md` - Implementation details
