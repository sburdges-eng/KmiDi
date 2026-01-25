# Spec 08: Output Verification

**Date:** January 18, 2026  
**Status:** Complete  
**Purpose:** Define format validation, quality checks, and export verification processes

## Overview

This specification defines how KmiDi verifies the correctness, quality, and format compliance of all outputs, including audio exports, MIDI exports, plugin state, and parameter values.

## Core Principles

1. **Format Compliance** - All exports must conform to industry-standard formats
2. **Quality Assurance** - Outputs must meet quality thresholds before being considered valid
3. **Verification Before Export** - Validate data before writing to disk
4. **Error Reporting** - Clear error messages when verification fails
5. **Reversibility** - Verified outputs can be re-imported successfully

## Format Validation

### 1. Audio Format Validation

All audio exports must be validated for format compliance.

#### Supported Audio Formats

- **WAV** (PCM, 16/24/32-bit, 44.1/48/88.2/96 kHz)
- **AIFF** (PCM, 16/24/32-bit, 44.1/48/88.2/96 kHz)
- **FLAC** (Lossless compression, 16/24-bit)
- **MP3** (Lossy compression, 128/192/256/320 kbps)
- **AAC** (Lossy compression, 128/192/256 kbps)

#### Validation Requirements

1. **Header Validation**
   - Verify file header matches format specification
   - Check magic numbers/identifiers
   - Validate metadata structure

2. **Sample Rate Validation**
   - Verify sample rate is within supported range
   - Check for valid sample rate values (no invalid rates)
   - Ensure consistency across channels

3. **Bit Depth Validation**
   - Verify bit depth matches format specification
   - Check for valid bit depth values
   - Ensure proper quantization

4. **Channel Configuration**
   - Verify channel count (mono, stereo, multi-channel)
   - Check channel layout matches specification
   - Validate interleaving for multi-channel

5. **Duration Validation**
   - Verify duration matches expected length
   - Check for truncated or corrupted data
   - Ensure file size matches content

#### Implementation

```typescript
interface AudioFormatValidation {
  format: 'WAV' | 'AIFF' | 'FLAC' | 'MP3' | 'AAC';
  sampleRate: number;
  bitDepth: number;
  channels: number;
  duration: number;
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

function validateAudioFormat(file: File): AudioFormatValidation {
  // Validate header, sample rate, bit depth, channels, duration
  // Return validation result with errors/warnings
}
```

### 2. MIDI Format Validation

All MIDI exports must be validated for format compliance.

#### MIDI Format Requirements

- **Standard MIDI File (SMF)** - Format 0, 1, or 2
- **MIDI 1.0 Specification** - Compliant with MIDI 1.0 spec
- **Event Validation** - All events must be valid MIDI events
- **Timing Validation** - Delta times must be valid
- **Track Structure** - Tracks must be properly structured

#### Validation Requirements

1. **File Header Validation**
   - Verify MThd chunk exists and is valid
   - Check format type (0, 1, or 2)
   - Validate track count
   - Verify time division (ticks per quarter note)

2. **Track Validation**
   - Verify MTrk chunks exist
   - Check track structure integrity
   - Validate event ordering

3. **Event Validation**
   - Verify all events are valid MIDI events
   - Check channel numbers (0-15)
   - Validate note numbers (0-127)
   - Verify velocity values (0-127)
   - Check control change values (0-127)

4. **Timing Validation**
   - Verify delta times are non-negative
   - Check for timing overflow
   - Ensure proper event sequencing

#### Implementation

```typescript
interface MIDIFormatValidation {
  format: 0 | 1 | 2;
  tracks: number;
  timeDivision: number;
  isValid: boolean;
  errors: string[];
  warnings: string[];
  eventCount: number;
  invalidEvents: MIDIEvent[];
}

function validateMIDIFormat(file: File): MIDIFormatValidation {
  // Validate header, tracks, events, timing
  // Return validation result with errors/warnings
}
```

### 3. Plugin State Validation

Plugin state exports must be validated for correctness and completeness.

#### Plugin State Requirements

- **Parameter Values** - All parameters must be within valid ranges
- **State Completeness** - All required state must be present
- **Format Compliance** - State format must match specification
- **Version Compatibility** - State must be compatible with plugin version

#### Validation Requirements

1. **Parameter Range Validation**
   - Verify all parameters are within min/max bounds
   - Check for NaN or Infinity values
   - Validate parameter types

2. **State Completeness**
   - Verify all required parameters are present
   - Check for missing critical state
   - Validate state structure

3. **Format Validation**
   - Verify state format matches specification
   - Check for corruption or invalid data
   - Validate serialization format

4. **Version Compatibility**
   - Verify state version matches plugin version
   - Check for backward/forward compatibility
   - Validate migration paths

#### Implementation

```typescript
interface PluginStateValidation {
  version: string;
  parameterCount: number;
  isValid: boolean;
  errors: string[];
  warnings: string[];
  invalidParameters: ParameterError[];
}

interface ParameterError {
  parameterId: string;
  value: number;
  expectedRange: { min: number; max: number };
  error: string;
}

function validatePluginState(state: PluginState): PluginStateValidation {
  // Validate parameters, completeness, format, version
  // Return validation result with errors/warnings
}
```

## Quality Checks

### 1. Audio Quality Checks

Audio outputs must meet quality thresholds.

#### Quality Metrics

1. **Signal Integrity**
   - No clipping (peak levels < 0 dBFS)
   - No DC offset
   - No silence where audio should exist
   - No corruption or artifacts

2. **Dynamic Range**
   - Verify dynamic range is appropriate
   - Check for excessive compression
   - Validate headroom

3. **Frequency Response**
   - Check for unwanted frequency content
   - Verify frequency response is flat (if expected)
   - Detect aliasing or artifacts

4. **Noise Floor**
   - Verify noise floor is acceptable
   - Check for unwanted noise or hum
   - Validate signal-to-noise ratio

#### Implementation

```typescript
interface AudioQualityCheck {
  peakLevel: number; // dBFS
  rmsLevel: number; // dBFS
  dynamicRange: number; // dB
  noiseFloor: number; // dBFS
  clippingDetected: boolean;
  dcOffset: number;
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

function checkAudioQuality(audio: AudioBuffer): AudioQualityCheck {
  // Analyze signal integrity, dynamic range, frequency response, noise
  // Return quality check result
}
```

### 2. MIDI Quality Checks

MIDI outputs must meet quality thresholds.

#### Quality Metrics

1. **Event Density**
   - Verify event density is reasonable
   - Check for excessive or missing events
   - Validate event distribution

2. **Timing Accuracy**
   - Verify timing is accurate
   - Check for timing jitter
   - Validate quantization

3. **Note Completeness**
   - Verify note-on has matching note-off
   - Check for hanging notes
   - Validate note durations

4. **Velocity Distribution**
   - Verify velocity values are appropriate
   - Check for velocity range
   - Validate velocity consistency

#### Implementation

```typescript
interface MIDIQualityCheck {
  eventDensity: number; // events per second
  timingAccuracy: number; // milliseconds
  hangingNotes: number;
  velocityRange: { min: number; max: number };
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

function checkMIDIQuality(midi: MIDIFile): MIDIQualityCheck {
  // Analyze event density, timing, notes, velocity
  // Return quality check result
}
```

### 3. Parameter Quality Checks

Parameter values must meet quality thresholds.

#### Quality Metrics

1. **Parameter Range Compliance**
   - All parameters within valid ranges
   - No invalid or out-of-range values
   - Proper parameter scaling

2. **Parameter Consistency**
   - Related parameters are consistent
   - No conflicting parameter values
   - Logical parameter relationships

3. **Parameter Completeness**
   - All required parameters are set
   - No missing critical parameters
   - Default values are appropriate

#### Implementation

```typescript
interface ParameterQualityCheck {
  outOfRangeParameters: ParameterError[];
  inconsistentParameters: ParameterConflict[];
  missingParameters: string[];
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

function checkParameterQuality(parameters: ParameterSet): ParameterQualityCheck {
  // Analyze range compliance, consistency, completeness
  // Return quality check result
}
```

## Export Verification

### 1. Pre-Export Validation

Before exporting, validate all data meets requirements.

#### Validation Steps

1. **Format Validation**
   - Verify export format is supported
   - Check format-specific requirements
   - Validate format compatibility

2. **Quality Checks**
   - Run audio/MIDI quality checks
   - Verify parameters meet quality thresholds
   - Check for warnings or errors

3. **Completeness Checks**
   - Verify all required data is present
   - Check for missing components
   - Validate data integrity

4. **Compatibility Checks**
   - Verify compatibility with target format
   - Check for format limitations
   - Validate export settings

#### Implementation

```typescript
interface ExportValidation {
  format: ExportFormat;
  isValid: boolean;
  canExport: boolean;
  errors: string[];
  warnings: string[];
  qualityChecks: QualityCheckResult[];
}

async function validateBeforeExport(
  data: ExportData,
  format: ExportFormat
): Promise<ExportValidation> {
  // Run format validation, quality checks, completeness, compatibility
  // Return validation result
}
```

### 2. Post-Export Verification

After exporting, verify the exported file is correct.

#### Verification Steps

1. **File Existence**
   - Verify file was created
   - Check file size is non-zero
   - Validate file permissions

2. **Format Verification**
   - Re-read and validate file format
   - Verify file structure is correct
   - Check for corruption

3. **Content Verification**
   - Verify exported content matches source
   - Check for data loss or corruption
   - Validate content integrity

4. **Re-import Test** (Optional)
   - Attempt to re-import exported file
   - Verify re-import succeeds
   - Check for data fidelity

#### Implementation

```typescript
interface PostExportVerification {
  fileExists: boolean;
  fileSize: number;
  formatValid: boolean;
  contentValid: boolean;
  reimportSuccessful: boolean;
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

async function verifyAfterExport(
  filePath: string,
  expectedContent: ExportData
): Promise<PostExportVerification> {
  // Verify file existence, format, content, re-import
  // Return verification result
}
```

### 3. Export Error Handling

When export verification fails, provide clear error messages and recovery options.

#### Error Types

1. **Format Errors**
   - Invalid format specification
   - Unsupported format
   - Format compatibility issues

2. **Quality Errors**
   - Quality thresholds not met
   - Signal integrity issues
   - Parameter validation failures

3. **File System Errors**
   - Permission denied
   - Disk full
   - Path invalid

4. **Data Errors**
   - Missing required data
   - Corrupted data
   - Invalid data values

#### Error Reporting

```typescript
interface ExportError {
  type: 'format' | 'quality' | 'filesystem' | 'data';
  severity: 'error' | 'warning';
  message: string;
  details?: string;
  recovery?: string; // Suggested recovery action
}

function handleExportError(error: ExportError): void {
  // Display error to user
  // Provide recovery options
  // Log error for debugging
}
```

## Testing Requirements

### Test 1: Format Validation

```typescript
test('Audio format validation rejects invalid formats', () => {
  const invalidFile = createInvalidAudioFile();
  const validation = validateAudioFormat(invalidFile);
  
  expect(validation.isValid).toBe(false);
  expect(validation.errors.length).toBeGreaterThan(0);
});
```

### Test 2: Quality Checks

```typescript
test('Audio quality check detects clipping', () => {
  const clippedAudio = createClippedAudio();
  const quality = checkAudioQuality(clippedAudio);
  
  expect(quality.clippingDetected).toBe(true);
  expect(quality.isValid).toBe(false);
});
```

### Test 3: Export Verification

```typescript
test('Export verification validates before export', async () => {
  const data = createExportData();
  const validation = await validateBeforeExport(data, 'WAV');
  
  expect(validation.canExport).toBe(true);
  expect(validation.errors.length).toBe(0);
});
```

### Test 4: Post-Export Verification

```typescript
test('Post-export verification checks file integrity', async () => {
  const filePath = await exportAudio(data, 'WAV');
  const verification = await verifyAfterExport(filePath, data);
  
  expect(verification.fileExists).toBe(true);
  expect(verification.formatValid).toBe(true);
  expect(verification.contentValid).toBe(true);
});
```

## Compliance Checklist

- [x] Audio format validation implemented
- [x] MIDI format validation implemented
- [x] Plugin state validation implemented
- [x] Audio quality checks implemented
- [x] MIDI quality checks implemented
- [x] Parameter quality checks implemented
- [x] Pre-export validation implemented
- [x] Post-export verification implemented
- [x] Error handling and reporting implemented
- [x] Re-import testing supported
- [x] Clear error messages provided
- [x] Recovery options available

## Related Specifications

- **Spec 07: Plugin Specific** - Defines plugin formats and state management
- **HOST_GLUE_ARCHITECTURE.md** - Defines host integration and format translation
- **INTENT_IR_SPEC.md** - Defines parameter structures

## References

- `docs/HOST_GLUE_ARCHITECTURE.md` - Host integration and format translation
- `docs/INTENT_IR_SPEC.md` - Parameter and state structures
- `docs/STRUCTURE_CROSS_EXAMINATION/07_SPEC_TO_IMPLEMENTATION_MATRIX.md` - Compliance matrix
