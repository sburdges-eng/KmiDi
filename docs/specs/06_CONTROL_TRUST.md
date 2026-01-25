# Spec 06: Control Trust

**Date:** January 18, 2026  
**Status:** Complete  
**Purpose:** Define AI trust mechanisms, override controls, and transparency requirements

## Overview

This specification defines how users maintain control and trust when interacting with AI/ML features in KmiDi. The system must provide clear mechanisms for users to understand, override, and control AI-generated suggestions and automation.

## Core Principles

1. **User Always Has Final Control** - AI can suggest, but never auto-apply without explicit user action
2. **Transparency First** - Users must understand what AI is doing and why
3. **Reversible Actions** - All AI suggestions can be undone or overridden
4. **Trust Through Visibility** - AI provenance and confidence must be visible

## Trust Mechanisms

### 1. AI Suggestion Visibility

All AI-generated suggestions must be clearly marked and distinguishable from user actions.

#### Visual Indicators

- **Provenance Badges**: Every AI suggestion displays its source:
  - `ML_TEXT` - Generated from text analysis
  - `ML_AUDIO` - Generated from audio analysis
  - `PRESET` - From preset library
  - `AUTOMATION` - From automation rules
  - `UI_DIRECT` - Direct user input (highest trust)
  - `UI_EDIT` - User-modified AI suggestion

- **Confidence Indicators**: Display confidence scores for AI predictions
  - High confidence (≥0.8): Green indicator
  - Medium confidence (0.5-0.8): Yellow indicator
  - Low confidence (<0.5): Red indicator with warning

#### Implementation

```typescript
interface AISuggestion {
  source: 'ML_TEXT' | 'ML_AUDIO' | 'PRESET' | 'AUTOMATION';
  confidence: number; // 0.0 to 1.0
  parameters: ParameterSet;
  explanation?: string; // Optional human-readable explanation
}
```

### 2. Override Controls

Users must be able to override any AI suggestion at any time.

#### Override Mechanisms

1. **Direct Parameter Control**
   - Users can manually adjust any parameter, overriding AI suggestions
   - Manual adjustments take precedence over AI suggestions
   - Override state is preserved until user changes it

2. **AI Suggestion Rejection**
   - Users can reject specific AI suggestions
   - Rejected suggestions are not re-applied automatically
   - Rejection history informs future suggestions

3. **AI Disable Toggle**
   - Global toggle to disable all AI suggestions
   - When disabled, only user input affects parameters
   - AI analysis continues in background but doesn't affect output

#### Implementation

```typescript
interface OverrideControl {
  parameterId: string;
  userValue: number;
  aiSuggestion?: AISuggestion;
  overrideActive: boolean;
  overrideTimestamp: number;
}

interface AISettings {
  enabled: boolean;
  autoApply: boolean; // Must be false per Spec 05
  showConfidence: boolean;
  showProvenance: boolean;
}
```

### 3. Transparency Requirements

Users must understand what AI is doing and why.

#### Required Information

1. **Source Attribution**
   - Every AI-generated value shows its source
   - Displayed in UI using provenance badges (see Visual Indicators)

2. **Confidence Scores**
   - AI predictions display confidence levels
   - Low confidence predictions show warnings
   - Users can see why confidence is low

3. **Explanation Text** (Optional but Recommended)
   - AI can provide human-readable explanations
   - Example: "Increased reverb based on 'atmospheric' emotion from text analysis"
   - Helps users understand AI reasoning

4. **History/Audit Trail**
   - Users can see what AI has suggested over time
   - Shows when suggestions were applied or rejected
   - Helps build trust through transparency

#### Implementation

```typescript
interface TransparencyInfo {
  source: IntentSource;
  confidence: number;
  explanation?: string;
  timestamp: number;
  applied: boolean;
  userModified: boolean;
}

interface AuditTrail {
  entries: TransparencyInfo[];
  getRecentEntries(count: number): TransparencyInfo[];
  getEntriesBySource(source: IntentSource): TransparencyInfo[];
}
```

## User Control Levels

### Level 1: Full AI Control (Not Allowed)

**Status:** ❌ Forbidden per Spec 05

AI cannot automatically apply changes without user approval. This level is explicitly forbidden.

### Level 2: AI Suggestions with Auto-Apply (Not Allowed)

**Status:** ❌ Forbidden per Spec 05

AI cannot automatically apply suggestions. All suggestions require explicit user action.

### Level 3: AI Suggestions with Manual Apply (Required)

**Status:** ✅ Required

- AI generates suggestions
- Suggestions are clearly marked
- User must explicitly accept each suggestion
- User can modify suggestions before applying
- User can reject suggestions

### Level 4: AI Analysis Only (Optional)

**Status:** ✅ Optional

- AI analyzes and provides information
- No parameter changes are suggested
- User sees analysis results
- User makes all decisions manually

## Override Priority

When multiple sources suggest values for the same parameter, priority order is:

1. **User Direct Input** (`UI_DIRECT`) - Highest priority
2. **User-Modified AI** (`UI_EDIT`) - User has reviewed and modified
3. **AI Suggestions** (`ML_TEXT`, `ML_AUDIO`) - Can be overridden
4. **Presets** (`PRESET`) - Can be overridden
5. **Automation** (`AUTOMATION`) - Can be overridden

## Trust Building Features

### 1. Confidence Visualization

- Visual indicators show AI confidence levels
- Low confidence suggestions are highlighted
- Users can see why confidence is low (e.g., "insufficient training data")

### 2. Learning from Rejections

- System tracks which suggestions users reject
- Rejected patterns inform future suggestions
- Helps AI learn user preferences

### 3. Comparison Mode

- Users can compare AI suggestion vs. current value
- Side-by-side preview of changes
- Helps users make informed decisions

### 4. Undo/Redo Support

- All AI-applied changes support undo
- Users can revert to previous state
- Builds trust through reversibility

## Implementation Requirements

### UI Components

1. **Provenance Badge Component**
   - Displays source of each parameter value
   - Color-coded by source type
   - Clickable for more information

2. **Confidence Indicator**
   - Visual indicator (color/shape) for confidence level
   - Tooltip shows exact confidence score
   - Warning for low confidence

3. **Override Controls**
   - Manual parameter sliders/controls
   - "Revert to AI" button (when overridden)
   - "Apply AI Suggestion" button (when available)

4. **Transparency Panel**
   - Shows current AI suggestions
   - Displays confidence scores
   - Shows explanation text
   - Provides audit trail

### Data Structures

```typescript
// IntentFrame with provenance (from INTENT_IR_SPEC.md)
interface IntentFrame {
  // ... existing fields ...
  provenance: {
    source: IntentSource;
    user_override_weight: number; // 0.0 = AI, 1.0 = user override
    confidence: number;
    explanation?: string;
  };
}

// Override state
interface ParameterOverride {
  parameterId: string;
  originalValue: number;
  aiSuggestion?: number;
  userValue: number;
  isOverridden: boolean;
  overrideTimestamp: number;
}
```

## Testing Requirements

### Test 1: Override Functionality

```typescript
test('User can override AI suggestion', () => {
  const aiSuggestion = { parameter: 'cutoff', value: 1200, source: 'ML_TEXT' };
  const userOverride = { parameter: 'cutoff', value: 800 };
  
  applyAISuggestion(aiSuggestion);
  applyUserOverride(userOverride);
  
  expect(getParameterValue('cutoff')).toBe(800);
  expect(getParameterProvenance('cutoff').source).toBe('UI_DIRECT');
});
```

### Test 2: Transparency Display

```typescript
test('AI suggestions show provenance and confidence', () => {
  const suggestion = createAISuggestion({
    source: 'ML_AUDIO',
    confidence: 0.75,
    explanation: 'Increased reverb based on audio analysis'
  });
  
  const display = renderSuggestion(suggestion);
  
  expect(display.provenance).toBe('ML_AUDIO');
  expect(display.confidence).toBe(0.75);
  expect(display.explanation).toBeDefined();
});
```

### Test 3: Rejection Tracking

```typescript
test('Rejected suggestions are tracked', () => {
  const suggestion = createAISuggestion({ source: 'ML_TEXT' });
  rejectSuggestion(suggestion);
  
  const rejectionHistory = getRejectionHistory();
  expect(rejectionHistory).toContain(suggestion);
  expect(getFutureSuggestions()).not.toContain(similarSuggestion);
});
```

## Compliance Checklist

- [x] AI suggestions are clearly marked with provenance
- [x] Confidence scores are displayed for AI predictions
- [x] Users can override any AI suggestion
- [x] Override state is preserved
- [x] AI cannot auto-apply changes (per Spec 05)
- [x] All AI actions are reversible (undo support)
- [x] Transparency information is available
- [x] Audit trail tracks AI suggestions and user actions
- [x] Low confidence suggestions show warnings
- [x] User preferences inform future suggestions

## Related Specifications

- **Spec 05: AI/ML Visibility** - Defines AI visibility requirements
- **AI_CONTROL_LAYER.md** - Defines AI architecture and boundaries
- **INTENT_IR_SPEC.md** - Defines IntentFrame structure with provenance

## References

- `docs/AI_CONTROL_LAYER.md` - AI architecture and boundaries
- `docs/INTENT_IR_SPEC.md` - IntentFrame specification
- `docs/STRUCTURE_CROSS_EXAMINATION/07_SPEC_TO_IMPLEMENTATION_MATRIX.md` - Compliance matrix
