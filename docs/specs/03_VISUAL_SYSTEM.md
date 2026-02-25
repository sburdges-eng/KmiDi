# 03. Visual System Specs

## Overview

These prevent "designer drift." Consistent visual rules ensure the UI feels cohesive and professional.

## Color Token & Theme Spec

### Dark-First Design Philosophy

**✓ REQUIRED:**
- All colors defined for dark backgrounds
- High contrast ratios (WCAG AA minimum)
- Semantic color tokens, not hardcoded values
- Emotion-appropriate color usage

**Implementation:**
```cpp
// KellyLookAndFeel.h - Dark-first theme
class KellyLookAndFeel : public juce::LookAndFeel_V4
{
public:
    KellyLookAndFeel() {
        // Dark background
        setColour (juce::DocumentWindow::backgroundColourId,
                  juce::Colour::fromRGB (24, 24, 26));

        // Semantic tokens
        setColour (backgroundPrimary,   juce::Colour::fromRGB (32, 32, 36));
        setColour (backgroundSecondary, juce::Colour::fromRGB (42, 42, 46));
        setColour (textPrimary,         juce::Colour::fromRGB (224, 224, 228));
        setColour (textSecondary,       juce::Colour::fromRGB (160, 160, 164));
        setColour (accentPrimary,       juce::Colour::fromRGB (0, 122, 255));
        setColour (errorColor,          juce::Colour::fromRGB (255, 59, 48));
    }

private:
    enum ColourIds {
        backgroundPrimary = 0x2000000,
        backgroundSecondary,
        textPrimary,
        textSecondary,
        accentPrimary,
        errorColor
    };
};
```

### Semantic Color Tokens

**Background Colors:**
- `bg.primary` - Main content areas (32, 32, 36)
- `bg.secondary` - Panels, cards (42, 42, 46)
- `bg.tertiary` - Inactive areas (52, 52, 56)

**Text Colors:**
- `text.primary` - Main labels, high importance (224, 224, 228)
- `text.secondary` - Secondary text, captions (160, 160, 164)
- `text.tertiary` - Disabled text, hints (112, 112, 116)

**Accent Colors:**
- `accent.primary` - Active controls, links (0, 122, 255)
- `accent.secondary` - Hover states (88, 166, 255)
- `accent.success` - Confirmation (52, 199, 89)
- `accent.warning` - Caution (255, 149, 0)
- `accent.error` - Errors (255, 59, 48)

**Functional Colors:**
- `border.light` - Subtle dividers (64, 64, 68)
- `border.medium` - Panel borders (84, 84, 88)
- `shadow` - Depth effects (0, 0, 0, 0.2)

### Emotion Color Restrictions

**✓ ALLOWED:**
- Emotion bars: Muted, desaturated colors
- Status indicators: Semantic colors (red=error, green=success)
- ML confidence: Blue gradient (low to high confidence)

**❌ FORBIDDEN:**
- Emotion-driven color schemes
- Color changes based on mood
- Emotion as primary visual language

**Implementation:**
```cpp
// Emotion colors - always muted
const juce::Colour emotionValenceColor   = juce::Colour::fromRGB (120, 120, 140); // Muted purple
const juce::Colour emotionArousalColor   = juce::Colour::fromRGB (140, 120, 120); // Muted red
const juce::Colour emotionDominanceColor = juce::Colour::fromRGB (120, 140, 120); // Muted green
const juce::Colour emotionComplexityColor = juce::Colour::fromRGB (140, 140, 120); // Muted yellow

// ML confidence - blue gradient
juce::Colour getMLConfidenceColor (float confidence) {
    return juce::Colour::fromHSV (0.6f, 0.3f, 0.4f + confidence * 0.4f, 1.0f);
    // Hue: blue, low saturation, brightness based on confidence
}
```

## Typography & Spacing Spec

### Font Usage Rules

**✓ ALLOWED:**
- System fonts only (San Francisco on macOS, system default elsewhere)
- Monospace for code/values
- Bold for emphasis, regular for body text

**❌ FORBIDDEN:**
- Custom fonts
- Font loading
- Font size < 11pt

**Implementation:**
```cpp
// Typography.h - System font only
class Typography
{
public:
    static juce::Font getPrimaryFont (float size = 13.0f) {
        return juce::Font (juce::Font::getDefaultSansSerifFontName(), size, juce::Font::plain);
    }

    static juce::Font getMonospaceFont (float size = 12.0f) {
        return juce::Font (juce::Font::getDefaultMonospacedFontName(), size, juce::Font::plain);
    }

    static juce::Font getBoldFont (float size = 13.0f) {
        return juce::Font (juce::Font::getDefaultSansSerifFontName(), size, juce::Font::bold);
    }
};
```

### Baseline Grid & Spacing

**✓ REQUIRED:**
- 4pt baseline grid
- Spacing units: 4pt, 8pt, 12pt, 16pt, 24pt, 32pt, 48pt
- Consistent vertical rhythm
- Component padding: 8pt minimum

**❌ FORBIDDEN:**
- Arbitrary spacing values
- Inconsistent padding
- Cramped layouts (< 8pt padding)

**Implementation:**
```cpp
// Spacing.h - Consistent spacing system
class Spacing
{
public:
    static constexpr int unit = 4;

    static int small()  { return 1 * unit; } // 4pt
    static int medium() { return 2 * unit; } // 8pt
    static int large()  { return 3 * unit; } // 12pt
    static int xl()     { return 4 * unit; } // 16pt
    static int xxl()    { return 6 * unit; } // 24pt
    static int xxxl()   { return 8 * unit; } // 32pt
    static int huge()   { return 12 * unit; } // 48pt
};

// Component with proper spacing
class SpacedComponent : public juce::Component
{
public:
    SpacedComponent() {
        setSize (200, 100);

        label.setBounds (Spacing::large(), Spacing::large(),
                        200 - 2 * Spacing::large(), 20);

        button.setBounds (Spacing::large(), 40,
                         200 - 2 * Spacing::large(), 24);
    }

private:
    juce::Label label;
    juce::TextButton button;
};
```

### Alignment Rules

**✓ REQUIRED:**
- Left-align text in Western locales
- Center-align numeric values
- Consistent component alignment
- Grid-based layout

**❌ FORBIDDEN:**
- Right-aligned body text
- Centered paragraphs
- Arbitrary positioning

**Implementation:**
```cpp
// Layout with proper alignment
void layoutComponents() {
    // Labels left-aligned
    titleLabel.setBounds (Spacing::large(), Spacing::large(),
                         getWidth() - 2 * Spacing::large(), 24);
    titleLabel.setJustificationType (juce::Justification::left);

    // Values center-aligned
    valueLabel.setBounds (Spacing::large(), 40,
                         getWidth() - 2 * Spacing::large(), 24);
    valueLabel.setJustificationType (juce::Justification::centred);

    // Buttons consistent spacing
    for (int i = 0; i < buttons.size(); ++i) {
        buttons[i]->setBounds (Spacing::large(), 70 + i * 32,
                              getWidth() - 2 * Spacing::large(), 24);
    }
}
```

### Illegal Cramming Prevention

**❌ FORBIDDEN:**
- Padding < 8pt
- Text size < 11pt
- Touch targets < 44pt
- Line height < 1.2x font size

**Implementation:**
```cpp
// Validation functions
bool validateSpacing (const juce::Component& component) {
    // Minimum padding check
    if (component.getBounds().getX() < Spacing::large() ||
        component.getBounds().getY() < Spacing::large()) {
        return false; // Too cramped
    }

    return true;
}

bool validateTypography (const juce::Label& label) {
    if (label.getFont().getHeight() < 11.0f) {
        return false; // Too small
    }

    if (label.getFont().getHeight() * 1.2f > label.getBounds().getHeight()) {
        return false; // Line height too tight
    }

    return true;
}
```

## Control Styling Spec (Hybrid Flat)

### Button Styling

**✓ REQUIRED:**
- Flat design with subtle depth
- Clear hover/active states
- Consistent sizing
- Accessible contrast

**States:**
- Default: Flat, subtle border
- Hover: Slight background change
- Active: Clear pressed state
- Disabled: Reduced opacity, no interaction

**Implementation:**
```cpp
// Button styling
void drawButtonBackground (juce::Graphics& g, juce::Button& button,
                          const juce::Colour& backgroundColour,
                          bool shouldDrawButtonAsHighlighted,
                          bool shouldDrawButtonAsDown)
{
    auto bounds = button.getLocalBounds().toFloat();

    // Base state - flat with subtle border
    g.setColour (backgroundColour);
    g.fillRoundedRectangle (bounds, 4.0f);

    g.setColour (juce::Colours::white.withAlpha (0.1f));
    g.drawRoundedRectangle (bounds, 4.0f, 1.0f);

    // Hover state
    if (shouldDrawButtonAsHighlighted) {
        g.setColour (juce::Colours::white.withAlpha (0.05f));
        g.fillRoundedRectangle (bounds, 4.0f);
    }

    // Active state
    if (shouldDrawButtonAsDown) {
        g.setColour (juce::Colours::black.withAlpha (0.1f));
        g.fillRoundedRectangle (bounds, 4.0f);
    }
}
```

### Slider/Knob Styling

**✓ REQUIRED:**
- Circular knobs for rotary controls
- Linear sliders for faders
- Clear value indication
- Smooth interaction

**❌ FORBIDDEN:**
- 3D bevel effects
- Excessive gradients
- Unclear value display

**Implementation:**
```cpp
// Knob styling - hybrid flat
void drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                      float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                      juce::Slider& slider)
{
    // Outer ring - subtle depth
    g.setColour (juce::Colours::white.withAlpha (0.1f));
    g.drawEllipse (x, y, width, height, 2.0f);

    // Value arc - clear indication
    juce::Path valueArc;
    valueArc.addArc (x + 4, y + 4, width - 8, height - 8,
                    rotaryStartAngle, rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle),
                    true);
    g.setColour (accentPrimary);
    g.strokePath (valueArc, juce::PathStrokeType (3.0f));

    // Center dot - subtle
    float centerX = x + width * 0.5f;
    float centerY = y + height * 0.5f;
    g.setColour (juce::Colours::white.withAlpha (0.8f));
    g.fillEllipse (centerX - 2, centerY - 2, 4, 4);
}
```

### Control Sizing

**✓ REQUIRED:**
- Buttons: Minimum 44pt height
- Knobs: 44pt diameter minimum
- Sliders: 44pt touch target
- Text fields: 24pt minimum height

**❌ FORBIDDEN:**
- Controls < 44pt in any dimension
- Inaccessible touch targets
- Cramped interfaces

**Implementation:**
```cpp
// Control sizing validation
bool validateControlSize (const juce::Component& control) {
    auto bounds = control.getBounds();

    // Minimum touch target
    if (bounds.getWidth() < 44 || bounds.getHeight() < 44) {
        return false;
    }

    return true;
}

// Proper control creation
class ControlFactory
{
public:
    static juce::TextButton* createButton (const juce::String& text) {
        auto* button = new juce::TextButton (text);
        button->setSize (120, 44); // Minimum size
        return button;
    }

    static juce::Slider* createKnob (const juce::String& name) {
        auto* slider = new juce::Slider (juce::Slider::Rotary,
                                       juce::Slider::TextBoxBelow,
                                       juce::Slider::NoTextBox);
        slider->setSize (44, 44); // Square knob
        slider->setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 20);
        return slider;
    }
};
```

### Depth and Shadow Rules

**✓ ALLOWED:**
- Subtle shadows for floating elements
- Layer separation in modals
- Minimal depth cues

**❌ FORBIDDEN:**
- Excessive drop shadows
- 3D bevel effects
- Heavy depth simulation

**Implementation:**
```cpp
// Subtle shadow for depth
void drawShadow (juce::Graphics& g, const juce::Rectangle<float>& bounds) {
    juce::DropShadow shadow (juce::Colours::black.withAlpha (0.2f), 4,
                           juce::Point<int> (0, 2));
    shadow.drawForRectangle (g, bounds);
}

// Modal with subtle depth
void drawModalBackground (juce::Graphics& g, const juce::Rectangle<float>& bounds) {
    // Backdrop blur effect
    g.setColour (juce::Colours::black.withAlpha (0.5f));
    g.fillRect (bounds);

    // Modal with subtle shadow
    auto modalBounds = bounds.reduced (40);
    drawShadow (g, modalBounds);

    g.setColour (backgroundPrimary);
    g.fillRoundedRectangle (modalBounds, 8.0f);

    g.setColour (borderMedium);
    g.drawRoundedRectangle (modalBounds, 8.0f, 1.0f);
}
```

## Audit Checklist

### Color Token Compliance
- [ ] All colors defined as semantic tokens
- [ ] Dark-first design (high contrast)
- [ ] Emotion colors always muted/desaturated
- [ ] WCAG AA compliance for text contrast
- [ ] No hardcoded color values in components

### Typography Compliance
- [ ] System fonts only (no custom fonts)
- [ ] Minimum 11pt font size
- [ ] Consistent font weights (regular, bold)
- [ ] Proper line heights (1.2x minimum)
- [ ] Monospace for code/numeric values

### Spacing Compliance
- [ ] 4pt baseline grid throughout
- [ ] Standard spacing units (4pt, 8pt, 12pt, 16pt, 24pt, 32pt, 48pt)
- [ ] Minimum 8pt padding on all components
- [ ] Consistent vertical rhythm
- [ ] No arbitrary spacing values

### Control Styling Compliance
- [ ] Flat design with subtle depth
- [ ] Clear hover/active/disabled states
- [ ] Minimum 44pt touch targets
- [ ] Accessible contrast ratios
- [ ] Consistent button/slider styling
- [ ] Subtle shadows only for depth

## Code Examples

### ✅ CORRECT: Semantic Color Usage
```cpp
// Color usage through tokens
class ThemedComponent : public juce::Component
{
    void paint (juce::Graphics& g) override {
        auto& lf = getLookAndFeel();

        // Use semantic tokens, not hardcoded colors
        g.setColour (lf.findColour (KellyLookAndFeel::backgroundPrimary));
        g.fillRect (getLocalBounds());

        g.setColour (lf.findColour (KellyLookAndFeel::textPrimary));
        g.drawText ("Label", getLocalBounds(), juce::Justification::left);
    }
};
```

### ❌ WRONG: Hardcoded Colors
```cpp
// WRONG - hardcoded colors
void paint (juce::Graphics& g) override {
    g.setColour (juce::Colours::darkgrey); // WRONG: hardcoded
    g.fillRect (getLocalBounds());

    g.setColour (juce::Colours::white); // WRONG: hardcoded
    g.drawText ("Label", getLocalBounds(), juce::Justification::left);
}
```

### ✅ CORRECT: Spacing System
```cpp
// Component with proper spacing
class WellSpacedComponent : public juce::Component
{
public:
    WellSpacedComponent() {
        addAndMakeVisible (titleLabel);
        addAndMakeVisible (valueField);
        addAndMakeVisible (actionButton);

        setSize (300, 120); // Appropriate size
    }

    void resized() override {
        auto bounds = getLocalBounds().reduced (Spacing::large()); // 16pt padding

        // Title - top
        titleLabel.setBounds (bounds.removeFromTop (24));

        // Spacing between elements
        bounds.removeFromTop (Spacing::medium()); // 8pt gap

        // Value field
        valueField.setBounds (bounds.removeFromTop (32));

        // Spacing before button
        bounds.removeFromTop (Spacing::large()); // 16pt gap

        // Button - bottom
        actionButton.setBounds (bounds.removeFromTop (44));
    }

private:
    juce::Label titleLabel;
    juce::TextEditor valueField;
    juce::TextButton actionButton;
};
```

### ❌ WRONG: Arbitrary Spacing
```cpp
// WRONG - arbitrary values
void resized() override {
    titleLabel.setBounds (10, 5, getWidth() - 20, 20);     // WRONG: 10, 5, 20
    valueField.setBounds (10, 30, getWidth() - 20, 25);   // WRONG: 30, 25
    actionButton.setBounds (10, 60, getWidth() - 20, 30); // WRONG: 60, 30
}
```

## Non-Compliance Fixes

### If Hardcoded Colors Found:
1. Define semantic tokens in LookAndFeel
2. Replace hardcoded colors with token lookups
3. Ensure dark-first design
4. Test contrast ratios

### If Poor Typography Found:
1. Replace with system font calls
2. Ensure minimum sizes (11pt)
3. Add proper line heights
4. Use monospace for numeric values

### If Inconsistent Spacing Found:
1. Replace arbitrary values with Spacing:: functions
2. Ensure 4pt grid alignment
3. Add minimum 8pt padding
4. Test touch target sizes

### If Poor Control Styling Found:
1. Implement hybrid flat design
2. Add proper state transitions
3. Ensure 44pt minimum sizes
4. Add accessibility labels