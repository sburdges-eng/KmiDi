// JUCEHostView.mm
// AppKit-side host that embeds a JUCE Component inside an NSView.
// AppKit owns windowing, menus, focus, and lifecycle. JUCE only draws/handles local input.

#import <Cocoa/Cocoa.h>

// JUCE headers (adjust include paths to your JUCE checkout)
#ifdef __cplusplus
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_events/juce_events.h>
#include "TimelineComponent.h"
#endif

// Forward declare the JUCE component we will host.
#ifdef __cplusplus
class TimelineComponent;
#endif

@interface JUCEHostView : NSView
@property (nonatomic, readonly) TimelineComponent *timelineComponent;
@end

@implementation JUCEHostView {
#ifdef __cplusplus
    std::unique_ptr<TimelineComponent> _timeline;
#endif
}

#ifdef __cplusplus
static juce::ModifierKeys convertModifiers(NSEventModifierFlags flags) {
    juce::ModifierKeys mods;
    if (flags & NSEventModifierFlagShift)   mods = mods.withFlags(juce::ModifierKeys::shiftModifier);
    if (flags & NSEventModifierFlagControl) mods = mods.withFlags(juce::ModifierKeys::ctrlModifier);
    if (flags & NSEventModifierFlagOption)  mods = mods.withFlags(juce::ModifierKeys::altModifier);
    if (flags & NSEventModifierFlagCommand) mods = mods.withFlags(juce::ModifierKeys::commandModifier);
    return mods;
}
#endif

- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        self.wantsLayer = YES; // Allow CA-backed for smoother repaints

#ifdef __cplusplus
        // Create JUCE TimelineComponent - owned by this NSView via unique_ptr
        // Lifecycle: Created here, destroyed automatically when NSView is deallocated
        // This is the insertion point for the JUCE timeline rendering
        _timeline = std::make_unique<TimelineComponent>();
        _timeline->setOpaque(true);
        _timeline->setPaintingIsUnclipped(false);
        _timeline->setBufferedToImage(false); // Avoid unnecessary buffering
        _timeline->setVisible(true);
        _timeline->setBounds(juce::Rectangle<int>(0, 0, (int)frameRect.size.width, (int)frameRect.size.height));
        _timeline->setInterceptsMouseClicks(true, true);

        // FUTURE ENGINE INTEGRATION POINT:
        // - Connect to audio engine for playhead position updates
        // - Connect to ML pipeline for visualization data
        // - Connect to project state for track/clip data
        // All connections should be made via callbacks/signals, not direct references
#endif
    }
    return self;
}

- (BOOL)isFlipped { return YES; }

- (void)layout {
    [super layout];
#ifdef __cplusplus
    if (_timeline) {
        // Handle Retina/scale factor - ensures JUCE renders at correct resolution
        const auto scale = (float)(self.window.backingScaleFactor ?: [NSScreen mainScreen].backingScaleFactor);
        const juce::Rectangle<int> bounds((int)self.bounds.size.width, (int)self.bounds.size.height);

        // Update JUCE component bounds to match NSView bounds
        // This is called automatically when the view resizes
        _timeline->setBounds(bounds);
        _timeline->setTransform(juce::AffineTransform::scale(scale, scale));
        _timeline->repaint();

        // FUTURE ENGINE INTEGRATION POINT:
        // - Notify engine of timeline size changes for zoom/scroll calculations
        // - Update ML visualization layer bounds if enabled
    }
#endif
}

#ifdef __cplusplus
- (TimelineComponent *)timelineComponent {
    return _timeline.get();
}
#endif

#pragma mark - Appearance

- (void)viewDidChangeEffectiveAppearance {
    [super viewDidChangeEffectiveAppearance];
#ifdef __cplusplus
    if (_timeline) {
        const bool isDark = self.effectiveAppearance.name.lowercaseString.containsString(@"dark");
        _timeline->setUseDarkMode(isDark);
    }
#endif
}

#pragma mark - Event forwarding

- (void)mouseDown:(NSEvent *)event    { [self forwardMouse:event type:juce::MouseInputSource::MouseEventType::down]; }
- (void)mouseUp:(NSEvent *)event      { [self forwardMouse:event type:juce::MouseInputSource::MouseEventType::up]; }
- (void)mouseDragged:(NSEvent *)event { [self forwardMouse:event type:juce::MouseInputSource::MouseEventType::drag]; }
- (void)mouseMoved:(NSEvent *)event   { [self forwardMouse:event type:juce::MouseInputSource::MouseEventType::move]; }
- (void)rightMouseDown:(NSEvent *)event { [self forwardMouse:event type:juce::MouseInputSource::MouseEventType::down]; }
- (void)rightMouseUp:(NSEvent *)event   { [self forwardMouse:event type:juce::MouseInputSource::MouseEventType::up]; }
- (void)scrollWheel:(NSEvent *)event {
#ifdef __cplusplus
    if (_timeline) {
        juce::Point<float> p(event.locationInWindow.x - self.frame.origin.x,
                             self.bounds.size.height - (event.locationInWindow.y - self.frame.origin.y));
        const float dx = (float)event.scrollingDeltaX;
        const float dy = (float)event.scrollingDeltaY;
        const bool isTrackpad = event.hasPreciseScrollingDeltas;
        _timeline->handleScroll(p, dx, dy, isTrackpad);
    }
#endif
}

- (void)keyDown:(NSEvent *)event {
    if ([self interpretKeyEvents:@[event]]) {
        return;
    }
#ifdef __cplusplus
    if (_timeline) {
        _timeline->handleKeyDown(event.keyCode, convertModifiers(event.modifierFlags));
    }
#endif
}

- (void)keyUp:(NSEvent *)event {
#ifdef __cplusplus
    if (_timeline) {
        _timeline->handleKeyUp(event.keyCode, convertModifiers(event.modifierFlags));
    }
#endif
}

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)acceptsFirstMouse:(NSEvent *)event { return YES; }

- (void)setEmotionSnapshotPath:(NSString *)path {
#ifdef __cplusplus
    if (_timeline && path) {
        std::string cppPath = [path UTF8String];
        _timeline->setEmotionSnapshotPath(cppPath);
    }
#endif
}

#pragma mark - Helpers

- (void)forwardMouse:(NSEvent *)event type:(juce::MouseInputSource::MouseEventType)type {
#ifdef __cplusplus
    if (!_timeline)
        return;
    NSPoint local = [self convertPoint:event.locationInWindow fromView:nil];
    // Flip Y to JUCE coords (origin top-left here because isFlipped YES)
    juce::Point<float> p((float)local.x, (float)local.y);
    _timeline->handleMouse(type, p, convertModifiers(event.modifierFlags), event.clickCount);
#endif
}

@end
