#!/usr/bin/env python3
"""Patch JUCE for macOS 15 CGWindowListCreateImage obsoletion (unavailable in SDK)."""
import sys

path = "modules/juce_gui_basics/native/juce_Windowing_mac.mm"
with open(path) as f:
    content = f.read()

# Skip if already patched
if "#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000" in content:
    sys.exit(0)

old = """static Image createNSWindowSnapshot (NSWindow* nsWindow)
{
    JUCE_AUTORELEASEPOOL
    {
        // CGWindowListCreateImage is replaced by functions in the ScreenCaptureKit framework, but
        // that framework is only available from macOS 12.3 onwards.
        // A suitable @available check should be added once the minimum build OS is 12.3 or greater,
        // so that ScreenCaptureKit can be weak-linked.
       #if defined (MAC_OS_VERSION_14_0) && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_14_0
        JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wdeprecated-declarations")
        #define JUCE_DEPRECATION_IGNORED 1
       #endif

        CGImageRef screenShot = CGWindowListCreateImage (CGRectNull,
                                                         kCGWindowListOptionIncludingWindow,
                                                         (CGWindowID) [nsWindow windowNumber],
                                                         kCGWindowImageBoundsIgnoreFraming);

       #if JUCE_DEPRECATION_IGNORED
        JUCE_END_IGNORE_WARNINGS_GCC_LIKE
        #undef JUCE_DEPRECATION_IGNORED
       #endif

        NSBitmapImageRep* bitmapRep = [[NSBitmapImageRep alloc] initWithCGImage: screenShot];

        Image result (Image::ARGB, (int) [bitmapRep size].width, (int) [bitmapRep size].height, true);

        selectImageForDrawing (result);
        [bitmapRep drawAtPoint: NSMakePoint (0, 0)];
        releaseImageAfterDrawing();

        [bitmapRep release];
        CGImageRelease (screenShot);

        return result;
    }
}"""

new = """static Image createNSWindowSnapshot (NSWindow* nsWindow)
{
    JUCE_AUTORELEASEPOOL
    {
       #if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        (void) nsWindow;
        return {};
       #else
        // CGWindowListCreateImage is replaced by functions in the ScreenCaptureKit framework, but
        // that framework is only available from macOS 12.3 onwards.
        // CGWindowListCreateImage is obsoleted/unavailable in macOS 15 SDK.
       #if defined (MAC_OS_VERSION_14_0) && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_14_0
        JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wdeprecated-declarations")
        #define JUCE_DEPRECATION_IGNORED 1
       #endif

        CGImageRef screenShot = CGWindowListCreateImage (CGRectNull,
                                                         kCGWindowListOptionIncludingWindow,
                                                         (CGWindowID) [nsWindow windowNumber],
                                                         kCGWindowImageBoundsIgnoreFraming);

       #if JUCE_DEPRECATION_IGNORED
        JUCE_END_IGNORE_WARNINGS_GCC_LIKE
        #undef JUCE_DEPRECATION_IGNORED
       #endif

        NSBitmapImageRep* bitmapRep = [[NSBitmapImageRep alloc] initWithCGImage: screenShot];

        Image result (Image::ARGB, (int) [bitmapRep size].width, (int) [bitmapRep size].height, true);

        selectImageForDrawing (result);
        [bitmapRep drawAtPoint: NSMakePoint (0, 0)];
        releaseImageAfterDrawing();

        [bitmapRep release];
        CGImageRelease (screenShot);

        return result;
       #endif
    }
}"""

if old not in content:
    print("JUCE patch: pattern not found (JUCE version may have changed)", file=sys.stderr)
    sys.exit(0)

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("JUCE patched for macOS 15")
