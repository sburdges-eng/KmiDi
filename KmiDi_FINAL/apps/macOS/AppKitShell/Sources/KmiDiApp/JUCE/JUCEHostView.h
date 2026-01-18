#import <Cocoa/Cocoa.h>

// ObjC-visible host view that embeds a JUCE Component. Implemented in JUCEHostView.mm.
@interface JUCEHostView : NSView
@property (nonatomic, readonly) TimelineComponent *timelineComponent;
- (void)setEmotionSnapshotPath:(NSString *)path;
@end
