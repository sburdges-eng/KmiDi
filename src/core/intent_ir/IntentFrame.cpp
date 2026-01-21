#include "IntentFrame.h"

namespace kelly {
namespace intent_ir {

IntentFrameSnapshot::IntentFrameSnapshot(const IntentFrame& frame)
    : frame_(frame)
{
}

void IntentFrameSnapshot::update(const IntentFrame& frame) {
    frame_ = frame;
}

} // namespace intent_ir
} // namespace kelly
