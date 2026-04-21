#include "midi/AffectUMP.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

#if JUCE_MODULE_AVAILABLE_juce_audio_basics
#include <juce_audio_basics/juce_audio_basics.h>
#endif

namespace kelly {
namespace AffectUMP {

uint32_t floatToUmp32Value(float value, float minVal, float maxVal) {
    if (minVal >= maxVal) {
        return 0;
    }
    float clamped = std::clamp(value, minVal, maxVal);
    float norm = (clamped - minVal) / (maxVal - minVal);
    // Cast via double and clamp before truncation to uint32_t to avoid UB when
    // the rounded value equals or exceeds 2^32 (implementation-defined / UB in C++).
    double scaled = std::round(static_cast<double>(norm) * static_cast<double>(0xFFFFFFFFu));
    scaled = std::clamp(scaled, 0.0, static_cast<double>(UINT32_MAX));
    auto v = static_cast<uint32_t>(scaled);
    return v;
}

void buildUmpAffectCC(uint8_t group,
                      uint8_t channel,
                      uint8_t indexCC,
                      float value,
                      float minVal,
                      float maxVal,
                      uint32_t outWords[2]) {
#if JUCE_MODULE_AVAILABLE_juce_audio_basics
    namespace ump = juce::universal_midi_packets;
    uint32_t data32 = floatToUmp32Value(value, minVal, maxVal);
    auto packet = ump::Factory::makeControlChangeV2(group, channel, indexCC & 0x7F, data32);
    outWords[0] = packet.front();
    outWords[1] = packet.back();
#else
    (void)group;
    (void)channel;
    (void)indexCC;
    (void)value;
    (void)minVal;
    (void)maxVal;
    outWords[0] = 0;
    outWords[1] = 0;
#endif
}

}  // namespace AffectUMP
}  // namespace kelly
