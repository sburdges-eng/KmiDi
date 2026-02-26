#include "adapters/legacy/EmotionThesaurusEnricher.h"

#include <algorithm>
#include <array>
#include <cctype>

namespace kelly {

namespace {

struct EmotionHint {
    const char* token;
    const char* canonicalName;
    float valence;
    float arousal;
    float weight;
};

std::string toLowerCopy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

const std::array<EmotionHint, 12> kHints = {{
    {"grief", "grief", -0.92f, 0.52f, 1.00f},
    {"betrayed", "betrayal", -0.88f, 0.66f, 0.98f},
    {"abandoned", "abandonment", -0.90f, 0.60f, 0.95f},
    {"regret", "regret", -0.74f, 0.47f, 0.90f},
    {"longing", "longing", -0.46f, 0.44f, 0.78f},
    {"hope", "hope", 0.60f, 0.58f, 0.86f},
    {"relief", "relief", 0.52f, 0.34f, 0.82f},
    {"calm", "calm", 0.35f, 0.20f, 0.72f},
    {"rage", "rage", -0.90f, 0.95f, 0.97f},
    {"terror", "terror", -0.98f, 0.98f, 0.99f},
    {"joy", "joy", 0.92f, 0.82f, 0.96f},
    {"ecstatic", "ecstatic", 0.98f, 0.95f, 0.99f},
}};

} // namespace

void EmotionThesaurusEnricher::enrich(const UserInput& input, IntentResult& intent) {
    if (input.text.empty()) {
        return;
    }

    const std::string lower = toLowerCopy(input.text);
    const EmotionHint* best = nullptr;

    for (const auto& hint : kHints) {
        if (lower.find(hint.token) == std::string::npos) {
            continue;
        }
        if (best == nullptr || hint.weight > best->weight) {
            best = &hint;
        }
    }

    if (best == nullptr) {
        return;
    }

    // Read-only enrichment: bias existing inferred emotion without changing control-flow state.
    constexpr float kBlend = 0.35f;
    intent.emotion.valence = std::clamp(
        ((1.0f - kBlend) * intent.emotion.valence) + (kBlend * best->valence), -1.0f, 1.0f);
    intent.emotion.arousal = std::clamp(
        ((1.0f - kBlend) * intent.emotion.arousal) + (kBlend * best->arousal), 0.0f, 1.0f);

    if (intent.emotion.name.empty() || intent.emotion.name == "neutral") {
        intent.emotion.name = best->canonicalName;
    }
}

} // namespace kelly
