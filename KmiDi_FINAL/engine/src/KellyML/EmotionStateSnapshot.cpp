#include "EmotionStateSnapshot.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>

namespace kelly::ml {

std::string EmotionStateSnapshot::toJson() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    oss << "{\n";
    oss << "  \"version\": " << version << ",\n";
    oss << "  \"timestamp\": " << timestamp << ",\n";
    oss << "  \"emotion\": {\n";
    oss << "    \"valence\": " << emotion.valence << ",\n";
    oss << "    \"arousal\": " << emotion.arousal << ",\n";
    oss << "    \"dominance\": " << emotion.dominance << ",\n";
    oss << "    \"complexity\": " << emotion.complexity << "\n";
    oss << "  }";

    if (!labelPrimary.empty() || !labelSecondary.empty()) {
        oss << ",\n";
        oss << "  \"labels\": {\n";
        if (!labelPrimary.empty()) {
            oss << "    \"primary\": \"" << labelPrimary << "\"";
            if (!labelSecondary.empty()) {
                oss << ",\n";
            }
        }
        if (!labelSecondary.empty()) {
            if (labelPrimary.empty()) {
                oss << "    ";
            }
            oss << "    \"secondary\": \"" << labelSecondary << "\"\n";
        } else {
            oss << "\n";
        }
        oss << "  }";
    }

    oss << "\n}";
    return oss.str();
}

bool EmotionStateSnapshot::writeToFile(const std::string& filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        return false;
    }

    file << toJson();
    file.close();
    return file.good();
}

EmotionStateSnapshot EmotionStateSnapshot::fromEmotionState(const EmotionState& state) {
    EmotionStateSnapshot snapshot;
    snapshot.emotion = state;

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto fractional = std::chrono::duration_cast<std::chrono::microseconds>(duration - seconds);
    snapshot.timestamp = static_cast<double>(seconds.count()) + (fractional.count() / 1000000.0);

    return snapshot;
}

} // namespace kelly::ml
