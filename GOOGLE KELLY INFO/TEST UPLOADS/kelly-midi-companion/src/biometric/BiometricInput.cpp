#include "biometric/BiometricInput.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace kelly {

BiometricInput::BiometricInput() : enabled_(false) {
    dataHistory_.reserve(HISTORY_SIZE);
}

BiometricInput::EmotionFromBiometrics BiometricInput::processBiometricData(const BiometricData& data) {
    if (!enabled_) {
        return {0.0f, 0.5f, 0.5f};  // Neutral default
    }
    
    EmotionFromBiometrics result;
    
    // Process heart rate -> arousal
    if (data.heartRate) {
        result.arousal = heartRateToArousal(*data.heartRate);
    } else {
        result.arousal = 0.5f;  // Default
    }
    
    // Process skin conductance -> intensity
    if (data.skinConductance) {
        result.intensity = conductanceToIntensity(*data.skinConductance);
    } else {
        result.intensity = 0.5f;  // Default
    }
    
    // Movement can also affect arousal
    if (data.movement) {
        float movementArousal = movementToArousal(*data.movement);
        result.arousal = (result.arousal + movementArousal) / 2.0f;  // Average
    }
    
    // Valence is harder to determine from basic biometrics
    // Could use temperature (warmer = more positive) or other signals
    if (data.temperature) {
        // Normal body temp ~37°C, warmer might indicate positive state
        float tempNorm = (*data.temperature - 36.0f) / 2.0f;  // Normalize around 36-38°C
        result.valence = std::clamp(tempNorm, -1.0f, 1.0f);
    } else {
        result.valence = 0.0f;  // Neutral
    }
    
    // Clamp all values
    result.valence = std::clamp(result.valence, -1.0f, 1.0f);
    result.arousal = std::clamp(result.arousal, 0.0f, 1.0f);
    result.intensity = std::clamp(result.intensity, 0.0f, 1.0f);
    
    // Add to history for smoothing
    addToHistory(data);
    
    // Trigger callback
    if (onDataCallback_) {
        onDataCallback_(data);
    }
    
    return result;
}

void BiometricInput::simulateInput(const BiometricData& data) {
    if (enabled_) {
        processBiometricData(data);
    }
}

BiometricInput::BiometricData BiometricInput::getSmoothedData() const {
    if (dataHistory_.empty()) {
        return {};
    }
    
    BiometricData smoothed;
    size_t count = dataHistory_.size();
    
    // Average heart rate
    float totalHR = 0.0f;
    int hrCount = 0;
    for (const auto& d : dataHistory_) {
        if (d.heartRate) {
            totalHR += *d.heartRate;
            hrCount++;
        }
    }
    if (hrCount > 0) smoothed.heartRate = totalHR / hrCount;
    
    // Average skin conductance
    float totalSC = 0.0f;
    int scCount = 0;
    for (const auto& d : dataHistory_) {
        if (d.skinConductance) {
            totalSC += *d.skinConductance;
            scCount++;
        }
    }
    if (scCount > 0) smoothed.skinConductance = totalSC / scCount;
    
    // Average temperature
    float totalTemp = 0.0f;
    int tempCount = 0;
    for (const auto& d : dataHistory_) {
        if (d.temperature) {
            totalTemp += *d.temperature;
            tempCount++;
        }
    }
    if (tempCount > 0) smoothed.temperature = totalTemp / tempCount;
    
    // Average movement
    float totalMove = 0.0f;
    int moveCount = 0;
    for (const auto& d : dataHistory_) {
        if (d.movement) {
            totalMove += *d.movement;
            moveCount++;
        }
    }
    if (moveCount > 0) smoothed.movement = totalMove / moveCount;
    
    // Use most recent timestamp
    smoothed.timestamp = dataHistory_.back().timestamp;
    
    return smoothed;
}

void BiometricInput::onBiometricData(std::function<void(const BiometricData&)> callback) {
    onDataCallback_ = callback;
}

void BiometricInput::addToHistory(const BiometricData& data) {
    dataHistory_.push_back(data);
    if (dataHistory_.size() > HISTORY_SIZE) {
        dataHistory_.erase(dataHistory_.begin());
    }
}

float BiometricInput::heartRateToArousal(float bpm) const {
    // Normal resting HR: 60-100 BPM
    // High arousal: >100 BPM
    // Low arousal: <60 BPM (or very calm)
    
    if (bpm < 60.0f) {
        return 0.2f;  // Very calm
    } else if (bpm < 80.0f) {
        return 0.3f + ((bpm - 60.0f) / 20.0f) * 0.2f;  // 0.3 to 0.5
    } else if (bpm < 100.0f) {
        return 0.5f + ((bpm - 80.0f) / 20.0f) * 0.3f;  // 0.5 to 0.8
    } else {
        return 0.8f + std::min((bpm - 100.0f) / 60.0f, 0.2f);  // 0.8 to 1.0
    }
}

float BiometricInput::conductanceToIntensity(float conductance) const {
    // Skin conductance typically 1-20 microsiemens
    // Higher = more intense emotional state
    return std::clamp(conductance / 20.0f, 0.0f, 1.0f);
}

float BiometricInput::movementToArousal(float movement) const {
    // Movement already normalized 0.0 to 1.0
    // More movement = higher arousal
    return movement;
}

} // namespace kelly
