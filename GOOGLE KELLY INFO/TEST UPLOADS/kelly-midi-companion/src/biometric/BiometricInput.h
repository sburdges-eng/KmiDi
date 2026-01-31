#pragma once

#include "common/Types.h"
#include <functional>
#include <optional>
#include <vector>

namespace kelly {

/**
 * Biometric Input - Real-time biometric data integration.
 * 
 * v2.0 feature: Integrates heart rate, skin conductance, etc. to influence
 * emotion parameters in real-time.
 */
class BiometricInput {
public:
    BiometricInput();
    ~BiometricInput() = default;
    
    /**
     * Biometric data structure
     */
    struct BiometricData {
        std::optional<float> heartRate;        // BPM
        std::optional<float> skinConductance;  // Microsiemens
        std::optional<float> temperature;      // Celsius
        std::optional<float> movement;         // 0.0 to 1.0 (accelerometer)
        double timestamp;                      // Seconds since start
    };
    
    /**
     * Process biometric data and convert to emotion parameters.
     * @param data The biometric reading
     * @return Emotion parameters (valence, arousal, intensity) derived from biometrics
     */
    struct EmotionFromBiometrics {
        float valence;    // -1.0 to 1.0
        float arousal;    // 0.0 to 1.0
        float intensity; // 0.0 to 1.0
    };
    
    EmotionFromBiometrics processBiometricData(const BiometricData& data);
    
    /**
     * Enable/disable biometric input
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
    /**
     * Set callback for when biometric data is received
     */
    void onBiometricData(std::function<void(const BiometricData&)> callback);
    
    /**
     * Simulate biometric input (for testing without hardware)
     */
    void simulateInput(const BiometricData& data);
    
    /**
     * Get smoothed/averaged biometric values
     */
    BiometricData getSmoothedData() const;
    
private:
    bool enabled_ = false;
    std::function<void(const BiometricData&)> onDataCallback_;
    
    // Smoothing/averaging
    std::vector<BiometricData> dataHistory_;
    static constexpr size_t HISTORY_SIZE = 10;
    
    void addToHistory(const BiometricData& data);
    
    /** Convert heart rate to arousal */
    float heartRateToArousal(float bpm) const;
    
    /** Convert skin conductance to intensity */
    float conductanceToIntensity(float conductance) const;
    
    /** Convert movement to arousal */
    float movementToArousal(float movement) const;
};

} // namespace kelly
