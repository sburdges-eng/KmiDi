#pragma once
/*
 * ColorFrequencyMapper.h - VAD → Color and Light Frequency
 * ==========================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: VADCalculator (emotion coordinates)
 * - UI Layer: EmotionWheel, visualization components
 *
 * Purpose: Maps affective state to visible spectrum proxies for UI theming.
 *
 * Features:
 * - Wavelength, THz frequency, and eV energy estimates
 * - RGB triplets and named colors
 */

#include "engine/VADCalculator.h"
#include <string>
#include <map>

namespace kelly {

struct ColorMapping {
    float wavelength;      // nm
    float frequency;       // THz
    float energy;          // eV
    float rgb[3];          // RGB values (0-1)
    std::string name;      // Color name
};

class ColorFrequencyMapper {
public:
    ColorFrequencyMapper();
    
    /**
     * Map emotion to color based on VAD
     */
    ColorMapping emotionToColor(const VADState& vad) const;
    
    /**
     * Map emotion name to color
     */
    ColorMapping emotionNameToColor(const std::string& emotionName) const;
    
    /**
     * Calculate color frequency from valence
     * f_color = f_min + (V+1)(f_max - f_min)/2
     */
    float valenceToFrequency(float valence, float fMin = 400.0f, float fMax = 750.0f) const;
    
    /**
     * Convert wavelength to RGB
     */
    void wavelengthToRGB(float wavelength, float rgb[3]) const;
    
    /**
     * Convert frequency to wavelength
     */
    float frequencyToWavelength(float frequencyTHz) const;
    
    /**
     * Convert wavelength to frequency
     */
    float wavelengthToFrequency(float wavelengthNm) const;
    
private:
    std::map<std::string, ColorMapping> emotionColorMap_;
    
    void initializeEmotionColors();
};

} // namespace kelly
