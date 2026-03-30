#include "ui/EQCurveView.h"
#include "ui/AIEQSuggestionEngine.h"
#include <cmath>

namespace kelly {

EQCurveView::EQCurveView() {
  // Initialize curves to flat (0 dB)
  userCurve_.fill(0.0f);
  aiCurve_.fill(0.0f);
}

void EQCurveView::paint(juce::Graphics &g) {
  auto bounds = getLocalBounds().toFloat();
  bounds.reduce(GRID_MARGIN, GRID_MARGIN);

  // Background
  g.fillAll(juce::Colour(0xFF1A1A1A));

  // Draw grid
  if (!gridPathValid_) {
    updateGridPaths();
  }
  g.setColour(juce::Colour(0xFF404040));
  g.strokePath(frequencyGridPath_, juce::PathStrokeType(0.5f));
  g.strokePath(gainGridPath_, juce::PathStrokeType(0.5f));

  // Draw gain axis (0 dB line)
  g.setColour(juce::Colour(0xFF606060));
  float zeroDbY = gainToY(0.0f, bounds.getHeight()) + bounds.getY();
  g.drawLine(bounds.getX(), zeroDbY, bounds.getX() + bounds.getWidth(), zeroDbY, 1.0f);

  // Draw user curve (solid, neutral light gray)
  if (!userCurvePathValid_) {
    updateUserCurvePath();
  }
  g.setColour(juce::Colour(0xFFCCCCCC));
  g.strokePath(userCurvePath_, juce::PathStrokeType(2.0f));

  // Draw AI curve (dashed, desaturated accent color, fades when disabled)
  if (aiCurveVisible_ && aiCurveIntensity_ > 0.0f) {
    if (!aiCurvePathValid_) {
      updateAICurvePath();
    }
    juce::Colour aiColor = juce::Colour(0xFF88AAFF).withAlpha(aiCurveIntensity_ * 0.7f);
    g.setColour(aiColor);
    juce::PathStrokeType stroke(1.5f);
    juce::Array<float> dashes;
    dashes.add(3.0f);
    dashes.add(3.0f);
    stroke.createDashedStroke(aiCurvePath_, aiCurvePath_, dashes.getRawDataPointer(), 2);
    g.strokePath(aiCurvePath_, stroke);
  }

  // Draw axes labels
  g.setColour(juce::Colour(0xFF888888));
  g.setFont(10.0f);

  // Frequency labels (major markers: 100Hz, 1kHz, 10kHz)
  float freqLabels[] = {100.0f, 1000.0f, 10000.0f};
  for (float freq : freqLabels) {
    float x = freqToX(freq, bounds.getWidth()) + bounds.getX();
    float y = bounds.getBottom() + 5.0f;
    juce::String label;
    if (freq >= 1000.0f) {
      label = juce::String(freq / 1000.0f, 1) + "k";
    } else {
      label = juce::String(static_cast<int>(freq));
    }
    g.drawText(label, x - 20.0f, y, 40.0f, 12.0f, juce::Justification::centred);
  }

  // Gain labels (major markers: -24, -12, 0, +12, +24 dB)
  float gainLabels[] = {-24.0f, -12.0f, 0.0f, 12.0f, 24.0f};
  for (float gain : gainLabels) {
    float y = gainToY(gain, bounds.getHeight()) + bounds.getY();
    juce::String label = juce::String(gain, 0);
    if (gain > 0.0f) {
      label = "+" + label;
    }
    g.drawText(label, 5.0f, y - 6.0f, 35.0f, 12.0f, juce::Justification::right);
  }
}

void EQCurveView::resized() {
  // Invalidate paths when size changes
  userCurvePathValid_ = false;
  aiCurvePathValid_ = false;
  gridPathValid_ = false;
  repaint();
}

void EQCurveView::updateUserCurve(const std::array<float, CURVE_POINTS> &curve) {
  userCurve_ = curve;
  userCurvePathValid_ = false;
  repaint();
}

void EQCurveView::updateAICurve(const std::array<float, CURVE_POINTS> &curve) {
  aiCurve_ = curve;
  aiCurvePathValid_ = false;
  repaint();
}

void EQCurveView::setAICurveVisible(bool visible) {
  if (aiCurveVisible_ != visible) {
    aiCurveVisible_ = visible;
    repaint();
  }
}

void EQCurveView::setAICurveIntensity(float intensity) {
  intensity = juce::jlimit(0.0f, 1.0f, intensity);
  if (aiCurveIntensity_ != intensity) {
    aiCurveIntensity_ = intensity;
    repaint();
  }
}

void EQCurveView::updateUserCurvePath() {
  auto bounds = getLocalBounds().toFloat();
  bounds.reduce(GRID_MARGIN, GRID_MARGIN);

  userCurvePath_.clear();
  const auto &freqGrid = AIEQSuggestionEngine::getFrequencyGrid();

  bool firstPoint = true;
  for (int i = 0; i < CURVE_POINTS; ++i) {
    float x = freqToX(freqGrid[i], bounds.getWidth()) + bounds.getX();
    float y = gainToY(userCurve_[i], bounds.getHeight()) + bounds.getY();

    // Clamp Y to bounds
    y = juce::jlimit(bounds.getY(), bounds.getBottom(), y);

    if (firstPoint) {
      userCurvePath_.startNewSubPath(x, y);
      firstPoint = false;
    } else {
      userCurvePath_.lineTo(x, y);
    }
  }

  userCurvePathValid_ = true;
}

void EQCurveView::updateAICurvePath() {
  auto bounds = getLocalBounds().toFloat();
  bounds.reduce(GRID_MARGIN, GRID_MARGIN);

  aiCurvePath_.clear();
  const auto &freqGrid = AIEQSuggestionEngine::getFrequencyGrid();

  bool firstPoint = true;
  for (int i = 0; i < CURVE_POINTS; ++i) {
    float x = freqToX(freqGrid[i], bounds.getWidth()) + bounds.getX();
    float y = gainToY(aiCurve_[i], bounds.getHeight()) + bounds.getY();

    // Clamp Y to bounds
    y = juce::jlimit(bounds.getY(), bounds.getBottom(), y);

    if (firstPoint) {
      aiCurvePath_.startNewSubPath(x, y);
      firstPoint = false;
    } else {
      aiCurvePath_.lineTo(x, y);
    }
  }

  aiCurvePathValid_ = true;
}

void EQCurveView::updateGridPaths() {
  auto bounds = getLocalBounds().toFloat();
  bounds.reduce(GRID_MARGIN, GRID_MARGIN);

  frequencyGridPath_.clear();
  gainGridPath_.clear();

  // Frequency grid lines (vertical): 100Hz, 1kHz, 10kHz
  float freqMarkers[] = {100.0f, 1000.0f, 10000.0f};
  for (float freq : freqMarkers) {
    float x = freqToX(freq, bounds.getWidth()) + bounds.getX();
    frequencyGridPath_.startNewSubPath(x, bounds.getY());
    frequencyGridPath_.lineTo(x, bounds.getBottom());
  }

  // Gain grid lines (horizontal): -24, -12, 0, +12, +24 dB
  float gainMarkers[] = {-24.0f, -12.0f, 0.0f, 12.0f, 24.0f};
  for (float gain : gainMarkers) {
    if (gain == 0.0f) {
      continue; // Skip 0 dB (drawn separately as axis)
    }
    float y = gainToY(gain, bounds.getHeight()) + bounds.getY();
    gainGridPath_.startNewSubPath(bounds.getX(), y);
    gainGridPath_.lineTo(bounds.getX() + bounds.getWidth(), y);
  }

  gridPathValid_ = true;
}

float EQCurveView::freqToX(float freq, float width) const {
  // Log scale: x = width * log(freq / fMin) / log(fMax / fMin)
  float logMin = std::log10(FREQ_MIN);
  float logMax = std::log10(FREQ_MAX);
  float logFreq = std::log10(std::max(FREQ_MIN, std::min(FREQ_MAX, freq)));
  float normalized = (logFreq - logMin) / (logMax - logMin);
  return normalized * width;
}

float EQCurveView::gainToY(float gainDb, float height) const {
  // Linear scale: y = height * (1 - (gain - gainMin) / (gainMax - gainMin))
  float normalized = (gainDb - GAIN_MIN) / (GAIN_MAX - GAIN_MIN);
  normalized = juce::jlimit(0.0f, 1.0f, normalized);
  return height * (1.0f - normalized);
}

} // namespace kelly
