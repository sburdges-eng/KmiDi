#include "MLVisualizationLayer.h"

namespace {
inline juce::Colour emotionColour(float valence, float arousal, bool isDark)
{
    // Map valence/arousal to hue/brightness; keep saturation low for subtlety.
    const float hue = juce::jlimit(0.0f, 1.0f, 0.55f + 0.25f * valence); // center around blue/cyan
    const float sat = 0.15f + 0.10f * std::abs(arousal);
    const float bri = isDark ? 0.18f : 0.85f;
    return juce::Colour::fromHSV(hue, sat, bri, 1.0f);
}

inline float safeProb(float v) { return juce::jlimit(0.0f, 1.0f, v); }
} // namespace

MLVisualizationLayer::MLVisualizationLayer()
{
    setInterceptsMouseClicks(false, false); // Non-intrusive overlay
    setOpaque(false);
}

void MLVisualizationLayer::setData(const Data& d)
{
    data = d;
    hasData = true;
    needsRebuild = true;
    repaint();
}

void MLVisualizationLayer::clearData()
{
    hasData = false;
    needsRebuild = true;
    melodyPath.clear();
    harmonyPath.clear();
    dynamicsPath.clear();
    grooveLines.clear();
    repaint();
}

void MLVisualizationLayer::setZoom(float zoom)
{
    zoomLevel = zoom;
    repaint();
}

void MLVisualizationLayer::setShowMelody(bool enable)   { showMelody = enable; repaint(); }
void MLVisualizationLayer::setShowHarmony(bool enable)  { showHarmony = enable; repaint(); }
void MLVisualizationLayer::setShowGroove(bool enable)   { showGroove = enable; repaint(); }
void MLVisualizationLayer::setShowDynamics(bool enable) { showDynamics = enable; repaint(); }
void MLVisualizationLayer::setShowEmotion(bool enable)  { showEmotion = enable; repaint(); }

void MLVisualizationLayer::setDarkMode(bool dark)
{
    isDark = dark;
    repaint();
}

void MLVisualizationLayer::resized()
{
    needsRebuild = true;
}

void MLVisualizationLayer::paint(juce::Graphics& g)
{
    if (!hasData)
        return;

    if (needsRebuild)
    {
        rebuildPaths();
        needsRebuild = false;
    }

    if (showEmotion)
        drawEmotionTint(g);

    if (showHarmony)
        drawHarmony(g);

    if (showGroove)
        drawGroove(g);

    if (showMelody && zoomLevel > 0.45f) // only when sufficiently zoomed in
        drawMelody(g);

    if (showDynamics)
        drawDynamics(g);
}

void MLVisualizationLayer::rebuildPaths()
{
    melodyPath.clear();
    harmonyPath.clear();
    dynamicsPath.clear();
    grooveLines.clear();

    const auto bounds = getLocalBounds().toFloat();
    if (bounds.isEmpty())
        return;

    // Melody ghosts: simple grid across width; alpha later set in paint.
    const float noteWidth = bounds.getWidth() / 64.0f; // coarse grid to avoid perf issues
    const float noteHeight = bounds.getHeight() * 0.35f;
    for (int i = 0; i < 128; i += 2) // sample every other note for density control
    {
        const float prob = safeProb(data.noteProbabilities[(size_t) i]);
        if (prob < 0.05f)
            continue;
        const float x = std::fmod((float) i * noteWidth * 0.5f, bounds.getWidth());
        const float y = bounds.getHeight() * 0.15f + (float)(i % 12) * 2.5f;
        melodyPath.addRectangle(x, y, noteWidth * 0.8f, noteHeight * 0.25f);
    }

    // Harmony: draw band per chord index chunk.
    const float chordBandHeight = bounds.getHeight() * 0.12f;
    for (int i = 0; i < 16 && i < (int) data.chordProbabilities.size(); ++i)
    {
        const float prob = safeProb(data.chordProbabilities[(size_t) i]);
        if (prob < 0.05f)
            continue;
        const float x0 = bounds.getWidth() * ((float) i / 16.0f);
        const float x1 = bounds.getWidth() * ((float) (i + 1) / 16.0f);
        harmonyPath.addRectangle(x0, bounds.getHeight() * 0.55f, x1 - x0, chordBandHeight);
    }

    // Groove: micro timing offsets rendered as short vertical lines.
    const float grooveY0 = bounds.getHeight() * 0.7f;
    const float grooveY1 = bounds.getHeight() * 0.9f;
    for (int i = 0; i < (int) data.groove.size(); ++i)
    {
        const float swing = juce::jlimit(-0.5f, 0.5f, data.groove[(size_t) i] - 0.5f);
        const float x = bounds.getWidth() * ((float) i / (float) data.groove.size());
        const float offset = swing * 6.0f; // pixels
        grooveLines.add (juce::Line<float>(x + offset, grooveY0, x + offset, grooveY1));
    }

    // Dynamics: polyline across width.
    const float dynBaseY = bounds.getHeight() * 0.35f;
    const float dynAmp = bounds.getHeight() * 0.18f;
    juce::Path dyn;
    dyn.startNewSubPath(bounds.getX(), dynBaseY);
    for (int i = 0; i < (int) data.dynamics.size(); ++i)
    {
        const float t = (float) i / (float) (data.dynamics.size() - 1);
        const float x = bounds.getWidth() * t;
        const float y = dynBaseY - dynAmp * juce::jlimit(0.0f, 1.0f, data.dynamics[(size_t) i]);
        dyn.lineTo(x, y);
    }
    dynamicsPath = dyn;
}

void MLVisualizationLayer::drawEmotionTint(juce::Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    const auto col = emotionColour(data.valence, data.arousal, isDark).withAlpha(isDark ? 0.06f : 0.05f);
    g.setColour(col);
    g.fillRect(bounds);
}

void MLVisualizationLayer::drawMelody(juce::Graphics& g)
{
    const auto col = emotionColour(data.valence, data.arousal, isDark);
    g.setColour(col.withAlpha(0.20f));
    g.fillPath(melodyPath);
}

void MLVisualizationLayer::drawHarmony(juce::Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    for (int i = 0; i < 16 && i < (int) data.chordProbabilities.size(); ++i)
    {
        const float prob = safeProb(data.chordProbabilities[(size_t) i]);
        if (prob < 0.05f)
            continue;
        const float x0 = bounds.getWidth() * ((float) i / 16.0f);
        const float x1 = bounds.getWidth() * ((float) (i + 1) / 16.0f);
        const float bandH = bounds.getHeight() * 0.12f;
        auto band = juce::Rectangle<float>(x0, bounds.getHeight() * 0.55f, x1 - x0, bandH);
        const float alpha = 0.08f + 0.15f * prob;
        g.setColour(juce::Colours::white.withAlpha(alpha));
        g.fillRect(band);
    }
}

void MLVisualizationLayer::drawGroove(juce::Graphics& g)
{
    g.setColour(juce::Colours::orange.withAlpha(0.18f));
    for (auto& ln : grooveLines)
        g.drawLine(ln, 1.0f);
}

void MLVisualizationLayer::drawDynamics(juce::Graphics& g)
{
    g.setColour(juce::Colours::lightgreen.withAlpha(0.25f));
    g.strokePath(dynamicsPath, juce::PathStrokeType(1.5f));
}
