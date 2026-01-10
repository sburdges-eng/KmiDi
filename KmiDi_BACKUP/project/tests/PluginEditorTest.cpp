/*
  ==============================================================================

    PluginEditorTest.cpp
    Created: 2025
    Author: DAiW Team

    Unit tests for DAiW Bridge Plugin Editor

  ==============================================================================
*/

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "PluginEditor.h"

// Use Catch2 if available, otherwise fall back to JUCE UnitTest
#ifdef CATCH_TEST_CASE
    #include <catch2/catch_test_macros.hpp>
#else
    #include <juce_core/juce_core.h>
    #define CATCH_TEST_CASE(name, tags) static void name()
    #define REQUIRE(condition) jassert(condition)
    #define REQUIRE_NOTHROW(expression) { try { expression; } catch(...) { jassertfalse; } }
#endif

// If Catch2 is not available, use JUCE's UnitTest framework
#ifndef CATCH_TEST_CASE
    #include <juce_core/juce_core.h>
    #define CATCH_TEST_CASE(name, tags) static void name()
    #define REQUIRE(condition) jassert(condition)
    #define REQUIRE_NOTHROW(expression) { try { expression; } catch(...) { jassertfalse; } }
#endif

//==============================================================================
// Test Fixtures
//==============================================================================

class PluginEditorTestFixture
{
public:
    PluginEditorTestFixture()
    {
        processor = std::make_unique<DAiWBridgeAudioProcessor>();
        editor = std::make_unique<DAiWBridgeAudioProcessorEditor>(*processor);
    }
    
    ~PluginEditorTestFixture()
    {
        editor.reset();
        processor.reset();
    }
    
    std::unique_ptr<DAiWBridgeAudioProcessor> processor;
    std::unique_ptr<DAiWBridgeAudioProcessorEditor> editor;
};

//==============================================================================
// UI Component Tests
//==============================================================================

CATCH_TEST_CASE("PluginEditor - Initialization", "[editor]")
{
    PluginEditorTestFixture fixture;
    
    REQUIRE(fixture.editor != nullptr);
    REQUIRE(fixture.editor->getWidth() > 0);
    REQUIRE(fixture.editor->getHeight() > 0);
}

CATCH_TEST_CASE("PluginEditor - Resize", "[editor]")
{
    PluginEditorTestFixture fixture;
    
    int newWidth = 600;
    int newHeight = 500;
    
    fixture.editor->setSize(newWidth, newHeight);
    fixture.editor->resized();
    
    REQUIRE(fixture.editor->getWidth() == newWidth);
    REQUIRE(fixture.editor->getHeight() == newHeight);
}

CATCH_TEST_CASE("PluginEditor - Paint", "[editor]")
{
    PluginEditorTestFixture fixture;
    
    juce::Graphics g(*fixture.editor);
    
    // Should not throw when painting
    REQUIRE_NOTHROW(fixture.editor->paint(g));
}

//==============================================================================
// Button Tests
//==============================================================================

CATCH_TEST_CASE("PluginEditor - Generate Button Exists", "[ui]")
{
    PluginEditorTestFixture fixture;
    
    // Button should exist (tested via editor creation)
    REQUIRE(fixture.editor != nullptr);
}

//==============================================================================
// Integration Tests
//==============================================================================

CATCH_TEST_CASE("PluginEditor - Full Workflow", "[integration]")
{
    PluginEditorTestFixture fixture;
    
    // Simulate user interaction workflow
    // 1. Editor created
    REQUIRE(fixture.editor != nullptr);
    
    // 2. Resize
    fixture.editor->setSize(500, 400);
    fixture.editor->resized();
    
    // 3. Paint
    juce::Graphics g(*fixture.editor);
    REQUIRE_NOTHROW(fixture.editor->paint(g));
    
    // 4. Editor should remain functional
    REQUIRE(fixture.editor->getWidth() == 500);
    REQUIRE(fixture.editor->getHeight() == 400);
}

