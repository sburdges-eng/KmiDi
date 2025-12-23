/*
  ==============================================================================

    RunTests.cpp
    Created: 2025
    Author: DAiW Team

    Main entry point for running all plugin tests

  ==============================================================================
*/

#include <JuceHeader.h>

// Option 1: Use JUCE's UnitTest framework
#include <juce_core/juce_core.h>

// Option 2: Use Catch2 if available
// #define CATCH_CONFIG_MAIN
// #include <catch2/catch_all.hpp>

// For JUCE UnitTest framework
class DAiWBridgeTestRunner : public juce::UnitTestRunner
{
public:
    void runAllTests()
    {
        // This will be populated when tests are registered
        // For now, tests are in separate files using CATCH_TEST_CASE
    }
};

// Main entry point
int main(int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI initialiser;
    
    // If using Catch2, it handles main()
    // Otherwise, use JUCE's test runner
    
    juce::UnitTestRunner runner;
    runner.runAllTests();
    
    return 0;
}

