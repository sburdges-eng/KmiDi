#include "plugin/PluginIRInspector.h"
#include "plugin/PluginLogger.h"
#include "plugin/PluginTestHarness.h"
#include "plugin/HostDebugger.h"
#include "core/intent_ir/IntentFrame.h"
#include <juce_core/juce_core.h>
#include <cassert>

using namespace kelly;
using namespace kelly::intent_ir;

void test_plugin_ir_inspector() {
    PluginIRInspector inspector;
    
    IntentFrame frame;
    frame.meta.intent_id = 0x123;
    frame.emotion.valence = 0.5f;
    
    inspector.updateFrame(frame);
    assert(inspector.getBounds().getWidth() > 0); // Component should have size
}

void test_plugin_logging() {
    PluginLogger logger;
    logger.setEnabled(true);
    
    logger.logLifecycle("prepareToPlay", "sampleRate=44100");
    logger.logParameterChange("valence", 0.5f, "UI");
    logger.logIRUpdate(0x123, "ML");
    
    auto entries = logger.getEntries();
    assert(entries.size() == 3);
    assert(entries[0].event_type == "lifecycle");
    assert(entries[1].event_type == "parameter_change");
    assert(entries[2].event_type == "ir_update");
}

void test_plugin_golden_test() {
    PluginTestHarness harness;
    
    IntentFrame staticFrame;
    staticFrame.meta.intent_id = 0xDEADBEEF;
    staticFrame.emotion.valence = 0.5f;
    
    harness.enableTestMode(staticFrame);
    assert(harness.isTestModeActive());
    assert(harness.shouldIgnoreUIEdits());
    assert(harness.shouldDisableML());
    assert(harness.shouldIgnoreAutomation());
    
    IntentFrame retrieved;
    assert(harness.getStaticFrame(retrieved));
    assert(retrieved.meta.intent_id == 0xDEADBEEF);
    
    harness.disableTestMode();
    assert(!harness.isTestModeActive());
}

void test_host_debugging() {
    HostDebugger debugger;
    
    HostType host = HostDebugger::detectHostType(nullptr);
    std::string name = HostDebugger::getHostName(host);
    assert(!name.empty());
    
    assert(debugger.hasQuirks(HostType::LogicPro));
    assert(debugger.hasQuirks(HostType::AbletonLive));
    assert(!debugger.hasQuirks(HostType::Reaper));
    
    bool sync_ok = debugger.verifyParameterSync(
        HostType::AbletonLive,
        "valence",
        0.5f,
        0.5001f
    );
    assert(sync_ok); // Should pass with small tolerance
}

int main() {
    juce::initialiseJuce_GUI();
    
    test_plugin_ir_inspector();
    test_plugin_logging();
    test_plugin_golden_test();
    test_host_debugging();
    
    juce::shutdownJuce_GUI();
    return 0;
}
