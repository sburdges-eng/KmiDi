#include "plugin/PluginLogger.h"
#include <sstream>
#include <iomanip>
#include <thread>

namespace kelly {

PluginLogger::PluginLogger() {
}

void PluginLogger::logLifecycle(const std::string& event, const std::string& details) {
    if (!enabled_.load()) return;
    
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.thread_name = getCurrentThreadName();
    entry.event_type = "lifecycle";
    entry.intent_id = 0;
    entry.parameter_source = "";
    entry.message = event + (details.empty() ? "" : ": " + details);
    
    addEntry(entry);
}

void PluginLogger::logParameterChange(
    const std::string& parameter_id,
    float new_value,
    const std::string& source
) {
    if (!enabled_.load()) return;
    
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.thread_name = getCurrentThreadName();
    entry.event_type = "parameter_change";
    entry.intent_id = 0;
    entry.parameter_source = source;
    
    std::ostringstream oss;
    oss << parameter_id << " = " << std::fixed << std::setprecision(3) << new_value;
    entry.message = oss.str();
    
    addEntry(entry);
}

void PluginLogger::logIRUpdate(uint64_t intent_id, const std::string& source) {
    if (!enabled_.load()) return;
    
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.thread_name = getCurrentThreadName();
    entry.event_type = "ir_update";
    entry.intent_id = intent_id;
    entry.parameter_source = source;
    entry.message = "IntentFrame updated";
    
    addEntry(entry);
}

std::vector<PluginLogger::LogEntry> PluginLogger::getEntries() const {
    std::lock_guard<std::mutex> lock(entriesMutex_);
    return entries_;
}

bool PluginLogger::exportToFile(const juce::File& file) const {
    auto entries = getEntries();
    
    std::ostringstream oss;
    oss << "Plugin Log Export\n";
    oss << "==================\n\n";
    
    for (const auto& entry : entries) {
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()
        ).count() % 1000;
        
        std::tm* local_time = std::localtime(&time_t);
        oss << std::put_time(local_time, "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms << " ";
        oss << "[" << entry.thread_name << "] ";
        oss << entry.event_type << " ";
        if (entry.intent_id != 0) {
            oss << "intent_id=0x" << std::hex << entry.intent_id << std::dec << " ";
        }
        if (!entry.parameter_source.empty()) {
            oss << "source=" << entry.parameter_source << " ";
        }
        oss << entry.message << "\n";
    }
    
    return file.replaceWithText(oss.str());
}

void PluginLogger::clear() {
    std::lock_guard<std::mutex> lock(entriesMutex_);
    entries_.clear();
}

std::string PluginLogger::getCurrentThreadName() const {
    // Try to get thread name (platform-specific)
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) == 0) {
        return std::string(name);
    }
    return "thread-" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

void PluginLogger::addEntry(const LogEntry& entry) {
    std::lock_guard<std::mutex> lock(entriesMutex_);
    entries_.push_back(entry);
    if (entries_.size() > MAX_ENTRIES) {
        entries_.erase(entries_.begin());
    }
}

} // namespace kelly
