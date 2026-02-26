#pragma once

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace kelly {

#ifndef KELLY_ENABLE_DEBUG_LOGS
#define KELLY_ENABLE_DEBUG_LOGS 0
#endif

#if KELLY_ENABLE_DEBUG_LOGS

inline const char* debugLogPath() {
    static std::string path;
    if (!path.empty()) {
        return path.c_str();
    }

    if (const char* configured = std::getenv("KELLY_DEBUG_LOG_PATH")) {
        if (*configured != '\0') {
            path = configured;
            if (std::filesystem::path configuredPath(path); !configuredPath.parent_path().empty()) {
                std::error_code ec;
                std::filesystem::create_directories(configuredPath.parent_path(), ec);
            }
            std::ofstream probe(path, std::ios::app);
            if (probe) {
                return path.c_str();
            }
            path.clear();
        }
    }

    const std::filesystem::path preferred =
        std::filesystem::current_path() / ".cursor" / "debug.log";
    {
        std::error_code ec;
        std::filesystem::create_directories(preferred.parent_path(), ec);
        std::ofstream probe(preferred, std::ios::app);
        if (probe) {
            path = preferred.string();
            return path.c_str();
        }
    }

    const std::filesystem::path fallback =
        std::filesystem::temp_directory_path() / "kelly-debug.log";
    path = fallback.string();
    return path.c_str();
}

inline std::string escapeDebugString(const std::string& dataVal) {
    std::string escaped;
    escaped.reserve(dataVal.size());
    for (const char c : dataVal) {
        if (c == '"') {
            escaped += "\\\"";
        } else if (c == '\\') {
            escaped += "\\\\";
        } else if (c >= 32 && c < 127) {
            escaped += c;
        } else {
            escaped += '?';
        }
    }
    return escaped;
}

inline void debugLog(const char* location,
                     const char* message,
                     const char* dataKey,
                     const std::string& dataVal,
                     const char* hypothesisId) {
    std::ofstream f(debugLogPath(), std::ios::app);
    if (!f) return;
    f << "{\"location\":\"" << location
      << "\",\"message\":\"" << message
      << "\",\"data\":{\"" << dataKey
      << "\":\"" << escapeDebugString(dataVal)
      << "\"},\"timestamp\":"
      << (std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count())
      << ",\"hypothesisId\":\"" << hypothesisId << "\"}\n";
}

inline void debugLogI(const char* location,
                      const char* message,
                      const char* dataKey,
                      int dataVal,
                      const char* hypothesisId) {
    std::ofstream f(debugLogPath(), std::ios::app);
    if (!f) return;
    f << "{\"location\":\"" << location
      << "\",\"message\":\"" << message
      << "\",\"data\":{\"" << dataKey
      << "\":" << dataVal
      << "},\"timestamp\":"
      << (std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count())
      << ",\"hypothesisId\":\"" << hypothesisId << "\"}\n";
}

inline void debugLogB(const char* location,
                      const char* message,
                      const char* dataKey,
                      bool dataVal,
                      const char* hypothesisId) {
    std::ofstream f(debugLogPath(), std::ios::app);
    if (!f) return;
    f << "{\"location\":\"" << location
      << "\",\"message\":\"" << message
      << "\",\"data\":{\"" << dataKey
      << "\":" << (dataVal ? "true" : "false")
      << "},\"timestamp\":"
      << (std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count())
      << ",\"hypothesisId\":\"" << hypothesisId << "\"}\n";
}

#else

inline const char* debugLogPath() { return ""; }
inline void debugLog(const char*, const char*, const char*, const std::string&, const char*) {}
inline void debugLogI(const char*, const char*, const char*, int, const char*) {}
inline void debugLogB(const char*, const char*, const char*, bool, const char*) {}

#endif

} // namespace kelly
