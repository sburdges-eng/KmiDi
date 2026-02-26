#pragma once

#include "common/DecisionTraceTypes.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace kelly {

class DecisionTrace {
public:
    static constexpr size_t kCapacity = 2048;

    void append(TraceRecord record);

    std::vector<TraceRecord> getBySessionId(uint64_t sessionId) const;
    std::vector<TraceRecord> getLastN(size_t n) const;
    SessionTranscript getSessionTranscript(uint64_t sessionId) const;

    size_t size() const;
    void clear();

    bool dumpToJsonl(const std::string& outputPath, uint64_t sessionId = 0) const;
    bool dumpSessionTranscriptToJsonl(const std::string& outputPath, uint64_t sessionId) const;

    mutable std::mutex mutex_;
    std::array<TraceRecord, kCapacity> buffer_{};
    size_t head_ = 0;
    size_t count_ = 0;
    uint64_t lastMonotonicMicros_ = 0;
    uint64_t lastHash_ = 0;
};

DecisionTrace& globalDecisionTrace();
std::string defaultDecisionTracePath();

} // namespace kelly

namespace kelly {

namespace detail {

inline uint64_t fnv1a(const std::string& value, uint64_t seed = 1469598103934665603ULL) {
    uint64_t hash = seed;
    for (const unsigned char c : value) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::string escapeJson(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 4);
    for (const char c : input) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

inline void writeRecordJson(std::ostream& out, const TraceRecord& r) {
    out << '{'
        << "\"timestamp\":" << r.timestamp << ','
        << "\"monotonicMicros\":" << r.monotonicMicros << ','
        << "\"sessionId\":" << r.sessionId << ','
        << "\"sequenceId\":" << r.sequenceId << ','
        << "\"fromState\":\"" << toString(r.fromState) << "\","
        << "\"toState\":\"" << toString(r.toState) << "\","
        << "\"transitionId\":\"" << escapeJson(r.transitionId) << "\","
        << "\"rulesEvaluated\":" << static_cast<unsigned>(r.rulesEvaluated) << ','
        << "\"ruleResults\":" << static_cast<unsigned>(r.ruleResults) << ','
        << "\"finalStatus\":\"" << toString(r.finalStatus) << "\","
        << "\"abstainReason\":\"" << toString(r.abstainReason) << "\","
        << "\"requestedGroupsMask\":" << r.requestedGroupsMask << ','
        << "\"modifiedGroupsMask\":" << r.modifiedGroupsMask << ','
        << "\"leakedGroupsMask\":" << r.leakedGroupsMask << ','
        << "\"outputChordCount\":" << r.outputChordCount << ','
        << "\"outputMaxVoices\":" << static_cast<unsigned>(r.outputMaxVoices) << ','
        << "\"outputHasMelody\":" << static_cast<unsigned>(r.outputHasMelody) << ','
        << "\"outputHasBass\":" << static_cast<unsigned>(r.outputHasBass) << ','
        << "\"playbackUserInitiated\":" << static_cast<unsigned>(r.playbackUserInitiated) << ','
        << "\"playbackAuthorized\":" << static_cast<unsigned>(r.playbackAuthorized) << ','
        << "\"playbackStarted\":" << static_cast<unsigned>(r.playbackStarted) << ','
        << "\"playbackStopped\":" << static_cast<unsigned>(r.playbackStopped) << ','
        << "\"intentMicros\":" << r.intentMicros << ','
        << "\"generationMicros\":" << r.generationMicros << ','
        << "\"totalMicros\":" << r.totalMicros << ','
        << "\"harnessViolation\":" << static_cast<unsigned>(r.harnessViolation) << ','
        << "\"harnessViolationCode\":\"" << escapeJson(r.harnessViolationCode) << "\","
        << "\"auditCommitted\":" << static_cast<unsigned>(r.auditCommitted) << ','
        << "\"tokenChainLinked\":" << static_cast<unsigned>(r.tokenChainLinked) << ','
        << "\"prevHash\":" << r.prevHash << ','
        << "\"recordHash\":" << r.recordHash << ','
        << "\"invalidViolations\":[";

    for (size_t i = 0; i < r.invalidViolations.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << '"' << escapeJson(r.invalidViolations[i]) << '"';
    }
    out << "]}";
}

inline uint64_t hashRecord(const TraceRecord& r) {
    std::ostringstream oss;
    oss << r.monotonicMicros << '|'
        << r.sessionId << '|'
        << r.sequenceId << '|'
        << static_cast<unsigned>(r.fromState) << '|'
        << static_cast<unsigned>(r.toState) << '|'
        << r.transitionId << '|'
        << static_cast<unsigned>(r.ruleResults) << '|'
        << static_cast<unsigned>(r.finalStatus) << '|'
        << static_cast<unsigned>(r.abstainReason) << '|'
        << r.requestedGroupsMask << '|'
        << r.modifiedGroupsMask << '|'
        << r.leakedGroupsMask << '|'
        << static_cast<unsigned>(r.playbackAuthorized) << '|'
        << r.prevHash;
    return fnv1a(oss.str());
}

} // namespace detail

inline void DecisionTrace::append(TraceRecord record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (record.monotonicMicros == 0) {
        record.monotonicMicros = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }
    if (record.monotonicMicros <= lastMonotonicMicros_) {
        record.monotonicMicros = lastMonotonicMicros_ + 1;
    }
    lastMonotonicMicros_ = record.monotonicMicros;

    if (record.timestamp == 0) {
        record.timestamp = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    record.prevHash = lastHash_;
    record.tokenChainLinked = 1;
    record.recordHash = detail::hashRecord(record);
    lastHash_ = record.recordHash;
    buffer_[head_] = std::move(record);
    head_ = (head_ + 1) % kCapacity;
    if (count_ < kCapacity) {
        ++count_;
    }
}

inline std::vector<TraceRecord> DecisionTrace::getBySessionId(uint64_t sessionId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TraceRecord> out;
    out.reserve(count_);
    for (size_t i = 0; i < count_; ++i) {
        const size_t idx = (head_ + kCapacity - count_ + i) % kCapacity;
        if (buffer_[idx].sessionId == sessionId) {
            out.push_back(buffer_[idx]);
        }
    }
    return out;
}

inline std::vector<TraceRecord> DecisionTrace::getLastN(size_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t take = std::min(n, count_);
    std::vector<TraceRecord> out;
    out.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        const size_t idx = (head_ + kCapacity - take + i) % kCapacity;
        out.push_back(buffer_[idx]);
    }
    return out;
}

inline SessionTranscript DecisionTrace::getSessionTranscript(uint64_t sessionId) const {
    SessionTranscript transcript;
    transcript.sessionId = sessionId;
    transcript.records = getBySessionId(sessionId);
    if (transcript.records.empty()) {
        return transcript;
    }

    transcript.stateSequence.reserve(transcript.records.size() + 1);
    transcript.stateSequence.push_back(transcript.records.front().fromState);
    for (const auto& record : transcript.records) {
        transcript.stateSequence.push_back(record.toState);
    }
    transcript.terminalState = transcript.records.back().toState;
    return transcript;
}

inline size_t DecisionTrace::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

inline void DecisionTrace::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    head_ = 0;
    count_ = 0;
    lastMonotonicMicros_ = 0;
    lastHash_ = 0;
}

inline bool DecisionTrace::dumpToJsonl(const std::string& outputPath, uint64_t sessionId) const {
    std::filesystem::path path(outputPath);
    if (!path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    const auto records = sessionId == 0 ? getLastN(kCapacity) : getBySessionId(sessionId);
    for (const auto& record : records) {
        detail::writeRecordJson(out, record);
        out << '\n';
    }
    return true;
}

inline bool DecisionTrace::dumpSessionTranscriptToJsonl(const std::string& outputPath, uint64_t sessionId) const {
    const auto transcript = getSessionTranscript(sessionId);
    std::filesystem::path path(outputPath);
    if (!path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    for (const auto& record : transcript.records) {
        detail::writeRecordJson(out, record);
        out << '\n';
    }
    return true;
}

inline DecisionTrace& globalDecisionTrace() {
    static DecisionTrace trace;
    return trace;
}

inline std::string defaultDecisionTracePath() {
    return "output/trace/decision_trace.jsonl";
}

} // namespace kelly
