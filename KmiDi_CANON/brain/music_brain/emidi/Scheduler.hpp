#pragma once

#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "Event.hpp"

namespace emidi {

/**
 * Scheduled event with EMIDI payload and scheduling metadata.
 */
struct ScheduledEvent {
    std::uint32_t start_tick = 0;
    Event event;
    std::uint8_t channel = 0;
    std::string event_id;
    std::unordered_map<std::string, std::string> metadata;
};

/**
 * Priority queue for scheduled events (min-heap keyed by start_tick).
 */
class EventScheduler {
public:
    void clear();
    std::size_t size() const;

    void schedule(const ScheduledEvent& event);
    void schedule_many(const std::vector<ScheduledEvent>& events);

    std::optional<std::uint32_t> peek_next_tick() const;
    std::vector<ScheduledEvent> pop_due_events(std::uint32_t current_tick, std::uint32_t lookahead_ticks);

private:
    struct CompareStartTick {
        bool operator()(const ScheduledEvent& lhs, const ScheduledEvent& rhs) const {
            return lhs.start_tick > rhs.start_tick;
        }
    };

    std::priority_queue<ScheduledEvent, std::vector<ScheduledEvent>, CompareStartTick> queue_;
};

}  // namespace emidi

