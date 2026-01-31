#include "Scheduler.hpp"

namespace emidi {

void EventScheduler::clear() {
    while (!queue_.empty()) {
        queue_.pop();
    }
}

std::size_t EventScheduler::size() const {
    return queue_.size();
}

void EventScheduler::schedule(const ScheduledEvent& event) {
    queue_.push(event);
}

void EventScheduler::schedule_many(const std::vector<ScheduledEvent>& events) {
    for (const auto& ev : events) {
        queue_.push(ev);
    }
}

std::optional<std::uint32_t> EventScheduler::peek_next_tick() const {
    if (queue_.empty()) {
        return std::nullopt;
    }
    return queue_.top().start_tick;
}

std::vector<ScheduledEvent> EventScheduler::pop_due_events(std::uint32_t current_tick, std::uint32_t lookahead_ticks) {
    std::vector<ScheduledEvent> due;
    const std::uint64_t window_tick = static_cast<std::uint64_t>(current_tick) + static_cast<std::uint64_t>(lookahead_ticks);

    while (!queue_.empty()) {
        const auto& next = queue_.top();
        if (static_cast<std::uint64_t>(next.start_tick) <= window_tick) {
            due.push_back(next);
            queue_.pop();
        } else {
            break;
        }
    }

    return due;
}

}  // namespace emidi

