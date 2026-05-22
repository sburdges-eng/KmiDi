// Cross-process test for daiw::ipc::SharedRegion.
//
// fork()s a child that attaches to the region created by the parent. Parent
// writes commands into cmds_in and publishes RTState fields; child verifies
// them and writes events back into events_out. Parent then verifies the
// child's events, unlinks the region, and reports.
//
// Standalone — no Catch2, no JUCE. Runs on POSIX (Linux + macOS).

#include "daiw/shared_ipc.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

int g_failures = 0;
void check(bool condition, const char *label) {
  if (!condition) {
    std::printf("FAIL: %s\n", label);
    ++g_failures;
  }
}

constexpr int kCommands = 16;

// Build a unique region name per run so concurrent test invocations don't
// collide on the global namespace. macOS PSHMNAMLEN is 31 (including the
// leading slash and the implicit nul) — keep the name short.
std::string make_region_name() {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/kmidi_ipc_%d", ::getpid());
  return std::string(buf);
}

int run_child(const std::string &region_name) {
  daiw::ipc::SharedRegion region;
  if (!region.attach(region_name, sizeof(daiw::ipc::SharedLayout), 2000)) {
    std::fprintf(stderr, "child: attach failed\n");
    return 1;
  }
  auto *layout = region.layout();
  if (!layout)
    return 2;

  // Wait for kCommands to arrive in cmds_in, then echo each one as an
  // EventMsg with code=1, copying the timestamp + data1.
  int received = 0;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (received < kCommands && std::chrono::steady_clock::now() < deadline) {
    auto cmd = layout->cmds_in.pop();
    if (!cmd) {
      std::this_thread::yield();
      continue;
    }
    // Validate: sender wrote sequential timestamps + data1 = index.
    if (cmd->timestamp != static_cast<uint64_t>(received) * 1000 ||
        cmd->data1 != static_cast<uint8_t>(received)) {
      std::fprintf(stderr, "child: cmd %d shape mismatch (ts=%llu d1=%u)\n",
                   received, (unsigned long long)cmd->timestamp,
                   (unsigned)cmd->data1);
      return 3;
    }
    // Echo back.
    daiw::ipc::EventMsg ev{};
    ev.timestamp = cmd->timestamp;
    ev.code = 1;
    ev.channel = cmd->channel;
    ev.data1 = cmd->data1;
    ev.data2 = cmd->data2;
    ev.data16 = cmd->data16;
    while (!layout->events_out.push(ev))
      std::this_thread::yield();
    ++received;
  }

  if (received != kCommands) {
    std::fprintf(stderr, "child: only %d/%d commands received\n", received,
                 kCommands);
    return 4;
  }

  // Read RTState fields and write them back into specific event payloads
  // so the parent can verify the seqlock-published values.
  daiw::ipc::EventMsg trailer{};
  trailer.timestamp = static_cast<uint64_t>(
      layout->state.bpm.load(std::memory_order_acquire) * 100.0);
  trailer.code = 2;
  trailer.data1 = static_cast<uint8_t>(
      layout->state.playing.load(std::memory_order_acquire) ? 1 : 0);
  trailer.data16 =
      static_cast<uint16_t>(layout->state.bar.load(std::memory_order_acquire));
  while (!layout->events_out.push(trailer))
    std::this_thread::yield();

  region.detach();
  return 0;
}

} // namespace

int main() {
  const std::string region_name = make_region_name();

  // Pre-unlink in case a previous failed run left it behind.
  daiw::ipc::SharedRegion::unlink_name(region_name);

  // Create the region in the parent BEFORE forking so the child can attach
  // immediately. SharedRegion takes ownership in the parent; we'll detach
  // and unlink after the child finishes.
  daiw::ipc::SharedRegion bootstrap;
  if (!bootstrap.create(region_name, sizeof(daiw::ipc::SharedLayout))) {
    std::printf("FAIL: bootstrap create\n");
    return 1;
  }
  bootstrap.detach(); // child will attach freshly; parent re-attaches below.

  pid_t pid = ::fork();
  if (pid < 0) {
    std::printf("FAIL: fork failed\n");
    daiw::ipc::SharedRegion::unlink_name(region_name);
    return 1;
  }
  if (pid == 0) {
    // Child path. Exit with non-zero on any failure.
    int rc = run_child(region_name);
    _exit(rc);
  }

  // Parent path. We already created the region but called detach(); we
  // need to re-attach (not re-create) since the SHM persists by name.
  // Reuse SharedRegion's attach() rather than create() so we don't trip
  // the O_EXCL.
  daiw::ipc::SharedRegion parent_region;
  if (!parent_region.attach(region_name, sizeof(daiw::ipc::SharedLayout),
                            2000)) {
    std::printf("FAIL: parent re-attach\n");
    ::kill(pid, SIGTERM);
    ::waitpid(pid, nullptr, 0);
    daiw::ipc::SharedRegion::unlink_name(region_name);
    return 1;
  }

  // Hand off to run_parent through a forwarding shim that doesn't try to
  // recreate the region. We replicate the run_parent body here using
  // parent_region.
  {
    auto *layout = parent_region.layout();
    check(layout != nullptr, "parent: layout pointer valid");

    layout->state.sequence.fetch_add(1, std::memory_order_acq_rel);
    layout->state.bpm.store(132.5, std::memory_order_relaxed);
    layout->state.playing.store(true, std::memory_order_relaxed);
    layout->state.bar.store(7u, std::memory_order_relaxed);
    layout->state.sequence.fetch_add(1, std::memory_order_release);

    int submitted = 0;
    while (submitted < kCommands) {
      daiw::ipc::Command cmd{};
      cmd.timestamp = static_cast<uint64_t>(submitted) * 1000ull;
      cmd.opcode = 42;
      cmd.channel = 3;
      cmd.data1 = static_cast<uint8_t>(submitted);
      cmd.data2 = 77;
      cmd.data16 = static_cast<uint16_t>(submitted * 3);
      if (layout->cmds_in.push(cmd)) {
        ++submitted;
      } else {
        std::this_thread::yield();
      }
    }
    check(submitted == kCommands, "parent: all commands submitted");

    int events_seen = 0;
    bool trailer_seen = false;
    daiw::ipc::EventMsg trailer{};
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(3000);
    while (std::chrono::steady_clock::now() < deadline &&
           !(events_seen == kCommands && trailer_seen)) {
      auto ev = layout->events_out.pop();
      if (!ev) {
        std::this_thread::yield();
        continue;
      }
      if (ev->code == 1)
        ++events_seen;
      else if (ev->code == 2) {
        trailer = *ev;
        trailer_seen = true;
      }
    }
    check(events_seen == kCommands, "parent: child echoed every command back");
    check(trailer_seen, "parent: child sent RTState trailer event");
    check(trailer.timestamp == static_cast<uint64_t>(132.5 * 100.0),
          "parent: trailer reflects bpm published via SHM");
    check(trailer.data1 == 1, "parent: trailer reflects playing=true");
    check(trailer.data16 == 7, "parent: trailer reflects bar=7");

    int status = 0;
    pid_t r = ::waitpid(pid, &status, 0);
    check(r == pid, "parent: waitpid succeeded");
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "parent: child exited 0");
  }

  parent_region.detach();
  daiw::ipc::SharedRegion::unlink_name(region_name);

  if (g_failures == 0) {
    std::printf("OK: all shared-ipc checks passed\n");
    return 0;
  }
  std::printf("FAIL: %d check(s) failed\n", g_failures);
  return 1;
}
