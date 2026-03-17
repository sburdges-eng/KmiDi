// MIDI-CI Microformat-to-SysEx daemon.
// Reads line-delimited JSON from stdin (e.g. {"op":"set","target":"cutoff","val":85}),
// wraps it in MIDI 2.0 / MIDI-CI Property Exchange "Set Property Data" SysEx,
// and sends via libremidi to the default MIDI output port.
//
// Build with: cmake -DBUILD_MIDI_CI_DAEMON=ON ...
// Run: ./midi_ci_daemon  (then feed JSON lines to stdin)
// Or: echo '{"op":"set","target":"cutoff","val":85}' | ./midi_ci_daemon

#include <libremidi/libremidi.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

// MIDI-CI Property Exchange (MMA/AMEI): SysEx header for Universal MIDI Packet / MIDI 1.0.
// Format: F0 7E 00 0D 09 <subtype> [payload] F7
// - 7E 00 = Universal Non-Real-Time, Device 00 (broadcast)
// - 0D = MIDI-CI (Configuration Inquiry)
// - 09 = Property Exchange
// - Subtype 0x03 = Set Property Data Request (we send JSON as resource data)
constexpr uint8_t kSysExStart = 0xF0;
constexpr uint8_t kSysExEnd = 0xF7;
constexpr uint8_t kUniNonRealtime = 0x7E;
constexpr uint8_t kDeviceIdBroadcast = 0x00;
constexpr uint8_t kMidiCI = 0x0D;
constexpr uint8_t kPropertyExchange = 0x09;
constexpr uint8_t kSetPropertyData = 0x03;

// Minimal microformat validation: require "op" and "target" keys (avoids sending garbage).
bool validate_microformat(const std::string& line) {
  return line.find("\"op\"") != std::string::npos && line.find("\"target\"") != std::string::npos;
}

// Build MIDI-CI Property Exchange "Set Property Data" SysEx with JSON payload.
// Payload is the raw JSON string (e.g. {"op":"set","target":"cutoff","val":85}).
std::vector<unsigned char> build_pe_set_property_sysex(const std::string& json_payload) {
  std::vector<unsigned char> out;
  out.reserve(8 + json_payload.size());
  out.push_back(kSysExStart);
  out.push_back(kUniNonRealtime);
  out.push_back(kDeviceIdBroadcast);
  out.push_back(kMidiCI);
  out.push_back(kPropertyExchange);
  out.push_back(kSetPropertyData);
  for (char c : json_payload) {
    out.push_back(static_cast<unsigned char>(c & 0x7F));
  }
  out.push_back(kSysExEnd);
  return out;
}

}  // namespace

int main() {
  libremidi::observer obs;
  std::vector<libremidi::output_port> ports = obs.get_output_ports();
  if (ports.empty()) {
    std::cerr << "midi_ci_daemon: no MIDI output ports found\n";
    return 1;
  }

  libremidi::midi_out out;
  auto err = out.open_port(ports.front());
  if (err != stdx::error{}) {
    std::cerr << "midi_ci_daemon: failed to open MIDI port\n";
    return 1;
  }
  std::cerr << "midi_ci_daemon: opened " << ports.front().display_name << "\n";

  std::string line;
  while (std::getline(std::cin, line)) {
    // Trim trailing CR if present
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    if (!validate_microformat(line)) {
      std::cerr << "midi_ci_daemon: skip (missing op/target): " << line.substr(0, 80) << "\n";
      continue;
    }
    std::vector<unsigned char> sysex = build_pe_set_property_sysex(line);
    auto send_err = out.send_message(sysex.data(), sysex.size());
    if (send_err != stdx::error{}) {
      std::cerr << "midi_ci_daemon: send failed\n";
    }
  }

  return 0;
}
