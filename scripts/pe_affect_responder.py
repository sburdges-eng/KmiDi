#!/usr/bin/env python3
"""
Standalone PE responder for com.sburdges.kmidi/affect.v1.

Listens on a MIDI port for MIDI-CI Property Exchange Get Resource (SysEx),
and responds with the affect resource JSON when the request matches
com.sburdges.kmidi/affect.v1. Validates schema and flow without requiring
host/DAW MIDI-CI support. See midi/README.md and docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md.
"""

import argparse
import json
import sys
from pathlib import Path

# Optional: rtmidi for SysEx I/O
try:
    import rtmidi
except ImportError:
    rtmidi = None

# MIDI-CI PE constants (MMA/AMEI)
SYSEX_START = 0xF0
SYSEX_END = 0xF7
UNIVERSAL_REALTIME = 0x7E
DEVICE_ID_BROADCAST = 0x7F
SUBID1_MIDI_CI = 0x0D
SUBID2_PE_GET_INQUIRY = 0x34
SUBID2_PE_GET_REPLY = 0x35
AFFECT_RESOURCE_ID = "com.sburdges.kmidi/affect.v1"


def load_affect_instance(schema_dir: Path) -> dict:
    """Load default affect instance (or from schema dir)."""
    default = {
        "valence": 0.0,
        "arousal": 0.5,
        "dynamics": 0.5,
        "mode": "ride",
    }
    instance_path = schema_dir / "pe_affect_instance.json"
    if instance_path.exists():
        with open(instance_path, encoding="utf-8") as f:
            data = json.load(f)
            return {**default, **data}
    return default


def is_pe_get_request(data: bytes) -> bool:
    """True if this SysEx is MIDI-CI PE Get Inquiry (0x34)."""
    if len(data) < 6 or data[0] != SYSEX_START or data[-1] != SYSEX_END:
        return False
    payload = data[1:-1]
    return (
        len(payload) >= 5
        and payload[0] == UNIVERSAL_REALTIME
        and payload[1] == DEVICE_ID_BROADCAST
        and payload[2] == SUBID1_MIDI_CI
        and payload[3] == SUBID2_PE_GET_INQUIRY
    )


def parse_pe_get_header(data: bytes) -> str | None:
    """Extract requested resource ID from PE Get header if present."""
    # Fixed: version(1), source MUID(4), dest MUID(4), request ID(1) => off 0..13, header at 14
    payload = data[1:-1]
    if len(payload) < 16:
        return None
    off = 14
    header_size = payload[off] | (payload[off + 1] << 7)
    off += 2
    if off + header_size > len(payload):
        return None
    try:
        header_json = bytes(payload[off : off + header_size]).decode("utf-8")
        obj = json.loads(header_json)
        return obj.get("resource") if isinstance(obj, dict) else None
    except (ValueError, UnicodeDecodeError):
        return None


def build_pe_get_reply(
    data: bytes, body_json: str, request_id: int, dest_muid: int, source_muid: int
) -> bytes:
    """Build PE Get Reply (0x35) SysEx with same header layout and body."""
    payload = bytearray(data[1:-1])
    if len(payload) < 14:
        return b""
    # Replace sub-id2 with Reply
    payload[3] = SUBID2_PE_GET_REPLY
    # Request ID at offset 13
    payload[13] = request_id & 0x7F
    body_b = body_json.encode("utf-8")
    # Append property data (single chunk). Full spec has chunk index/count.
    header_size = payload[14] | (payload[15] << 7)
    header_end = 16 + header_size
    new_payload = bytes(payload[:header_end]) + body_b
    return bytes([SYSEX_START]) + new_payload + bytes([SYSEX_END])


def run_responder(port_name: str | None, schema_dir: Path, verbose: bool) -> None:
    """Open MIDI in/out and respond to PE Get for affect.v1."""
    if rtmidi is None:
        print("rtmidi not installed. pip install rtmidi", file=sys.stderr)
        sys.exit(1)
    affect = load_affect_instance(schema_dir)
    body_json = json.dumps(affect, separators=(",", ":"))
    midi_in = rtmidi.MidiIn()
    midi_out = rtmidi.MidiOut()
    ports_in = midi_in.get_ports()
    ports_out = midi_out.get_ports()
    if not ports_in or not ports_out:
        print("No MIDI ports found.", file=sys.stderr)
        sys.exit(1)
    in_port = port_name or ports_in[0]
    out_port = port_name or ports_out[0]
    try:
        idx_in = next(i for i, p in enumerate(ports_in) if port_name in p or p == in_port)
    except StopIteration:
        idx_in = 0
    try:
        idx_out = next(i for i, p in enumerate(ports_out) if port_name in p or p == out_port)
    except StopIteration:
        idx_out = 0
    midi_in.open_port(idx_in)
    midi_out.open_port(idx_out)
    if verbose:
        print(f"Listening on: {ports_in[idx_in]}")
        print(f"Sending replies to: {ports_out[idx_out]}")
        print(f"Resource: {AFFECT_RESOURCE_ID}")
    try:
        while True:
            msg = midi_in.get_message()
            if msg is None:
                continue
            data = msg[0]
            if not is_pe_get_request(data):
                continue
            resource = parse_pe_get_header(data)
            if resource != AFFECT_RESOURCE_ID:
                if verbose and resource:
                    print(f"Ignoring request for resource: {resource}")
                continue
            payload = data[1:-1]
            request_id = payload[13] & 0x7F
            source_muid = payload[5] | (payload[6] << 8) | (payload[7] << 16) | (payload[8] << 24)
            dest_muid = payload[9] | (payload[10] << 8) | (payload[11] << 16) | (payload[12] << 24)
            reply = build_pe_get_reply(data, body_json, request_id, dest_muid, source_muid)
            if reply:
                midi_out.send_message(list(reply))
                if verbose:
                    print(f"Replied with affect resource (request_id={request_id})")
    except KeyboardInterrupt:
        pass
    finally:
        midi_in.close_port()
        midi_out.close_port()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PE responder for com.sburdges.kmidi/affect.v1 (MIDI-CI Property Exchange)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="MIDI port name (substring match) or use first available",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "midi",
        help="Directory containing pe_affect_schema.json / pe_affect_instance.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log requests and replies")
    args = parser.parse_args()
    run_responder(args.port, args.schema_dir, args.verbose)


if __name__ == "__main__":
    main()
