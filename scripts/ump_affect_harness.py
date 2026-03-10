#!/usr/bin/env python3
"""
UMP affect test harness: stream synthetic valence/arousal/dynamics curves to a MIDI port.

Sends MIDI 2.0 UMP 32-bit CC messages (vendor indices 0x28, 0x29, 0x2A) at ~100 Hz
for DAW verification. See midi/README.md and docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md.

Usage:
  python3 scripts/ump_affect_harness.py --port "KmiDi Virtual" --duration 30 --curve sine
  python3 scripts/ump_affect_harness.py --list-ports

Requires: rtmidi (pip install rtmidi) or rtmidi2 (pip install rtmidi2).
UMP is sent as raw 32-bit words; the port must be UMP-capable (e.g. Logic Pro X with
a virtual port that supports MIDI 2.0, or a loopback). If your driver only accepts
MIDI 1.0, use the fallback: run the Kelly plugin in a host and drive affect from
the Music Brain API (e.g. /audio/valence-arousal) or use a C++ harness that links
AffectUMP and sends to a UMP port.
"""

import argparse
import math
import sys
import time

# rtmidi (MidiOut.send_raw) or rtmidi2 (MidiOut.send_raw)
try:
    import rtmidi2
    HAS_RTMIDI2 = True
except ImportError:
    HAS_RTMIDI2 = False
try:
    import rtmidi
    HAS_RTMIDI = True
except ImportError:
    HAS_RTMIDI = False

from ump_affect_utils import (
    CC_VALENCE,
    CC_AROUSAL,
    CC_DYNAMICS,
    CONTROL_RATE_HZ,
    float_to_ump32,
    build_ump_cc_v2,
)


def curve_sine(t: float, period: float = 4.0, low: float = 0.0, high: float = 1.0) -> float:
    """Sine in [low, high] with given period (seconds)."""
    x = math.sin(2 * math.pi * t / period)
    return low + (high - low) * (x + 1) / 2


def curve_ramp(t: float, period: float = 8.0, low: float = 0.0, high: float = 1.0) -> float:
    """Sawtooth ramp in [low, high] with given period (seconds)."""
    x = (t % period) / period
    return low + (high - low) * x


def curve_step(t: float, period: float = 2.0, low: float = 0.0, high: float = 1.0) -> float:
    """Step: low for first half of period, high for second half."""
    return high if (t % period) >= period / 2 else low


def run_harness(
    port_spec,
    duration: float,
    curve: str,
    group: int = 0,
    channel: int = 0,
    verbose: bool = False,
) -> None:
    """Open port and stream affect curves at CONTROL_RATE_HZ until duration (seconds)."""
    if HAS_RTMIDI2:
        midi_out = rtmidi2.MidiOut()
        ports = rtmidi2.get_out_ports()
    elif HAS_RTMIDI:
        midi_out = rtmidi.MidiOut()
        ports = midi_out.get_ports()
    else:
        print("Need rtmidi or rtmidi2: pip install rtmidi", file=sys.stderr)
        sys.exit(1)

    if not ports:
        print("No MIDI output ports found.", file=sys.stderr)
        sys.exit(1)

    # Resolve port by name or index
    if isinstance(port_spec, int):
        port_index = port_spec
    else:
        port_index = None
        for i, name in enumerate(ports):
            if port_spec in name or (isinstance(port_spec, str) and name == port_spec):
                port_index = i
                break
        if port_index is None:
            print(f"Port '{port_spec}' not found. Available: {ports}", file=sys.stderr)
            sys.exit(1)

    midi_out.open_port(port_index)
    port_name = ports[port_index] if isinstance(ports[port_index], str) else str(port_index)
    print(f"Sending UMP affect to '{port_name}' for {duration}s, curve={curve} @ {CONTROL_RATE_HZ} Hz")

    dt = 1.0 / CONTROL_RATE_HZ
    t0 = time.monotonic()
    n = 0
    try:
        while True:
            t = time.monotonic() - t0
            if t >= duration:
                break
            # Valence -1..1 (sine)
            valence = curve_sine(t, period=5.0, low=-1.0, high=1.0)
            if curve == "ramp":
                arousal = curve_ramp(t, period=6.0)
                dynamics = curve_ramp(t, period=7.0)
            elif curve == "step":
                arousal = curve_step(t, period=2.0)
                dynamics = curve_step(t, period=3.0)
            else:
                arousal = curve_sine(t, period=4.0)
                dynamics = curve_sine(t, period=6.0)

            v32 = float_to_ump32(valence, -1.0, 1.0)
            a32 = float_to_ump32(arousal, 0.0, 1.0)
            d32 = float_to_ump32(dynamics, 0.0, 1.0)

            raw_val = build_ump_cc_v2(group, channel, CC_VALENCE, v32)
            raw_aro = build_ump_cc_v2(group, channel, CC_AROUSAL, a32)
            raw_dyn = build_ump_cc_v2(group, channel, CC_DYNAMICS, d32)

            if HAS_RTMIDI2:
                midi_out.send_raw(*raw_val)
                midi_out.send_raw(*raw_aro)
                midi_out.send_raw(*raw_dyn)
            else:
                midi_out.send_message(list(raw_val))
                midi_out.send_message(list(raw_aro))
                midi_out.send_message(list(raw_dyn))

            n += 1
            if verbose and n % 100 == 0:
                print(f"  t={t:.2f} valence={valence:.3f} arousal={arousal:.3f} dynamics={dynamics:.3f}")
            time.sleep(max(0, dt - (time.monotonic() - t0 - t)))
    finally:
        midi_out.close_port()

    print(f"Sent {n} frames ({n * 3} UMP CC messages)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream synthetic affect curves as UMP 32-bit CC to a MIDI port (see midi/README.md)."
    )
    ap.add_argument("--port", type=str, default=None, help="Output port name or index (required unless --list-ports)")
    ap.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds (default 30)")
    ap.add_argument("--curve", type=str, choices=["sine", "ramp", "step"], default="sine", help="Curve type (default sine)")
    ap.add_argument("--group", type=int, default=0, help="UMP group 0..15 (default 0)")
    ap.add_argument("--channel", type=int, default=0, help="UMP channel 0..15 (default 0)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print sample values every 100 frames")
    ap.add_argument("--list-ports", action="store_true", help="List output ports and exit")
    args = ap.parse_args()

    if args.list_ports:
        if HAS_RTMIDI2:
            ports = rtmidi2.get_out_ports()
        elif HAS_RTMIDI:
            midi_out = rtmidi.MidiOut()
            ports = midi_out.get_ports()
            del midi_out
        else:
            print("Need rtmidi or rtmidi2", file=sys.stderr)
            sys.exit(1)
        for i, name in enumerate(ports):
            print(f"  {i}: {name}")
        return

    if args.port is None:
        print("--port is required (or use --list-ports)", file=sys.stderr)
        sys.exit(1)

    try:
        port_index = int(args.port)
    except ValueError:
        port_index = args.port

    run_harness(
        port_spec=port_index,
        duration=args.duration,
        curve=args.curve,
        group=args.group,
        channel=args.channel,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
