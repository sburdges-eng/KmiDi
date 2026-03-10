#!/usr/bin/env python3
"""
Morph affect bridge: normalize Sensel Morph (or MIDI CC) to KmiDi affect, optional UMP forward.

Reads from (1) MIDI input (Morph sending CC 40/41/42) or (2) Sensel API. Outputs valence/arousal/
dynamics at ~100 Hz and optionally forwards as UMP. See midi/sensel_kmidi_profile.md and
docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md section 5.
"""

import argparse
import json
import sys
import time

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

# Shared UMP helpers (must match src/midi/AffectUMP.h)
from ump_affect_utils import (
    CC_VALENCE,
    CC_AROUSAL,
    CC_DYNAMICS,
    CONTROL_RATE_HZ,
    float_to_ump32,
    build_ump_cc_v2,
)

# MIDI CC numbers for affect (decimal; same as 0x28, 0x29, 0x2A)
CC_VALENCE_7BIT = 40
CC_AROUSAL_7BIT = 41
CC_DYNAMICS_7BIT = 42

# Sensel Morph sensor bounds (mm, grams) per API docs
SENSEL_WIDTH_MM = 240.0
SENSEL_HEIGHT_MM = 139.0
SENSEL_FORCE_MAX_G = 8192.0


def _cc_to_valence(cc7: int) -> float:
    """Map 7-bit CC (0–127) to valence -1..1 with 64 = 0."""
    return (cc7 - 64) / 64.0 if cc7 <= 127 else 0.0


def _cc_to_01(cc7: int) -> float:
    """Map 7-bit CC (0–127) to 0..1."""
    return cc7 / 127.0 if cc7 <= 127 else 0.0


def _smooth(current: float, target: float, alpha: float) -> float:
    """Exponential moving average."""
    return current * (1 - alpha) + target * alpha


def _parse_cc(message: list) -> tuple[int, int] | None:
    """Return (controller, value) for a MIDI CC message, else None."""
    if len(message) < 3:
        return None
    status = message[0]
    if (status & 0xF0) != 0xB0:
        return None
    return (message[1], message[2])


class AffectState:
    """KmiDi affect: valence (-1..1), arousal (0..1), dynamics (0..1)."""

    __slots__ = ("valence", "arousal", "dynamics")

    def __init__(self, valence: float = 0.0, arousal: float = 0.5, dynamics: float = 0.5):
        self.valence = valence
        self.arousal = arousal
        self.dynamics = dynamics

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dynamics": round(self.dynamics, 4),
        }

    def clamp(self) -> None:
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dynamics = max(0.0, min(1.0, self.dynamics))


def run_midi_in(
    midi_in_port,
    midi_out_port,
    forward_ump: bool,
    jsonl: bool,
    rate_hz: int,
    smooth_alpha: float,
    group: int,
    channel: int,
) -> None:
    """Read CC from MIDI input, update affect state, optionally forward as UMP."""
    state = AffectState()
    dt = 1.0 / rate_hz
    n = 0
    try:
        while True:
            t0 = time.monotonic()
            # Poll MIDI (non-blocking)
            msg = midi_in_port.get_message()
            if msg:
                parsed = _parse_cc(msg[0] if isinstance(msg, tuple) else msg)
                if parsed:
                    cc_num, cc_val = parsed
                    if cc_num == CC_VALENCE_7BIT or cc_num == CC_VALENCE:
                        state.valence = _smooth(state.valence, _cc_to_valence(cc_val), smooth_alpha)
                    elif cc_num == CC_AROUSAL_7BIT or cc_num == CC_AROUSAL:
                        state.arousal = _smooth(state.arousal, _cc_to_01(cc_val), smooth_alpha)
                    elif cc_num == CC_DYNAMICS_7BIT or cc_num == CC_DYNAMICS:
                        state.dynamics = _smooth(state.dynamics, _cc_to_01(cc_val), smooth_alpha)
            state.clamp()

            if forward_ump and midi_out_port is not None:
                v32 = float_to_ump32(state.valence, -1.0, 1.0)
                a32 = float_to_ump32(state.arousal, 0.0, 1.0)
                d32 = float_to_ump32(state.dynamics, 0.0, 1.0)
                for raw in (
                    build_ump_cc_v2(group, channel, CC_VALENCE, v32),
                    build_ump_cc_v2(group, channel, CC_AROUSAL, a32),
                    build_ump_cc_v2(group, channel, CC_DYNAMICS, d32),
                ):
                    if HAS_RTMIDI2:
                        midi_out_port.send_raw(*raw)
                    else:
                        midi_out_port.send_message(list(raw))

            if jsonl:
                print(json.dumps(state.to_dict()), flush=True)

            n += 1
            elapsed = time.monotonic() - t0
            time.sleep(max(0, dt - elapsed))
    except KeyboardInterrupt:
        pass


def run_sensel(
    midi_out_port,
    forward_ump: bool,
    jsonl: bool,
    rate_hz: int,
    smooth_alpha: float,
    group: int,
    channel: int,
) -> None:
    """Poll Sensel API, map contacts to affect, optionally forward as UMP."""
    try:
        import sensel
    except ImportError:
        print(
            "Sensel API not installed. Install from https://github.com/sensel/sensel-api "
            "(sensel-lib-wrappers/sensel-lib-python).",
            file=sys.stderr,
        )
        sys.exit(1)

    state = AffectState()
    dt = 1.0 / rate_hz
    sensel.open_connection()
    try:
        sensel.set_content_control(sensel.SENSEL_CONTACT_MASK)
        sensel.start_scanning()
        n = 0
        while True:
            t0 = time.monotonic()
            frame = sensel.read_frame()
            if frame and frame.contacts:
                # Centroid and max force across contacts
                x_sum = sum(c.x_pos for c in frame.contacts)
                y_sum = sum(c.y_pos for c in frame.contacts)
                force_max = max(c.total_force for c in frame.contacts)
                n_cont = len(frame.contacts)
                x_avg = x_sum / n_cont
                y_avg = y_sum / n_cont
                # Map to KmiDi ranges
                valence_t = (x_avg / SENSEL_WIDTH_MM) * 2.0 - 1.0
                arousal_t = y_avg / SENSEL_HEIGHT_MM
                dynamics_t = min(1.0, force_max / SENSEL_FORCE_MAX_G)
                state.valence = _smooth(state.valence, valence_t, smooth_alpha)
                state.arousal = _smooth(state.arousal, arousal_t, smooth_alpha)
                state.dynamics = _smooth(state.dynamics, dynamics_t, smooth_alpha)
            state.clamp()

            if forward_ump and midi_out_port is not None:
                v32 = float_to_ump32(state.valence, -1.0, 1.0)
                a32 = float_to_ump32(state.arousal, 0.0, 1.0)
                d32 = float_to_ump32(state.dynamics, 0.0, 1.0)
                for raw in (
                    build_ump_cc_v2(group, channel, CC_VALENCE, v32),
                    build_ump_cc_v2(group, channel, CC_AROUSAL, a32),
                    build_ump_cc_v2(group, channel, CC_DYNAMICS, d32),
                ):
                    if HAS_RTMIDI2:
                        midi_out_port.send_raw(*raw)
                    else:
                        midi_out_port.send_message(list(raw))

            if jsonl:
                print(json.dumps(state.to_dict()), flush=True)

            n += 1
            elapsed = time.monotonic() - t0
            time.sleep(max(0, dt - elapsed))
    finally:
        sensel.stop_scanning()
        sensel.close_connection()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Morph affect bridge: MIDI or Sensel API to KmiDi affect, optional UMP."
    )
    ap.add_argument("--midi-in", type=str, default=None, help="MIDI input port (Morph output)")
    ap.add_argument("--midi-out", type=str, default=None, help="Output port for --forward-ump")
    ap.add_argument("--forward-ump", action="store_true", help="Send affect as UMP to --midi-out")
    ap.add_argument("--jsonl", action="store_true", help="Print affect as JSONL")
    ap.add_argument("--sensel", action="store_true", help="Use Sensel API instead of MIDI in")
    ap.add_argument("--rate", type=int, default=CONTROL_RATE_HZ, help="Output rate Hz")
    ap.add_argument("--smooth", type=float, default=0.3, help="Smoothing alpha 0..1")
    ap.add_argument("--group", type=int, default=0, help="UMP group 0..15")
    ap.add_argument("--channel", type=int, default=0, help="UMP channel 0..15")
    ap.add_argument("--list-ports", action="store_true", help="List MIDI ports and exit")
    args = ap.parse_args()

    if args.list_ports:
        if HAS_RTMIDI2:
            in_ports = rtmidi2.get_in_ports()
            out_ports = rtmidi2.get_out_ports()
        elif HAS_RTMIDI:
            mi = rtmidi.MidiIn()
            mo = rtmidi.MidiOut()
            in_ports = mi.get_ports()
            out_ports = mo.get_ports()
            del mi, mo
        else:
            print("Need rtmidi or rtmidi2: pip install rtmidi", file=sys.stderr)
            sys.exit(1)
        print("MIDI input ports:")
        for i, name in enumerate(in_ports):
            print(f"  {i}: {name}")
        print("MIDI output ports:")
        for i, name in enumerate(out_ports):
            print(f"  {i}: {name}")
        return

    if args.forward_ump and not args.midi_out:
        print("--forward-ump requires --midi-out", file=sys.stderr)
        sys.exit(1)
    if not args.sensel and not args.midi_in:
        print("Use --midi-in PORT or --sensel. See --list-ports.", file=sys.stderr)
        sys.exit(1)
    if args.sensel and args.midi_in:
        print("Use either --midi-in or --sensel, not both.", file=sys.stderr)
        sys.exit(1)

    midi_out_port = None
    if args.forward_ump:
        if HAS_RTMIDI2:
            midi_out_port = rtmidi2.MidiOut()
            ports = rtmidi2.get_out_ports()
        elif HAS_RTMIDI:
            midi_out_port = rtmidi.MidiOut()
            ports = midi_out_port.get_ports()
        else:
            print("Need rtmidi or rtmidi2 for --forward-ump", file=sys.stderr)
            sys.exit(1)
        port_index = None
        for i, name in enumerate(ports):
            if args.midi_out in name or name == args.midi_out:
                port_index = i
                break
        if port_index is None:
            try:
                port_index = int(args.midi_out)
            except ValueError:
                pass
        if port_index is None:
            print(f"Output port '{args.midi_out}' not found.", file=sys.stderr)
            sys.exit(1)
        midi_out_port.open_port(port_index)
        print(f"Forwarding UMP to {ports[port_index]}", file=sys.stderr)

    if args.sensel:
        try:
            run_sensel(
                midi_out_port,
                args.forward_ump,
                args.jsonl,
                args.rate,
                args.smooth,
                args.group,
                args.channel,
            )
        except KeyboardInterrupt:
            pass
        finally:
            if midi_out_port is not None:
                midi_out_port.close_port()
        return

    try:
        if HAS_RTMIDI2:
            midi_in = rtmidi2.MidiIn()
            in_ports = rtmidi2.get_in_ports()
        else:
            midi_in = rtmidi.MidiIn()
            in_ports = midi_in.get_ports()
        port_index = None
        for i, name in enumerate(in_ports):
            if args.midi_in in name or name == args.midi_in:
                port_index = i
                break
        if port_index is None:
            try:
                port_index = int(args.midi_in)
            except ValueError:
                pass
        if port_index is None:
            print(f"Input port '{args.midi_in}' not found.", file=sys.stderr)
            sys.exit(1)
        midi_in.open_port(port_index)
        print(f"Reading MIDI from {in_ports[port_index]}", file=sys.stderr)
        try:
            run_midi_in(
                midi_in,
                midi_out_port,
                args.forward_ump,
                args.jsonl,
                args.rate,
                args.smooth,
                args.group,
                args.channel,
            )
        except KeyboardInterrupt:
            pass
        finally:
            midi_in.close_port()
    finally:
        if midi_out_port is not None:
            midi_out_port.close_port()


if __name__ == "__main__":
    main()
